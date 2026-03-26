"""
Kani TTS API - Main Application
================================

FastAPI application providing endpoints for:
- Text-to-Speech inference via the Kani TTS model
- LoRA fine-tuning job management
- Model hot-swapping at runtime

The TTS model and NeMo audio codec are loaded once at startup via the
FastAPI lifespan handler, then shared across all request handlers.

Run with:
    uvicorn app.main:app --host 0.0.0.0 --port 8000

Environment variables (all prefixed with KANI_):
    KANI_MODEL_NAME          HuggingFace repo or local path for the TTS model
    KANI_DEVICE_MAP          PyTorch device map strategy (default: "auto")
    KANI_USE_FLASH_ATTENTION Enable Flash Attention 2 (default: true)
    KANI_USE_TORCH_COMPILE   Enable torch.compile (default: false)
    KANI_HF_TOKEN            HuggingFace token for Hub push after training
"""

import asyncio
import io
import uuid

import numpy as np
import scipy.io.wavfile
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from app.config import settings
from app.schemas import (
    TTSRequest,
    TrainRequest,
    TrainResponse,
    TrainStatusResponse,
    ModelLoadRequest,
    HealthResponse,
    ModelLoadResponse,
    HubUploadRequest,
    HubUploadResponse,
    DataPrepRequest,
    DataPrepResponse,
    DataPrepStatusResponse,
)

# ---------------------------------------------------------------------------
# Global model singletons -- populated during the lifespan startup phase.
# ---------------------------------------------------------------------------
kani_model = None
audio_player = None


# ---------------------------------------------------------------------------
# Lifespan: load model on startup, release on shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan event handler.

    On startup:
        1. Initializes the NeMo audio codec (NemoAudioPlayer).
        2. Loads the Kani causal-LM model (KaniModel) with optional
           Flash Attention 2 and torch.compile.

    On shutdown:
        Releases model references so GPU memory is freed.
    """
    global kani_model, audio_player

    from app.models.audio_player import NemoAudioPlayer
    from app.models.kani_model import KaniModel

    print("Loading Kani TTS model...")
    audio_player = NemoAudioPlayer(settings)
    kani_model = KaniModel(settings, audio_player)
    print("Model loaded and ready!")

    yield

    del kani_model, audio_player


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

API_DESCRIPTION = """
## Kani TTS API

A production-ready Text-to-Speech API built on the
[Kani TTS](https://huggingface.co/nineninesix/kani-tts-400m-0.3-pt) model
(~400M params) with NVIDIA NeMo Nano Codec for audio decoding.

### Key features

- **Multi-speaker TTS** -- condition speech generation on a `speaker_id`
  for voice cloning / selection.
- **LoRA fine-tuning** -- launch background training jobs with custom
  HuggingFace datasets and track their status.
- **Model hot-swap** -- load a different checkpoint without restarting.
- **Hub upload** -- push model checkpoints to HuggingFace Hub (auto after
  training or via dedicated endpoint).
- **Frame-level position encoding** -- optimized RoPE positions where all
  4 codec tokens per audio frame share a single position, improving
  long-form coherence and KV-cache efficiency.

### Architecture overview

```
Text prompt
    |
    v
[Tokenizer] --> [Causal LM  (generate)] --> interleaved text + audio tokens
                                                     |
                                                     v
                                    [NeMo Nano Codec (decode)] --> PCM waveform
                                                                      |
                                                                      v
                                                                  WAV response
```

### Audio format

All audio is returned as **16-bit PCM WAV at 22 050 Hz** (mono).
"""

app = FastAPI(
    title="Kani TTS API",
    description=API_DESCRIPTION,
    version="1.0.0",
    lifespan=lifespan,
    openapi_tags=[
        {
            "name": "Inference",
            "description": "Generate speech audio from text input.",
        },
        {
            "name": "Data Preparation",
            "description": (
                "Encode raw audio datasets into NeMo Nano Codec tokens "
                "for Kani TTS training. Runs in the background."
            ),
        },
        {
            "name": "Training",
            "description": (
                "Launch and monitor LoRA fine-tuning jobs. "
                "Training runs in a background thread so the API stays responsive."
            ),
        },
        {
            "name": "Model Management",
            "description": (
                "Hot-swap the loaded model checkpoint at runtime "
                "and upload checkpoints to the HuggingFace Hub."
            ),
        },
        {
            "name": "Health",
            "description": "Liveness / readiness probes.",
        },
    ],
)


# ═══════════════════════════════════════════════════════════════════════════
# Health
# ═══════════════════════════════════════════════════════════════════════════

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Health check",
    description=(
        "Returns the API status and which model is currently loaded. "
        "Use this endpoint for liveness/readiness probes in container orchestrators."
    ),
)
async def health():
    """
    Returns:
        HealthResponse with current model name and load status.
    """
    return HealthResponse(
        status="ok",
        model=settings.model_name,
        model_loaded=kani_model is not None,
    )


# ═══════════════════════════════════════════════════════════════════════════
# TTS Inference
# ═══════════════════════════════════════════════════════════════════════════

@app.post(
    "/tts",
    tags=["Inference"],
    summary="Generate speech from text",
    description=(
        "Accepts a text prompt (and optional speaker ID) and returns a "
        "synthesized **WAV audio file** (16-bit PCM, 22 050 Hz, mono).\n\n"
        "The model uses nucleus sampling with configurable temperature, "
        "top-p, and repetition penalty. If not provided, server defaults "
        "from the `KANI_*` environment variables are used.\n\n"
        "**Speaker conditioning**: pass a `speaker_id` that the model was "
        "fine-tuned with (e.g. `\"0047599005d8\"`) to generate speech in "
        "that speaker's voice."
    ),
    response_class=StreamingResponse,
    responses={
        200: {
            "content": {"audio/wav": {}},
            "description": "Synthesized speech as a WAV file.",
        },
        422: {"description": "Model failed to produce valid audio tokens."},
        503: {"description": "Model not loaded yet (startup in progress)."},
    },
)
async def tts(req: TTSRequest):
    """
    Text-to-Speech generation endpoint.

    Workflow:
        1. Tokenize the text, prepend speaker_id if provided.
        2. Run causal-LM generate with sampling parameters.
        3. Extract audio codec tokens from the generated sequence.
        4. Decode tokens to a PCM waveform via the NeMo Nano Codec.
        5. Pack the waveform into a WAV byte-stream and return it.

    Args:
        req: TTSRequest containing the text prompt and optional overrides.

    Returns:
        StreamingResponse with Content-Type audio/wav.

    Raises:
        HTTPException 503: If the model hasn't finished loading.
        HTTPException 422: If the model output is malformed (e.g. missing
            speech tokens, invalid codec sequence).
    """
    if kani_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    try:
        audio, text = await asyncio.to_thread(
            kani_model.run_model,
            req.text,
            speaker_id=req.speaker_id,
            temperature=req.temperature,
            top_p=req.top_p,
            max_new_tokens=req.max_new_tokens,
            repetition_penalty=req.repetition_penalty,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    # Convert float32 numpy audio to 16-bit PCM WAV
    buf = io.BytesIO()
    audio_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
    scipy.io.wavfile.write(buf, 22050, audio_int16)
    buf.seek(0)

    return StreamingResponse(
        buf,
        media_type="audio/wav",
        headers={"Content-Disposition": 'attachment; filename="output.wav"'},
    )


# ═══════════════════════════════════════════════════════════════════════════
# Data Preparation
# ═══════════════════════════════════════════════════════════════════════════

@app.post(
    "/data/prepare",
    response_model=DataPrepResponse,
    tags=["Data Preparation"],
    summary="Encode audio dataset into NeMo codec tokens",
    description=(
        "Encodes raw audio from a HuggingFace dataset into NeMo Nano Codec "
        "tokens (4 codebook layers), producing a dataset ready for Kani TTS "
        "fine-tuning.\n\n"
        "The source dataset must have an **audio column** (HF audio format) "
        "and a **text column** with transcriptions.\n\n"
        "Processing runs in the background -- the endpoint returns immediately "
        "with a `job_id` that can be polled via `GET /data/prepare/{job_id}`.\n\n"
        "**Output**: A JSON file with `nano_layer_1..4`, `encoded_len`, `text`, "
        "and `speaker` fields per sample. Optionally uploaded to HuggingFace Hub."
    ),
)
async def prepare_data(req: DataPrepRequest):
    """
    Launch a background data preparation job.

    Returns:
        DataPrepResponse with the assigned job_id and initial status.
    """
    from app.training.data_prep import data_prep_jobs, run_data_preparation

    job_id = str(uuid.uuid4())[:8]
    data_prep_jobs[job_id] = {
        "status": "starting",
        "error": None,
        "total": None,
        "processed": 0,
        "failed_samples": 0,
        "output_path": None,
        "hub_repo": None,
    }

    asyncio.get_event_loop().run_in_executor(
        None,
        run_data_preparation,
        job_id,
        req.dataset_name,
        req.split,
        req.text_column,
        req.audio_column,
        req.speaker_column,
        req.speaker_id,
        req.output_dir,
        req.hf_token,
        req.hub_repo,
    )

    return DataPrepResponse(job_id=job_id, status="started")


@app.get(
    "/data/prepare/{job_id}",
    response_model=DataPrepStatusResponse,
    tags=["Data Preparation"],
    summary="Check data preparation job status",
    description=(
        "Poll the status of a data preparation job.\n\n"
        "Returns the number of samples processed so far, the total, "
        "and the output path when completed."
    ),
    responses={
        404: {"description": "No data preparation job found with the given ID."},
    },
)
async def data_prep_status(job_id: str):
    """
    Query the status of a data preparation job.

    Args:
        job_id: The 8-character UUID returned by POST /data/prepare.

    Returns:
        DataPrepStatusResponse with progress info.
    """
    from app.training.data_prep import data_prep_jobs

    if job_id not in data_prep_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = data_prep_jobs[job_id]
    return DataPrepStatusResponse(
        job_id=job_id,
        status=job["status"],
        total=job.get("total"),
        processed=job.get("processed", 0),
        failed_samples=job.get("failed_samples", 0),
        output_path=job.get("output_path"),
        hub_repo=job.get("hub_repo"),
        error=job.get("error"),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════════

@app.post(
    "/train",
    response_model=TrainResponse,
    tags=["Training"],
    summary="Start a LoRA fine-tuning job",
    description=(
        "Launches a background LoRA + SFT training run.\n\n"
        "The request body specifies:\n"
        "- One or more HuggingFace datasets to merge (with optional "
        "speaker IDs and categorical filters).\n"
        "- LoRA hyperparameters (rank, alpha, dropout, target modules).\n"
        "- SFT training config (epochs, batch size, LR schedule, etc.).\n\n"
        "Training runs asynchronously -- the endpoint returns immediately "
        "with a `job_id` that can be polled via `GET /train/{job_id}`.\n\n"
        "After training completes the LoRA weights are merged into the base "
        "model and saved to `output_dir`. If `hf_token` and `dataset_name` "
        "are provided, the merged model is automatically uploaded to the "
        "HuggingFace Hub."
    ),
)
async def train(req: TrainRequest):
    """
    Start a background training job.

    Returns:
        TrainResponse with the assigned job_id and initial status.
    """
    from app.training.trainer import training_jobs, run_training

    job_id = str(uuid.uuid4())[:8]
    training_jobs[job_id] = {"status": "starting", "error": None}

    asyncio.get_event_loop().run_in_executor(None, run_training, job_id, req)

    return TrainResponse(job_id=job_id, status="started")


@app.get(
    "/train/{job_id}",
    response_model=TrainStatusResponse,
    tags=["Training"],
    summary="Check training job status",
    description=(
        "Poll the status of a previously started training job.\n\n"
        "Possible statuses:\n"
        "- `starting` -- job accepted, resources being allocated.\n"
        "- `running`  -- training loop in progress.\n"
        "- `completed` -- model merged and saved successfully.\n"
        "- `failed`   -- an error occurred; check the `error` field for the traceback."
    ),
    responses={
        404: {"description": "No training job found with the given ID."},
    },
)
async def train_status(job_id: str):
    """
    Query the status of a training job by its ID.

    Args:
        job_id: The 8-character UUID returned by POST /train.

    Returns:
        TrainStatusResponse with current status and optional error message.
    """
    from app.training.trainer import training_jobs

    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = training_jobs[job_id]
    return TrainStatusResponse(
        job_id=job_id,
        status=job["status"],
        error=job.get("error"),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Model Management
# ═══════════════════════════════════════════════════════════════════════════

@app.post(
    "/model/load",
    response_model=ModelLoadResponse,
    tags=["Model Management"],
    summary="Hot-swap the loaded model",
    description=(
        "Replace the currently loaded model with a different checkpoint.\n\n"
        "Accepts either:\n"
        "- A **local filesystem path** (e.g. `./checkpoints/lora_kani_model_ft_exp`)\n"
        "- A **HuggingFace repo ID** (e.g. `jsbeaudry/haitian-kani-ht-v3`)\n\n"
        "The old model is unloaded and GPU memory is freed before the new "
        "model is loaded. This endpoint blocks until the new model is ready."
    ),
    responses={
        503: {"description": "Model not initialized (startup hasn't completed)."},
        500: {"description": "Failed to load the requested model."},
    },
)
async def load_model(req: ModelLoadRequest):
    """
    Hot-swap the TTS model at runtime.

    Args:
        req: ModelLoadRequest with the path/repo of the new model.

    Returns:
        Confirmation with the new model name.
    """
    if kani_model is None:
        raise HTTPException(status_code=503, detail="Model not initialized")

    try:
        await asyncio.to_thread(kani_model.reload_model, req.model_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return ModelLoadResponse(status="ok", model=req.model_path)


@app.post(
    "/model/upload",
    response_model=HubUploadResponse,
    tags=["Model Management"],
    summary="Upload a checkpoint to HuggingFace Hub",
    description=(
        "Upload a local model checkpoint (weights + tokenizer) to the "
        "HuggingFace Hub.\n\n"
        "Use this to upload any checkpoint -- for example a model produced "
        "by a training job, or an externally saved checkpoint.\n\n"
        "The Hub repository is created automatically if it doesn't exist. "
        "This endpoint blocks until the upload is complete."
    ),
    responses={
        404: {"description": "Model directory not found at the given path."},
        500: {"description": "Upload failed (auth error, network issue, etc.)."},
    },
)
async def upload_model(req: HubUploadRequest):
    """
    Upload a local model checkpoint to the HuggingFace Hub.

    Args:
        req: HubUploadRequest with model_path, hf_token, and dataset_name.

    Returns:
        HubUploadResponse confirming the upload.

    Raises:
        HTTPException 404: If the model_path directory doesn't exist.
        HTTPException 500: On upload failure.
    """
    from app.training.trainer import upload_to_hub

    try:
        await asyncio.to_thread(
            upload_to_hub, req.model_path, req.hf_token, req.dataset_name
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return HubUploadResponse(
        status="ok",
        repo=req.dataset_name,
        model_path=req.model_path,
    )
