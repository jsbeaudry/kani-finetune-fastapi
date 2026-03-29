"""
Kani TTS API - Main Application
================================

FastAPI application providing endpoints for:
- Text-to-Speech inference via the Kani TTS model (``kani_tts`` library)
- Dataset preparation (NeMo Nano Codec encoding)
- LoRA fine-tuning job management
- Model evaluation (acoustic similarity metrics)
- Model hot-swapping at runtime and HuggingFace Hub upload

The TTS model is loaded once at startup via the ``kani_tts.KaniTTS``
wrapper, which handles tokenizer, causal LM, and NeMo codec internally.

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
import os
import uuid

import numpy as np
import soundfile as sf
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
    EvalRequest,
    EvalResponse,
    EvalStatusResponse,
)

# ---------------------------------------------------------------------------
# Global model singleton -- populated during the lifespan startup phase.
# Uses the kani_tts library which bundles tokenizer + LM + NeMo codec.
# ---------------------------------------------------------------------------
tts_model = None
_current_model_name = None


def _load_kani_model(model_name: str):
    """
    Load (or reload) a KaniTTS model.

    Args:
        model_name: HuggingFace repo ID or local path.

    Returns:
        Loaded KaniTTS instance.
    """
    from kani_tts import KaniTTS
    return KaniTTS(
        model_name,
        suppress_logs=True,
        show_info=True,
    )


# ---------------------------------------------------------------------------
# Lifespan: load model on startup, release on shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan event handler.

    On startup:
        Loads the Kani TTS model via ``kani_tts.KaniTTS``, which
        initializes the causal LM, tokenizer, and NeMo audio codec.

    On shutdown:
        Releases the model reference so GPU memory is freed.
    """
    global tts_model, _current_model_name

    print(f"Loading Kani TTS model: {settings.model_name}")
    tts_model = _load_kani_model(settings.model_name)
    _current_model_name = settings.model_name
    print("Model loaded and ready!")

    yield

    del tts_model


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
            "name": "Evaluation",
            "description": (
                "Evaluate TTS quality by comparing generated audio against "
                "reference recordings using acoustic similarity metrics."
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
        model=_current_model_name or settings.model_name,
        model_loaded=tts_model is not None,
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

    Uses the ``kani_tts.KaniTTS`` library for end-to-end synthesis.

    Workflow:
        1. If ``model`` is specified and differs from the current model,
           hot-swap to the requested model.
        2. Call ``KaniTTS.__call__()`` with the text and optional params.
        3. Pack the float32 numpy audio into a 16-bit PCM WAV stream.

    Args:
        req: TTSRequest containing the text prompt and optional overrides.

    Returns:
        StreamingResponse with Content-Type audio/wav.

    Raises:
        HTTPException 503: If the model hasn't finished loading.
        HTTPException 422: If the model output is malformed.
        HTTPException 500: If model loading fails.
    """
    global tts_model, _current_model_name

    if tts_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    # Hot-swap model if the request specifies a different one
    requested = req.model_name
    if requested and requested != _current_model_name:
        try:
            tts_model = await asyncio.to_thread(_load_kani_model, requested)
            _current_model_name = requested
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to load model '{requested}': {e}",
            )

    # Build kwargs -- only pass overrides that are explicitly set
    gen_kwargs = {}
    if req.speaker_id is not None:
        gen_kwargs["speaker_id"] = req.speaker_id
    if req.temperature is not None:
        gen_kwargs["temperature"] = req.temperature
    if req.top_p is not None:
        gen_kwargs["top_p"] = req.top_p
    if req.max_new_tokens is not None:
        gen_kwargs["max_new_tokens"] = req.max_new_tokens
    if req.repetition_penalty is not None:
        gen_kwargs["repetition_penalty"] = req.repetition_penalty

    try:
        audio, text = await asyncio.to_thread(
            tts_model, req.text, **gen_kwargs
        )
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))

    # Convert float32 numpy audio to 16-bit PCM WAV
    buf = io.BytesIO()
    sf.write(buf, audio, 22050, subtype="PCM_16", format="WAV")
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
    summary="Encode audio dataset(s) into NeMo codec tokens",
    description=(
        "Encodes raw audio from **one or more** HuggingFace datasets into "
        "NeMo Nano Codec tokens (4 codebook layers), merges the results, "
        "and produces a single dataset ready for Kani TTS fine-tuning.\n\n"
        "Each entry in `datasets` can point to a different HF repo with "
        "its own column names and speaker settings.\n\n"
        "Processing runs in the background -- the endpoint returns immediately "
        "with a `job_id` that can be polled via `GET /data/prepare/{job_id}`.\n\n"
        "**Output**: A merged JSON file with `nano_layer_1..4`, `encoded_len`, "
        "`text`, and `speaker` fields per sample. Optionally uploaded to a "
        "single HuggingFace Hub repository."
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
        "current_dataset": None,
        "datasets_done": 0,
        "datasets_total": len(req.datasets),
    }

    # Serialize dataset sources to plain dicts for the background thread
    sources = [ds.model_dump() for ds in req.datasets]

    asyncio.get_event_loop().run_in_executor(
        None,
        run_data_preparation,
        job_id,
        sources,
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
        current_dataset=job.get("current_dataset"),
        datasets_done=job.get("datasets_done", 0),
        datasets_total=job.get("datasets_total", 0),
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

    Loads a new ``KaniTTS`` instance, replacing the current one.

    Args:
        req: ModelLoadRequest with the path/repo of the new model.

    Returns:
        Confirmation with the new model name.
    """
    global tts_model, _current_model_name

    if tts_model is None:
        raise HTTPException(status_code=503, detail="Model not initialized")

    try:
        tts_model = await asyncio.to_thread(_load_kani_model, req.model_path)
        _current_model_name = req.model_path
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


# ═══════════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════════


@app.post(
    "/evaluate",
    response_model=EvalResponse,
    tags=["Evaluation"],
    summary="Evaluate TTS quality against a reference dataset",
    description=(
        "Compares TTS-generated audio against reference (human) recordings "
        "from a HuggingFace dataset using acoustic similarity metrics:\n\n"
        "- **MFCC Cosine Similarity** -- timbre & vocal quality\n"
        "- **Chroma Cosine Similarity** -- pitch & harmonic content\n"
        "- **Spectral Centroid Similarity** -- brightness match\n"
        "- **DTW (Dynamic Time Warping)** -- temporal structure\n"
        "- **Overall Score** -- weighted combination "
        "(MFCC 0.35 + Chroma 0.25 + Spectral 0.15 + DTW 0.25)\n\n"
        "All scores are on a **0-1 scale** where 1.0 = perfect match.\n\n"
        "Processing runs in the background -- returns a `job_id` to poll "
        "via `GET /evaluate/{job_id}`."
    ),
)
async def evaluate(req: EvalRequest):
    """
    Launch a background evaluation job.

    If ``model`` is specified and differs from the current model,
    the model is hot-swapped before evaluation begins.

    Returns:
        EvalResponse with the assigned job_id and initial status.
    """
    global tts_model, _current_model_name
    from app.evaluation.evaluator import eval_jobs, run_evaluation

    if tts_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    # Hot-swap model if requested
    requested = req.model_name
    if requested and requested != _current_model_name:
        try:
            tts_model = await asyncio.to_thread(_load_kani_model, requested)
            _current_model_name = requested
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to load model '{requested}': {e}",
            )

    job_id = str(uuid.uuid4())[:8]
    eval_jobs[job_id] = {
        "status": "starting",
        "error": None,
        "total": None,
        "processed": 0,
        "results": None,
        "summary": None,
    }

    asyncio.get_event_loop().run_in_executor(
        None,
        run_evaluation,
        job_id,
        tts_model,
        req.dataset_name,
        req.split,
        req.text_column,
        req.audio_column,
        req.speaker_column,
        req.speaker_id,
        req.hf_token,
    )

    return EvalResponse(job_id=job_id, status="started")


@app.get(
    "/evaluate/{job_id}",
    response_model=EvalStatusResponse,
    tags=["Evaluation"],
    summary="Check evaluation job status and results",
    description=(
        "Poll the status of an evaluation job.\n\n"
        "When completed, returns per-sample scores and a dataset-level "
        "summary with averaged metrics."
    ),
    responses={
        404: {"description": "No evaluation job found with the given ID."},
    },
)
async def eval_status(job_id: str):
    """
    Query the status and results of an evaluation job.

    Args:
        job_id: The 8-character UUID returned by POST /evaluate.

    Returns:
        EvalStatusResponse with progress, per-sample results, and summary.
    """
    from app.evaluation.evaluator import eval_jobs

    if job_id not in eval_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = eval_jobs[job_id]
    return EvalStatusResponse(
        job_id=job_id,
        status=job["status"],
        total=job.get("total"),
        processed=job.get("processed", 0),
        results=job.get("results"),
        summary=job.get("summary"),
        error=job.get("error"),
    )
