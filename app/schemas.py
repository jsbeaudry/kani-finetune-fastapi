"""
Kani TTS API - Request & Response Schemas
==========================================

Pydantic models used for request validation and OpenAPI documentation
across all API endpoints. Organized by domain:

- **Inference**: TTSRequest
- **Data Preparation**: DataPrepRequest, DataPrepResponse, DataPrepStatusResponse
- **Training**: TrainRequest, TrainResponse, TrainStatusResponse,
  CategoricalFilterSchema, HFDatasetSchema
- **Model Management**: ModelLoadRequest, ModelLoadResponse
- **Hub Upload**: HubUploadRequest, HubUploadResponse
- **Health**: HealthResponse
"""

from pydantic import BaseModel, Field
from typing import Optional, List


# ═══════════════════════════════════════════════════════════════════════════
# Health
# ═══════════════════════════════════════════════════════════════════════════

class HealthResponse(BaseModel):
    """Response from the /health endpoint."""

    status: str = Field(
        ...,
        description="Server status. Always `\"ok\"` if the server is reachable.",
        examples=["ok"],
    )
    model: str = Field(
        ...,
        description="Name or path of the currently configured model.",
        examples=["nineninesix/kani-tts-400m-0.3-pt"],
    )
    model_loaded: bool = Field(
        ...,
        description="Whether the model has finished loading and is ready for inference.",
    )


# ═══════════════════════════════════════════════════════════════════════════
# Inference
# ═══════════════════════════════════════════════════════════════════════════

class TTSRequest(BaseModel):
    """
    Request body for the POST /tts endpoint.

    Only `text` is required -- all other fields fall back to the server's
    default configuration (set via KANI_* environment variables).
    """

    text: str = Field(
        ...,
        min_length=1,
        description="The text to synthesize into speech.",
        examples=[
            "Hello, how are you today?",
            "Gen peyi ki gen bon sistèm siveyans maladi.",
        ],
    )
    speaker_id: Optional[str] = Field(
        None,
        description=(
            "Optional speaker identity string. When provided, it is prepended "
            "to the text as `\"<speaker_id>: <text>\"` to condition the model on "
            "a specific voice. Must match a speaker ID the model was trained with."
        ),
        examples=["0047599005d8", "alice"],
    )
    temperature: Optional[float] = Field(
        None,
        ge=0.0,
        le=2.0,
        description=(
            "Sampling temperature for speech generation. Higher values produce "
            "more varied / expressive speech; lower values are more deterministic. "
            "Default: 0.8."
        ),
    )
    top_p: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description=(
            "Nucleus sampling probability mass. Only the smallest set of tokens "
            "whose cumulative probability exceeds top_p are considered. Default: 0.95."
        ),
    )
    max_new_tokens: Optional[int] = Field(
        None,
        ge=1,
        le=4096,
        description=(
            "Maximum number of new tokens to generate (text + audio). "
            "Longer values allow for longer audio output but increase latency. "
            "Default: 1200."
        ),
    )
    repetition_penalty: Optional[float] = Field(
        None,
        ge=1.0,
        le=3.0,
        description=(
            "Penalty applied to tokens that have already appeared in the sequence. "
            "Values > 1.0 reduce repetitive patterns in the generated audio. "
            "Default: 1.1."
        ),
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "text": "Hello, how are you today?",
                    "speaker_id": "0047599005d8",
                },
                {
                    "text": "Bonjour, comment allez-vous?",
                    "speaker_id": None,
                    "temperature": 0.6,
                    "max_new_tokens": 800,
                },
            ]
        }
    }


# ═══════════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════════

class CategoricalFilterSchema(BaseModel):
    """
    Filter a HuggingFace dataset to include only rows where a
    categorical column matches a specific value.

    Commonly used to extract a single speaker from a multi-speaker dataset.

    Example:
        To keep only rows where the `speaker` column equals `"ex02"`:
        ```json
        {"column_name": "speaker", "value": "ex02"}
        ```
    """

    column_name: str = Field(
        "speaker",
        description="Name of the dataset column to filter on.",
        examples=["speaker", "language", "gender"],
    )
    value: str = Field(
        "ex02",
        description="The value to match in the specified column.",
        examples=["ex02", "spk_001", "en"],
    )


class HFDatasetSchema(BaseModel):
    """
    Configuration for a single HuggingFace dataset to include in training.

    Multiple datasets can be combined in a TrainRequest to enable
    multi-speaker or multi-language fine-tuning.

    The dataset must contain pre-encoded audio codec token columns
    (nano_layer_1 through nano_layer_4) produced by the NeMo Nano Codec.
    """

    reponame: str = Field(
        ...,
        description="HuggingFace dataset repository ID (e.g. `jsbeaudry/kani-pretrain-data`).",
        examples=["jsbeaudry/kani-pretrain-data"],
    )
    name: Optional[str] = Field(
        None,
        description="Dataset configuration/subset name. None for single-config datasets.",
    )
    split: str = Field(
        "train",
        description="Which split to load (`train`, `test`, `validation`).",
    )
    text_col_name: str = Field(
        "text",
        description="Column name containing the text transcription.",
    )
    nano_layer_1: str = Field(
        "nano_layer_1",
        description="Column name for the 1st NeMo codec codebook tokens.",
    )
    nano_layer_2: str = Field(
        "nano_layer_2",
        description="Column name for the 2nd NeMo codec codebook tokens.",
    )
    nano_layer_3: str = Field(
        "nano_layer_3",
        description="Column name for the 3rd NeMo codec codebook tokens.",
    )
    nano_layer_4: str = Field(
        "nano_layer_4",
        description="Column name for the 4th NeMo codec codebook tokens.",
    )
    encoded_len: str = Field(
        "encoded_len",
        description=(
            "Column name containing the encoded audio length (in frames). "
            "Used for duration filtering (frame_count / 12.5 = seconds)."
        ),
    )
    speaker_id: Optional[str] = Field(
        None,
        description=(
            "Speaker identity label. When set, every text sample is prefixed "
            "with `\"<speaker_id>: <text>\"` to condition the model on this voice. "
            "Omit for single-speaker training."
        ),
        examples=["simon", "alice", "0047599005d8"],
    )
    max_len: Optional[int] = Field(
        None,
        ge=1,
        description=(
            "Maximum number of samples to randomly select from this dataset. "
            "Useful for balancing datasets of different sizes. "
            "Omit to use all available samples."
        ),
        examples=[2000, 5000],
    )
    categorical_filter: Optional[CategoricalFilterSchema] = Field(
        None,
        description=(
            "Optional filter to select a subset of rows. "
            "Common use: extracting one speaker from a multi-speaker dataset."
        ),
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "reponame": "jsbeaudry/kani-pretrain-data",
                    "split": "train",
                    "speaker_id": "simon",
                    "max_len": 2000,
                },
            ]
        }
    }


class TrainRequest(BaseModel):
    """
    Request body for the POST /train endpoint.

    Configures the entire training pipeline:
    1. **Dataset** -- which HF datasets to load, merge, and preprocess.
    2. **LoRA** -- adapter hyperparameters.
    3. **SFT** -- supervised fine-tuning schedule.
    4. **Output** -- where to save the merged model and whether to push to Hub.
    """

    # ── Base model ──
    base_model_id: str = Field(
        "nineninesix/kani-tts-400m-0.3-pt",
        description="HuggingFace model ID or local path for the base model to fine-tune.",
    )

    # ── Dataset ──
    hf_datasets: List[HFDatasetSchema] = Field(
        ...,
        min_length=1,
        description=(
            "One or more HuggingFace datasets to merge for training. "
            "Each entry can target a different speaker or language."
        ),
    )
    max_duration_sec: int = Field(
        30,
        ge=1,
        description=(
            "Global maximum audio duration in seconds. Samples longer than "
            "this are filtered out before training. Lower values reduce GPU "
            "memory usage."
        ),
    )
    n_shards_per_dataset: int = Field(
        4,
        ge=1,
        le=32,
        description="Number of parallel shards for dataset preprocessing per dataset.",
    )

    # ── LoRA config ──
    lora_r: int = Field(
        16,
        ge=1,
        description=(
            "LoRA rank (r). Lower values = fewer trainable params & faster "
            "training; higher values = more capacity. Original notebook used 32."
        ),
    )
    lora_alpha: int = Field(
        16,
        ge=1,
        description="LoRA scaling factor. Typically matched to `lora_r`.",
    )
    lora_dropout: float = Field(
        0.1,
        ge=0.0,
        le=1.0,
        description="Dropout probability applied to LoRA layers.",
    )
    target_modules: List[str] = Field(
        ["q_proj", "k_proj", "v_proj", "out_proj"],
        description=(
            "Which model modules to apply LoRA to. The optimized default "
            "targets attention projections only. Add `w1`, `w2`, `w3`, "
            "`in_proj` for broader adaptation (at the cost of speed)."
        ),
    )

    # ── SFT config ──
    num_train_epochs: int = Field(
        4,
        ge=1,
        description="Number of complete passes over the training dataset.",
    )
    per_device_train_batch_size: int = Field(
        2,
        ge=1,
        description="Training batch size per GPU. Reduce if running out of VRAM.",
    )
    gradient_accumulation_steps: int = Field(
        4,
        ge=1,
        description=(
            "Number of forward passes before a weight update. "
            "Effective batch size = per_device_train_batch_size x gradient_accumulation_steps."
        ),
    )
    learning_rate: float = Field(
        5e-5,
        gt=0,
        description="Peak learning rate for the optimizer.",
    )
    lr_scheduler_type: str = Field(
        "cosine",
        description="Learning rate scheduler type (e.g. `cosine`, `linear`, `constant`).",
    )
    warmup_ratio: float = Field(
        0.1,
        ge=0.0,
        le=1.0,
        description="Fraction of total training steps used for LR warmup.",
    )
    weight_decay: float = Field(
        0.02,
        ge=0.0,
        description="Weight decay (L2 regularization) coefficient.",
    )

    # ── Output ──
    output_dir: str = Field(
        "./checkpoints",
        description="Directory where training checkpoints and the final merged model are saved.",
    )

    # ── HuggingFace Hub upload (optional, runs automatically after training) ──
    hf_token: Optional[str] = Field(
        None,
        description=(
            "HuggingFace API token for uploading the merged model to the Hub "
            "after training completes. If omitted, the model is only saved locally."
        ),
        examples=["hf_xxxxxxxxxxxxxxxxxxxx"],
    )
    dataset_name: Optional[str] = Field(
        None,
        description=(
            "HuggingFace Hub repository ID to upload the merged model to "
            "(e.g. `user/model-name`). Required if `hf_token` is provided. "
            "The repo will be created automatically if it doesn't exist."
        ),
        examples=["jsbeaudry/haitian-kani-ht-v3"],
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "base_model_id": "nineninesix/kani-tts-400m-0.3-pt",
                    "hf_datasets": [
                        {
                            "reponame": "jsbeaudry/kani-pretrain-data",
                            "split": "train",
                        }
                    ],
                    "max_duration_sec": 30,
                    "num_train_epochs": 4,
                    "lora_r": 16,
                    "output_dir": "./checkpoints",
                    "hf_token": "hf_xxxxxxxxxxxxxxxxxxxx",
                    "dataset_name": "jsbeaudry/haitian-kani-ht-v3",
                }
            ]
        }
    }


class TrainResponse(BaseModel):
    """Response returned immediately when a training job is submitted."""

    job_id: str = Field(
        ...,
        description="Unique 8-character identifier for this training job.",
        examples=["a1b2c3d4"],
    )
    status: str = Field(
        ...,
        description="Initial status of the job (always `\"started\"`).",
        examples=["started"],
    )


class TrainStatusResponse(BaseModel):
    """Response from GET /train/{job_id} showing current training progress."""

    job_id: str = Field(
        ...,
        description="The training job identifier.",
        examples=["a1b2c3d4"],
    )
    status: str = Field(
        ...,
        description=(
            "Current job status: `starting`, `running`, `completed`, or `failed`."
        ),
        examples=["running", "completed", "failed"],
    )
    error: Optional[str] = Field(
        None,
        description=(
            "Python traceback if the job failed. `null` for non-failed jobs."
        ),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Model Management
# ═══════════════════════════════════════════════════════════════════════════

class ModelLoadRequest(BaseModel):
    """Request body for the POST /model/load endpoint."""

    model_path: str = Field(
        ...,
        description=(
            "Path to the model to load. Can be a local filesystem path "
            "(e.g. `./checkpoints/lora_kani_model_ft_exp`) or a HuggingFace "
            "repo ID (e.g. `jsbeaudry/haitian-kani-ht-v3`)."
        ),
        examples=[
            "./checkpoints/lora_kani_model_ft_exp",
            "jsbeaudry/haitian-kani-ht-v3",
        ],
    )


class ModelLoadResponse(BaseModel):
    """Response from POST /model/load after successfully swapping the model."""

    status: str = Field(
        ...,
        description="Operation status.",
        examples=["ok"],
    )
    model: str = Field(
        ...,
        description="The model path/repo that is now loaded.",
        examples=["jsbeaudry/haitian-kani-ht-v3"],
    )


# ═══════════════════════════════════════════════════════════════════════════
# Hub Upload
# ═══════════════════════════════════════════════════════════════════════════

class HubUploadRequest(BaseModel):
    """
    Request body for the POST /model/upload endpoint.

    Uploads a local model checkpoint (model weights + tokenizer) to the
    HuggingFace Hub.
    """

    model_path: str = Field(
        ...,
        description=(
            "Local filesystem path to the model checkpoint directory. "
            "Must contain the model weights and tokenizer files "
            "(e.g. `./checkpoints/lora_kani_model_ft_exp`)."
        ),
        examples=["./checkpoints/lora_kani_model_ft_exp"],
    )
    hf_token: str = Field(
        ...,
        description="HuggingFace API token with write access.",
        examples=["hf_xxxxxxxxxxxxxxxxxxxx"],
    )
    dataset_name: str = Field(
        ...,
        description=(
            "HuggingFace Hub repository ID to upload the model to "
            "(e.g. `user/model-name`). Created automatically if it doesn't exist."
        ),
        examples=["jsbeaudry/haitian-kani-ht-v3"],
    )


class HubUploadResponse(BaseModel):
    """Response from POST /model/upload after a successful Hub upload."""

    status: str = Field(
        ...,
        description="Operation status.",
        examples=["ok"],
    )
    repo: str = Field(
        ...,
        description="The HuggingFace Hub repo the model was uploaded to.",
        examples=["jsbeaudry/haitian-kani-ht-v3"],
    )
    model_path: str = Field(
        ...,
        description="The local path that was uploaded.",
        examples=["./checkpoints/lora_kani_model_ft_exp"],
    )


# ═══════════════════════════════════════════════════════════════════════════
# Data Preparation
# ═══════════════════════════════════════════════════════════════════════════

class DatasetSourceSchema(BaseModel):
    """
    A single HuggingFace dataset source for data preparation.

    Each source specifies which HF dataset to load and how its columns
    are named. Multiple sources can be combined in a single data
    preparation job to produce one merged encoded dataset.
    """

    dataset_name: str = Field(
        ...,
        description=(
            "HuggingFace dataset repository ID containing raw audio "
            "(e.g. `mozilla-foundation/common_voice_17_0`)."
        ),
        examples=["jsbeaudry/my-audio-dataset"],
    )
    split: str = Field(
        "train",
        description="Dataset split to encode (`train`, `test`, `validation`).",
    )
    audio_column: str = Field(
        "audio",
        description="Name of the column containing audio data (HF audio format).",
    )
    text_column: str = Field(
        "text",
        description="Name of the column containing text transcriptions.",
    )
    speaker_column: Optional[str] = Field(
        None,
        description=(
            "Name of the column containing speaker IDs. "
            "If set, each sample's speaker value is read from this column. "
            "Ignored if `speaker_id` is also provided."
        ),
        examples=["speaker_id", "speaker"],
    )
    speaker_id: Optional[str] = Field(
        None,
        description=(
            "Fixed speaker ID to assign to ALL samples in this dataset. "
            "Use this for single-speaker datasets. "
            "Overrides `speaker_column` if both are provided."
        ),
        examples=["alice", "0047599005d8"],
    )


class DataPrepRequest(BaseModel):
    """
    Request body for the POST /data/prepare endpoint.

    Encodes raw audio from one or more HuggingFace datasets into NeMo
    Nano Codec tokens, producing a single merged dataset ready for
    Kani TTS fine-tuning.

    Each entry in ``datasets`` specifies a HF dataset and its column
    mappings. All encoded samples are merged into one output file and
    optionally uploaded to a single HuggingFace Hub repository.
    """

    datasets: list[DatasetSourceSchema] = Field(
        ...,
        min_length=1,
        description=(
            "One or more HuggingFace dataset sources to encode and merge. "
            "Each source can have different column names and speaker settings."
        ),
    )
    output_dir: str = Field(
        "./encoded_data",
        description="Local directory to save the encoded JSON output.",
    )

    # Optional Hub upload
    hf_token: Optional[str] = Field(
        None,
        description=(
            "HuggingFace API token. When provided along with `hub_repo`, "
            "the merged encoded dataset is automatically uploaded to the Hub."
        ),
        examples=["hf_xxxxxxxxxxxxxxxxxxxx"],
    )
    hub_repo: Optional[str] = Field(
        None,
        description=(
            "HuggingFace Hub repo ID to upload the merged encoded dataset to "
            "(e.g. `user/encoded-dataset`). Requires `hf_token`."
        ),
        examples=["jsbeaudry/kani-pretrain-data"],
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "datasets": [
                        {
                            "dataset_name": "jsbeaudry/dataset-a",
                            "split": "train",
                            "audio_column": "audio",
                            "text_column": "text",
                            "speaker_id": "alice",
                        },
                        {
                            "dataset_name": "jsbeaudry/dataset-b",
                            "split": "train",
                            "audio_column": "audio",
                            "text_column": "sentence",
                            "speaker_column": "speaker_id",
                        },
                    ],
                    "output_dir": "./encoded_data",
                    "hf_token": "hf_xxxxxxxxxxxxxxxxxxxx",
                    "hub_repo": "jsbeaudry/kani-pretrain-data",
                }
            ]
        }
    }


class DataPrepResponse(BaseModel):
    """Response returned immediately when a data preparation job is submitted."""

    job_id: str = Field(
        ...,
        description="Unique 8-character identifier for this data prep job.",
        examples=["e5f6g7h8"],
    )
    status: str = Field(
        ...,
        description="Initial status of the job (always `\"started\"`).",
        examples=["started"],
    )


class DataPrepStatusResponse(BaseModel):
    """Response from GET /data/prepare/{job_id} showing encoding progress."""

    job_id: str = Field(
        ...,
        description="The data preparation job identifier.",
        examples=["e5f6g7h8"],
    )
    status: str = Field(
        ...,
        description="Current job status: `starting`, `running`, `completed`, or `failed`.",
        examples=["running", "completed", "failed"],
    )
    current_dataset: Optional[str] = Field(
        None,
        description="The dataset currently being processed (None when completed/failed).",
    )
    datasets_done: int = Field(
        0,
        description="Number of datasets fully processed so far.",
    )
    datasets_total: int = Field(
        0,
        description="Total number of datasets to process.",
    )
    total: Optional[int] = Field(
        None,
        description="Total number of samples across all datasets.",
    )
    processed: int = Field(
        0,
        description="Number of samples successfully encoded so far (across all datasets).",
    )
    failed_samples: int = Field(
        0,
        description="Number of samples that failed to encode (across all datasets).",
    )
    output_path: Optional[str] = Field(
        None,
        description="Path to the merged output JSON file (set when completed).",
    )
    hub_repo: Optional[str] = Field(
        None,
        description="Hub repo the merged encoded dataset was uploaded to (if applicable).",
    )
    error: Optional[str] = Field(
        None,
        description="Python traceback if the job failed.",
    )
