"""
Kani TTS API - Request & Response Schemas
==========================================

Pydantic models used for request validation and OpenAPI documentation
across all API endpoints. Organized by domain:

- **Inference**: TTSRequest
- **Training**: TrainRequest, TrainResponse, TrainStatusResponse,
  CategoricalFilterSchema, HFDatasetSchema
- **Model Management**: ModelLoadRequest, ModelLoadResponse
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
    push_to_hub: Optional[str] = Field(
        None,
        description=(
            "HuggingFace Hub repository ID to push the merged model to "
            "after training (e.g. `user/model-name`). Requires KANI_HF_TOKEN "
            "to be set. Omit to skip Hub upload."
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
