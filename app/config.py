"""
Kani TTS API - Configuration
=============================

Central configuration via Pydantic Settings with environment variable support.
All settings are prefixed with ``KANI_`` (e.g. ``KANI_MODEL_NAME``).

Example .env file::

    KANI_MODEL_NAME=jsbeaudry/haitian-kani-ht-v3
    KANI_DEVICE_MAP=auto
    KANI_USE_FLASH_ATTENTION=true
    KANI_HF_TOKEN=hf_xxxxxxxxxxxxx
    KANI_TEMPERATURE=0.6
    KANI_MAX_NEW_TOKENS=1500
"""

from pydantic import Field
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """
    Application-wide settings loaded from environment variables.

    All fields can be overridden at runtime by setting the corresponding
    ``KANI_<FIELD_NAME>`` environment variable.
    """

    # ── Model identity ──

    model_name: str = Field(
        "nineninesix/kani-tts-400m-0.3-pt",
        description=(
            "HuggingFace repo ID or local filesystem path for the Kani TTS model. "
            "This is loaded at startup and used for all /tts requests."
        ),
    )
    device_map: str = Field(
        "auto",
        description=(
            "PyTorch device mapping strategy passed to "
            "AutoModelForCausalLM.from_pretrained(). "
            "Use 'auto' for automatic GPU/CPU placement, 'cpu' to force CPU, "
            "or 'cuda:0' for a specific GPU."
        ),
    )
    checkpoint_dir: str = Field(
        "./checkpoints",
        description="Default directory for saving training checkpoints and merged models.",
    )
    hf_token: Optional[str] = Field(
        None,
        description=(
            "HuggingFace API token used to push trained models to the Hub. "
            "Required only if using the push_to_hub feature in /train."
        ),
    )

    # ── Inference defaults ──
    # These are used when the /tts request does not provide explicit overrides.

    tokeniser_length: int = Field(
        64400,
        description=(
            "Size of the text tokenizer vocabulary. Audio codec tokens and "
            "special control tokens are indexed starting from this offset. "
            "Must match the tokenizer used by the loaded model."
        ),
    )
    start_of_text: int = Field(
        1,
        description="Token ID marking the beginning of a text sequence.",
    )
    end_of_text: int = Field(
        2,
        description="Token ID marking the end of a text sequence.",
    )
    max_new_tokens: int = Field(
        1200,
        ge=1,
        le=4096,
        description=(
            "Default maximum number of new tokens (text + audio) the model "
            "generates per request. ~1200 tokens produces roughly 10-15 seconds "
            "of audio depending on content."
        ),
    )
    temperature: float = Field(
        0.8,
        ge=0.0,
        le=2.0,
        description=(
            "Default sampling temperature. 0.8 provides a good balance between "
            "expressiveness and coherence."
        ),
    )
    top_p: float = Field(
        0.95,
        ge=0.0,
        le=1.0,
        description="Default nucleus sampling probability threshold.",
    )
    repetition_penalty: float = Field(
        1.1,
        ge=1.0,
        description="Default penalty for repeated tokens. Reduces audio looping artifacts.",
    )
    use_torch_compile: bool = Field(
        False,
        description=(
            "Whether to apply torch.compile() to the model for kernel fusion. "
            "Can improve throughput on supported GPUs (A100+) but adds a "
            "one-time compilation delay on first inference."
        ),
    )
    use_flash_attention: bool = Field(
        True,
        description=(
            "Whether to load the model with Flash Attention 2. "
            "Significantly reduces memory usage and speeds up generation. "
            "Requires the flash-attn package and an Ampere+ GPU."
        ),
    )

    # ── Training defaults ──
    # Used as fallback values; the /train endpoint accepts per-request overrides.

    base_model_id: str = Field(
        "nineninesix/kani-tts-400m-0.3-pt",
        description="Default base model for fine-tuning (can be overridden per training job).",
    )
    lora_r: int = Field(
        16,
        ge=1,
        description="Default LoRA rank.",
    )
    lora_alpha: int = Field(
        16,
        ge=1,
        description="Default LoRA scaling factor.",
    )
    lora_dropout: float = Field(
        0.1,
        ge=0.0,
        le=1.0,
        description="Default LoRA dropout probability.",
    )
    num_train_epochs: int = Field(
        4,
        ge=1,
        description="Default number of training epochs.",
    )
    per_device_train_batch_size: int = Field(
        2,
        ge=1,
        description="Default per-GPU training batch size.",
    )
    gradient_accumulation_steps: int = Field(
        4,
        ge=1,
        description="Default gradient accumulation steps.",
    )
    learning_rate: float = Field(
        5e-5,
        gt=0,
        description="Default peak learning rate.",
    )

    model_config = {"env_prefix": "KANI_"}


settings = Settings()
