"""
Kani TTS - LoRA Fine-Tuning Trainer
=====================================

Orchestrates the complete training pipeline when triggered via ``POST /train``:

    1. **Dataset preparation** -- Load and merge HuggingFace datasets with
       parallel shard processing and frame-level position encoding.
    2. **Model loading** -- Load base model with Flash Attention 2.
    3. **LoRA injection** -- Apply Low-Rank Adaptation to target modules.
    4. **SFT training** -- Run Supervised Fine-Tuning with the custom
       ``FramePosCollator`` that preserves position_ids.
    5. **Merge & save** -- Merge LoRA weights back into the base model
       and save a standalone checkpoint.
    6. **Hub push** (optional) -- Upload the merged model to HuggingFace Hub.

Job Management
--------------

Training jobs run in a background thread (via ``asyncio.to_thread`` in
the FastAPI route). Job state is tracked in the ``training_jobs`` dict:

.. code-block:: python

    training_jobs[job_id] = {
        "status": "starting" | "running" | "completed" | "failed",
        "error": None | "<traceback>",
        "model_path": None | "<path_to_merged_model>",
    }

OmegaConf Bridge
-----------------

The dataset processing pipeline uses OmegaConf for config handling.
Since the API receives Pydantic models, this module provides dataclass
wrappers (``CategoricalFilter``, ``HFDatasetDC``, ``DatasetConfig``)
and a ``_build_dataset_config()`` converter function.
"""

import os
import uuid
import traceback
from dataclasses import dataclass, field
from typing import Optional, List, Dict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig, get_peft_model

from app.schemas import TrainRequest, HFDatasetSchema, CategoricalFilterSchema
from app.training.dataset import DatasetProcessor
from app.training.collator import FramePosCollator


# ---------------------------------------------------------------------------
# In-memory job store
# ---------------------------------------------------------------------------

training_jobs: Dict[str, dict] = {}
"""
Dict mapping job_id -> job state dict.

Keys in each job dict:
    - status (str): "starting", "running", "completed", "failed"
    - error (str | None): Python traceback if failed
    - model_path (str | None): Path to merged model if completed
"""


# ---------------------------------------------------------------------------
# Dataclass wrappers for OmegaConf compatibility
# ---------------------------------------------------------------------------

@dataclass
class CategoricalFilter:
    """OmegaConf-compatible mirror of ``CategoricalFilterSchema``."""
    column_name: str = "speaker"
    value: str = "ex02"


@dataclass
class HFDatasetDC:
    """
    OmegaConf-compatible mirror of ``HFDatasetSchema``.

    Field names and semantics match the Pydantic schema exactly.
    See ``app.schemas.HFDatasetSchema`` for field documentation.
    """
    reponame: str = ""
    name: Optional[str] = None
    split: str = "train"
    text_col_name: str = "text"
    nano_layer_1: str = "nano_layer_1"
    nano_layer_2: str = "nano_layer_2"
    nano_layer_3: str = "nano_layer_3"
    nano_layer_4: str = "nano_layer_4"
    encoded_len: str = "encoded_len"
    speaker_id: Optional[str] = None
    max_len: Optional[int] = None
    categorical_filter: Optional[CategoricalFilter] = None


@dataclass
class DatasetConfig:
    """
    OmegaConf-compatible top-level dataset configuration.

    Attributes:
        max_duration_sec: Global filter -- exclude samples longer than this.
        hf_datasets: List of dataset definitions to merge.
    """
    max_duration_sec: Optional[int] = 30
    hf_datasets: List[HFDatasetDC] = field(default_factory=list)


def _build_dataset_config(req: TrainRequest) -> DatasetConfig:
    """
    Convert a Pydantic ``TrainRequest`` into a ``DatasetConfig`` dataclass
    that can be consumed by ``OmegaConf.structured()``.

    Args:
        req: The incoming training request from the API.

    Returns:
        A DatasetConfig ready for the dataset processing pipeline.
    """
    hf_list = []
    for ds in req.hf_datasets:
        cat_filter = None
        if ds.categorical_filter:
            cat_filter = CategoricalFilter(
                column_name=ds.categorical_filter.column_name,
                value=ds.categorical_filter.value,
            )
        hf_list.append(
            HFDatasetDC(
                reponame=ds.reponame,
                name=ds.name,
                split=ds.split,
                text_col_name=ds.text_col_name,
                nano_layer_1=ds.nano_layer_1,
                nano_layer_2=ds.nano_layer_2,
                nano_layer_3=ds.nano_layer_3,
                nano_layer_4=ds.nano_layer_4,
                encoded_len=ds.encoded_len,
                speaker_id=ds.speaker_id,
                max_len=ds.max_len,
                categorical_filter=cat_filter,
            )
        )
    return DatasetConfig(max_duration_sec=req.max_duration_sec, hf_datasets=hf_list)


# ---------------------------------------------------------------------------
# Training entrypoint
# ---------------------------------------------------------------------------

def run_training(job_id: str, req: TrainRequest) -> None:
    """
    Execute the full LoRA fine-tuning pipeline.

    This function is designed to be called in a background thread via
    ``asyncio.get_event_loop().run_in_executor()``. It updates
    ``training_jobs[job_id]`` as it progresses through each stage.

    Pipeline stages:
        1. Build and preprocess the merged training dataset.
        2. Load the base model with Flash Attention 2.
        3. Inject LoRA adapters into the specified target modules.
        4. Run SFT training with the FramePosCollator.
        5. Merge LoRA weights and save the standalone checkpoint.
        6. (Optional) Push merged model to HuggingFace Hub.

    Args:
        job_id: Unique identifier for this training job (used for status tracking).
        req: The full training configuration from the API request.

    Side effects:
        - Updates ``training_jobs[job_id]["status"]`` through the stages.
        - Writes model files to ``req.output_dir``.
        - On failure, stores the traceback in ``training_jobs[job_id]["error"]``.
    """
    try:
        training_jobs[job_id]["status"] = "running"

        # ── 1) Build dataset ──
        ds_config = _build_dataset_config(req)
        processor = DatasetProcessor(
            ds_config, req.base_model_id, n_shards_per_dataset=req.n_shards_per_dataset
        )
        train_dataset = processor()
        print(f"[{job_id}] Dataset ready: {len(train_dataset)} samples")

        # ── 2) Load base model ──
        tokenizer = AutoTokenizer.from_pretrained(req.base_model_id)
        model = AutoModelForCausalLM.from_pretrained(
            req.base_model_id,
            device_map="auto",
            torch_dtype="bfloat16",
            attn_implementation="flash_attention_2",
        )

        # ── 3) Apply LoRA ──
        lora_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=req.lora_r,
            lora_alpha=req.lora_alpha,
            lora_dropout=req.lora_dropout,
            target_modules=req.target_modules,
            bias="none",
            modules_to_save=None,
            use_rslora=True,
        )
        lora_model = get_peft_model(model, lora_config)

        # ── 4) SFT training ──
        sft_config = SFTConfig(
            num_train_epochs=req.num_train_epochs,
            per_device_train_batch_size=req.per_device_train_batch_size,
            gradient_accumulation_steps=req.gradient_accumulation_steps,
            learning_rate=req.learning_rate,
            lr_scheduler_type=req.lr_scheduler_type,
            warmup_ratio=req.warmup_ratio,
            weight_decay=req.weight_decay,
            optim="adamw_torch",
            overwrite_output_dir=True,
            output_dir=req.output_dir,
            save_strategy="no",
            remove_unused_columns=False,  # Keep position_ids column
        )

        trainer = SFTTrainer(
            model=lora_model,
            args=sft_config,
            train_dataset=train_dataset,
            data_collator=FramePosCollator(),
        )

        print(f"[{job_id}] Starting training...")
        trainer.train()
        print(f"[{job_id}] Training complete!")

        # ── 5) Merge LoRA weights & save ──
        merged_model = lora_model.merge_and_unload()
        merged_path = os.path.join(req.output_dir, "lora_kani_model_ft_exp")
        os.makedirs(merged_path, exist_ok=True)
        merged_model.save_pretrained(merged_path)
        tokenizer.save_pretrained(merged_path)
        print(f"[{job_id}] Merged model saved to {merged_path}")

        # ── 6) Optional Hub upload ──
        if req.hf_token and req.dataset_name:
            merged_model.push_to_hub(req.dataset_name, token=req.hf_token)
            tokenizer.push_to_hub(req.dataset_name, token=req.hf_token)
            print(f"[{job_id}] Uploaded to HuggingFace Hub: {req.dataset_name}")
            training_jobs[job_id]["hub_repo"] = req.dataset_name

        training_jobs[job_id]["status"] = "completed"
        training_jobs[job_id]["model_path"] = merged_path

    except Exception as e:
        training_jobs[job_id]["status"] = "failed"
        training_jobs[job_id]["error"] = traceback.format_exc()
        print(f"[{job_id}] Training failed: {e}")


# ---------------------------------------------------------------------------
# Standalone Hub upload
# ---------------------------------------------------------------------------

def upload_to_hub(model_path: str, hf_token: str, dataset_name: str) -> None:
    """
    Upload a local model checkpoint to the HuggingFace Hub.

    Loads the model and tokenizer from the given path, then pushes both
    to the specified Hub repository. The repo is created automatically
    if it doesn't exist.

    This function is designed to be called in a background thread via
    ``asyncio.to_thread()``.

    Args:
        model_path: Local filesystem path to the checkpoint directory
            (must contain model weights and tokenizer files).
        hf_token: HuggingFace API token with write permissions.
        dataset_name: Target Hub repository ID (e.g. ``user/model-name``).

    Raises:
        FileNotFoundError: If model_path does not exist.
        Exception: On Hub authentication or upload failure.
    """
    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"Model directory not found: {model_path}")

    print(f"Loading model from {model_path} for Hub upload...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="bfloat16",
        device_map="cpu",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    print(f"Uploading to {dataset_name}...")
    model.push_to_hub(dataset_name, token=hf_token)
    tokenizer.push_to_hub(dataset_name, token=hf_token)
    print(f"Upload complete: {dataset_name}")
