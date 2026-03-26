"""
Kani TTS - Dataset Preparation (Audio Encoding)
=================================================

Encodes raw audio from a HuggingFace dataset into NeMo Nano Codec tokens,
producing a dataset ready for Kani TTS fine-tuning.

Pipeline Overview
-----------------

::

    HuggingFace Dataset (with audio column)
        |
        v
    [Resample to 22050 Hz]
        |
        v
    [NeMo Nano Codec encode] --> 4 codebook token layers + encoded_len
        |
        v
    [Attach text + speaker metadata]
        |
        v
    Encoded JSON  -->  (optionally pushed to HuggingFace Hub)

The NeMo Nano Codec operates at:
    - Sample rate: 22 050 Hz
    - Bitrate: 0.6 kbps
    - Frame rate: 12.5 fps
    - Codebooks: 4 layers x 4032 codes each

Output Format
-------------

Each encoded sample contains::

    {
        "text": "transcription...",
        "speaker": "speaker_id",
        "nano_layer_1": [int, int, ...],
        "nano_layer_2": [int, int, ...],
        "nano_layer_3": [int, int, ...],
        "nano_layer_4": [int, int, ...],
        "encoded_len": 125
    }

This output format is directly compatible with the Kani TTS training
pipeline (``app.training.dataset.DatasetProcessor``).
"""

import io
import json
import os
import warnings
import traceback
from typing import Dict

import librosa
import numpy as np
import soundfile as sf
import torch

from nemo.collections.tts.models import AudioCodecModel

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# In-memory job store for data preparation jobs
# ---------------------------------------------------------------------------

data_prep_jobs: Dict[str, dict] = {}
"""
Dict mapping job_id -> job state dict.

Keys in each job dict:
    - status (str): "starting", "running", "completed", "failed"
    - error (str | None): Python traceback if failed
    - total (int | None): Total samples in the dataset
    - processed (int): Number of successfully encoded samples
    - failed_samples (int): Number of samples that failed to encode
    - output_path (str | None): Path to the output JSON file
    - hub_repo (str | None): Hub repo ID if uploaded
"""

# ---------------------------------------------------------------------------
# Codec model singleton
# ---------------------------------------------------------------------------

_codec_model = None
_codec_device = None


def _get_codec_model():
    """
    Load the NeMo Nano Codec model (singleton).

    The model is loaded once on first call and cached for subsequent uses.
    Runs on CUDA if available, otherwise CPU.

    The discriminator is intentionally skipped -- it is only needed for
    codec training, not for encode/decode. Skipping it avoids the
    ``torch.load`` CVE-2025-32434 error on PyTorch < 2.6.

    Returns:
        Tuple of (codec_model, device_string).
    """
    global _codec_model, _codec_device
    if _codec_model is None:
        print("Loading NeMo Nano Codec for data preparation...")

        # Load config first, disable discriminator, then restore
        _codec_cfg = AudioCodecModel.from_pretrained(
            "nvidia/nemo-nano-codec-22khz-0.6kbps-12.5fps",
            return_config=True,
        )
        _codec_cfg.discriminator = None
        _codec_model = AudioCodecModel.from_pretrained(
            "nvidia/nemo-nano-codec-22khz-0.6kbps-12.5fps",
            override_config_path=_codec_cfg,
            strict=False,
            map_location="cpu",
        )
        _codec_model.eval()

        # Try CUDA first; fall back to CPU if the GPU arch is unsupported
        if torch.cuda.is_available():
            try:
                _codec_model.to("cuda")
                # Smoke-test: run a tiny tensor op to catch arch mismatches early
                _test = torch.zeros(1, device="cuda") + 1
                del _test
                _codec_device = "cuda"
            except RuntimeError:
                print("CUDA kernel not available for this GPU -- falling back to CPU")
                _codec_model.to("cpu")
                _codec_device = "cpu"
        else:
            _codec_device = "cpu"

        print(f"NeMo Codec loaded on {_codec_device}")
    return _codec_model, _codec_device


SAMPLE_RATE = 22050


def load_audio_from_raw(audio_value) -> np.ndarray:
    """
    Load audio from a HuggingFace dataset audio column (raw / undecoded).

    When the audio column is cast with ``decode=False``, each row contains::

        {"bytes": b"...", "path": "filename.mp3"}

    We decode the bytes with ``soundfile`` (or ``librosa`` as fallback)
    and resample to 22 050 Hz. This avoids the ``torchcodec`` dependency
    that newer ``datasets`` versions require for auto-decoding.

    Also handles the legacy decoded format (dict with ``array`` key) as a
    fallback for datasets that are already decoded.

    Args:
        audio_value: The raw audio column value from a HF dataset row.

    Returns:
        Float32 numpy array at 22050 Hz.
    """
    # Legacy decoded format: {"array": np.ndarray, "sampling_rate": int}
    if isinstance(audio_value, dict) and "array" in audio_value:
        audio_array = np.array(audio_value["array"], dtype=np.float32)
        sr = audio_value["sampling_rate"]
        if sr != SAMPLE_RATE:
            audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=SAMPLE_RATE)
        return audio_array

    # Raw / undecoded format: {"bytes": b"...", "path": "..."}
    raw_bytes = audio_value.get("bytes") if isinstance(audio_value, dict) else None
    if raw_bytes is None:
        raise ValueError(
            f"Unsupported audio format: expected dict with 'bytes' or 'array' key, "
            f"got {type(audio_value)}"
        )

    buf = io.BytesIO(raw_bytes)

    # Try soundfile first, fall back to librosa for non-WAV formats
    try:
        audio_array, sr = sf.read(buf, dtype="float32")
    except Exception:
        buf.seek(0)
        audio_array, sr = librosa.load(buf, sr=None, mono=True)

    # Convert stereo to mono if needed
    if audio_array.ndim > 1:
        audio_array = audio_array.mean(axis=1)

    if sr != SAMPLE_RATE:
        audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=SAMPLE_RATE)

    return audio_array.astype(np.float32)


def encode_audio(audio_value, codec_model, device: str) -> dict:
    """
    Encode a single audio sample into 4 NeMo codec token layers.

    Args:
        audio_value: HF dataset audio column value (raw bytes or decoded dict).
        codec_model: Loaded NeMo AudioCodecModel.
        device: Device string ("cuda" or "cpu").

    Returns:
        Dict with keys: nano_layer_1..4 (list[int]), encoded_len (int).

    Raises:
        Exception: On codec encoding failure.
    """
    audio = load_audio_from_raw(audio_value)

    # NeMo codec encode() expects [B, T] (batch, time) -- NOT [B, C, T]
    audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(device)
    audio_len = torch.tensor([audio_tensor.shape[-1]], dtype=torch.int32).to(device)

    with torch.inference_mode():
        tokens, tokens_len = codec_model.encode(audio=audio_tensor, audio_len=audio_len)

    # tokens shape: [B, num_codebooks, num_frames] -> squeeze batch
    tokens_np = tokens.cpu().numpy().squeeze(0).astype(int).tolist()

    return {
        "nano_layer_1": tokens_np[0],
        "nano_layer_2": tokens_np[1],
        "nano_layer_3": tokens_np[2],
        "nano_layer_4": tokens_np[3],
        "encoded_len": int(tokens_len.cpu().numpy()[0]),
    }


def run_data_preparation(
    job_id: str,
    dataset_sources: list[dict],
    output_dir: str,
    hf_token: str | None,
    hub_repo: str | None,
) -> None:
    """
    Encode one or more HuggingFace datasets into NeMo codec tokens and
    merge the results into a single output.

    Designed to run in a background thread via ``asyncio.to_thread()``.

    Steps:
        1. For each dataset source: load, encode audio, attach metadata.
        2. Merge all encoded samples into one list.
        3. Save merged results as a single JSON file.
        4. (Optional) Push the merged dataset to HuggingFace Hub.

    Args:
        job_id: Unique job identifier for status tracking.
        dataset_sources: List of dicts, each containing:
            - dataset_name (str): HF dataset repo ID.
            - split (str): Dataset split.
            - audio_column (str): Audio column name.
            - text_column (str): Text column name.
            - speaker_column (str | None): Speaker ID column.
            - speaker_id (str | None): Fixed speaker ID.
        output_dir: Directory to write the output JSON file.
        hf_token: HF API token for Hub upload (or None to skip).
        hub_repo: HF Hub repo ID for the merged encoded dataset (or None).

    Side effects:
        Updates ``data_prep_jobs[job_id]`` throughout processing.
    """
    from datasets import load_dataset, Dataset, Audio

    try:
        job = data_prep_jobs[job_id]
        job["status"] = "running"
        job["datasets_total"] = len(dataset_sources)

        # Load codec once for all datasets
        codec_model, device = _get_codec_model()

        all_results = []
        total_samples = 0
        total_failed = 0

        for ds_idx, src in enumerate(dataset_sources):
            ds_name = src["dataset_name"]
            split = src["split"]
            audio_col = src["audio_column"]
            text_col = src["text_column"]
            spk_col = src.get("speaker_column")
            spk_id = src.get("speaker_id")

            job["current_dataset"] = ds_name

            # 1) Load dataset -- disable audio auto-decoding to avoid
            #    the torchcodec dependency.  We decode with soundfile.
            print(f"[{job_id}] [{ds_idx+1}/{len(dataset_sources)}] Loading: {ds_name} (split={split})")
            dataset = load_dataset(ds_name, split=split)
            dataset = dataset.cast_column(audio_col, Audio(decode=False))
            ds_len = len(dataset)
            total_samples += ds_len
            job["total"] = total_samples
            print(f"[{job_id}] Loaded {ds_len} samples from {ds_name}")

            # 2) Encode all samples in this dataset
            ds_failed = 0
            for idx, sample in enumerate(dataset):
                try:
                    encoded = encode_audio(sample[audio_col], codec_model, device)

                    # Attach metadata
                    encoded["text"] = sample.get(text_col, "")

                    if spk_id is not None:
                        encoded["speaker"] = spk_id
                    elif spk_col and spk_col in sample:
                        encoded["speaker"] = sample[spk_col]
                    else:
                        encoded["speaker"] = "anon"

                    all_results.append(encoded)

                    if (idx + 1) % 50 == 0 or idx == ds_len - 1:
                        job["processed"] = len(all_results)
                        print(f"[{job_id}] [{ds_name}] Encoded {idx+1}/{ds_len} samples")

                except Exception as e:
                    ds_failed += 1
                    total_failed += 1
                    if total_failed <= 3:
                        print(f"[{job_id}] Error encoding {ds_name} sample {idx}:\n{traceback.format_exc()}")
                    else:
                        print(f"[{job_id}] Error encoding {ds_name} sample {idx}: {e}")

            job["datasets_done"] = ds_idx + 1
            job["failed_samples"] = total_failed
            print(f"[{job_id}] Finished {ds_name}: {ds_len - ds_failed} encoded, {ds_failed} failed")

        job["processed"] = len(all_results)
        job["current_dataset"] = None

        # 3) Save merged JSON
        os.makedirs(output_dir, exist_ok=True)
        if hub_repo:
            safe_name = hub_repo.replace("/", "-")
        else:
            safe_name = "-".join(
                s["dataset_name"].replace("/", "-") for s in dataset_sources
            )
        output_path = os.path.join(output_dir, f"{safe_name}-merged-nemo-encoded.json")

        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        job["output_path"] = output_path
        print(f"[{job_id}] Saved {len(all_results)} merged samples to {output_path}")

        # 4) Optional: push merged dataset to Hub
        if hf_token and hub_repo:
            print(f"[{job_id}] Uploading merged dataset to {hub_repo}...")
            encoded_dataset = Dataset.from_list(all_results)
            encoded_dataset.push_to_hub(hub_repo, split="train", token=hf_token)
            job["hub_repo"] = hub_repo
            print(f"[{job_id}] Uploaded to {hub_repo}")

        job["status"] = "completed"
        print(
            f"[{job_id}] Data preparation complete: "
            f"{len(all_results)}/{total_samples} samples from "
            f"{len(dataset_sources)} dataset(s)"
        )

    except Exception as e:
        data_prep_jobs[job_id]["status"] = "failed"
        data_prep_jobs[job_id]["error"] = traceback.format_exc()
        print(f"[{job_id}] Data preparation failed: {e}")
