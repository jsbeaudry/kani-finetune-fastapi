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

import json
import os
import warnings
import traceback
from typing import Dict

import librosa
import numpy as np
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

    Returns:
        Tuple of (codec_model, device_string).
    """
    global _codec_model, _codec_device
    if _codec_model is None:
        print("Loading NeMo Nano Codec for data preparation...")
        _codec_device = "cuda" if torch.cuda.is_available() else "cpu"
        _codec_model = (
            AudioCodecModel
            .from_pretrained("nvidia/nemo-nano-codec-22khz-0.6kbps-12.5fps")
            .eval()
            .to(_codec_device)
        )
        print(f"NeMo Codec loaded on {_codec_device}")
    return _codec_model, _codec_device


SAMPLE_RATE = 22050


def load_audio_from_array(audio_dict: dict) -> np.ndarray:
    """
    Extract and resample audio from a HuggingFace dataset audio column.

    HF datasets store audio as::

        {"array": np.ndarray, "sampling_rate": int, "path": str}

    This function extracts the array and resamples to 22050 Hz if needed.

    Args:
        audio_dict: The audio column value from a HF dataset row.

    Returns:
        Float32 numpy array at 22050 Hz.
    """
    audio_array = audio_dict["array"]
    sr = audio_dict["sampling_rate"]

    if sr != SAMPLE_RATE:
        audio_array = librosa.resample(
            np.array(audio_array, dtype=np.float32), orig_sr=sr, target_sr=SAMPLE_RATE
        )

    return np.array(audio_array, dtype=np.float32)


def encode_audio(audio_dict: dict, codec_model, device: str) -> dict:
    """
    Encode a single audio sample into 4 NeMo codec token layers.

    Args:
        audio_dict: HF dataset audio column value.
        codec_model: Loaded NeMo AudioCodecModel.
        device: Device string ("cuda" or "cpu").

    Returns:
        Dict with keys: nano_layer_1..4 (list[int]), encoded_len (int).

    Raises:
        Exception: On codec encoding failure.
    """
    audio = load_audio_from_array(audio_dict)

    # NeMo codec expects [B, C, T] tensor
    audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
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
    dataset_name: str,
    split: str,
    text_column: str,
    audio_column: str,
    speaker_column: str | None,
    speaker_id: str | None,
    output_dir: str,
    hf_token: str | None,
    hub_repo: str | None,
) -> None:
    """
    Encode an entire HuggingFace dataset into NeMo codec tokens.

    Designed to run in a background thread via ``asyncio.to_thread()``.

    Steps:
        1. Load the HF dataset.
        2. Load / reuse the NeMo codec model.
        3. Iterate over all samples, encode audio, attach text/speaker metadata.
        4. Save results as JSON.
        5. (Optional) Push the encoded dataset to HuggingFace Hub.

    Args:
        job_id: Unique job identifier for status tracking.
        dataset_name: HF dataset repo ID (e.g. ``user/dataset``).
        split: Dataset split to process.
        text_column: Name of the text transcription column.
        audio_column: Name of the audio column.
        speaker_column: Name of the speaker ID column (or None).
        speaker_id: Fixed speaker ID to assign to all samples (or None).
        output_dir: Directory to write the output JSON file.
        hf_token: HF API token for Hub upload (or None to skip).
        hub_repo: HF Hub repo ID for the encoded dataset (or None).

    Side effects:
        Updates ``data_prep_jobs[job_id]`` throughout processing.
    """
    from datasets import load_dataset, Dataset

    try:
        data_prep_jobs[job_id]["status"] = "running"

        # 1) Load dataset
        print(f"[{job_id}] Loading dataset: {dataset_name} (split={split})")
        dataset = load_dataset(dataset_name, split=split)
        total = len(dataset)
        data_prep_jobs[job_id]["total"] = total
        print(f"[{job_id}] Loaded {total} samples")

        # 2) Load codec
        codec_model, device = _get_codec_model()

        # 3) Encode all samples
        results = []
        failed_count = 0

        for idx, sample in enumerate(dataset):
            try:
                encoded = encode_audio(sample[audio_column], codec_model, device)

                # Attach metadata
                encoded["text"] = sample.get(text_column, "")

                if speaker_id is not None:
                    encoded["speaker"] = speaker_id
                elif speaker_column and speaker_column in sample:
                    encoded["speaker"] = sample[speaker_column]
                else:
                    encoded["speaker"] = "anon"

                results.append(encoded)

                if (idx + 1) % 50 == 0 or idx == total - 1:
                    data_prep_jobs[job_id]["processed"] = len(results)
                    print(f"[{job_id}] Encoded {len(results)}/{total} samples")

            except Exception as e:
                failed_count += 1
                print(f"[{job_id}] Error encoding sample {idx}: {e}")

        data_prep_jobs[job_id]["processed"] = len(results)
        data_prep_jobs[job_id]["failed_samples"] = failed_count

        # 4) Save JSON
        os.makedirs(output_dir, exist_ok=True)
        safe_name = dataset_name.replace("/", "-")
        output_path = os.path.join(output_dir, f"{safe_name}-{split}-nemo-encoded.json")

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        data_prep_jobs[job_id]["output_path"] = output_path
        print(f"[{job_id}] Saved {len(results)} encoded samples to {output_path}")

        # 5) Optional: push to Hub
        if hf_token and hub_repo:
            print(f"[{job_id}] Uploading encoded dataset to {hub_repo}...")
            encoded_dataset = Dataset.from_list(results)
            encoded_dataset.push_to_hub(hub_repo, split=split, token=hf_token)
            data_prep_jobs[job_id]["hub_repo"] = hub_repo
            print(f"[{job_id}] Uploaded to {hub_repo}")

        data_prep_jobs[job_id]["status"] = "completed"
        print(f"[{job_id}] Data preparation complete: {len(results)}/{total} samples encoded")

    except Exception as e:
        data_prep_jobs[job_id]["status"] = "failed"
        data_prep_jobs[job_id]["error"] = traceback.format_exc()
        print(f"[{job_id}] Data preparation failed: {e}")
