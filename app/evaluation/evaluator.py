"""
Kani TTS - Model Evaluation
=============================

Compares TTS-generated audio against reference (human) recordings from a
HuggingFace test dataset using acoustic similarity metrics.

Metrics
-------

+-------------------------------+-------------------------------------------+
| Metric                        | What it measures                          |
+===============================+===========================================+
| MFCC Cosine Similarity        | Timbre & vocal quality                    |
+-------------------------------+-------------------------------------------+
| Chroma Cosine Similarity      | Pitch & harmonic content                  |
+-------------------------------+-------------------------------------------+
| Spectral Centroid Similarity  | Brightness match                          |
+-------------------------------+-------------------------------------------+
| DTW Distance (normalized)     | Temporal structure alignment              |
+-------------------------------+-------------------------------------------+
| Overall Score                 | Weighted combination of the above         |
+-------------------------------+-------------------------------------------+

Weights for the overall score::

    MFCC  0.35  +  Chroma  0.25  +  Spectral  0.15  +  DTW  0.25

All metrics are on a 0-1 scale where **1.0 = perfect match**.
"""

import traceback
from typing import Dict

import librosa
import numpy as np
from scipy.spatial.distance import cosine as cosine_distance

from app.training.data_prep import load_audio_from_raw

# ---------------------------------------------------------------------------
# In-memory job store
# ---------------------------------------------------------------------------

eval_jobs: Dict[str, dict] = {}
"""
Dict mapping job_id -> evaluation job state.

Keys:
    - status (str): "starting", "running", "completed", "failed"
    - error (str | None): Python traceback if failed
    - total (int | None): Total samples in the test dataset
    - processed (int): Samples evaluated so far
    - results (list[dict]): Per-sample metric dicts
    - summary (dict | None): Dataset-level metric averages
"""

SAMPLE_RATE = 22050

# Weights for the overall score
WEIGHTS = {
    "mfcc_similarity": 0.35,
    "chroma_similarity": 0.25,
    "spectral_centroid_similarity": 0.15,
    "dtw_similarity": 0.25,
}


# ---------------------------------------------------------------------------
# Audio comparison metrics
# ---------------------------------------------------------------------------


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity between two feature vectors.

    Flattens multi-dimensional features (e.g. MFCC matrices) by taking
    the mean across the time axis first, then computing 1 - cosine_distance.

    Args:
        a: Feature array, shape (n_features,) or (n_features, n_frames).
        b: Feature array, same shape convention as *a*.

    Returns:
        Similarity score in [0, 1]. 1.0 = identical.
    """
    if a.ndim > 1:
        a = a.mean(axis=1)
    if b.ndim > 1:
        b = b.mean(axis=1)

    # Ensure same length (truncate to shorter)
    min_len = min(len(a), len(b))
    a, b = a[:min_len], b[:min_len]

    # Guard against zero vectors
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0

    return float(1.0 - cosine_distance(a, b))


def _dtw_similarity(ref_audio: np.ndarray, gen_audio: np.ndarray) -> float:
    """
    Dynamic Time Warping similarity on MFCC features.

    Computes the DTW cost between two MFCC sequences, then normalizes
    it to a 0-1 similarity score using ``1 / (1 + normalized_cost)``.

    Args:
        ref_audio: Reference audio, float32, 22050 Hz.
        gen_audio: Generated audio, float32, 22050 Hz.

    Returns:
        Similarity score in [0, 1]. 1.0 = identical temporal structure.
    """
    ref_mfcc = librosa.feature.mfcc(y=ref_audio, sr=SAMPLE_RATE, n_mfcc=13)
    gen_mfcc = librosa.feature.mfcc(y=gen_audio, sr=SAMPLE_RATE, n_mfcc=13)

    D, _wp = librosa.sequence.dtw(X=ref_mfcc, Y=gen_mfcc, metric="euclidean")

    # Normalize by path length
    path_len = D.shape[0] + D.shape[1]
    normalized_cost = D[-1, -1] / path_len if path_len > 0 else 0.0

    return float(1.0 / (1.0 + normalized_cost))


def compare_audio(ref_audio: np.ndarray, gen_audio: np.ndarray) -> dict:
    """
    Compute a full similarity report between reference and generated audio.

    Both inputs must be float32 numpy arrays at 22050 Hz.

    Args:
        ref_audio: Reference (human) audio waveform.
        gen_audio: TTS-generated audio waveform.

    Returns:
        Dict with keys:
            - mfcc_similarity (float): Timbre & vocal quality [0-1]
            - chroma_similarity (float): Pitch & harmonic content [0-1]
            - spectral_centroid_similarity (float): Brightness match [0-1]
            - dtw_similarity (float): Temporal structure [0-1]
            - overall_score (float): Weighted combination [0-1]
    """
    # MFCC — timbre & vocal quality
    ref_mfcc = librosa.feature.mfcc(y=ref_audio, sr=SAMPLE_RATE, n_mfcc=13)
    gen_mfcc = librosa.feature.mfcc(y=gen_audio, sr=SAMPLE_RATE, n_mfcc=13)
    mfcc_sim = _cosine_similarity(ref_mfcc, gen_mfcc)

    # Chroma — pitch & harmonic content
    ref_chroma = librosa.feature.chroma_stft(y=ref_audio, sr=SAMPLE_RATE)
    gen_chroma = librosa.feature.chroma_stft(y=gen_audio, sr=SAMPLE_RATE)
    chroma_sim = _cosine_similarity(ref_chroma, gen_chroma)

    # Spectral Centroid — brightness
    ref_cent = librosa.feature.spectral_centroid(y=ref_audio, sr=SAMPLE_RATE)
    gen_cent = librosa.feature.spectral_centroid(y=gen_audio, sr=SAMPLE_RATE)
    spectral_sim = _cosine_similarity(ref_cent, gen_cent)

    # DTW — temporal structure
    dtw_sim = _dtw_similarity(ref_audio, gen_audio)

    # Overall weighted score
    overall = (
        WEIGHTS["mfcc_similarity"] * mfcc_sim
        + WEIGHTS["chroma_similarity"] * chroma_sim
        + WEIGHTS["spectral_centroid_similarity"] * spectral_sim
        + WEIGHTS["dtw_similarity"] * dtw_sim
    )

    return {
        "mfcc_similarity": round(mfcc_sim, 4),
        "chroma_similarity": round(chroma_sim, 4),
        "spectral_centroid_similarity": round(spectral_sim, 4),
        "dtw_similarity": round(dtw_sim, 4),
        "overall_score": round(overall, 4),
    }


# ---------------------------------------------------------------------------
# Background evaluation job
# ---------------------------------------------------------------------------


def run_evaluation(
    job_id: str,
    tts_model,
    dataset_name: str,
    split: str,
    text_column: str,
    audio_column: str,
    speaker_column: str | None,
    speaker_id: str | None,
    hf_token: str | None,
) -> None:
    """
    Evaluate TTS quality against a reference dataset.

    Designed to run in a background thread via ``asyncio.to_thread()``.

    For each sample in the dataset:
        1. Decode the reference audio from the HF dataset.
        2. Generate TTS audio using the loaded KaniTTS model.
        3. Compare the two using acoustic similarity metrics.

    Args:
        job_id: Unique job identifier for status tracking.
        tts_model: The loaded KaniTTS instance (callable).
        dataset_name: HF dataset repo ID containing test audio/text pairs.
        split: Dataset split to evaluate (e.g. "test").
        text_column: Column name for transcription text.
        audio_column: Column name for reference audio.
        speaker_column: Column with speaker IDs (or None).
        speaker_id: Fixed speaker ID for all samples (or None).
        hf_token: HF API token for private datasets (or None).

    Side effects:
        Updates ``eval_jobs[job_id]`` throughout processing.
    """
    from datasets import load_dataset, Audio

    job = eval_jobs[job_id]

    try:
        job["status"] = "running"

        # 1) Load dataset with raw audio (no torchcodec)
        print(f"[eval:{job_id}] Loading dataset: {dataset_name} (split={split})")
        dataset = load_dataset(dataset_name, split=split, token=hf_token)
        dataset = dataset.cast_column(audio_column, Audio(decode=False))
        total = len(dataset)
        job["total"] = total
        print(f"[eval:{job_id}] Loaded {total} samples")

        results = []

        for idx, sample in enumerate(dataset):
            try:
                # 2) Decode reference audio
                ref_audio = load_audio_from_raw(sample[audio_column])

                # 3) Get text + speaker
                text = sample.get(text_column, "")
                if not text:
                    print(f"[eval:{job_id}] Skipping sample {idx}: empty text")
                    continue

                spk = speaker_id
                if spk is None and speaker_column and speaker_column in sample:
                    spk = sample[speaker_column]

                # 4) Generate TTS audio via KaniTTS (callable)
                gen_kwargs = {}
                if spk is not None:
                    gen_kwargs["speaker_id"] = spk
                gen_audio, _ = tts_model(text, **gen_kwargs)

                # 5) Compare
                metrics = compare_audio(ref_audio, gen_audio)
                metrics["sample_index"] = idx
                metrics["text"] = text[:100]  # truncate for readability
                results.append(metrics)

                job["processed"] = len(results)

                if (idx + 1) % 10 == 0 or idx == total - 1:
                    print(
                        f"[eval:{job_id}] Evaluated {len(results)}/{total} "
                        f"(latest overall: {metrics['overall_score']:.4f})"
                    )

            except Exception as e:
                print(f"[eval:{job_id}] Error on sample {idx}: {e}")
                results.append({
                    "sample_index": idx,
                    "text": sample.get(text_column, "")[:100],
                    "error": str(e),
                    "mfcc_similarity": None,
                    "chroma_similarity": None,
                    "spectral_centroid_similarity": None,
                    "dtw_similarity": None,
                    "overall_score": None,
                })
                job["processed"] = len(results)

        # 6) Compute summary averages (exclude failed samples)
        valid = [r for r in results if r.get("overall_score") is not None]
        if valid:
            summary = {
                "mfcc_similarity": round(
                    sum(r["mfcc_similarity"] for r in valid) / len(valid), 4
                ),
                "chroma_similarity": round(
                    sum(r["chroma_similarity"] for r in valid) / len(valid), 4
                ),
                "spectral_centroid_similarity": round(
                    sum(r["spectral_centroid_similarity"] for r in valid) / len(valid), 4
                ),
                "dtw_similarity": round(
                    sum(r["dtw_similarity"] for r in valid) / len(valid), 4
                ),
                "overall_score": round(
                    sum(r["overall_score"] for r in valid) / len(valid), 4
                ),
                "samples_evaluated": len(valid),
                "samples_failed": len(results) - len(valid),
            }
        else:
            summary = {
                "mfcc_similarity": 0.0,
                "chroma_similarity": 0.0,
                "spectral_centroid_similarity": 0.0,
                "dtw_similarity": 0.0,
                "overall_score": 0.0,
                "samples_evaluated": 0,
                "samples_failed": len(results),
            }

        job["results"] = results
        job["summary"] = summary
        job["status"] = "completed"
        print(
            f"[eval:{job_id}] Evaluation complete: "
            f"{summary['samples_evaluated']}/{total} samples, "
            f"overall score: {summary['overall_score']:.4f}"
        )

    except Exception as e:
        job["status"] = "failed"
        job["error"] = traceback.format_exc()
        print(f"[eval:{job_id}] Evaluation failed: {e}")
