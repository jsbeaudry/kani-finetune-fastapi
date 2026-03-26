"""
Kani TTS - Dataset Processing Pipeline
========================================

Converts raw HuggingFace datasets (with pre-encoded NeMo codec tokens)
into training-ready datasets with ``input_ids``, ``labels``,
``attention_mask``, and ``position_ids``.

Pipeline Overview
-----------------

::

    HuggingFace Dataset(s)
        |
        v
    [Duration Filter]  -- remove samples > max_duration_sec
        |
        v
    [Add Audio Codes]  -- interleave 4 codec layers, remove consecutive dupes
        |
        v
    [Create Input IDs] -- wrap in control tokens, build frame-level position_ids
        |
        v
    [Clean Columns]    -- keep only input_ids, labels, attention_mask, position_ids
        |
        v
    Training-ready HF Dataset

Multi-Speaker Support
---------------------

Each dataset can be assigned a ``speaker_id``. When set, every text sample
is prefixed with ``"<speaker_id>: <text>"`` before tokenization. This
conditions the model to generate speech in that speaker's voice.

Parallel Processing
-------------------

Datasets are split into shards and processed in parallel using
``ProcessPoolExecutor``. Each shard gets its own ``TrainDataPreProcessor``
instance (which loads a copy of the tokenizer) to avoid GIL contention.

Frame-Level Position Encoding
-----------------------------

Unlike standard sequential positions, audio tokens are assigned positions
at the **frame level**: all 4 codec tokens within one audio frame share
the same position ID. This reduces the effective RoPE distance by 4x,
improving long-form generation quality.

::

    Standard:    Frame0=[pos20,pos21,pos22,pos23]  Frame1=[pos24,pos25,pos26,pos27]
    Frame-level: Frame0=[pos20,pos20,pos20,pos20]  Frame1=[pos21,pos21,pos21,pos21]
"""

import locale
import math
import multiprocessing as mp
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from typing import Optional

import numpy as np
import torch
from datasets import load_dataset, concatenate_datasets, Dataset
from omegaconf import OmegaConf
from transformers import AutoTokenizer


class TrainDataPreProcessor:
    """
    Processes a single dataset shard: filters by duration, encodes audio
    tokens, builds input_ids with frame-level position_ids.

    This class is instantiated once per shard in a worker process.

    Args:
        tokenizer_name: HuggingFace model/tokenizer ID for text encoding.
        max_dur: Maximum audio duration in seconds (samples over this are dropped).
        speaker_id: Optional speaker label to prepend to all text samples.

    Token vocabulary constants (matching the Kani model):
        - tokeniser_length: 64400 (text vocab size)
        - codebook_size: 4032 (codes per NeMo codebook)
        - audio_tokens_start: tokeniser_length + 10
    """

    def __init__(self, tokenizer_name: str, max_dur: int, speaker_id: str = None) -> None:
        self.text_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_dur = max_dur
        self.speaker_id = speaker_id
        locale.getpreferredencoding = lambda: "UTF-8"

        self.tokeniser_length = 64400
        self.start_of_text = 1
        self.end_of_text = 2
        self.start_of_speech = self.tokeniser_length + 1
        self.end_of_speech = self.tokeniser_length + 2
        self.start_of_human = self.tokeniser_length + 3
        self.end_of_human = self.tokeniser_length + 4
        self.start_of_ai = self.tokeniser_length + 5
        self.end_of_ai = self.tokeniser_length + 6
        self.pad_token = self.tokeniser_length + 7
        self.audio_tokens_start = self.tokeniser_length + 10
        self.codebook_size = 4032

    def add_codes(self, example: dict) -> dict:
        """
        Interleave 4 NeMo codec layers into a flat token sequence.

        Steps:
            1. Stack the 4 layer columns into [num_frames, 4].
            2. Add per-codebook offsets (0, 4032, 8064, 12096).
            3. Remove consecutive duplicate frames (codec dedup).
            4. Add the audio_tokens_start offset.
            5. Flatten to a 1-D list.

        Also stores ``num_audio_frames`` for position_id construction.

        Args:
            example: A single dataset row (dict with nano_layer_1..4 columns).

        Returns:
            The same dict with ``codes_list`` (list[int]) and
            ``num_audio_frames`` (int) added.
        """
        snac_layers = ["nano_layer_1", "nano_layer_2", "nano_layer_3", "nano_layer_4"]
        codes = [example[i] for i in snac_layers]
        codes = np.array(codes).T
        all_codes = codes + np.array([self.codebook_size * i for i in range(4)])

        all_codes = self.remove_consecutive_duplicates_np(all_codes)

        all_codes = all_codes + self.audio_tokens_start
        example["codes_list"] = all_codes.flatten().tolist()
        example["num_audio_frames"] = all_codes.shape[0]
        return example

    def remove_consecutive_duplicates_np(self, arr: np.ndarray) -> np.ndarray:
        """
        Remove consecutive duplicate frames from a 2-D array.

        Two frames are considered duplicates if ALL 4 codebook values match.
        This is a lossy compression step that reduces sequence length without
        significant audio quality loss.

        Args:
            arr: Array of shape [num_frames, 4].

        Returns:
            De-duplicated array with shape [new_num_frames, 4].

        Raises:
            ValueError: If the input is not 2-D.
        """
        if arr.ndim != 2:
            raise ValueError("2D array expected [num_frames, frame_size]")
        mask = np.any(arr[1:] != arr[:-1], axis=1)
        keep = np.insert(mask, 0, True)
        return arr[keep]

    def create_input_ids(self, example: dict) -> dict:
        """
        Build the full training sequence with frame-level position IDs.

        Sequence structure::

            [START_OF_HUMAN] + text_tokens + [END_OF_HUMAN]
            + [START_OF_AI] + [START_OF_SPEECH]
            + audio_codes
            + [END_OF_SPEECH] + [END_OF_AI]

        Position ID strategy:
            - Prefix (text + control tokens): sequential (0, 1, 2, ...)
            - Audio tokens: groups of 4 share one position per frame
            - Suffix (END_OF_SPEECH, END_OF_AI): continue from last audio pos

        Args:
            example: Dict with ``text``, ``codes_list``, ``num_audio_frames``.

        Returns:
            Dict with ``input_ids``, ``labels``, ``attention_mask``,
            ``position_ids`` added.
        """
        if self.speaker_id is not None:
            text_prompt = f"{self.speaker_id.lower()}: {example['text']}"
        elif example.get("speaker") is not None:
            text_prompt = f"{example['speaker'].lower()}: {example['text']}"
        else:
            text_prompt = example["text"]

        text_ids = self.text_tokenizer.encode(text_prompt, add_special_tokens=True)
        text_ids.append(self.end_of_text)
        example["text_tokens"] = text_ids

        prefix = (
            [self.start_of_human]
            + example["text_tokens"]
            + [self.end_of_human]
            + [self.start_of_ai]
            + [self.start_of_speech]
        )
        suffix = [self.end_of_speech, self.end_of_ai]
        audio_codes = example["codes_list"]

        input_ids = prefix + audio_codes + suffix

        # ── Frame-level position_ids ──
        prefix_len = len(prefix)
        num_audio_tokens = len(audio_codes)
        num_frames = example["num_audio_frames"]
        suffix_len = len(suffix)

        # Text/control: sequential
        text_positions = list(range(prefix_len))

        # Audio: each group of 4 shares one position
        audio_start_pos = prefix_len
        audio_positions = []
        for frame_idx in range(num_frames):
            frame_pos = audio_start_pos + frame_idx
            audio_positions.extend([frame_pos] * 4)

        # Handle partial frames from dedup remainder
        remainder = num_audio_tokens - (num_frames * 4)
        if remainder > 0:
            last_pos = audio_start_pos + num_frames
            audio_positions.extend([last_pos] * remainder)

        # Suffix: continue sequentially
        suffix_start = audio_positions[-1] + 1 if audio_positions else audio_start_pos
        suffix_positions = list(range(suffix_start, suffix_start + suffix_len))

        position_ids = text_positions + audio_positions + suffix_positions

        example["input_ids"] = input_ids
        example["labels"] = input_ids
        example["attention_mask"] = [1] * len(input_ids)
        example["position_ids"] = position_ids
        return example

    def __call__(self, dataset: Dataset) -> Dataset:
        """
        Run the full preprocessing pipeline on a dataset shard.

        Steps:
            1. Filter by max audio duration.
            2. Encode audio tokens (add_codes).
            3. Remove samples with empty/null codes.
            4. Build input_ids and position_ids (create_input_ids).
            5. Drop all columns except the 4 training columns.

        Args:
            dataset: HuggingFace Dataset shard with columns:
                text, nano_layer_1..4, encoded_len, (optional) speaker.

        Returns:
            Processed Dataset with columns: input_ids, labels,
            attention_mask, position_ids.
        """
        if self.max_dur:
            dataset_len = len(dataset)
            dataset = dataset.filter(lambda i: i["encoded_len"] / 12.5 <= self.max_dur)
            print(f"Duration filter: {len(dataset)} rows from {dataset_len}")

        dataset = dataset.map(
            self.add_codes,
            remove_columns=["nano_layer_1", "nano_layer_2", "nano_layer_3", "nano_layer_4"],
            desc="Add Audio Codes",
        )
        dataset = dataset.filter(lambda x: x["codes_list"] is not None, desc="Check codes list")
        dataset = dataset.filter(lambda x: len(x["codes_list"]) > 0, desc="Check codes list length")
        dataset = dataset.map(
            self.create_input_ids,
            remove_columns=["text", "codes_list", "num_audio_frames"],
            desc="Create input ids",
        )

        columns_to_keep = ["input_ids", "labels", "attention_mask", "position_ids"]
        columns_to_remove = [col for col in dataset.column_names if col not in columns_to_keep]
        dataset = dataset.remove_columns(columns_to_remove)
        return dataset


def process_shard(
    shard_idx: int, shard_data: Dataset, tokenizer_name: str, max_dur: int, speaker_id: str
) -> Dataset:
    """
    Worker function for parallel shard processing.

    Instantiates a fresh TrainDataPreProcessor in the worker process
    and runs it on the given shard.

    Args:
        shard_idx: Index of this shard (for logging).
        shard_data: HuggingFace Dataset shard.
        tokenizer_name: HF tokenizer to load in the worker.
        max_dur: Max audio duration filter.
        speaker_id: Speaker label (or None).

    Returns:
        Processed Dataset shard.
    """
    processor = TrainDataPreProcessor(tokenizer_name, max_dur, speaker_id)
    return processor(shard_data)


class ItemDataset:
    """
    Loads and processes a single HuggingFace dataset entry.

    Handles:
    - Loading the dataset from HF Hub.
    - Applying optional categorical filters (e.g. select one speaker).
    - Renaming columns to the standard names expected by the preprocessor.
    - Splitting into shards and processing in parallel.
    - Optional random subsampling (max_len).

    Args:
        item_cfg: OmegaConf config for this dataset (reponame, split, etc.).
        tokenizer_name: HF model/tokenizer ID.
        max_dur: Max audio duration in seconds.
        n_shards: Number of parallel processing shards.
    """

    def __init__(self, item_cfg, tokenizer_name: str, max_dur: int, n_shards: int = None):
        self.item_cfg = item_cfg
        self.tokenizer_name = tokenizer_name
        self.max_dur = max_dur
        self.speaker_id = self.item_cfg.get("speaker_id")
        self.max_len = self.item_cfg.get("max_len")

        if n_shards is None:
            self.n_shards = min(mp.cpu_count(), 8)
        else:
            self.n_shards = n_shards

        self.dataset = load_dataset(
            self.item_cfg.reponame,
            self.item_cfg.name,
            split=self.item_cfg.split,
            num_proc=10,
        )

        # Optional categorical filter (e.g. select speaker=="ex02")
        if self.item_cfg.get("categorical_filter"):
            cf = self.item_cfg.categorical_filter
            self.dataset = self.dataset.filter(
                lambda x: x[cf.column_name] == cf.value
            )

        # Rename columns to standard names
        rename_dict = {
            self.item_cfg.text_col_name: "text",
            self.item_cfg.nano_layer_1: "nano_layer_1",
            self.item_cfg.nano_layer_2: "nano_layer_2",
            self.item_cfg.nano_layer_3: "nano_layer_3",
            self.item_cfg.nano_layer_4: "nano_layer_4",
            self.item_cfg.encoded_len: "encoded_len",
        }
        self.dataset = self.dataset.rename_columns(rename_dict)

    def __call__(self) -> Dataset:
        """
        Process the dataset using parallel shards.

        Returns:
            Fully processed HuggingFace Dataset ready for training,
            optionally subsampled to ``max_len`` rows.
        """
        shards = []
        for i in range(self.n_shards):
            shard = self.dataset.shard(num_shards=self.n_shards, index=i)
            shards.append((shard, i))

        processed_shards = []
        with ProcessPoolExecutor(max_workers=self.n_shards) as executor:
            future_to_shard = {
                executor.submit(
                    process_shard, shard_idx, shard, self.tokenizer_name, self.max_dur, self.speaker_id
                ): shard_idx
                for shard, shard_idx in shards
            }
            for future in as_completed(future_to_shard):
                shard_idx = future_to_shard[future]
                processed_shard = future.result()
                processed_shards.append((shard_idx, processed_shard))

        processed_shards.sort(key=lambda x: x[0])
        final_shards = [shard for _, shard in processed_shards]

        final_dataset = concatenate_datasets(final_shards)
        if self.max_len is not None:
            final_dataset = final_dataset.shuffle(seed=42).select(range(self.max_len))
        return final_dataset


class DatasetProcessor:
    """
    Top-level dataset orchestrator that merges multiple HF datasets.

    Iterates over all dataset entries in the config, processes each via
    ``ItemDataset``, and concatenates the results into a single shuffled
    training dataset.

    Args:
        dataset_config: Dataclass/OmegaConf config with ``hf_datasets`` list
            and ``max_duration_sec``.
        tokenizer_name: HF tokenizer ID (shared across all datasets).
        n_shards_per_dataset: Parallel shards per dataset (overrides auto-detect).

    Example::

        processor = DatasetProcessor(config, "nineninesix/kani-tts-400m-0.3-pt", n_shards_per_dataset=4)
        train_dataset = processor()
        # train_dataset has columns: input_ids, labels, attention_mask, position_ids
    """

    def __init__(self, dataset_config, tokenizer_name: str, n_shards_per_dataset: int = None):
        self.cfg = OmegaConf.structured(dataset_config)
        self.tokenizer_name = tokenizer_name
        self.n_shards_per_dataset = n_shards_per_dataset

    def __call__(self) -> Dataset:
        """
        Load, process, and merge all configured datasets.

        Returns:
            Shuffled HuggingFace Dataset combining all sources.
        """
        datasets = []
        for item_cfg in self.cfg.hf_datasets:
            item_ds_maker = ItemDataset(
                item_cfg=item_cfg,
                tokenizer_name=self.tokenizer_name,
                max_dur=self.cfg.max_duration_sec,
                n_shards=self.n_shards_per_dataset,
            )
            datasets.append(item_ds_maker())

        final_dataset = concatenate_datasets(datasets)
        final_dataset = final_dataset.shuffle()
        return final_dataset
