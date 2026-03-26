"""
Kani TTS - Frame-Level Position Collator
==========================================

Custom data collator for the SFT trainer that pads **four** columns:
``input_ids``, ``labels``, ``attention_mask``, and ``position_ids``.

The standard HuggingFace collators only handle the first three, but the
frame-level position encoding optimization requires ``position_ids`` to
be present in every batch so that the model receives explicit positional
information during training.

Padding strategy:
    - ``input_ids``:     padded with PAD token (tokeniser_length + 7 = 64407)
    - ``labels``:        padded with -100 (ignored by cross-entropy loss)
    - ``attention_mask``: padded with 0 (masked)
    - ``position_ids``:  padded with 0 (irrelevant since attention_mask is 0)
"""

import torch
from dataclasses import dataclass

PAD_TOKEN_ID = 64400 + 7  # tokeniser_length + 7


@dataclass
class FramePosCollator:
    """
    Data collator that pads all four training columns to the longest
    sequence in the batch and converts them to PyTorch tensors.

    This is required because the Kani TTS training pipeline uses custom
    frame-level ``position_ids`` where groups of 4 audio tokens share
    one position. Without this collator, the default SFTTrainer collator
    would discard the ``position_ids`` column.

    Args:
        pad_token_id: Token ID used to pad ``input_ids``. Defaults to
            the model's PAD token (64407).

    Example::

        collator = FramePosCollator()
        batch = collator([
            {"input_ids": [1, 2, 3], "labels": [1, 2, 3],
             "attention_mask": [1, 1, 1], "position_ids": [0, 1, 2]},
            {"input_ids": [4, 5],    "labels": [4, 5],
             "attention_mask": [1, 1],    "position_ids": [0, 1]},
        ])
        # batch["input_ids"].shape == torch.Size([2, 3])
        # Second sample padded: [4, 5, 64407]
    """

    pad_token_id: int = PAD_TOKEN_ID

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        """
        Pad a list of feature dicts to uniform length and stack into tensors.

        Args:
            features: List of dicts, each with keys ``input_ids``,
                ``labels``, ``attention_mask``, ``position_ids`` (all lists of ints).

        Returns:
            Dict of batched tensors ready for the model's forward pass.
        """
        max_len = max(len(f["input_ids"]) for f in features)

        batch = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
            "position_ids": [],
        }
        for f in features:
            pad_len = max_len - len(f["input_ids"])
            batch["input_ids"].append(f["input_ids"] + [self.pad_token_id] * pad_len)
            batch["attention_mask"].append(f["attention_mask"] + [0] * pad_len)
            batch["labels"].append(f["labels"] + [-100] * pad_len)
            batch["position_ids"].append(f["position_ids"] + [0] * pad_len)

        return {k: torch.tensor(v) for k, v in batch.items()}
