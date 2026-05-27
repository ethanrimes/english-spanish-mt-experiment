"""Padding collator for causal-LM translation training.

Pads input_ids / labels / attention_mask to the longest in a batch.
Padding tokens in labels are masked to -100 (HF loss ignore index).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class PadCollator:
    pad_token_id: int
    label_pad_id: int = -100
    pad_to_multiple_of: int | None = 8     # nice for tensor cores

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        max_len = max(len(f["input_ids"]) for f in features)
        if self.pad_to_multiple_of:
            m = self.pad_to_multiple_of
            max_len = ((max_len + m - 1) // m) * m

        input_ids, labels, attn = [], [], []
        for f in features:
            n = len(f["input_ids"])
            pad = max_len - n
            input_ids.append(f["input_ids"] + [self.pad_token_id] * pad)
            labels.append(f["labels"] + [self.label_pad_id] * pad)
            attn.append(f["attention_mask"] + [0] * pad)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }
