"""Prompt formatting and tokenization for training.

We train a single bidirectional translation model by sampling directions
uniformly. Each example is rendered as:

    English: <text>
    Spanish: <text><eos>

…or its reverse. The loss is computed only on the **target** side (the
text after the second "<lang>: " marker) so the model learns to generate,
not just memorize the prompt.

Public entrypoints:
- `make_training_example(record, *, tokenizer, ...)` → dict with input_ids,
  labels (target-only), attention_mask.
- `make_inference_prompt(text, direction, ...)` → string to feed the model.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal

Direction = Literal["en2es", "es2en"]


@dataclass(frozen=True)
class PromptFormatter:
    src_lang_names: dict[str, str]  # {"en": "English", "es": "Spanish"}
    template: str                   # uses {src_lang}, {src_text}, {tgt_lang}, {tgt_text}
    direction_sampling: str = "balanced"

    def pick_direction(self, rng: random.Random) -> Direction:
        if self.direction_sampling == "balanced":
            return "en2es" if rng.random() < 0.5 else "es2en"
        if self.direction_sampling == "en2es":
            return "en2es"
        if self.direction_sampling == "es2en":
            return "es2en"
        raise ValueError(self.direction_sampling)

    def render(self, en: str, es: str, direction: Direction) -> tuple[str, str, str]:
        """Return (prompt_prefix, target_text, full_text). prompt_prefix is
        the part we mask in the loss; target_text is what the model must
        produce; full_text is what we tokenize."""
        if direction == "en2es":
            src_text, tgt_text = en, es
            src_lang, tgt_lang = "en", "es"
        else:
            src_text, tgt_text = es, en
            src_lang, tgt_lang = "es", "en"
        src_name = self.src_lang_names[src_lang]
        tgt_name = self.src_lang_names[tgt_lang]
        # The prefix ends right before tgt_text. The model must learn to emit tgt_text.
        prefix = f"{src_name}: {src_text}\n{tgt_name}: "
        full = prefix + tgt_text
        return prefix, tgt_text, full


def make_training_example(
    record: dict,
    *,
    tokenizer,
    formatter: PromptFormatter,
    max_seq_len: int,
    rng: random.Random,
    mask_loss_on_prompt: bool = True,
) -> dict:
    """Returns dict with input_ids, labels, attention_mask (lists, not padded).
    The collator handles right-padding to the longest in a batch."""
    direction = formatter.pick_direction(rng)
    prefix, _tgt, full = formatter.render(record["en"], record["es"], direction)

    # Tokenize prefix and full separately so we know exactly where to start the loss.
    prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full, add_special_tokens=False)["input_ids"]

    # Append EOS so generation knows when to stop.
    eos_id = tokenizer.eos_token_id
    if eos_id is not None and (not full_ids or full_ids[-1] != eos_id):
        full_ids = full_ids + [eos_id]

    # Truncate from the end if too long.
    if len(full_ids) > max_seq_len:
        full_ids = full_ids[:max_seq_len]

    input_ids = full_ids
    labels = list(input_ids)
    if mask_loss_on_prompt:
        # Mask everything up to and including the prefix.
        cutoff = min(len(prefix_ids), len(labels))
        for i in range(cutoff):
            labels[i] = -100

    attention_mask = [1] * len(input_ids)
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "direction": direction,
    }


def make_inference_prompt(
    text: str,
    direction: Direction,
    *,
    formatter: PromptFormatter,
) -> str:
    if direction == "en2es":
        src_name = formatter.src_lang_names["en"]
        tgt_name = formatter.src_lang_names["es"]
    else:
        src_name = formatter.src_lang_names["es"]
        tgt_name = formatter.src_lang_names["en"]
    return f"{src_name}: {text}\n{tgt_name}: "
