"""Translation inference. Batched beam search by default.

The model emits text starting from a prompt of the form
   "English: <text>\nSpanish: "
We stop at the next newline or EOS, strip whitespace, and return.
"""
from __future__ import annotations

import logging
from typing import Iterable

from ..data.format import PromptFormatter, make_inference_prompt

log = logging.getLogger(__name__)


def translate_batch(
    texts: list[str],
    *,
    direction: str,                # 'en2es' or 'es2en'
    model,
    tokenizer,
    formatter: PromptFormatter,
    max_new_tokens: int = 192,
    num_beams: int = 4,
    length_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
    do_sample: bool = False,
    batch_size: int = 16,
) -> list[str]:
    import torch

    device = next(model.parameters()).device
    out: list[str] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        prompts = [make_inference_prompt(t, direction, formatter=formatter) for t in batch]
        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        enc = {k: v.to(device) for k, v in enc.items()}
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "num_beams": num_beams,
            "length_penalty": length_penalty,
            "no_repeat_ngram_size": no_repeat_ngram_size,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        with torch.inference_mode():
            ids = model.generate(**enc, **gen_kwargs)
        for j, full_ids in enumerate(ids):
            # Strip the prompt prefix from the decoded output.
            prompt_len = enc["input_ids"][j].shape[0]
            new_ids = full_ids[prompt_len:]
            text = tokenizer.decode(new_ids, skip_special_tokens=True)
            out.append(_clean_completion(text))
    return out


def _clean_completion(s: str) -> str:
    # Cut at first newline (translation should be a single sentence after the prompt).
    s = s.split("\n", 1)[0]
    return s.strip()


def translate_iter(
    texts: Iterable[str],
    *,
    direction: str,
    model,
    tokenizer,
    formatter: PromptFormatter,
    **kwargs,
) -> list[str]:
    return translate_batch(list(texts), direction=direction, model=model, tokenizer=tokenizer, formatter=formatter, **kwargs)
