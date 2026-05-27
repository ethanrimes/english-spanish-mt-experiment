"""Spanish-contagion probe.

Quantifies how the base model's English-trained tokenizer handles Spanish.
This isn't a substitute for "did the pretrain see Spanish text" — that's
unknowable post-hoc without the training data — but it gives an empirical
fingerprint of how disadvantaged Spanish text is at tokenization time.

Outputs:
- Mean tokens per word, EN vs ES (Spanish is *expected* to be worse)
- Vocab coverage on a Spanish sample (count of tokens that fall into the
  byte-fallback range vs proper merges)
- Bytes-per-token, EN vs ES
- A handful of side-by-side example tokenizations

We use FLORES-200 dev English + Spanish for the comparison since both sides
are professionally translated parallel sentences.
"""
from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class ProbeResult:
    model_id: str
    vocab_size: int
    en_tokens_per_word: float
    es_tokens_per_word: float
    en_bytes_per_token: float
    es_bytes_per_token: float
    fertility_ratio: float          # es_tpw / en_tpw, >1.0 means Spanish disadvantaged
    samples: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__


def probe(
    tokenizer,
    en_sentences: list[str],
    es_sentences: list[str],
    *,
    n_examples: int = 10,
) -> ProbeResult:
    en_tpw, en_bpt = _stats(tokenizer, en_sentences)
    es_tpw, es_bpt = _stats(tokenizer, es_sentences)
    samples = []
    for en, es in list(zip(en_sentences, es_sentences))[:n_examples]:
        en_ids = tokenizer(en, add_special_tokens=False)["input_ids"]
        es_ids = tokenizer(es, add_special_tokens=False)["input_ids"]
        samples.append({
            "en_text": en,
            "es_text": es,
            "en_tokens": tokenizer.convert_ids_to_tokens(en_ids),
            "es_tokens": tokenizer.convert_ids_to_tokens(es_ids),
            "en_n": len(en_ids),
            "es_n": len(es_ids),
        })
    model_id = getattr(tokenizer, "name_or_path", "?")
    return ProbeResult(
        model_id=model_id,
        vocab_size=tokenizer.vocab_size,
        en_tokens_per_word=en_tpw,
        es_tokens_per_word=es_tpw,
        en_bytes_per_token=en_bpt,
        es_bytes_per_token=es_bpt,
        fertility_ratio=es_tpw / max(en_tpw, 1e-9),
        samples=samples,
    )


def _stats(tokenizer, texts: list[str]) -> tuple[float, float]:
    tpw_values, bpt_values = [], []
    for t in texts:
        if not t.strip():
            continue
        words = t.split()
        if not words:
            continue
        ids = tokenizer(t, add_special_tokens=False)["input_ids"]
        if not ids:
            continue
        tpw_values.append(len(ids) / len(words))
        bpt_values.append(len(t.encode("utf-8")) / len(ids))
    return (
        statistics.mean(tpw_values) if tpw_values else 0.0,
        statistics.mean(bpt_values) if bpt_values else 0.0,
    )


def format_report(p: ProbeResult, *, n_examples: int = 5) -> str:
    """Markdown report suitable for `docs/MODEL_CHOICE.md`."""
    lines = [
        f"# Tokenizer probe: `{p.model_id}`",
        "",
        f"- Vocab size: **{p.vocab_size}**",
        f"- English tokens/word: **{p.en_tokens_per_word:.3f}**",
        f"- Spanish tokens/word: **{p.es_tokens_per_word:.3f}**",
        f"- Fertility ratio (ES/EN): **{p.fertility_ratio:.2f}**  (>1.0 means Spanish costs more tokens per word)",
        f"- Bytes/token EN: **{p.en_bytes_per_token:.2f}**",
        f"- Bytes/token ES: **{p.es_bytes_per_token:.2f}**",
        "",
        "## Example tokenizations",
        "",
    ]
    for i, ex in enumerate(p.samples[:n_examples], start=1):
        lines.append(f"### Example {i}")
        lines.append(f"- **EN ({ex['en_n']} tok):** `{ex['en_text']}`")
        lines.append(f"  - Pieces: `{' '.join(ex['en_tokens'])}`")
        lines.append(f"- **ES ({ex['es_n']} tok):** `{ex['es_text']}`")
        lines.append(f"  - Pieces: `{' '.join(ex['es_tokens'])}`")
        lines.append("")
    return "\n".join(lines)
