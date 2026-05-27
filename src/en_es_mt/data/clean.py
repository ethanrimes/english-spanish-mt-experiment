"""Per-source cleaning + filtering.

Steps applied per record:
1. Normalize whitespace, optionally strip HTML/XML tag residue.
2. Reject empty / too-short / too-long sentences.
3. Reject pairs whose length ratio is implausible (likely misaligned).
4. Reject pairs with identical en/es text.
5. Optional: language-ID check (fasttext lid.176) — keep only if EN side is
   English and ES side is Spanish above a confidence threshold.
6. Deduplicate within a source by exact (en, es) match (case-folded).

Writes per-source cleaned jsonl to data/interim/{name}.jsonl with the same
schema plus `en_len`, `es_len`, and a stable record id.
"""
from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Any

from tqdm import tqdm

from ..utils.io import read_jsonl, write_jsonl

log = logging.getLogger(__name__)

_WS = re.compile(r"\s+")
_TAG = re.compile(r"<[^>]+>")


def clean_source(
    in_path: Path,
    out_path: Path,
    *,
    rules: dict[str, Any],
    langid: "LangID | None" = None,
) -> dict:
    """Clean one source. Returns a small stats dict."""
    in_path = Path(in_path)
    out_path = Path(out_path)
    stats = {
        "in": 0,
        "out": 0,
        "drop_too_short": 0,
        "drop_too_long": 0,
        "drop_length_ratio": 0,
        "drop_identical": 0,
        "drop_langid": 0,
        "drop_dup": 0,
    }
    seen: set[str] = set()

    def _records():
        for rec in tqdm(read_jsonl(in_path), desc=f"  clean {in_path.stem}", unit="rec", mininterval=1.0):
            stats["in"] += 1
            en = _normalize(rec.get("en", ""), rules)
            es = _normalize(rec.get("es", ""), rules)
            if not en or not es:
                stats["drop_too_short"] += 1
                continue
            en_len, es_len = len(en), len(es)
            if en_len < rules.get("min_chars", 1) or es_len < rules.get("min_chars", 1):
                stats["drop_too_short"] += 1
                continue
            if en_len > rules.get("max_chars", 10_000) or es_len > rules.get("max_chars", 10_000):
                stats["drop_too_long"] += 1
                continue
            ratio = max(en_len, es_len) / max(1, min(en_len, es_len))
            if ratio > rules.get("max_length_ratio", 99.0):
                stats["drop_length_ratio"] += 1
                continue
            if rules.get("drop_if_identical", True) and en.lower() == es.lower():
                stats["drop_identical"] += 1
                continue
            if langid is not None:
                lp_en = langid.score(en, "en")
                lp_es = langid.score(es, "es")
                thr = rules.get("langid_min_prob", 0.5)
                if lp_en < thr or lp_es < thr:
                    stats["drop_langid"] += 1
                    continue
            key = _key(en, es)
            if key in seen:
                stats["drop_dup"] += 1
                continue
            seen.add(key)
            stats["out"] += 1
            yield {
                "id": _stable_id(rec.get("source", "?"), en, es),
                "en": en,
                "es": es,
                "source": rec.get("source", in_path.stem),
                "en_len": en_len,
                "es_len": es_len,
            }

    write_jsonl(out_path, _records())
    return stats


def _normalize(s: str, rules: dict) -> str:
    if not s:
        return ""
    if rules.get("strip_html_tags", True):
        s = _TAG.sub(" ", s)
    if rules.get("normalize_whitespace", True):
        s = _WS.sub(" ", s).strip()
    return s


def _key(en: str, es: str) -> str:
    return en.lower() + "||" + es.lower()


def _stable_id(source: str, en: str, es: str) -> str:
    h = hashlib.blake2b(digest_size=8)
    h.update(source.encode("utf-8"))
    h.update(b"\x00")
    h.update(en.encode("utf-8"))
    h.update(b"\x00")
    h.update(es.encode("utf-8"))
    return h.hexdigest()


class LangID:
    """Thin wrapper over fasttext lid.176. Lazy-loaded so the package
    imports without the model. Pass `enabled=False` to no-op."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._model = None

    def _load(self):
        if self._model is not None:
            return self._model
        # `fasttext-langdetect` ships lid.176 and gives a simple API.
        from ftlangdetect import detect  # noqa

        self._model = detect
        return self._model

    def score(self, text: str, expected: str) -> float:
        if not self.enabled or not text:
            return 1.0
        try:
            res = self._load()(text=text.replace("\n", " "), low_memory=False)
            return res["score"] if res["lang"] == expected else 1.0 - res["score"]
        except Exception:
            return 1.0  # be lenient on errors — better to keep than drop
