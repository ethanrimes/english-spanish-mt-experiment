"""Downloaders for parallel corpora.

Two flavors:
- `opus_tmx_zip`: streams a .tmx.gz file from OPUS, parses TMX <tu> records
  on the fly, and writes a jsonl of {en, es, source} pairs. Honors a
  per-source cap so we can take a slice on the laptop and the full thing on
  Azure where bandwidth + disk are plentiful.
- `hf_dataset`: pulls a HuggingFace dataset and emits jsonl.

Downloads are idempotent — if the output jsonl already exists with at least
the requested cap, we skip. To force re-download, delete the file.
"""
from __future__ import annotations

import gzip
import logging
import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

from tqdm import tqdm

from ..utils.io import write_jsonl
from .sources import Source

log = logging.getLogger(__name__)


def download_source(
    source: Source,
    *,
    raw_dir: Path,
    environment: Literal["local", "azure"],
) -> Path:
    """Download a single source to {raw_dir}/{source.name}.jsonl and return the path."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_path = raw_dir / f"{source.name}.jsonl"

    cap = source.max_pairs_local if environment == "local" else source.max_pairs_azure

    if out_path.exists():
        existing = _count_lines(out_path)
        if cap is None or existing >= cap:
            log.info("skip %s — %s already has %d pairs (cap %s)", source.name, out_path, existing, cap)
            return out_path

    log.info("downloading %s (cap=%s, env=%s)", source.name, cap, environment)

    if source.loader == "opus_tmx_zip":
        n = write_jsonl(out_path, _stream_opus_tmx(source.url, source.name, cap=cap))
    elif source.loader == "hf_dataset":
        n = write_jsonl(out_path, _stream_hf_dataset(source, cap=cap))
    else:
        raise ValueError(f"unknown loader: {source.loader}")
    log.info("wrote %d pairs to %s", n, out_path)
    return out_path


def _stream_opus_tmx(url: str, source_name: str, *, cap: int | None) -> Iterator[dict]:
    """Stream an OPUS-style .tmx.gz file and yield {en, es, source}."""
    req = urllib.request.Request(url, headers={"User-Agent": "en-es-mt/0.1"})
    with urllib.request.urlopen(req) as resp:  # noqa: S310 - controlled URL list
        with gzip.GzipFile(fileobj=resp) as gz:
            yield from _parse_tmx_stream(gz, source_name, cap=cap)


def _parse_tmx_stream(file_obj, source_name: str, *, cap: int | None) -> Iterator[dict]:
    """Streaming TMX parser. Yields one pair per <tu>.

    TMX schema (simplified):
        <tmx>
          <body>
            <tu>
              <tuv xml:lang="en"><seg>...</seg></tuv>
              <tuv xml:lang="es"><seg>...</seg></tuv>
            </tu>
            ...
    """
    n = 0
    iter_ = ET.iterparse(file_obj, events=("end",))
    pbar = tqdm(desc=f"  parse {source_name}", unit="pair", mininterval=1.0)
    for event, elem in iter_:
        # iterparse strips namespaces in attribute names; the xml:lang attr
        # comes through as '{http://www.w3.org/XML/1998/namespace}lang'.
        tag = _strip_ns(elem.tag)
        if tag != "tu":
            continue
        en_text = None
        es_text = None
        for tuv in elem:
            if _strip_ns(tuv.tag) != "tuv":
                continue
            lang = _get_lang(tuv)
            seg_text = _first_seg_text(tuv)
            if not seg_text:
                continue
            if lang.startswith("en"):
                en_text = seg_text
            elif lang.startswith("es"):
                es_text = seg_text
        elem.clear()  # free memory — critical for multi-GB files
        if en_text and es_text:
            n += 1
            pbar.update(1)
            yield {"en": en_text, "es": es_text, "source": source_name}
            if cap is not None and n >= cap:
                break
    pbar.close()


def _stream_hf_dataset(source: Source, *, cap: int | None) -> Iterator[dict]:
    """Pull from HuggingFace `datasets`. Lazy import so package import is cheap."""
    from datasets import load_dataset

    ds = load_dataset(source.hf_id, source.hf_config, split=source.hf_split)
    n = 0
    for ex in ds:
        # FLORES schema: 'sentence_eng_Latn' and 'sentence_spa_Latn' fields.
        en = ex.get("sentence_eng_Latn") or ex.get("en") or ex.get("translation", {}).get("en")
        es = ex.get("sentence_spa_Latn") or ex.get("es") or ex.get("translation", {}).get("es")
        if not en or not es:
            continue
        n += 1
        yield {"en": en, "es": es, "source": source.name}
        if cap is not None and n >= cap:
            break


def _strip_ns(tag: str) -> str:
    return tag.split("}", 1)[1] if "}" in tag else tag


def _get_lang(tuv_elem) -> str:
    for k, v in tuv_elem.attrib.items():
        if k.endswith("lang"):
            return v
    return ""


def _first_seg_text(tuv_elem) -> str | None:
    for child in tuv_elem:
        if _strip_ns(child.tag) == "seg":
            # `itertext` joins child text nodes (TMX can embed inline markup).
            return "".join(child.itertext()).strip()
    return None


def _count_lines(path: Path) -> int:
    with open(path, encoding="utf-8") as f:
        return sum(1 for _ in f)
