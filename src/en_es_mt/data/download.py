"""Downloaders for parallel corpora.

Three flavors:
- `opus_tmx_zip`: streams a .tmx.gz file from OPUS, parses TMX <tu> records
  on the fly, and writes a jsonl of {en, es, source} pairs. Honors a
  per-source cap so we can take a slice on the laptop and the full thing on
  Azure where bandwidth + disk are plentiful.
- `flores_tarball`: pulls Meta's FLORES-200 .tar.gz from their CDN and
  extracts the en/es .dev[test] files; lines are 1:1 parallel.
- `hf_dataset`: pulls a HuggingFace Parquet-based dataset. (Script-based
  HF datasets are unsupported by datasets>=3.0; FLORES is handled via
  flores_tarball instead.)

Downloads are idempotent — if the output jsonl already exists with at least
the requested cap, we skip. To force re-download, delete the file.
"""
from __future__ import annotations

import gzip
import io
import logging
import tarfile
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
    elif source.loader == "flores_tarball":
        n = write_jsonl(out_path, _stream_flores_tarball(source, cap=cap))
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


_FLORES_CACHE: dict[str, dict[str, list[str]]] = {}


def _stream_flores_tarball(source: Source, *, cap: int | None) -> Iterator[dict]:
    """Pull FLORES-200 from Meta's CDN tarball.

    The tarball contains:
      flores200_dataset/dev/{lang}_Latn.dev
      flores200_dataset/devtest/{lang}_Latn.devtest
    Lines are 1:1 parallel across languages.

    `source.extra["split"]` must be "dev" or "devtest". The tarball is
    downloaded once per process and held in memory (only ~30MB extracted).
    """
    # `split` may come via the known `split:` yaml key (→ source.hf_split)
    # or via `extra["split"]` if the user added it under any other name.
    split = source.hf_split or source.extra.get("split") or "devtest"
    url = source.url or "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"
    en_lang = source.extra.get("en_lang", "eng_Latn")
    es_lang = source.extra.get("es_lang", "spa_Latn")

    if url not in _FLORES_CACHE:
        log.info("fetching FLORES tarball %s", url)
        req = urllib.request.Request(url, headers={"User-Agent": "en-es-mt/0.1"})
        with urllib.request.urlopen(req) as resp:  # noqa: S310 - controlled URL
            data = resp.read()
        _FLORES_CACHE[url] = _parse_flores_tar(data)

    members = _FLORES_CACHE[url]
    en_key = f"{split}/{en_lang}.{split}"
    es_key = f"{split}/{es_lang}.{split}"
    if en_key not in members or es_key not in members:
        raise KeyError(f"FLORES tarball missing {en_key} or {es_key}; have keys: "
                       f"{sorted(k for k in members if split in k)[:6]}...")
    en_lines = members[en_key]
    es_lines = members[es_key]
    if len(en_lines) != len(es_lines):
        raise ValueError(f"FLORES {split}: en/es line count mismatch ({len(en_lines)} vs {len(es_lines)})")

    n = 0
    for en, es in zip(en_lines, es_lines):
        en, es = en.strip(), es.strip()
        if not en or not es:
            continue
        n += 1
        yield {"en": en, "es": es, "source": source.name}
        if cap is not None and n >= cap:
            break


def _parse_flores_tar(data: bytes) -> dict[str, list[str]]:
    """Extract every {split}/{lang}.{split} member to a list of lines."""
    out: dict[str, list[str]] = {}
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        for m in tar.getmembers():
            if not m.isfile():
                continue
            # Member paths look like "flores200_dataset/dev/eng_Latn.dev"
            parts = m.name.split("/")
            if len(parts) < 3:
                continue
            split = parts[-2]
            fname = parts[-1]
            if not fname.endswith(f".{split}"):
                continue
            key = f"{split}/{fname}"
            f = tar.extractfile(m)
            if f is None:
                continue
            out[key] = f.read().decode("utf-8").splitlines()
    return out


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
