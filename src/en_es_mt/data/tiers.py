"""Build nested data tiers (T10k ⊂ T50k ⊂ ... ⊂ T5M).

Strategy:
1. Concatenate every cleaned source's jsonl into one in-memory pool.
2. Global dedupe by (en, es).
3. For each tier size N, compute a *target count per source* from
   `source_weights_per_tier[N]` (renormalized to sum to N).
4. Within each source, distribute the per-source quota uniformly across
   length buckets so the tier isn't dominated by short OpenSubtitles lines
   or long Europarl speeches.
5. Sample without replacement, deterministic via seed.
6. **Nesting:** sample the largest tier first, then take subsets for the
   smaller tiers from the larger one — guarantees T_i ⊂ T_{i+1}. The
   smaller-tier draw still respects its own (different) source mix by
   stratified subsampling of the larger pool when possible, falling back
   to global-random where the per-source quota would exceed the available.
7. Write train/val splits + a manifest per tier.

Manifest contents (small JSON, committed to git):
- tier size
- seed
- source distribution achieved
- length-bucket distribution achieved
- train/val count, sha256 of each split file
- created-at timestamp
"""
from __future__ import annotations

import datetime as dt
import logging
import random
from collections import defaultdict
from pathlib import Path

from ..utils.io import read_jsonl, sha256_file, write_json, write_jsonl

log = logging.getLogger(__name__)


def build_tiers(
    cleaned_paths: dict[str, Path],
    *,
    out_dir: Path,
    manifest_dir: Path,
    tier_sizes: list[int],
    source_weights_per_tier: dict[int, dict[str, float]],
    length_buckets: list[int],
    val_fraction: float,
    seed: int,
) -> dict[int, dict]:
    """Build all tiers. Returns {tier_size: manifest}."""
    out_dir = Path(out_dir)
    manifest_dir = Path(manifest_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    log.info("loading cleaned sources into memory pool...")
    pool: list[dict] = []
    for name, p in cleaned_paths.items():
        n_added = 0
        for rec in read_jsonl(p):
            pool.append(rec)
            n_added += 1
        log.info("  %s: %d records", name, n_added)
    log.info("pool size: %d", len(pool))

    # Global dedupe (keep first occurrence to be deterministic with sorting).
    pool.sort(key=lambda r: r["id"])
    seen = set()
    deduped = []
    for rec in pool:
        key = (rec["en"], rec["es"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(rec)
    log.info("after global dedupe: %d", len(deduped))
    pool = deduped

    by_source: dict[str, list[dict]] = defaultdict(list)
    for rec in pool:
        by_source[rec["source"]].append(rec)
    for src in by_source:
        by_source[src].sort(key=lambda r: r["id"])  # determinism

    sorted_sizes = sorted(set(tier_sizes), reverse=True)
    manifests: dict[int, dict] = {}
    sampled_for_tier: dict[int, list[dict]] = {}

    # Largest tier first.
    largest = sorted_sizes[0]
    sampled_for_tier[largest] = _stratified_sample(
        by_source,
        target_n=largest,
        source_weights=_normalize_weights(source_weights_per_tier[largest]),
        length_buckets=length_buckets,
        seed=seed,
    )

    # Smaller tiers — sample from within the larger one for strict nesting.
    for size in sorted_sizes[1:]:
        parent = sampled_for_tier[sorted_sizes[sorted_sizes.index(size) - 1]]
        parent_by_source = defaultdict(list)
        for rec in parent:
            parent_by_source[rec["source"]].append(rec)
        sampled_for_tier[size] = _stratified_sample(
            parent_by_source,
            target_n=size,
            source_weights=_normalize_weights(source_weights_per_tier[size]),
            length_buckets=length_buckets,
            seed=seed + size,  # different seed offset per tier so order varies
            allow_fallback=True,
        )

    for size, records in sampled_for_tier.items():
        manifests[size] = _write_tier(
            records,
            size=size,
            out_dir=out_dir,
            manifest_dir=manifest_dir,
            val_fraction=val_fraction,
            length_buckets=length_buckets,
            seed=seed,
        )

    return manifests


def _normalize_weights(w: dict[str, float]) -> dict[str, float]:
    total = sum(w.values()) or 1.0
    return {k: v / total for k, v in w.items()}


def _stratified_sample(
    by_source: dict[str, list[dict]],
    *,
    target_n: int,
    source_weights: dict[str, float],
    length_buckets: list[int],
    seed: int,
    allow_fallback: bool = False,
) -> list[dict]:
    """Sample target_n records, respecting source quotas and length-bucket
    spread. Deterministic given (seed, by_source ordering).
    """
    rng = random.Random(seed)
    picked: list[dict] = []
    seen_ids: set[str] = set()

    for src, weight in source_weights.items():
        if src not in by_source:
            continue
        src_quota = round(target_n * weight)
        candidates = by_source[src]
        # Bucket by length on the EN side.
        buckets: dict[int, list[dict]] = defaultdict(list)
        for rec in candidates:
            buckets[_bucket(rec["en_len"], length_buckets)].append(rec)
        per_bucket_target = max(1, src_quota // max(1, len(buckets)))
        src_picked: list[dict] = []
        for _, recs in sorted(buckets.items()):
            if not recs:
                continue
            k = min(per_bucket_target, len(recs))
            sample = rng.sample(recs, k)
            src_picked.extend(sample)
        # Fill any remaining quota with random source samples.
        if len(src_picked) < src_quota:
            remaining = [r for r in candidates if r["id"] not in {x["id"] for x in src_picked}]
            extra = min(src_quota - len(src_picked), len(remaining))
            if extra > 0:
                src_picked.extend(rng.sample(remaining, extra))
        for r in src_picked:
            if r["id"] in seen_ids:
                continue
            seen_ids.add(r["id"])
            picked.append(r)

    # If we under-shot the target (a source had less data than the quota),
    # top up from any source. allow_fallback governs whether this is used.
    if len(picked) < target_n and allow_fallback:
        all_recs = [r for recs in by_source.values() for r in recs if r["id"] not in seen_ids]
        deficit = target_n - len(picked)
        if all_recs:
            picked.extend(rng.sample(all_recs, min(deficit, len(all_recs))))
    elif len(picked) < target_n:
        all_recs = [r for recs in by_source.values() for r in recs if r["id"] not in seen_ids]
        deficit = target_n - len(picked)
        if all_recs:
            picked.extend(rng.sample(all_recs, min(deficit, len(all_recs))))

    if len(picked) > target_n:
        picked = rng.sample(picked, target_n)

    rng.shuffle(picked)
    return picked


def _bucket(length: int, edges: list[int]) -> int:
    for i, edge in enumerate(edges):
        if length < edge:
            return i
    return len(edges)


def _write_tier(
    records: list[dict],
    *,
    size: int,
    out_dir: Path,
    manifest_dir: Path,
    val_fraction: float,
    length_buckets: list[int],
    seed: int,
) -> dict:
    """Write tier train/val + manifest. Returns the manifest dict."""
    tier_dir = out_dir / f"T{size}"
    tier_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed + size + 1)
    shuffled = list(records)
    rng.shuffle(shuffled)
    n_val = max(50, int(round(len(shuffled) * val_fraction)))
    val_records = shuffled[:n_val]
    train_records = shuffled[n_val:]

    train_path = tier_dir / "train.jsonl"
    val_path = tier_dir / "val.jsonl"
    write_jsonl(train_path, train_records)
    write_jsonl(val_path, val_records)

    source_counts: dict[str, int] = defaultdict(int)
    bucket_counts: dict[int, int] = defaultdict(int)
    for r in records:
        source_counts[r["source"]] += 1
        bucket_counts[_bucket(r["en_len"], length_buckets)] += 1

    manifest = {
        "tier": size,
        "created_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "seed": seed,
        "counts": {
            "train": len(train_records),
            "val": len(val_records),
            "total": len(records),
        },
        "source_distribution": dict(source_counts),
        "length_bucket_distribution": {str(k): v for k, v in bucket_counts.items()},
        "length_bucket_edges": length_buckets,
        "files": {
            "train": {"path": str(train_path.relative_to(out_dir.parent.parent)), "sha256": sha256_file(train_path)},
            "val": {"path": str(val_path.relative_to(out_dir.parent.parent)), "sha256": sha256_file(val_path)},
        },
    }
    manifest_path = manifest_dir / f"T{size}.json"
    write_json(manifest_path, manifest)
    log.info("tier T%d: %d train + %d val → %s", size, len(train_records), len(val_records), manifest_path)
    return manifest
