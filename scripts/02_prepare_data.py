"""Clean, dedupe, and build tiered training sets.

Usage:
  uv run python scripts/02_prepare_data.py
  uv run python scripts/02_prepare_data.py --tiers 10000 50000 100000 500000 1000000 5000000
  uv run python scripts/02_prepare_data.py --no-langid     # skip fasttext lang-id check

Pipeline:
  raw/{source}.jsonl
    → clean.py
        → interim/{source}.jsonl   (filtered, deduped per-source, normalized)
    → tiers.py
        → processed/T{size}/{train,val}.jsonl  +  processed/manifests/T{size}.json
"""
from __future__ import annotations

import argparse
from pathlib import Path

from en_es_mt.data.clean import LangID, clean_source
from en_es_mt.data.sources import active_sources, load_data_config
from en_es_mt.data.tiers import build_tiers
from en_es_mt.utils.io import repo_root, write_json
from en_es_mt.utils.logging import configure as configure_logging


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=["local", "azure"], default="local")
    parser.add_argument("--tiers", nargs="*", type=int, default=None,
                        help="Override tier sizes from data.yaml.")
    parser.add_argument("--no-langid", action="store_true",
                        help="Skip fasttext lang-id filtering (faster).")
    parser.add_argument("--only", nargs="*", default=None,
                        help="Clean only these source names.")
    args = parser.parse_args()

    configure_logging()
    cfg = load_data_config()
    root = repo_root()
    raw_dir = root / cfg.paths["raw_dir"]
    interim_dir = root / cfg.paths["interim_dir"]
    processed_dir = root / cfg.paths["processed_dir"]
    manifest_dir = root / cfg.paths["manifest_dir"]

    interim_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    sources = active_sources(cfg, environment=args.env)
    if args.only:
        keep = set(args.only)
        sources = [s for s in sources if s.name in keep]

    # --- Clean per source ---
    langid = LangID(enabled=not args.no_langid)
    cleaned_paths: dict[str, Path] = {}
    all_stats: dict[str, dict] = {}
    for s in sources:
        in_path = raw_dir / f"{s.name}.jsonl"
        out_path = interim_dir / f"{s.name}.jsonl"
        if not in_path.exists():
            print(f"# skipping {s.name}: {in_path} not present (run 01_download_data.py)")
            continue
        print(f"# cleaning {s.name} → {out_path}")
        stats = clean_source(in_path, out_path, rules=cfg.clean, langid=langid)
        all_stats[s.name] = stats
        cleaned_paths[s.name] = out_path
        print(f"  in={stats['in']} out={stats['out']} "
              f"(short={stats['drop_too_short']} long={stats['drop_too_long']} "
              f"ratio={stats['drop_length_ratio']} dup={stats['drop_dup']} "
              f"identical={stats['drop_identical']} langid={stats['drop_langid']})")

    write_json(manifest_dir / "clean_stats.json", all_stats)

    # --- Build tiers ---
    tier_sizes = args.tiers or cfg.tiers["sizes"]
    print(f"# building tiers: {tier_sizes}")
    weights = cfg.tiers["source_weights_per_tier"]
    # YAML may load int keys as str
    weights = {int(k): v for k, v in weights.items()}

    build_tiers(
        cleaned_paths=cleaned_paths,
        out_dir=processed_dir,
        manifest_dir=manifest_dir,
        tier_sizes=tier_sizes,
        source_weights_per_tier=weights,
        length_buckets=cfg.tiers["length_buckets"],
        val_fraction=cfg.tiers["val_fraction"],
        seed=cfg.tiers["seed"],
    )

    # --- English-only holdout for catastrophic-forgetting probe ---
    # Source: English side of FLORES-200 dev (held out from all training).
    _build_english_holdout(
        flores_dev=root / cfg.paths.get("eval_dir", "data/eval") / "flores200_dev.jsonl",
        out_path=root / cfg.paths.get("eval_dir", "data/eval") / "english_holdout.jsonl",
    )


def _build_english_holdout(flores_dev: Path, out_path: Path) -> None:
    from en_es_mt.utils.io import read_jsonl, write_jsonl

    if not flores_dev.exists():
        print(f"# skip english_holdout: {flores_dev} not present (run 01_download_data.py --include-eval)")
        return
    records = ({"text": r["en"]} for r in read_jsonl(flores_dev))
    n = write_jsonl(out_path, records)
    print(f"# wrote english_holdout.jsonl ({n} records) → {out_path}")


if __name__ == "__main__":
    main()
