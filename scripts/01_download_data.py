"""Download every enabled source corpus to data/raw/.

Usage:
  uv run python scripts/01_download_data.py            # local sample
  uv run python scripts/01_download_data.py --env azure   # full pull

Idempotent: re-running skips sources whose output jsonl already exists at
the configured cap. Delete the .jsonl to force a re-download.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from en_es_mt.data.download import download_source
from en_es_mt.data.sources import active_sources, load_data_config
from en_es_mt.utils.io import repo_root
from en_es_mt.utils.logging import configure as configure_logging


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=["local", "azure"], default="local",
                        help="Selects which per-source cap to apply.")
    parser.add_argument("--only", nargs="*", default=None,
                        help="Restrict to a subset of source names.")
    parser.add_argument("--include-eval", action="store_true",
                        help="Also download FLORES eval sets (HuggingFace).")
    args = parser.parse_args()

    configure_logging()
    cfg = load_data_config()
    raw_dir = repo_root() / cfg.paths["raw_dir"]
    eval_dir = repo_root() / cfg.paths.get("eval_dir", "data/eval")

    sources = active_sources(cfg, environment=args.env)
    if args.only:
        keep = set(args.only)
        sources = [s for s in sources if s.name in keep]

    print(f"# Downloading {len(sources)} source(s) to {raw_dir} (env={args.env})")
    for s in sources:
        download_source(s, raw_dir=raw_dir, environment=args.env)

    if args.include_eval:
        print(f"# Downloading eval sets to {eval_dir}")
        for s in cfg.eval:
            download_source(s, raw_dir=eval_dir, environment=args.env)


if __name__ == "__main__":
    main()
