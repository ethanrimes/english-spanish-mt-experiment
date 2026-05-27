"""Train one or more tiers sequentially.

Each tier writes a RunRecord to runs/registry.jsonl which the ETA estimator
reads. You can pick any subset of tiers — run small ones first, see results,
then decide whether the large ones are worth the GPU time.

Usage:
  uv run python scripts/04_train.py --tiers 10000
  uv run python scripts/04_train.py --tiers 10000 50000 100000
  uv run python scripts/04_train.py --tiers 500000 --dry-run   # validate without GPU time

After running a small tier, use scripts/00_estimate_runtime.py to project
how long a larger tier will take given the measured throughput.
"""
from __future__ import annotations

import argparse

from en_es_mt.train.trainer import run_tier
from en_es_mt.utils.logging import configure as configure_logging


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiers", nargs="+", type=int, required=True,
                        help="One or more tier sizes (e.g. 10000 50000).")
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--model-config", default="configs/model.yaml")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--output-root", default="checkpoints")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate config + data without training.")
    parser.add_argument("--data-dir", default=None,
                        help="(Azure ML) Override processed-data root.")
    args = parser.parse_args()
    configure_logging()

    # Azure ML pumps the input data mount path via --data-dir; if set,
    # we substitute the processed-data root in the data config.
    if args.data_dir:
        import yaml
        from en_es_mt.utils.io import repo_root

        cfg_path = repo_root() / args.data_config
        cfg = yaml.safe_load(cfg_path.read_text())
        cfg["paths"]["processed_dir"] = args.data_dir
        tmp = cfg_path.with_suffix(".azure.yaml")
        tmp.write_text(yaml.safe_dump(cfg))
        args.data_config = str(tmp)

    for tier in args.tiers:
        print(f"\n===== T{tier} =====\n")
        run_tier(
            tier=tier,
            train_cfg_path=args.config,
            model_cfg_path=args.model_config,
            data_cfg_path=args.data_config,
            output_root=args.output_root,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
