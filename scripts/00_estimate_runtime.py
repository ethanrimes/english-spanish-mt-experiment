"""Project wall-clock for one or more tiers using runs/registry.jsonl.

Run this BEFORE submitting a large tier to Azure ML. It uses the
measured throughput from any completed prior run to extrapolate.

Usage:
  uv run python scripts/00_estimate_runtime.py 10000 50000 100000 500000 1000000 5000000

Output is a table with columns:
  Tier | Epochs | Examples seen | Basis run | ex/s | Train ETA | + Eval | Confidence

Confidence:
  measured     — we have a completed run for that exact tier
  extrapolated — using throughput from a different tier (usually fine for
                 same model + hardware)
  no_data      — no prior runs; submit a small tier first
"""
from __future__ import annotations

import argparse

from en_es_mt.train.eta import print_forecast_table
from en_es_mt.utils.io import load_yaml, repo_root
from en_es_mt.utils.logging import configure as configure_logging


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("tiers", nargs="+", type=int)
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--eval-seconds-per-pass", type=float, default=90.0,
                        help="Overhead added per eval cycle (FLORES-dev generation).")
    args = parser.parse_args()
    configure_logging()

    cfg = load_yaml(repo_root() / args.config)
    defaults = cfg["defaults"]

    epochs_per_tier: dict[int, float] = {}
    eval_passes_per_tier: dict[int, int] = {}
    for t in args.tiers:
        per = cfg.get("tiers", {}).get(t) or cfg.get("tiers", {}).get(str(t)) or {}
        ep = per.get("num_train_epochs", defaults["num_train_epochs"])
        eval_steps = per.get("eval_steps", defaults["eval_steps"])
        bs = defaults["per_device_train_batch_size"] * defaults["gradient_accumulation_steps"]
        total_steps = max(1, (t * ep) // bs)
        passes = max(1, total_steps // eval_steps)
        epochs_per_tier[t] = ep
        eval_passes_per_tier[t] = passes

    print_forecast_table(
        args.tiers,
        epochs_per_tier=epochs_per_tier,
        eval_passes_per_tier=eval_passes_per_tier,
        eval_seconds_per_pass=args.eval_seconds_per_pass,
    )


if __name__ == "__main__":
    main()
