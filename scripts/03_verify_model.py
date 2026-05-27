"""Smoke-test the base model: load it, run the Spanish-tokenization probe,
and write a markdown report to docs/MODEL_PROBE.md.

Usage:
  uv run python scripts/03_verify_model.py
  uv run python scripts/03_verify_model.py --n-samples 50
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from en_es_mt.model.loader import load_model_and_tokenizer
from en_es_mt.model.tokenizer_probe import format_report, probe
from en_es_mt.utils.io import load_yaml, read_jsonl, repo_root, write_json
from en_es_mt.utils.logging import configure as configure_logging


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-cfg", default="configs/model.yaml")
    parser.add_argument("--flores", default="data/eval/flores200_dev.jsonl",
                        help="Parallel jsonl used to probe ES tokenization.")
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--out-md", default="docs/MODEL_PROBE.md")
    parser.add_argument("--out-json", default="docs/MODEL_PROBE.json")
    parser.add_argument("--load-model", action="store_true",
                        help="Also load the full model weights (slow; tokenizer-only by default).")
    args = parser.parse_args()
    configure_logging()

    root = repo_root()
    model_cfg = load_yaml(root / args.model_cfg)
    flores_path = root / args.flores
    if not flores_path.exists():
        raise SystemExit(
            f"missing {flores_path} — run scripts/01_download_data.py --include-eval first"
        )

    if args.load_model:
        bundle = load_model_and_tokenizer(model_cfg, for_training=False)
        tokenizer = bundle.tokenizer
    else:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_cfg["model"]["hf_id"])

    records = list(read_jsonl(flores_path))[: args.n_samples]
    en = [r["en"] for r in records]
    es = [r["es"] for r in records]

    result = probe(tokenizer, en, es, n_examples=10)

    out_md = root / args.out_md
    out_json = root / args.out_json
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(format_report(result), encoding="utf-8")
    write_json(out_json, result.to_dict())
    print(f"# wrote {out_md}")
    print(f"# wrote {out_json}")
    print(f"# fertility_ratio (es/en) = {result.fertility_ratio:.2f}")


if __name__ == "__main__":
    main()
