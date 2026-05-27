"""Evaluate a trained checkpoint on FLORES devtest (both directions).

Usage:
  uv run python scripts/05_evaluate.py --checkpoint checkpoints/T10k/best
  uv run python scripts/05_evaluate.py --checkpoint checkpoints/T100k/checkpoint-2000 --comet
"""
from __future__ import annotations

import argparse
from pathlib import Path

from en_es_mt.data.format import PromptFormatter
from en_es_mt.eval.metrics import score_translations
from en_es_mt.eval.translate import translate_batch
from en_es_mt.utils.io import load_yaml, read_jsonl, repo_root, write_json
from en_es_mt.utils.logging import configure as configure_logging


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--eval-set", default="flores200_devtest")
    parser.add_argument("--model-config", default="configs/model.yaml")
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--comet", action="store_true",
                        help="Also compute COMET (downloads ~1.5GB model).")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    configure_logging()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    mcfg = load_yaml(repo_root() / args.model_config)
    formatter = PromptFormatter(
        src_lang_names=mcfg["prompt"]["src_lang_names"],
        template=mcfg["prompt"]["template"],
    )

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint, torch_dtype=dtype)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    records = list(read_jsonl(Path("data/eval") / f"{args.eval_set}.jsonl"))
    en = [r["en"] for r in records]
    es = [r["es"] for r in records]
    print(f"# evaluating {args.checkpoint} on {args.eval_set} ({len(records)} pairs)")

    results = {}
    for direction, src, ref in [("en2es", en, es), ("es2en", es, en)]:
        hyps = translate_batch(
            src, direction=direction, model=model, tokenizer=tokenizer, formatter=formatter,
            max_new_tokens=args.max_new_tokens, num_beams=args.num_beams,
        )
        scores = score_translations(hyps, ref, sources=src, compute_comet=args.comet)
        results[direction] = {
            "bleu": scores.bleu, "chrf": scores.chrf, "chrf_pp": scores.chrf_pp,
            "comet": scores.comet, "n": scores.n,
        }
        print(f"  {direction}: BLEU={scores.bleu:.2f}  chrF={scores.chrf:.2f}  chrF++={scores.chrf_pp:.2f}"
              + (f"  COMET={scores.comet:.4f}" if scores.comet is not None else ""))

    out = {
        "checkpoint": args.checkpoint,
        "eval_set": args.eval_set,
        "num_beams": args.num_beams,
        "results": results,
    }
    out_path = args.output or f"eval_{Path(args.checkpoint).name}_{args.eval_set}.json"
    write_json(out_path, out)
    print(f"# wrote {out_path}")


if __name__ == "__main__":
    main()
