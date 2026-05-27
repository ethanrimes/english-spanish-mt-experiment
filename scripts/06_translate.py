"""Interactive translation REPL / batch translator using a checkpoint.

Usage:
  # Translate stdin:
  echo "Hello, world." | uv run python scripts/06_translate.py --checkpoint checkpoints/T10k/best --to es

  # Translate a file (one sentence per line):
  uv run python scripts/06_translate.py --checkpoint ... --to es < src.txt > tgt.txt
"""
from __future__ import annotations

import argparse
import sys

from en_es_mt.data.format import PromptFormatter
from en_es_mt.eval.translate import translate_batch
from en_es_mt.utils.io import load_yaml, repo_root
from en_es_mt.utils.logging import configure as configure_logging


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--to", choices=["en", "es"], required=True,
                        help="Target language (input direction inferred as the other).")
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--model-config", default="configs/model.yaml")
    args = parser.parse_args()
    configure_logging("WARNING")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    mcfg = load_yaml(repo_root() / args.model_config)
    formatter = PromptFormatter(
        src_lang_names=mcfg["prompt"]["src_lang_names"],
        template=mcfg["prompt"]["template"],
    )
    direction = "en2es" if args.to == "es" else "es2en"

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint, torch_dtype=dtype)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    lines = [line.rstrip("\n") for line in sys.stdin if line.strip()]
    if not lines:
        print("# no input on stdin", file=sys.stderr)
        return
    outs = translate_batch(
        lines, direction=direction, model=model, tokenizer=tokenizer, formatter=formatter,
        max_new_tokens=args.max_new_tokens, num_beams=args.num_beams,
    )
    for o in outs:
        print(o)


if __name__ == "__main__":
    main()
