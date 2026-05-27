"""Top-level Typer CLI: `en-es-mt <command>`.

This is a convenience wrapper. The numbered scripts under `scripts/` are
the recommended way to drive the project from a shell — they exist so a
shell history shows the experiment progression clearly. The CLI exists
for ad-hoc / IDE / Azure ML use.
"""
from __future__ import annotations

import logging

import typer

from .utils.logging import configure as configure_logging

app = typer.Typer(add_completion=False, help="English↔Spanish MT experiment driver.")

log = logging.getLogger(__name__)


@app.callback()
def _root(verbose: bool = typer.Option(False, "--verbose", "-v")):
    configure_logging("DEBUG" if verbose else "INFO")


@app.command()
def estimate(
    tiers: list[int] = typer.Argument(..., help="Tier sizes to estimate (e.g. 10000 50000)"),
    train_cfg: str = typer.Option("configs/train.yaml"),
):
    """Print a runtime forecast for the given tiers from runs/registry.jsonl."""
    from .train.eta import print_forecast_table
    from .utils.io import load_yaml

    cfg = load_yaml(train_cfg)
    defaults = cfg["defaults"]
    epochs = {}
    eval_passes = {}
    for t in tiers:
        per = cfg.get("tiers", {}).get(t) or cfg.get("tiers", {}).get(str(t)) or {}
        ep = per.get("num_train_epochs", defaults["num_train_epochs"])
        ev_every = per.get("eval_steps", defaults["eval_steps"])
        # estimate total steps for tier
        bs = defaults["per_device_train_batch_size"] * defaults["gradient_accumulation_steps"]
        total_steps = max(1, (t * ep) // bs)
        passes = max(1, total_steps // ev_every)
        epochs[t] = ep
        eval_passes[t] = passes
    print_forecast_table(tiers, epochs_per_tier=epochs, eval_passes_per_tier=eval_passes)


@app.command()
def train(
    tiers: list[int] = typer.Argument(..., help="Tier sizes to train (sequentially)"),
    train_cfg: str = typer.Option("configs/train.yaml"),
    model_cfg: str = typer.Option("configs/model.yaml"),
    data_cfg: str = typer.Option("configs/data.yaml"),
    output_root: str = typer.Option("checkpoints"),
    dry_run: bool = typer.Option(False, "--dry-run"),
):
    """Train one or more tiers sequentially. Each emits a RunRecord."""
    from .train.trainer import run_tier

    for t in tiers:
        log.info("=== T%d ===", t)
        run_tier(
            tier=t,
            train_cfg_path=train_cfg,
            model_cfg_path=model_cfg,
            data_cfg_path=data_cfg,
            output_root=output_root,
            dry_run=dry_run,
        )


@app.command()
def evaluate(
    checkpoint: str = typer.Argument(..., help="Path to a trained checkpoint dir"),
    eval_set: str = typer.Option("flores200_devtest"),
    model_cfg: str = typer.Option("configs/model.yaml"),
    output_json: str = typer.Option("eval.json"),
):
    """Evaluate a checkpoint on FLORES dev/devtest. Writes a JSON results file."""
    from pathlib import Path

    from .data.format import PromptFormatter
    from .eval.metrics import score_translations
    from .eval.translate import translate_batch
    from .utils.io import load_yaml, read_jsonl, write_json

    mcfg = load_yaml(model_cfg)
    prompt_cfg = mcfg["prompt"]
    formatter = PromptFormatter(
        src_lang_names=prompt_cfg["src_lang_names"],
        template=prompt_cfg["template"],
        direction_sampling=prompt_cfg.get("direction_sampling", "balanced"),
    )

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    import torch

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=dtype).to(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model.eval()

    eval_path = Path("data/eval") / f"{eval_set}.jsonl"
    records = list(read_jsonl(eval_path))
    en = [r["en"] for r in records]
    es = [r["es"] for r in records]

    results = {}
    for direction, src, ref in [("en2es", en, es), ("es2en", es, en)]:
        hyps = translate_batch(src, direction=direction, model=model, tokenizer=tokenizer, formatter=formatter)
        scores = score_translations(hyps, ref, sources=src, compute_comet=False)
        results[direction] = scores.__dict__

    write_json(output_json, {"checkpoint": checkpoint, "eval_set": eval_set, "results": results})
    log.info("wrote %s", output_json)
