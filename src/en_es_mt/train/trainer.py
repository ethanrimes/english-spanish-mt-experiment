"""End-to-end training entrypoint. Public API: `run_tier(...)`.

What this module does, in order:
1. Resolve per-tier training args by merging `defaults` ← `tiers[N]`.
2. Load base model + tokenizer (model/loader.py).
3. Stream the tier's jsonl files into a `datasets.Dataset`, formatted with
   our bidirectional prompt template (data/format.py).
4. Instantiate the HF Trainer with W&B + MLflow + TensorBoard reporting.
5. Attach generation-eval, forgetting-probe, and throughput callbacks.
6. Train. Save snapshots per the configured cadence.
7. Append a RunRecord to runs/registry.jsonl with throughput, durations,
   and the best eval scores so the ETA estimator can use them later.

The training script `scripts/04_train.py` is a thin CLI wrapper around
`run_tier` that supports `--tiers 10000 50000 100000` to queue multiple
runs sequentially.
"""
from __future__ import annotations

import logging
import os
import random
import time
from dataclasses import asdict
from pathlib import Path

from ..data.format import PromptFormatter, make_training_example
from ..model.loader import load_model_and_tokenizer
from ..obs.tracking import TrackingConfig, configure_environment, log_extra_run_metadata, make_run_name
from ..utils.io import load_yaml, read_jsonl, repo_root
from ..utils.seed import set_seed
from .callbacks import ForgettingProbeCallback, GenerationEvalCallback, ThroughputTracker, load_jsonl_field
from .collator import PadCollator
from .eta import RunRecord, append_run, current_git_sha, detect_hardware, now_utc

log = logging.getLogger(__name__)


def run_tier(
    tier: int,
    *,
    train_cfg_path: str | None = None,
    model_cfg_path: str | None = None,
    data_cfg_path: str | None = None,
    output_root: str | None = None,
    dry_run: bool = False,
) -> dict:
    """Run a single tier end-to-end. Returns a summary dict.

    Pass `dry_run=True` to validate config + data without training.
    """
    root = repo_root()
    train_cfg_path = train_cfg_path or str(root / "configs" / "train.yaml")
    model_cfg_path = model_cfg_path or str(root / "configs" / "model.yaml")
    data_cfg_path = data_cfg_path or str(root / "configs" / "data.yaml")
    train_cfg_raw = load_yaml(train_cfg_path)
    model_cfg = load_yaml(model_cfg_path)
    data_cfg = load_yaml(data_cfg_path)

    args_dict = _resolve_tier_args(train_cfg_raw, tier)
    seed = args_dict.get("seed", 20260526)
    set_seed(seed)

    output_root = output_root or args_dict["output_dir"]
    tier_output_dir = Path(output_root) / f"T{tier}"
    tier_output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Data ----
    processed_dir = Path(data_cfg["paths"]["processed_dir"]) / f"T{tier}"
    train_path = processed_dir / "train.jsonl"
    val_path = processed_dir / "val.jsonl"
    if not train_path.exists():
        raise FileNotFoundError(
            f"missing {train_path} — run scripts/02_prepare_data.py first to build tier T{tier}"
        )
    eval_dir = Path(data_cfg["paths"].get("eval_dir", "data/eval"))
    flores_dev_path = eval_dir / "flores200_dev.jsonl"
    flores_devtest_path = eval_dir / "flores200_devtest.jsonl"
    english_holdout_path = eval_dir / "english_holdout.jsonl"

    # ---- Model + tokenizer ----
    bundle = load_model_and_tokenizer(model_cfg, for_training=True)
    model, tokenizer = bundle.model, bundle.tokenizer

    # ---- Prompt formatter ----
    prompt_cfg = model_cfg["prompt"]
    formatter = PromptFormatter(
        src_lang_names=prompt_cfg["src_lang_names"],
        template=prompt_cfg["template"],
        direction_sampling=prompt_cfg.get("direction_sampling", "balanced"),
    )

    # ---- Build datasets via streaming generator ----
    train_records = list(read_jsonl(train_path))
    val_records = list(read_jsonl(val_path))
    log.info("loaded T%d: %d train / %d val", tier, len(train_records), len(val_records))

    rng = random.Random(seed)
    max_seq_len = args_dict.get("max_seq_len", 384)
    mask_loss = prompt_cfg.get("mask_loss_on_prompt", True)

    def _featurize(records: list[dict]) -> list[dict]:
        return [
            make_training_example(
                r, tokenizer=tokenizer, formatter=formatter,
                max_seq_len=max_seq_len, rng=rng, mask_loss_on_prompt=mask_loss,
            )
            for r in records
        ]

    train_feats = _featurize(train_records)
    val_feats = _featurize(val_records)

    if dry_run:
        return {
            "tier": tier, "train": len(train_feats), "val": len(val_feats),
            "first": {k: v[:32] if isinstance(v, list) else v for k, v in train_feats[0].items()},
        }

    # ---- Tracking ----
    git_sha = current_git_sha()
    run_name = make_run_name(tier, git_sha)
    tracking = TrackingConfig(
        project=os.environ.get("WANDB_PROJECT", "en-es-mt"),
        entity=os.environ.get("WANDB_ENTITY"),
        run_name=run_name,
        tags=train_cfg_raw.get("observability", {}).get("wandb", {}).get("tags", []),
        report_to=args_dict.get("report_to", ["wandb", "mlflow", "tensorboard"]),
    )
    configure_environment(tracking)

    # ---- HF Trainer ----
    from transformers import Trainer, TrainingArguments

    targs = TrainingArguments(
        output_dir=str(tier_output_dir),
        run_name=run_name,
        overwrite_output_dir=False,
        seed=seed,
        max_seq_length=max_seq_len if hasattr(TrainingArguments, "max_seq_length") else None,
        num_train_epochs=args_dict["num_train_epochs"],
        per_device_train_batch_size=args_dict["per_device_train_batch_size"],
        per_device_eval_batch_size=args_dict["per_device_eval_batch_size"],
        gradient_accumulation_steps=args_dict["gradient_accumulation_steps"],
        learning_rate=args_dict["learning_rate"],
        weight_decay=args_dict["weight_decay"],
        adam_beta1=args_dict["adam_beta1"],
        adam_beta2=args_dict["adam_beta2"],
        adam_epsilon=args_dict["adam_epsilon"],
        max_grad_norm=args_dict["max_grad_norm"],
        warmup_ratio=args_dict["warmup_ratio"],
        lr_scheduler_type=args_dict["lr_scheduler_type"],
        bf16=args_dict.get("bf16", True),
        fp16=args_dict.get("fp16", False),
        tf32=args_dict.get("tf32", True),
        gradient_checkpointing=args_dict.get("gradient_checkpointing", True),
        logging_steps=args_dict.get("logging_steps", 25),
        save_strategy=args_dict.get("save_strategy", "steps"),
        save_steps=args_dict["save_steps"],
        save_total_limit=args_dict["save_total_limit"],
        eval_strategy=args_dict.get("eval_strategy", "steps"),
        eval_steps=args_dict["eval_steps"],
        load_best_model_at_end=args_dict.get("load_best_model_at_end", True),
        metric_for_best_model=args_dict.get("metric_for_best_model", "eval_avg_bleu"),
        greater_is_better=args_dict.get("greater_is_better", True),
        report_to=tracking.report_to,
        disable_tqdm=args_dict.get("disable_tqdm", False),
        dataloader_drop_last=False,
        dataloader_num_workers=2,
        remove_unused_columns=False,
    )

    # ---- Callbacks ----
    callbacks = [ThroughputTracker()]
    gen_eval = args_dict.get("generation_eval", {})
    if gen_eval.get("enabled", True) and flores_dev_path.exists():
        flores_dev = list(read_jsonl(flores_dev_path))
        callbacks.append(GenerationEvalCallback(
            flores_dev_records=flores_dev,
            tokenizer=tokenizer,
            formatter=formatter,
            every_n_steps=gen_eval.get("every_n_steps", 500),
            num_samples=gen_eval.get("num_samples", 200),
            max_new_tokens=gen_eval.get("max_new_tokens", 192),
            num_beams=gen_eval.get("num_beams", 4),
            log_sample_translations=gen_eval.get("log_sample_translations", 20),
        ))
    else:
        log.warning("generation eval disabled or missing %s — only loss eval will run", flores_dev_path)

    fp = args_dict.get("forgetting_probe", {})
    if fp.get("enabled", True) and english_holdout_path.exists():
        en_texts = load_jsonl_field(str(english_holdout_path), "text")[: fp.get("num_samples", 500)]
        callbacks.append(ForgettingProbeCallback(
            english_holdout=en_texts,
            tokenizer=tokenizer,
            every_n_steps=fp.get("every_n_steps", 500),
        ))
    else:
        log.info("forgetting probe disabled or missing %s — skipping", english_holdout_path)

    collator = PadCollator(pad_token_id=tokenizer.pad_token_id)

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_feats,
        eval_dataset=val_feats,
        tokenizer=tokenizer,
        data_collator=collator,
        callbacks=callbacks,
    )

    # ---- Run-registry entry: started ----
    started_at = now_utc()
    run_started = time.time()

    log_extra_run_metadata({
        "tier": tier,
        "model_id": model_cfg["model"]["hf_id"],
        "train_examples": len(train_feats),
        "val_examples": len(val_feats),
        "git_sha": git_sha,
        "hardware": detect_hardware(),
        "max_seq_len": max_seq_len,
    })

    log.info("starting training — tier T%d, run %s", tier, run_name)
    train_output = trainer.train()
    log.info("training finished — %s", train_output.metrics)

    # ---- Final FLORES devtest eval if available ----
    final_metrics: dict[str, float] = {}
    if flores_devtest_path.exists():
        log.info("running final FLORES devtest eval...")
        from ..eval.metrics import score_translations
        from ..eval.translate import translate_batch

        devtest = list(read_jsonl(flores_devtest_path))
        en_src = [r["en"] for r in devtest]
        es_src = [r["es"] for r in devtest]
        for direction, src, ref in [("en2es", en_src, es_src), ("es2en", es_src, en_src)]:
            hyps = translate_batch(
                src, direction=direction, model=model, tokenizer=tokenizer,
                formatter=formatter,
                max_new_tokens=gen_eval.get("max_new_tokens", 192),
                num_beams=gen_eval.get("num_beams", 4),
            )
            scores = score_translations(hyps, ref, sources=src, compute_comet=False)
            final_metrics[f"final_{direction}_bleu"] = scores.bleu
            final_metrics[f"final_{direction}_chrf"] = scores.chrf
            final_metrics[f"final_{direction}_chrf++"] = scores.chrf_pp
        log.info("final FLORES devtest scores: %s", final_metrics)

    # ---- Throughput summary + registry write ----
    tp = next(cb for cb in callbacks if isinstance(cb, ThroughputTracker))
    tp_summary = tp.summary()

    record = RunRecord(
        tier=tier,
        started_at_utc=started_at,
        ended_at_utc=now_utc(),
        duration_sec=time.time() - run_started,
        status="completed",
        model=model_cfg["model"]["hf_id"],
        git_sha=git_sha,
        hardware=detect_hardware(),
        examples_per_sec=tp_summary["examples_per_sec"],
        steps_per_sec=tp_summary["steps_per_sec"],
        total_steps=int(tp_summary["total_steps"]),
        total_examples=int(tp_summary["total_examples"]),
        train_examples=len(train_feats),
        epochs=args_dict["num_train_epochs"],
        eval_bleu_en2es=final_metrics.get("final_en2es_bleu"),
        eval_bleu_es2en=final_metrics.get("final_es2en_bleu"),
        eval_chrf_en2es=final_metrics.get("final_en2es_chrf"),
        eval_chrf_es2en=final_metrics.get("final_es2en_chrf"),
        best_checkpoint=str(tier_output_dir / "best"),
        extras={"train_output": train_output.metrics, "args": args_dict},
    )
    append_run(record)
    log.info("registered run T%d → runs/registry.jsonl", tier)
    return asdict(record)


def _resolve_tier_args(train_cfg: dict, tier: int) -> dict:
    """Merge defaults with per-tier overrides. Nested dicts merged shallowly."""
    base = dict(train_cfg["defaults"])
    per_tier = train_cfg.get("tiers", {}).get(tier, {})
    if not per_tier:
        # YAML keys may load as int; try str as fallback.
        per_tier = train_cfg.get("tiers", {}).get(str(tier), {})
    for k, v in per_tier.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            merged = dict(base[k])
            merged.update(v)
            base[k] = merged
        else:
            base[k] = v
    return base
