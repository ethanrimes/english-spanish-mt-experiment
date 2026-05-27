"""Custom HF Trainer callbacks.

1. `GenerationEvalCallback` — on every save/eval cycle, runs beam-search
   generation on a small FLORES-dev slice in both directions and logs
   BLEU/chrF to the active trackers. Also logs a handful of side-by-side
   sample translations.
2. `ForgettingProbeCallback` — every eval cycle, computes mean negative
   log-likelihood (== perplexity proxy) on a held-out English-only set so
   we can spot catastrophic forgetting of English as Spanish ramps up.
3. `ThroughputTracker` — accumulates examples/sec, tokens/sec, steps/sec
   into a moving average. Read at end-of-run to write the run registry.

These callbacks live in their own module so the main trainer.py stays
focused on optimization config.
"""
from __future__ import annotations

import logging
import statistics
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import torch
from transformers import TrainerCallback, TrainerControl, TrainerState
from transformers.training_args import TrainingArguments

from ..data.format import PromptFormatter
from ..eval.metrics import score_translations
from ..eval.translate import translate_batch
from ..utils.io import read_jsonl

log = logging.getLogger(__name__)


@dataclass
class GenerationEvalCallback(TrainerCallback):
    """Run BLEU/chrF generation eval on the FLORES dev set."""
    flores_dev_records: list[dict]         # records with 'en' and 'es' fields
    tokenizer: Any
    formatter: PromptFormatter
    every_n_steps: int
    num_samples: int
    max_new_tokens: int = 192
    num_beams: int = 4
    log_sample_translations: int = 20
    history: list[dict] = field(default_factory=list)

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if state.global_step == 0 or state.global_step % self.every_n_steps != 0:
            return control
        return self._run(args, state, control, **kwargs)

    def on_train_end(self, args, state, control, **kwargs):
        # Always run a final eval at end of training.
        return self._run(args, state, control, final=True, **kwargs)

    def _run(self, args, state, control, *, final: bool = False, **kwargs) -> TrainerControl:
        model = kwargs["model"]
        model.eval()
        sample = self.flores_dev_records[: self.num_samples] if not final else self.flores_dev_records
        en_texts = [r["en"] for r in sample]
        es_texts = [r["es"] for r in sample]

        results = {}
        samples_for_log = []
        for direction, src, ref in [("en2es", en_texts, es_texts), ("es2en", es_texts, en_texts)]:
            hyps = translate_batch(
                src, direction=direction, model=model, tokenizer=self.tokenizer,
                formatter=self.formatter, max_new_tokens=self.max_new_tokens,
                num_beams=self.num_beams,
            )
            scores = score_translations(hyps, ref)
            results[f"eval_{direction}_bleu"] = scores.bleu
            results[f"eval_{direction}_chrf"] = scores.chrf
            results[f"eval_{direction}_chrf++"] = scores.chrf_pp

            # A few samples for trackers.
            for s_i, (s, h, r) in enumerate(zip(src, hyps, ref)):
                if s_i >= self.log_sample_translations:
                    break
                samples_for_log.append({"direction": direction, "src": s, "hyp": h, "ref": r})

        avg_bleu = (results["eval_en2es_bleu"] + results["eval_es2en_bleu"]) / 2
        results["eval_avg_bleu"] = avg_bleu

        model.train()

        # Push into tracker(s). HF Trainer also logs via state.log_history.
        if hasattr(state, "log_history"):
            state.log_history.append({"step": state.global_step, **results})
        self.history.append({"step": state.global_step, **results})
        log.info("step %d | %s", state.global_step, {k: round(v, 2) for k, v in results.items()})

        _log_to_trackers(results, step=state.global_step)
        _log_samples(samples_for_log, step=state.global_step)
        return control


@dataclass
class ForgettingProbeCallback(TrainerCallback):
    """Compute mean NLL on a held-out English-only set every N steps."""
    english_holdout: list[str]
    tokenizer: Any
    every_n_steps: int
    history: list[dict] = field(default_factory=list)

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 0 or state.global_step % self.every_n_steps != 0:
            return control
        model = kwargs["model"]
        model.eval()
        device = next(model.parameters()).device
        nlls = []
        with torch.inference_mode():
            for text in self.english_holdout:
                enc = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
                out = model(**enc, labels=enc["input_ids"])
                nlls.append(out.loss.item())
        model.train()
        mean_nll = statistics.mean(nlls)
        ppl = float(torch.exp(torch.tensor(mean_nll)))
        log.info("step %d | english_holdout_ppl=%.2f", state.global_step, ppl)
        _log_to_trackers({"english_holdout_ppl": ppl, "english_holdout_nll": mean_nll}, step=state.global_step)
        self.history.append({"step": state.global_step, "ppl": ppl, "nll": mean_nll})
        return control


@dataclass
class ThroughputTracker(TrainerCallback):
    """Records examples/sec, tokens/sec, steps/sec across training.

    HF Trainer logs train_runtime + train_samples_per_second at end, but
    we want a running view + a clean read-back at end so we can persist
    to the run registry.
    """
    window: int = 50
    step_times: deque = field(default_factory=lambda: deque(maxlen=50))
    last_t: float = 0.0
    total_steps: int = 0
    total_examples: int = 0
    total_tokens: int = 0
    start_time: float = 0.0
    end_time: float = 0.0

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        self.last_t = self.start_time
        return control

    def on_step_end(self, args, state, control, **kwargs):
        now = time.time()
        dt_ = now - self.last_t
        self.last_t = now
        self.step_times.append(dt_)
        self.total_steps += 1
        # Effective per-step examples = per_device_bs * grad_accum * world_size
        bs = args.per_device_train_batch_size * max(1, args.gradient_accumulation_steps)
        try:
            world = max(1, args.world_size)
        except AttributeError:
            world = 1
        self.total_examples += bs * world
        return control

    def on_train_end(self, args, state, control, **kwargs):
        self.end_time = time.time()
        return control

    def summary(self) -> dict[str, float]:
        elapsed = max(1e-9, (self.end_time or time.time()) - self.start_time)
        return {
            "duration_sec": elapsed,
            "steps_per_sec": self.total_steps / elapsed,
            "examples_per_sec": self.total_examples / elapsed,
            "total_steps": float(self.total_steps),
            "total_examples": float(self.total_examples),
        }


# ---------- tracker fan-out ----------

def _log_to_trackers(values: dict[str, float], *, step: int) -> None:
    try:
        import wandb  # type: ignore

        if wandb.run is not None:
            wandb.log(values, step=step)
    except Exception:
        pass
    try:
        import mlflow  # type: ignore

        if mlflow.active_run() is not None:
            for k, v in values.items():
                try:
                    mlflow.log_metric(k, float(v), step=step)
                except Exception:
                    pass
    except Exception:
        pass


def _log_samples(samples: list[dict], *, step: int) -> None:
    if not samples:
        return
    try:
        import wandb  # type: ignore

        if wandb.run is not None:
            table = wandb.Table(columns=["direction", "src", "hypothesis", "reference"])
            for s in samples:
                table.add_data(s["direction"], s["src"], s["hyp"], s["ref"])
            wandb.log({"samples": table}, step=step)
    except Exception:
        pass


def load_jsonl_field(path: str, field: str) -> list[str]:
    return [rec[field] for rec in read_jsonl(path) if field in rec]
