"""W&B + MLflow + TensorBoard wiring.

HF Trainer already understands `report_to=["wandb","mlflow","tensorboard"]`,
so this module's main job is:
- read .env and surface the right env vars for HF Trainer
- name runs consistently (T{tier}-{git_sha[:7]}-{timestamp})
- log a handful of things HF Trainer doesn't log out of the box:
  - tokenizer-probe stats once at start
  - example translations every eval cycle
  - English-only perplexity (forgetting probe) at each eval cycle

The Trainer Callback for the above lives in train/callbacks.py — this
module is the place to centralize start-of-run setup.
"""
from __future__ import annotations

import datetime as dt
import logging
import os
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class TrackingConfig:
    project: str = "en-es-mt"
    entity: str | None = None
    run_name: str = ""
    tags: list[str] = field(default_factory=list)
    report_to: list[str] = field(default_factory=lambda: ["wandb", "mlflow", "tensorboard"])


def make_run_name(tier: int, git_sha: str | None = None) -> str:
    sha = (git_sha or "")[:7] or "nogit"
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%d-%H%M")
    return f"T{tier}-{sha}-{stamp}"


def configure_environment(tracking: TrackingConfig) -> None:
    """Set env vars Trainer reads at construction time. Safe to call repeatedly."""
    if "wandb" in tracking.report_to:
        # HF Trainer reads these
        os.environ.setdefault("WANDB_PROJECT", tracking.project)
        if tracking.entity:
            os.environ.setdefault("WANDB_ENTITY", tracking.entity)
        os.environ.setdefault("WANDB_RUN_GROUP", tracking.project)
        os.environ.setdefault("WANDB_LOG_MODEL", "false")  # we ship checkpoints to blob, not W&B
        os.environ.setdefault("WANDB_WATCH", "false")
    if "mlflow" in tracking.report_to:
        # If MLFLOW_TRACKING_URI isn't set, MLflow logs to ./mlruns by default.
        # Azure ML auto-sets it to the workspace tracking URI when running there.
        os.environ.setdefault("MLFLOW_EXPERIMENT_NAME", tracking.project)


def log_extra_run_metadata(extra: dict[str, Any]) -> None:
    """Best-effort: log a dict of run-level metadata to whichever
    tracker(s) are active. Called once at start of training."""
    try:
        import wandb  # type: ignore

        if wandb.run is not None:
            wandb.config.update(extra, allow_val_change=True)
    except Exception:
        pass
    try:
        import mlflow  # type: ignore

        if mlflow.active_run() is not None:
            for k, v in extra.items():
                try:
                    mlflow.log_param(k, v)
                except Exception:
                    pass
    except Exception:
        pass
