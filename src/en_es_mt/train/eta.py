"""Run registry + ETA estimator.

Every training run appends one line to `runs/registry.jsonl` with measured
throughput. Given that registry, we can project how long a future run
(typically a larger data tier) will take *before* committing to it.

Why this matters: a T5M run on a single A100 is ~10–30x longer than T500k.
You want to see T10k → T500k results land first, look at the
quality-vs-data curve, then *decide* whether T1M and T5M are worth the
GPU budget. This module makes that decision data-driven instead of
"how do I feel about it."
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from ..utils.io import repo_root

log = logging.getLogger(__name__)


def registry_path() -> Path:
    return repo_root() / "runs" / "registry.jsonl"


@dataclass
class RunRecord:
    tier: int
    started_at_utc: str
    ended_at_utc: str | None = None
    duration_sec: float | None = None
    status: str = "running"            # running | completed | failed | aborted
    model: str = ""
    git_sha: str = ""
    hardware: str = ""                  # e.g. "1x A100 80GB"
    examples_per_sec: float | None = None
    tokens_per_sec: float | None = None
    steps_per_sec: float | None = None
    total_steps: int | None = None
    total_examples: int | None = None
    train_examples: int | None = None
    epochs: float | None = None
    eval_bleu_en2es: float | None = None
    eval_bleu_es2en: float | None = None
    eval_chrf_en2es: float | None = None
    eval_chrf_es2en: float | None = None
    best_checkpoint: str | None = None
    notes: str = ""
    extras: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


def append_run(record: RunRecord) -> None:
    p = registry_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")


def load_registry() -> list[dict]:
    p = registry_path()
    if not p.exists():
        return []
    out = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                log.warning("skipping malformed registry line: %r", line[:80])
    return out


def current_git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root(), stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return ""


def now_utc() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def detect_hardware() -> str:
    """Best-effort hardware string for the registry."""
    try:
        import torch

        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            name = torch.cuda.get_device_name(0)
            mem_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3))
            return f"{n}x {name} {mem_gb}GB"
    except Exception:
        pass
    return "cpu"


# ---------- ETA estimation ----------

@dataclass
class TierForecast:
    tier: int
    epochs: float
    examples_seen: int
    basis_run_tier: int | None
    basis_examples_per_sec: float | None
    est_duration_sec: float | None
    eval_overhead_sec: float
    confidence: str  # "measured" | "extrapolated" | "no_data"

    def pretty_duration(self) -> str:
        if self.est_duration_sec is None:
            return "?"
        return _fmt_duration(self.est_duration_sec + self.eval_overhead_sec)


def estimate_tier_duration(
    target_tier: int,
    *,
    target_epochs: float,
    eval_passes: int,
    eval_seconds_per_pass: float = 90.0,
) -> TierForecast:
    """Project wall time for a target tier from registry history.

    Algorithm:
    - Find the most recent *completed* run with measured `examples_per_sec`.
    - examples_seen = target_tier * target_epochs
    - est_duration = examples_seen / examples_per_sec
    - eval_overhead = eval_passes * eval_seconds_per_pass

    Confidence:
    - "measured" if we have a completed run on the same tier
    - "extrapolated" if only smaller-tier runs are available
    - "no_data" if registry is empty
    """
    runs = [r for r in load_registry() if r.get("status") == "completed" and r.get("examples_per_sec")]
    examples_seen = int(target_tier * target_epochs)
    eval_overhead = eval_passes * eval_seconds_per_pass

    if not runs:
        return TierForecast(
            tier=target_tier, epochs=target_epochs, examples_seen=examples_seen,
            basis_run_tier=None, basis_examples_per_sec=None,
            est_duration_sec=None, eval_overhead_sec=eval_overhead,
            confidence="no_data",
        )

    same_tier = [r for r in runs if r.get("tier") == target_tier]
    if same_tier:
        chosen = same_tier[-1]
        confidence = "measured"
    else:
        chosen = runs[-1]   # most recent completed run, any tier
        confidence = "extrapolated"

    eps = float(chosen["examples_per_sec"])
    return TierForecast(
        tier=target_tier, epochs=target_epochs, examples_seen=examples_seen,
        basis_run_tier=int(chosen["tier"]), basis_examples_per_sec=eps,
        est_duration_sec=examples_seen / max(eps, 1e-9),
        eval_overhead_sec=eval_overhead,
        confidence=confidence,
    )


def _fmt_duration(seconds: float) -> str:
    hours, rem = divmod(int(seconds), 3600)
    mins, secs = divmod(rem, 60)
    if hours >= 24:
        days, h = divmod(hours, 24)
        return f"{days}d{h}h{mins:02d}m"
    if hours:
        return f"{hours}h{mins:02d}m"
    if mins:
        return f"{mins}m{secs:02d}s"
    return f"{secs}s"


def print_forecast_table(
    tiers: list[int],
    *,
    epochs_per_tier: dict[int, float],
    eval_passes_per_tier: dict[int, int],
    eval_seconds_per_pass: float = 90.0,
) -> None:
    """Pretty-print a decision table for the requested tiers. Cheap; no W&B."""
    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(title="Tier runtime forecast", show_lines=False)
        table.add_column("Tier", justify="right")
        table.add_column("Epochs", justify="right")
        table.add_column("Examples seen", justify="right")
        table.add_column("Basis", justify="left")
        table.add_column("ex/s", justify="right")
        table.add_column("Train ETA", justify="right")
        table.add_column("+ Eval", justify="right")
        table.add_column("Confidence", justify="left")

        for t in tiers:
            f = estimate_tier_duration(
                t,
                target_epochs=epochs_per_tier.get(t, 1.0),
                eval_passes=eval_passes_per_tier.get(t, 1),
                eval_seconds_per_pass=eval_seconds_per_pass,
            )
            basis = "—" if f.basis_run_tier is None else f"T{f.basis_run_tier}"
            eps = "—" if f.basis_examples_per_sec is None else f"{f.basis_examples_per_sec:.1f}"
            train_eta = _fmt_duration(f.est_duration_sec) if f.est_duration_sec else "?"
            full_eta = f.pretty_duration() if f.est_duration_sec else "?"
            table.add_row(
                f"T{t}", f"{f.epochs:g}", f"{f.examples_seen:,}",
                basis, eps, train_eta, full_eta, f.confidence,
            )
        console.print(table)
    except ImportError:
        # Plain-text fallback
        print("tier\tepochs\texamples\tbasis\tex/s\ttrain_eta\teval_overhead\tconfidence")
        for t in tiers:
            f = estimate_tier_duration(
                t,
                target_epochs=epochs_per_tier.get(t, 1.0),
                eval_passes=eval_passes_per_tier.get(t, 1),
                eval_seconds_per_pass=eval_seconds_per_pass,
            )
            print(
                f"T{t}\t{f.epochs}\t{f.examples_seen}\t"
                f"{'T' + str(f.basis_run_tier) if f.basis_run_tier else '-'}\t"
                f"{f.basis_examples_per_sec or '-'}\t{f.est_duration_sec or '-'}\t"
                f"{f.eval_overhead_sec}\t{f.confidence}"
            )


def env_str(key: str, default: str = "") -> str:
    return os.environ.get(key, default)
