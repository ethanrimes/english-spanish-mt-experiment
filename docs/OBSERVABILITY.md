# Observability + rollback

## What's logged where

Every training run logs to three places simultaneously:

| Target | What lands there | When |
| --- | --- | --- |
| **Weights & Biases** | Train loss, eval BLEU/chrF, English forgetting PPL, side-by-side sample translations, GPU/CPU/RAM utilization | Live during training |
| **MLflow** (Azure ML auto-attached) | Same metrics + run params + final artifacts pointer | Live during training |
| **TensorBoard** | Train loss curves | Local file + Azure ML viewer |
| **`runs/registry.jsonl`** | One line per run: tier, throughput, durations, best eval scores, git SHA, hardware | At run end (or crash, where possible) |
| **Per-checkpoint dir** | Model weights, tokenizer, optimizer state, `trainer_state.json` | Every `save_steps` |
| **HF Trainer state log** | All metrics in `trainer_state.json` | Saved with each checkpoint |

## Metrics that matter

| Metric | Where | Why |
| --- | --- | --- |
| `train/loss` | W&B / MLflow / TB | Standard. Watch for spikes (LR too high) or plateaus (bad data). |
| `eval_en2es_bleu` / `eval_es2en_bleu` | W&B / MLflow | The headline numbers. Reported on FLORES dev during training. |
| `eval_avg_bleu` | W&B / MLflow | Mean of the two directions. Drives `metric_for_best_model`. |
| `english_holdout_ppl` | W&B / MLflow | **Forgetting probe.** Mean PPL on held-out English. If this climbs while BLEU goes up, the model is sacrificing English to gain Spanish — that's catastrophic forgetting. |
| `examples_per_sec` | TB + registry | Throughput. Feeds the ETA estimator. |
| `samples` (W&B table) | W&B | 20 side-by-side translations per eval cycle. Quickest qualitative check. |

## Rollback path (catastrophic forgetting)

The point of the snapshot trail is that *every* training run produces multiple checkpoints — by default the rolling last `save_total_limit` plus the best-by-eval-BLEU plus the final. If the forgetting probe spikes mid-run, the workflow is:

1. Identify the step where English PPL started climbing (W&B chart).
2. Find the highest-step checkpoint **before** the spike whose eval BLEU is still acceptable.
3. Restore that checkpoint via `--checkpoint checkpoints/T{tier}/checkpoint-<step>` for eval, or resume training from it with a lower LR.

Concrete commands:

```powershell
# Evaluate a specific snapshot
uv run python scripts/05_evaluate.py --checkpoint checkpoints/T100k/checkpoint-2500

# Translate sample text with a specific snapshot
"Hello, world." | uv run python scripts/06_translate.py --checkpoint checkpoints/T100k/checkpoint-2500 --to es
```

## Per-run registry

`runs/registry.jsonl` is the single source of truth for "what was actually run." Sample entry:

```json
{
  "tier": 10000,
  "started_at_utc": "2026-05-27T01:23:45+00:00",
  "ended_at_utc":   "2026-05-27T01:54:11+00:00",
  "duration_sec":   1826,
  "status": "completed",
  "model": "HuggingFaceTB/SmolLM2-360M",
  "git_sha": "9b3d2cf...",
  "hardware": "1x NVIDIA A100-SXM4-80GB 80GB",
  "examples_per_sec": 78.2,
  "total_examples": 64000,
  "epochs": 8,
  "eval_bleu_en2es": 12.4,
  "eval_bleu_es2en": 14.1,
  "best_checkpoint": "checkpoints/T10000/best"
}
```

The ETA estimator (`scripts/00_estimate_runtime.py`) uses `examples_per_sec` from the most recent completed run to project larger tiers.

## Inspecting a finished run

```powershell
# 1. Decision table for future tiers given history
uv run python scripts/00_estimate_runtime.py 100000 500000 1000000 5000000

# 2. Per-checkpoint loss/BLEU history (raw)
cat checkpoints/T10000/trainer_state.json | jq '.log_history'

# 3. Open W&B project (CLI helper)
wandb projects en-es-mt
```

## Crash recovery

If a job crashes mid-training, the partial state is on disk:

- The latest `checkpoint-<step>` directory is fully resumable.
- Re-run `scripts/04_train.py --tiers <N>` from the same output dir; HF Trainer finds the latest checkpoint and resumes.
- The run record won't be in `runs/registry.jsonl` for the crashed run — that's intentional, since `examples_per_sec` from a crashed run is misleading for ETA.
