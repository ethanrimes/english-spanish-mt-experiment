# vast.ai setup — step-by-step

A cheaper, no-quota alternative to Azure for this project. ~25% the cost per GPU-hour, instant provisioning, same SKUs (A10, A100, H100).

**Time required:** ~15 min the first time, ~5 min per subsequent training run.

---

## What you need

| Tool | How to verify | Install link |
| --- | --- | --- |
| vast.ai account with funds (≥ $10 to start) | https://cloud.vast.ai/billing/ | https://cloud.vast.ai |
| `vastai` CLI | `vastai --version` (installed by `uv sync --extra vastai`) | auto |
| `ssh` client | `ssh -V` (Windows 10+ has OpenSSH built-in) | OS |
| Either: PowerShell + native `scp`, OR `rsync` via WSL/Git-Bash | for data upload | OS |

> **Funding tip.** Start with $20. A full T10k–T1M sweep on an A10 costs ~$10 of GPU time + bootstrap overhead. T5M alone is ~$15.

---

## Step 1 — Sign up and get an API key

1. https://cloud.vast.ai/account/ — register (Google/GitHub OAuth works fine).
2. Add a payment method and put $20 on the account.
3. https://cloud.vast.ai/cli/ — copy your **API key**. It's a 64-char hex string starting with `<your-username>:`.

You'll paste this into `.env` in step 5. Don't share it elsewhere — anyone with this key can spend your money.

---

## Step 2 — Install the vast CLI locally

The CLI is included as an optional extra in the project:

```powershell
uv sync --extra vastai --extra cpu --extra dev
uv run vastai --version
```

If that fails, the CLI is also available via pip standalone:
```powershell
uv tool install vastai
```

Authenticate the CLI once with your API key:
```powershell
uv run vastai set api-key <your-key>
```

---

## Step 3 — Pick a GPU instance

Search the marketplace for available offers. The helper script applies our defaults (A10 24GB or A100 40/80GB, datacenter-quality, recent driver):

```powershell
uv run python scripts/vastai_search.py
```

Sample output:

```
ID         GPU             VRAM   Price/hr  DLPerf  Country  Internet  Reliability
12345678   RTX A10         24GB   $0.31     45.2    US       1000Mbps  0.97
23456789   RTX A6000       48GB   $0.45     58.1    US       2500Mbps  0.99
34567890   A100 SXM4       80GB   $1.04     112.4   DE        500Mbps  0.98
```

Picking criteria:
- **GPU model** — A10 24GB is the sweet spot for SmolLM2-360M. A6000 48GB or A100 if you want faster.
- **Reliability** > 0.95
- **Internet** ≥ 500 Mbps (matters for data upload and HF downloads)
- **Price/hr** sane (< $0.50 for A10, < $1.50 for A100)

> Note your chosen offer ID — you'll pass it to `vastai_provision.py` next.

---

## Step 4 — Provision the instance

```powershell
uv run python scripts/vastai_provision.py --offer-id 12345678
```

What it does:
1. Creates an instance from the offer ID with our PyTorch image (`pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime`).
2. Polls until it's running (~60–120 s) and SSH is reachable.
3. Writes `VAST_INSTANCE_ID`, `VAST_SSH_HOST`, `VAST_SSH_PORT` into `.env`.
4. Prints the SSH command you can use to connect manually.

Sample output:

```
[OK] created instance 87654321
[wait] provisioning... 30s
[wait] provisioning... 60s
[OK] running. SSH: ssh -p 32100 root@ssh4.vast.ai
[OK] wrote VAST_* keys to .env
```

The instance is **now billing**. Cost meter starts immediately. Set a teardown reminder for yourself, or use `--auto-destroy-hours N` on the provision command to set a hard timeout.

---

## Step 5 — Fill `.env`

Most fields are now filled by `vastai_provision.py`. The remaining ones you set manually:

```ini
# Vast.ai (filled mostly by scripts/vastai_provision.py)
VAST_API_KEY=...                # from step 1
VAST_INSTANCE_ID=87654321       # filled
VAST_SSH_HOST=ssh4.vast.ai      # filled
VAST_SSH_PORT=32100             # filled
VAST_SSH_USER=root              # default

# W&B (cross-platform, same as Azure path)
WANDB_API_KEY=...               # from wandb.ai/authorize
WANDB_PROJECT=en-es-mt
WANDB_ENTITY=                   # blank for personal account
```

---

## Step 6 — Upload data + bootstrap the instance

Three data-transfer options, pick one:

### Option A — `scp` upload from local laptop (recommended for first run)
~500 MB of processed tiers + eval → 5–10 min depending on uplink:

```powershell
uv run python scripts/vastai_sync.py --upload
```

This wraps `scp` with the SSH coordinates from `.env`. PowerShell-native, no rsync needed.

### Option B — Re-download fresh on the remote
No upload needed; the bootstrap re-runs `01_download_data.py` + `02_prepare_data.py` on the rented box (fast datacenter bandwidth). Adds ~10 min of GPU time at start (~$0.05).

Skip step 6 entirely and pass `--download-fresh` to the bootstrap in step 7.

### Option C — HuggingFace Hub private dataset (best for repeated runs)
Upload once via `scripts/upload_to_hf.py`, then any future instance (vast, Modal, etc.) just `huggingface-cli download <repo>`. Best if you'll spin up multiple instances over weeks.

---

## Step 7 — Run training

```powershell
# Bootstrap the remote (one-time per instance lifetime)
uv run python scripts/vastai_run.py --bootstrap

# Run a tier
uv run python scripts/vastai_run.py --tiers 10000

# Stream live logs (same as if local)
uv run python scripts/vastai_run.py --tail-logs
```

`vastai_run.py` is a thin wrapper around ssh-and-run. Under the hood:

1. `ssh root@host -p PORT 'cd /workspace/en-es-mt && uv run python scripts/04_train.py --tiers 10000'`
2. Streams stdout live to your local terminal
3. Reads the run registry that gets written under `/workspace/en-es-mt/runs/registry.jsonl`

After a run finishes:
```powershell
# Pull checkpoints + manifests back locally
uv run python scripts/vastai_sync.py --download checkpoints runs
```

---

## Step 8 — Tear down when done

**Critical.** vast.ai bills per second the instance is running. Forget to stop it and you wake up to a much bigger bill.

```powershell
# Stop (keeps storage, can resume; cheap idle ~$0.001/hr)
uv run python scripts/vastai_destroy.py --stop

# Destroy (irreversible, wipes storage, no further billing)
uv run python scripts/vastai_destroy.py --destroy
```

Or in the web UI: https://cloud.vast.ai/instances/ → click the trash icon.

---

## Cost reference

A typical full sweep with A10 24GB at ~$0.32/hr on vast:

| Tier  | Wall time | Cost   |
| ----- | --------- | ------ |
| T10k  | ~75 min   | ~$0.40 |
| T50k  | ~3.5 hr   | ~$1.10 |
| T100k | ~6 hr     | ~$1.90 |
| T500k | ~15 hr    | ~$4.80 |
| T1M   | ~25 hr    | ~$8.00 |
| T5M   | ~110 hr   | ~$35   |
| **Full sweep T10k–T5M** | ~165 hr | **~$50** |

That's ~3× cheaper than Azure A10 list price for the same compute.

---

## Trade-offs vs Azure

| | vast.ai | Azure ML |
|---|---|---|
| Provisioning time | ~60 s | ~5 min (compute cluster) |
| GPU cost per hour | $0.30–$0.40 (A10) | $1.20 (A10) |
| Quota / approval | none | required, can take days |
| Managed MLflow / artifact tracking | no (use W&B) | yes (auto-attached) |
| SSH access | yes | no (jobs only) |
| Reliability | per-host basis (filter > 0.97) | very high |
| Multi-GPU jobs | yes (offers list dual/quad A6000/A100) | yes |
| Spot pricing | yes (called "Interruptible") | yes |
| Data persistence | per-instance disk + their cloud sync | blob storage |

vast.ai is the right call for research / one-shot experiments. Azure ML is the right call for production / long-lived MLops.
