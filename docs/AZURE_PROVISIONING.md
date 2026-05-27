# Azure provisioning — step-by-step

End-to-end walkthrough: you provision the Azure resources from your account, paste a handful of values into `.env`, and I (or whoever's running training) picks them up automatically. Nothing here touches secrets that need to leave your machine.

**Time required:** ~15 min active, +20–30 min for the GPU compute cluster to provision in the background.

---

## What you need before you start

| Tool | How to verify | Install link |
| --- | --- | --- |
| Azure subscription with quota for **NV A10 v5** family (or NC A100 v4) | Run `az vm list-usage -l eastus -o table` after step 2; check `aml` quota too | n/a — assume yes |
| `az` CLI ≥ 2.60 | `az --version` | https://learn.microsoft.com/cli/azure/install-azure-cli |
| `az ml` extension | `az extension show -n ml` (will install on first use) | auto |
| PowerShell 7 | `pwsh --version` (you have it; the repo already uses it) | n/a |
| W&B account (free tier OK) | `wandb login` after registering | https://wandb.ai |

> **Quota note.** GPU SKUs are quota-gated per region. If `az vm list-usage` shows 0 cores available for `Standard NVADSA10v5 Family vCPUs`, request an increase: Portal → Subscription → Usage + quotas → search "NVADSA10v5" → New Quota Request. Approval is usually under an hour. Same applies to `Standard NCADSA100v4 Family vCPUs` for the A100 fallback.

---

## Step 1 — Pick your names

You'll need names for four things. Pick now so the rest of the steps just substitute. Defaults shown are fine.

| Variable | Meaning | Suggested |
| --- | --- | --- |
| `$RG` | Resource group | `rg-en-es-mt` |
| `$LOC` | Azure region | `eastus` (cheap, has A10) |
| `$WS` | Azure ML workspace | `aml-en-es-mt` |
| `$STORAGE` | Storage account (must be **3–24 lowercase chars, globally unique**) | `enesmt<yourinitials><suffix>` e.g. `enesmtek09` |

> Storage account names are globally unique across all of Azure. If yours collides, append a random 2-digit suffix.

---

## Step 2 — Run the one-shot provisioning script

The repo ships a PowerShell script that creates everything in the right order. Run it as a regular user (not admin needed); it just calls `az`.

```powershell
# from the repo root
pwsh -File scripts/setup_azure.ps1 `
    -ResourceGroup    rg-en-es-mt `
    -Location         eastus `
    -Workspace        aml-en-es-mt `
    -StorageAccount   enesmtek09 `
    -ComputeName      gpu-a10-1x `
    -VmSize           Standard_NV36ads_A10_v5
```

What it does (each step is idempotent — safe to re-run):

1. `az login` if you're not already signed in.
2. Creates the resource group.
3. Creates the Azure ML workspace (this also creates a Key Vault, App Insights, and a default storage account under the hood — we'll *use* a separate storage account for our data so it survives if the workspace is deleted).
4. Creates the storage account + the `en-es-mt` blob container.
5. Creates the GPU compute cluster (`min=0, max=1, idle scale-down 30 min` — only burns money while jobs run).
6. Registers the AML environment from `azure/environment.yml`.
7. Prints the values you need to paste into `.env`.

Expected runtime: ~3 min for the AML workspace, ~1 min for storage, the compute cluster provisions in the background — script returns immediately once it's submitted.

> **If `setup_azure.ps1` errors with "quota":** see the quota note in the prerequisites. The script will tell you exactly which SKU is short.

---

## Step 3 — Get a W&B API key

W&B logs all the training metrics. The free tier covers solo research more than enough.

1. Go to https://wandb.ai/authorize.
2. Copy the key (a 40-char hex string starting with the user prefix).
3. You'll paste it into `.env` in step 5. Don't paste it anywhere else.

Optional: `pwsh -c "wandb login <KEY>"` — caches the key in `~/.netrc` so subsequent runs don't need it in env, but having it in `.env` is fine for AML job submissions.

---

## Step 4 — (Optional) Get a HuggingFace token

Only required if you ever want to pull a *gated* HF model. SmolLM2-360M is open, so you can skip. If you do want it:

1. https://huggingface.co/settings/tokens → New token → Read scope.
2. Copy the `hf_…` string.

---

## Step 5 — Fill `.env`

The repo has `.env.example`. Copy it to `.env` (which is gitignored) and paste the values that `setup_azure.ps1` printed in step 2, plus the W&B key from step 3.

```powershell
Copy-Item .env.example .env
notepad .env   # or your editor of choice
```

You're filling in:

```ini
AZURE_SUBSCRIPTION_ID=...      # printed by setup_azure.ps1
AZURE_RESOURCE_GROUP=rg-en-es-mt
AZURE_ML_WORKSPACE=aml-en-es-mt
AZURE_REGION=eastus

AZURE_STORAGE_ACCOUNT=enesmtek09
AZURE_STORAGE_CONTAINER=en-es-mt

AZURE_COMPUTE_NAME=gpu-a10-1x
AZURE_VM_SIZE=Standard_NV36ads_A10_v5

WANDB_API_KEY=...              # from wandb.ai/authorize
WANDB_PROJECT=en-es-mt
WANDB_ENTITY=                  # blank for personal account; fill in for team accounts

HF_TOKEN=                      # blank unless you have a gated model
HF_HOME=./data/hf_cache
```

> **Security:** `.env` is in `.gitignore`. Never paste your subscription ID or W&B key into chat, the repo, commits, PRs, or notebooks. Anyone with the W&B key can post to your dashboard. Anyone with sufficient Azure permissions can spend your money.

---

## Step 6 — Validate

Run the checker. It uses `DefaultAzureCredential` (your `az login` session), reads `.env`, and confirms each resource exists and is reachable.

```powershell
uv run python scripts/check_azure.py
```

Sample expected output:

```
[OK] .env loaded (10 vars)
[OK] az credential acquired (DefaultAzureCredential)
[OK] subscription: 'My Subscription' (xxxxxxxx-xxxx-…)
[OK] resource group: rg-en-es-mt exists in eastus
[OK] AML workspace: aml-en-es-mt accessible
[OK] storage account: enesmtek09 reachable
[OK] container 'en-es-mt' exists
[OK] compute cluster: gpu-a10-1x state=Succeeded (current=0, max=1)
[OK] environment 'en-es-mt-env' registered (version=1)
[OK] W&B API key looks valid (40 chars)
SUMMARY: 9 ok, 0 warnings, 0 errors
```

Any `[ERROR]` lines spell out exactly what to fix. Most common fixes:

| Error | Fix |
| --- | --- |
| `subscription not found` | `az account set --subscription <id>` |
| `compute cluster: …state=Creating` | Wait 5 min, re-run checker. Cluster provisioning is async. |
| `environment not registered` | Re-run `setup_azure.ps1` — the env registration is the last step and sometimes fails on first try because the workspace isn't quite ready. |
| `W&B API key too short` | You probably pasted with surrounding quotes. Strip them. |

---

## Step 7 — Upload prepared data + register as a data asset

Once your local `02_prepare_data.py` run is done (it builds `data/processed/T{tier}/`), upload it once so all subsequent training jobs can mount it:

```powershell
uv run python scripts/upload_data.py --include processed eval
```

This uses `azure.storage.blob` (no `azcopy` install required), reads `.env`, and writes:
- `data/processed/` → blob `<container>/en-es-mt/data/processed/`
- `data/eval/`      → blob `<container>/en-es-mt/data/eval/`

Then registers an AML data asset pointing at the processed folder:

```
[OK] data asset 'en-es-parallel-processed' version 1 registered
```

After that, `azure/submit_job.py` works without further setup. Verify with a dry run:

```powershell
uv run python azure/submit_job.py --tiers 10000 --dry-run
```

This prints the job spec Azure would receive without actually submitting.

---

## Step 8 — Submit your first run

```powershell
# Confirm runtime expectation (will show 'no_data' confidence until T10k runs at least once)
uv run python scripts/00_estimate_runtime.py 10000

# Submit T10k. Returns a Studio URL.
uv run python azure/submit_job.py --tiers 10000 --wait
```

After T10k completes:
- `runs/registry.jsonl` gets a measured throughput record.
- `00_estimate_runtime.py` produces real ETAs for T50k → T5M.
- W&B shows the BLEU/chrF/forgetting-PPL curves live.

That's the steady state. Submit larger tiers when you're ready; the registry will keep refining the ETA after each completion.

---

## What I (Claude) need from you

Just **finish step 5** (fill `.env`) and **run step 6** (validator). Once `check_azure.py` reports all-green, paste the validator output back to me. That output contains *no* secret values — just confirmations like "resource group exists" and "compute cluster state=Succeeded".

If you'd rather hand me the values directly, paste the **output of `scripts/setup_azure.ps1`** — it ends with a "paste these into .env" block that includes the subscription ID, resource group, etc. but **not** the W&B key. The W&B key goes only in `.env`.

I will never ask you for the W&B key, the HF token, or storage account keys in chat. They live in `.env` and nowhere else.

---

## Tearing it all down

```powershell
# Stop accidental burn (compute auto-scales to 0; this is belt-and-suspenders)
az ml compute update --name gpu-a10-1x --min-instances 0 `
    --workspace-name aml-en-es-mt --resource-group rg-en-es-mt

# Or nuke the whole RG when the experiment is done
az group delete -n rg-en-es-mt --yes --no-wait
```
