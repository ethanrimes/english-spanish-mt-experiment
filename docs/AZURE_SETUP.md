# Azure setup

One-time setup before running training jobs on Azure ML. The repo's code is parameterized via `.env` so concrete subscription / RG / workspace names are never committed.

## What you need

| Resource | Purpose | SKU / spec |
| --- | --- | --- |
| Azure ML workspace | Job tracking, environment registry, MLflow | any tier; same region as compute |
| Azure ML compute cluster | GPU training | `Standard_NV36ads_A10_v5` (1× A10 24GB, **recommended default**) or `Standard_NC24ads_A100_v4` (1× A100 80GB, ~3× faster, ~3× cost) |
| Storage account + container | Data assets, checkpoints | Standard, with HNS for blob speed |
| Azure Key Vault (auto-created with AML) | Secrets (W&B, HF tokens) | included |
| Application Insights (auto-created with AML) | Logging | included |
| W&B account | Training dashboards | free tier is fine |

## One-time provisioning (CLI)

```powershell
# 1. Login + set defaults
az login
az account set --subscription <subscription-id>
$RG = "rg-en-es-mt"
$LOC = "eastus"
$WS = "aml-en-es-mt"

# 2. Resource group + AML workspace
az group create -n $RG -l $LOC
az ml workspace create --name $WS --resource-group $RG --location $LOC

# 3a. GPU compute cluster — cheap option (A10 24GB, recommended default)
az ml compute create `
    --name gpu-a10-1x `
    --type AmlCompute `
    --size Standard_NV36ads_A10_v5 `
    --min-instances 0 --max-instances 1 `
    --idle-time-before-scale-down 1800 `
    --workspace-name $WS --resource-group $RG

# 3b. (Optional) faster option for T5M / debugging
az ml compute create `
    --name gpu-a100-1x `
    --type AmlCompute `
    --size Standard_NC24ads_A100_v4 `
    --min-instances 0 --max-instances 1 `
    --idle-time-before-scale-down 1800 `
    --workspace-name $WS --resource-group $RG

# 4. Register the environment (after .env is populated and repo is cloned)
az ml environment create -f azure/environment.yml `
    --workspace-name $WS --resource-group $RG
```

## Data assets

Once data prep has run on Azure (or you've uploaded the local prep output to blob), register the processed corpus as a data asset so jobs can mount it read-only:

```powershell
az ml data create `
    --name en-es-parallel-processed `
    --type uri_folder `
    --path "https://<storage_account>.blob.core.windows.net/<container>/en-es-mt/data/processed" `
    --workspace-name $WS --resource-group $RG
```

Submission jobs reference it as `azureml:en-es-parallel-processed@latest`.

## Secrets

Two secrets are needed at job time, surfaced via environment variables:

| Secret | Used by | How to set |
| --- | --- | --- |
| `WANDB_API_KEY` | W&B logging | Add as Key Vault secret OR pass via `submit_job.py` env block (it reads from your local `.env`) |
| `HF_TOKEN` | (optional) gated HF model pulls | Same |

`submit_job.py` reads `.env` on the *submitting* machine and passes the values to the job's environment block. They are never written to the repo.

## Running a job

```powershell
# Sanity check the runtime forecast before spending GPU time
uv run python scripts/00_estimate_runtime.py 10000

# Submit one or more tiers
uv run python azure/submit_job.py --tiers 10000
uv run python azure/submit_job.py --tiers 50000 100000
uv run python azure/submit_job.py --tiers 500000 --wait      # stream logs until done
```

Each submission prints a Studio URL where you can watch real-time logs / MLflow metrics. After the run completes, the run record is appended to `runs/registry.jsonl` *on the compute node* — pull it down with:

```powershell
az ml job download --name <job_name> --download-path ./runs/T<tier>-<job_name> --workspace-name $WS --resource-group $RG
```

## Cost estimate (Pay-as-you-go, East US, list prices)

A10 ~$1.20/hr (NV36ads_A10_v5) is the recommended default; A100 ~$3.67/hr (NC24ads_A100_v4) for ~3× speed.

| Tier  | A10 time | A10 cost | A100 time | A100 cost |
| ----- | -------- | -------- | --------- | --------- |
| T10k  | ~75 min  | ~$1.50   | ~30 min   | ~$2       |
| T50k  | ~3.5 hr  | ~$4      | ~1.5 hr   | ~$6       |
| T100k | ~6 hr    | ~$7      | ~2.5 hr   | ~$9       |
| T500k | ~15 hr   | ~$18     | ~6 hr     | ~$22      |
| T1M   | ~25–30 hr| ~$30–36  | ~10–12 hr | ~$40      |
| T5M   | ~100–130 hr | ~$120–155 | ~40–55 hr | ~$180  |

Recommended strategy: run T10k → T1M on A10 (cheaper, total ~$60 across all five tiers), then decide whether T5M is worth a burst on A100 (~$180) or A10 (~$130–155). Refresh these numbers after T10k completes — `00_estimate_runtime.py` produces a measured projection from the actual throughput you observed.

## Cleanup

```powershell
# Stop accidental burn — set min-instances to 0 (idle scale-down handles this anyway)
az ml compute update --name gpu-a100-1x --min-instances 0 --workspace-name $WS --resource-group $RG

# Or nuke the whole RG when the experiment is done
az group delete -n $RG --yes --no-wait
```
