"""Validate the local Azure + W&B configuration end-to-end.

Reads .env, then checks every resource the training pipeline needs:
- Subscription is reachable with DefaultAzureCredential
- Resource group exists
- AML workspace accessible
- Storage account + container reachable
- GPU compute cluster registered (and reports state)
- AML environment registered
- W&B API key is the right shape

NO secrets are ever printed. Output is intentionally safe to paste back.

Usage:
  uv run python scripts/check_azure.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

OK = "[OK]"
WARN = "[WARN]"
ERR = "[ERROR]"

stats = {"ok": 0, "warn": 0, "err": 0}


def ok(msg: str) -> None:
    stats["ok"] += 1
    print(f"{OK} {msg}")


def warn(msg: str) -> None:
    stats["warn"] += 1
    print(f"{WARN} {msg}")


def err(msg: str) -> None:
    stats["err"] += 1
    print(f"{ERR} {msg}")


def main() -> int:
    # --- .env ---
    try:
        from dotenv import dotenv_values, load_dotenv
    except ImportError:
        err("python-dotenv not installed. Run `uv sync` first.")
        return 2

    env_path = Path(".env")
    if not env_path.exists():
        err(".env not found. Copy .env.example to .env and fill in the values.")
        return 2

    load_dotenv()
    env = dotenv_values(env_path)
    ok(f".env loaded ({sum(1 for v in env.values() if v)} non-empty vars)")

    required = [
        "AZURE_SUBSCRIPTION_ID", "AZURE_RESOURCE_GROUP", "AZURE_ML_WORKSPACE",
        "AZURE_STORAGE_ACCOUNT", "AZURE_STORAGE_CONTAINER", "AZURE_COMPUTE_NAME",
    ]
    missing = [k for k in required if not env.get(k)]
    if missing:
        err(f"missing required keys in .env: {missing}")
        return _summary()

    sub_id = env["AZURE_SUBSCRIPTION_ID"]
    rg     = env["AZURE_RESOURCE_GROUP"]
    ws     = env["AZURE_ML_WORKSPACE"]
    region = env.get("AZURE_REGION", "")
    storage = env["AZURE_STORAGE_ACCOUNT"]
    container = env["AZURE_STORAGE_CONTAINER"]
    compute = env["AZURE_COMPUTE_NAME"]

    # --- credential ---
    try:
        from azure.identity import DefaultAzureCredential
    except ImportError:
        err("azure-identity not installed. Run `uv sync`.")
        return _summary()

    try:
        cred = DefaultAzureCredential()
        # Force a token acquisition to surface auth errors here, not later.
        _ = cred.get_token("https://management.azure.com/.default")
        ok("az credential acquired (DefaultAzureCredential)")
    except Exception as e:
        err(f"could not acquire Azure credential: {type(e).__name__}: {e}\n"
            f"        Try: az login")
        return _summary()

    # --- subscription / RG ---
    try:
        from azure.mgmt.resource import ResourceManagementClient

        rm = ResourceManagementClient(cred, sub_id)
        sub = rm.subscriptions.get(sub_id) if hasattr(rm, "subscriptions") else None
        ok(f"subscription accessible ({_short(sub_id)})")
    except Exception as e:
        warn(f"could not load azure-mgmt-resource ({e!s}); skipping RG check via SDK")
    try:
        from azure.mgmt.resource import ResourceManagementClient

        rm = ResourceManagementClient(cred, sub_id)
        rgo = rm.resource_groups.get(rg)
        ok(f"resource group: {rg} exists in {rgo.location}")
    except Exception as e:
        err(f"resource group {rg} not found: {e}")
        return _summary()

    # --- AML workspace ---
    try:
        from azure.ai.ml import MLClient

        ml = MLClient(cred, sub_id, rg, ws)
        wso = ml.workspaces.get(ws)
        ok(f"AML workspace: {ws} accessible (location={wso.location})")
    except Exception as e:
        err(f"AML workspace {ws} not accessible: {e}")
        return _summary()

    # --- storage ---
    try:
        from azure.storage.blob import BlobServiceClient

        bsc = BlobServiceClient(
            account_url=f"https://{storage}.blob.core.windows.net",
            credential=cred,
        )
        # Light-weight call that only requires reader on the storage account.
        _ = bsc.get_service_properties()
        ok(f"storage account: {storage} reachable")
        cc = bsc.get_container_client(container)
        if cc.exists():
            ok(f"container '{container}' exists")
        else:
            warn(f"container '{container}' missing (setup_azure.ps1 normally creates it)")
    except Exception as e:
        err(f"storage {storage}/{container} not reachable: {type(e).__name__}: {e}")

    # --- compute cluster ---
    try:
        compute_obj = ml.compute.get(compute)
        state = getattr(compute_obj, "provisioning_state", "?")
        cur = getattr(compute_obj, "current_instance_count", None)
        mx = getattr(compute_obj, "max_instances", None) or getattr(compute_obj, "scale_settings", None)
        size = getattr(compute_obj, "size", "?")
        ok(f"compute cluster: {compute} state={state} size={size} max={mx} current={cur}")
        if state.lower() not in ("succeeded", "creating", "updating"):
            warn(f"compute state is unusual: {state}")
    except Exception as e:
        err(f"compute cluster {compute} not found: {e}")

    # --- environment ---
    try:
        e = ml.environments.get(name="en-es-mt-env", label="latest")
        ok(f"environment 'en-es-mt-env' registered (version={e.version})")
    except Exception as ex:
        warn(f"environment 'en-es-mt-env' not yet registered ({type(ex).__name__})")

    # --- W&B ---
    wb_key = env.get("WANDB_API_KEY") or os.environ.get("WANDB_API_KEY")
    if not wb_key:
        warn("WANDB_API_KEY not set in .env — training jobs will skip W&B logging")
    elif len(wb_key) == 40 and all(c in "0123456789abcdef" for c in wb_key.lower()):
        ok("W&B API key looks valid (40 hex chars)")
    else:
        warn(f"WANDB_API_KEY length is {len(wb_key)} (expected 40 hex). Trim quotes/whitespace?")

    return _summary()


def _short(s: str) -> str:
    if len(s) <= 12:
        return s
    return f"{s[:8]}...{s[-4:]}"


def _summary() -> int:
    print(f"\nSUMMARY: {stats['ok']} ok, {stats['warn']} warnings, {stats['err']} errors")
    return 0 if stats["err"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
