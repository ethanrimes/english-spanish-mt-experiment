"""Upload prepared data to Azure Blob Storage and register an AML data asset.

Reads .env for storage account + container + workspace coordinates. Uses
AAD credential (your `az login` identity) — no account keys needed.

Usage:
  # Upload processed + eval folders (default)
  uv run python scripts/upload_data.py

  # Pick what to include
  uv run python scripts/upload_data.py --include processed
  uv run python scripts/upload_data.py --include processed eval raw   # also push raw

  # Override the blob prefix (default 'en-es-mt')
  uv run python scripts/upload_data.py --prefix en-es-mt-v2
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--include", nargs="+", default=["processed", "eval"],
                        choices=["raw", "interim", "processed", "eval"])
    parser.add_argument("--prefix", default="en-es-mt",
                        help="Blob path prefix under the container.")
    parser.add_argument("--register-asset", action="store_true", default=True,
                        help="Also register an AML data asset pointing at the uploaded processed/.")
    parser.add_argument("--asset-name", default="en-es-parallel-processed")
    args = parser.parse_args()

    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        print("[!] python-dotenv missing — run `uv sync` first.", file=sys.stderr)
        return 2

    storage = _require("AZURE_STORAGE_ACCOUNT")
    container = _require("AZURE_STORAGE_CONTAINER")

    from azure.identity import DefaultAzureCredential
    from azure.storage.blob import BlobServiceClient

    cred = DefaultAzureCredential()
    bsc = BlobServiceClient(
        account_url=f"https://{storage}.blob.core.windows.net",
        credential=cred,
    )
    container_client = bsc.get_container_client(container)
    if not container_client.exists():
        print(f"[!] container '{container}' does not exist — run setup_azure.ps1 first", file=sys.stderr)
        return 2

    for folder in args.include:
        local = Path("data") / folder
        if not local.exists():
            print(f"[skip] {local} does not exist")
            continue
        blob_prefix = f"{args.prefix}/data/{folder}"
        _upload_folder(local, container_client, blob_prefix)

    if args.register_asset and "processed" in args.include:
        _register_asset(args.asset_name, args.prefix, storage, container)
    return 0


def _upload_folder(local: Path, container_client, blob_prefix: str) -> None:
    files = [p for p in local.rglob("*") if p.is_file()]
    total_bytes = sum(p.stat().st_size for p in files)
    print(f"[upload] {local} → {blob_prefix}  ({len(files)} files, {_fmt(total_bytes)})")
    uploaded = 0
    sent = 0
    for f in files:
        rel = f.relative_to(local).as_posix()
        blob_name = f"{blob_prefix}/{rel}"
        size = f.stat().st_size
        with open(f, "rb") as fh:
            container_client.upload_blob(name=blob_name, data=fh, overwrite=True, max_concurrency=4)
        uploaded += 1
        sent += size
        if uploaded % 10 == 0 or uploaded == len(files):
            print(f"  {uploaded}/{len(files)}  {_fmt(sent)}/{_fmt(total_bytes)}")


def _register_asset(name: str, prefix: str, storage: str, container: str) -> None:
    sub_id = _require("AZURE_SUBSCRIPTION_ID")
    rg = _require("AZURE_RESOURCE_GROUP")
    ws = _require("AZURE_ML_WORKSPACE")

    from azure.ai.ml import MLClient
    from azure.ai.ml.constants import AssetTypes
    from azure.ai.ml.entities import Data
    from azure.identity import DefaultAzureCredential

    ml = MLClient(DefaultAzureCredential(), sub_id, rg, ws)
    blob_uri = f"https://{storage}.blob.core.windows.net/{container}/{prefix}/data/processed"
    asset = Data(
        name=name,
        description="Cleaned + tiered EN-ES parallel corpus (T10k/T50k/T100k/T500k/T1M/T5M).",
        type=AssetTypes.URI_FOLDER,
        path=blob_uri,
    )
    out = ml.data.create_or_update(asset)
    print(f"[ok] registered data asset '{out.name}' version {out.version}")
    print(f"     azureml:{out.name}@latest  →  {blob_uri}")


def _require(key: str) -> str:
    v = os.environ.get(key, "")
    if not v:
        print(f"[!] missing env var: {key} (set in .env)", file=sys.stderr)
        sys.exit(2)
    return v


def _fmt(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


if __name__ == "__main__":
    sys.exit(main())
