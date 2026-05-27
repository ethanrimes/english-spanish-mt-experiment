"""Rent a vast.ai instance and wait for SSH to come up.

After this returns:
  - The instance is running and billable.
  - SSH coordinates are written to .env (VAST_INSTANCE_ID, _HOST, _PORT, _USER).
  - You can ssh in directly or use vastai_run.py for training.

Usage:
  uv run python scripts/vastai_provision.py --offer-id 12345678
  uv run python scripts/vastai_provision.py --offer-id 12345678 --disk-gb 60
  uv run python scripts/vastai_provision.py --offer-id 12345678 --auto-destroy-hours 24
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


# A current PyTorch image — bumped as PyTorch releases. Cuda 12.x matches A10/A100/H100.
DEFAULT_IMAGE = "pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--offer-id", required=True, type=int)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--disk-gb", type=int, default=40,
                        help="Persistent disk in GB. Bumps cost slightly.")
    parser.add_argument("--ssh", action="store_true", default=True,
                        help="Provision with SSH enabled (default).")
    parser.add_argument("--auto-destroy-hours", type=float, default=0.0,
                        help="If > 0, schedule an auto-destroy this many hours after creation. "
                             "Belt-and-suspenders against forgetting to tear down.")
    parser.add_argument("--label", default="en-es-mt")
    parser.add_argument("--no-update-env", action="store_true")
    args = parser.parse_args()

    if not _has_cli():
        print("[!] vastai CLI missing. Run: uv sync --extra vastai", file=sys.stderr)
        return 2

    print(f"[create] offer={args.offer_id} image={args.image} disk={args.disk_gb}GB")
    cmd = [
        "vastai", "create", "instance", str(args.offer_id),
        "--image", args.image,
        "--disk", str(args.disk_gb),
        "--label", args.label,
        "--ssh",
        "--raw",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"[!] create failed: {proc.stderr}", file=sys.stderr)
        return proc.returncode
    try:
        result = json.loads(proc.stdout)
    except json.JSONDecodeError:
        print("[!] could not parse create response: " + proc.stdout[:500], file=sys.stderr)
        return 1
    if not result.get("success"):
        print(f"[!] vast returned: {result}", file=sys.stderr)
        return 1
    instance_id = result["new_contract"]
    print(f"[ok] created instance {instance_id}")

    # Wait for state=running + SSH coords populated
    host, port, user = _wait_for_ready(instance_id, timeout_s=300)
    if not host:
        print("[!] instance did not become ready within 5 min — check vast.ai console", file=sys.stderr)
        return 1
    print(f"[ok] running. SSH: ssh -p {port} {user}@{host}")

    if not args.no_update_env:
        _update_env({
            "VAST_INSTANCE_ID": str(instance_id),
            "VAST_SSH_HOST": host,
            "VAST_SSH_PORT": str(port),
            "VAST_SSH_USER": user,
        })
        print("[ok] wrote VAST_* keys to .env")

    if args.auto_destroy_hours > 0:
        print(f"[note] auto-destroy NOT scheduled by vast (no native field). Set a calendar reminder.")
        # vast.ai does not currently have a self-destruct field; the schedule
        # has to be enforced client-side. Future: spawn a local cron.

    print("\nNext:")
    print("  uv run python scripts/vastai_sync.py --upload")
    print("  uv run python scripts/vastai_run.py --bootstrap")
    print("  uv run python scripts/vastai_run.py --tiers 10000")
    return 0


def _wait_for_ready(instance_id: int, *, timeout_s: int = 300) -> tuple[str | None, int | None, str | None]:
    start = time.time()
    while time.time() - start < timeout_s:
        proc = subprocess.run(
            ["vastai", "show", "instance", str(instance_id), "--raw"],
            capture_output=True, text=True,
        )
        if proc.returncode == 0:
            try:
                inst = json.loads(proc.stdout)
            except json.JSONDecodeError:
                inst = {}
            status = inst.get("actual_status") or inst.get("intended_status") or "?"
            host = inst.get("public_ipaddr") or inst.get("ssh_host")
            port = inst.get("ssh_port") or _first_port(inst.get("ports", {}))
            user = "root"
            if status == "running" and host and port:
                return host, int(port), user
            elapsed = int(time.time() - start)
            print(f"[wait] {elapsed:>3}s  state={status}  host={host or 'pending'}  port={port or 'pending'}")
        time.sleep(8)
    return None, None, None


def _first_port(ports_dict: dict) -> int | None:
    for binds in (ports_dict or {}).values():
        for b in binds or []:
            hp = b.get("HostPort")
            if hp:
                try:
                    return int(hp)
                except ValueError:
                    continue
    return None


def _update_env(updates: dict) -> None:
    env_path = Path(".env")
    if not env_path.exists():
        env_path.write_text("# vast.ai config\n", encoding="utf-8")
    lines = env_path.read_text(encoding="utf-8").splitlines()
    seen = set()
    new_lines = []
    for line in lines:
        if "=" in line and not line.strip().startswith("#"):
            k = line.split("=", 1)[0].strip()
            if k in updates:
                new_lines.append(f"{k}={updates[k]}")
                seen.add(k)
                continue
        new_lines.append(line)
    for k, v in updates.items():
        if k not in seen:
            new_lines.append(f"{k}={v}")
    env_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


def _has_cli() -> bool:
    return subprocess.run(["vastai", "--version"], capture_output=True).returncode == 0


if __name__ == "__main__":
    sys.exit(main())
