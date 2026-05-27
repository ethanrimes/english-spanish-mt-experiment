"""Stop or destroy the vast.ai instance referenced in .env.

Stop:   keeps disk + state, very cheap idle (~$0.001/hr). Resumable.
Destroy: wipes everything, no further billing. Irreversible.

Usage:
  uv run python scripts/vastai_destroy.py --stop
  uv run python scripts/vastai_destroy.py --start            # resume a stopped instance
  uv run python scripts/vastai_destroy.py --destroy --yes    # nuke for good
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--stop", action="store_true")
    g.add_argument("--start", action="store_true")
    g.add_argument("--destroy", action="store_true")
    parser.add_argument("--yes", action="store_true", help="Skip confirmation prompt for --destroy.")
    args = parser.parse_args()

    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    inst = os.environ.get("VAST_INSTANCE_ID")
    if not inst:
        print("[!] VAST_INSTANCE_ID not set in .env", file=sys.stderr)
        return 2

    if args.stop:
        return subprocess.call(["vastai", "stop", "instance", inst])
    if args.start:
        return subprocess.call(["vastai", "start", "instance", inst])
    if args.destroy:
        if not args.yes:
            print(f"This will permanently destroy vast.ai instance {inst} and wipe its disk.")
            confirm = input("Type the instance ID to confirm: ").strip()
            if confirm != inst:
                print("Aborted.")
                return 1
        rc = subprocess.call(["vastai", "destroy", "instance", inst])
        if rc == 0:
            # Blank out the SSH coords so we don't accidentally try to use them
            _blank_env_keys(["VAST_INSTANCE_ID", "VAST_SSH_HOST", "VAST_SSH_PORT"])
            print("[ok] destroyed; cleared VAST_INSTANCE_ID + SSH coords from .env")
        return rc
    return 2


def _blank_env_keys(keys: list[str]) -> None:
    from pathlib import Path

    p = Path(".env")
    if not p.exists():
        return
    lines = p.read_text(encoding="utf-8").splitlines()
    new = []
    for line in lines:
        if "=" in line and not line.strip().startswith("#"):
            k = line.split("=", 1)[0].strip()
            if k in keys:
                new.append(f"{k}=")
                continue
        new.append(line)
    p.write_text("\n".join(new) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
