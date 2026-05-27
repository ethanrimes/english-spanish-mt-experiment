"""Run commands on the vast.ai instance over SSH.

Reads SSH coords from .env. Three modes:

  --bootstrap        Provision Python + deps + clone the repo on the remote.
                     Idempotent: re-running is cheap.

  --tiers N [N ...]  Run scripts/04_train.py on the remote with these tiers,
                     streaming logs back live.

  --shell            Open an interactive SSH session in the project directory.

Usage:
  uv run python scripts/vastai_run.py --bootstrap
  uv run python scripts/vastai_run.py --tiers 10000
  uv run python scripts/vastai_run.py --tiers 10000 50000 100000
  uv run python scripts/vastai_run.py --shell
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys


REMOTE_BASE = "/workspace/en-es-mt"
GIT_REPO_DEFAULT = "https://github.com/ethanrimes/english-spanish-mt-experiment.git"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", action="store_true",
                        help="Set up the remote: clone, install uv + deps.")
    parser.add_argument("--tiers", nargs="+", type=int, default=None,
                        help="Train one or more tiers on the remote.")
    parser.add_argument("--shell", action="store_true", help="Open an interactive SSH session.")
    parser.add_argument("--git-repo", default=GIT_REPO_DEFAULT)
    parser.add_argument("--git-branch", default="main")
    parser.add_argument("--download-fresh", action="store_true",
                        help="During bootstrap, also run scripts/01_download_data.py + 02_prepare_data.py "
                             "(instead of expecting an upload from your laptop).")
    parser.add_argument("--remote-base", default=REMOTE_BASE)
    args = parser.parse_args()

    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    host = os.environ.get("VAST_SSH_HOST")
    port = os.environ.get("VAST_SSH_PORT")
    user = os.environ.get("VAST_SSH_USER", "root")
    if not (host and port):
        print("[!] VAST_SSH_HOST/PORT not set in .env. Run vastai_provision.py first.", file=sys.stderr)
        return 2

    if not (args.bootstrap or args.tiers or args.shell):
        print("[!] specify --bootstrap, --tiers ..., or --shell", file=sys.stderr)
        return 2

    if args.shell:
        return _ssh_interactive(host, port, user, args.remote_base)
    if args.bootstrap:
        rc = _ssh_run(host, port, user,
                      _bootstrap_script(args.git_repo, args.git_branch, args.remote_base, args.download_fresh))
        if rc != 0:
            return rc
    if args.tiers:
        wandb_key = os.environ.get("WANDB_API_KEY", "")
        wandb_project = os.environ.get("WANDB_PROJECT", "en-es-mt")
        wandb_entity = os.environ.get("WANDB_ENTITY", "")
        hf_token = os.environ.get("HF_TOKEN", "")
        tier_args = " ".join(str(t) for t in args.tiers)
        cmd = _training_script(args.remote_base, tier_args, wandb_key, wandb_project, wandb_entity, hf_token)
        return _ssh_run(host, port, user, cmd)
    return 0


def _ssh_run(host: str, port: str, user: str, cmd: str) -> int:
    ssh_target = f"{user}@{host}"
    full = ["ssh", "-p", str(port), "-o", "StrictHostKeyChecking=accept-new", ssh_target, "bash -lc", _quote(cmd)]
    return subprocess.call(full)


def _ssh_interactive(host: str, port: str, user: str, remote_base: str) -> int:
    ssh_target = f"{user}@{host}"
    init = f"cd {remote_base} 2>/dev/null || true; exec bash -l"
    full = ["ssh", "-t", "-p", str(port), "-o", "StrictHostKeyChecking=accept-new", ssh_target, init]
    return subprocess.call(full)


def _bootstrap_script(repo: str, branch: str, base: str, download_fresh: bool) -> str:
    download_block = (
        "uv run python scripts/01_download_data.py --env azure --include-eval && "
        "uv run python scripts/02_prepare_data.py --no-langid --tiers 10000 50000 100000 500000 1000000 5000000"
        if download_fresh
        else "echo '[bootstrap] skipping data download — expecting upload via scripts/vastai_sync.py --upload'"
    )
    return f"""\
set -euo pipefail
echo '[bootstrap] apt deps'
apt-get update -qq && apt-get install -y -qq git curl ca-certificates build-essential
echo '[bootstrap] install uv'
if ! command -v uv >/dev/null 2>&1; then curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"; fi
echo '[bootstrap] clone repo'
mkdir -p {base}
if [ ! -d {base}/.git ]; then
  git clone -b {branch} {repo} {base}
else
  cd {base} && git fetch && git checkout {branch} && git pull --ff-only
fi
cd {base}
echo '[bootstrap] uv sync'
~/.local/bin/uv sync --extra cuda --extra dev || uv sync --extra cuda --extra dev
{download_block}
echo '[bootstrap] done'
"""


def _training_script(base: str, tier_args: str, wandb_key: str, wandb_project: str, wandb_entity: str, hf_token: str) -> str:
    return f"""\
set -euo pipefail
export WANDB_API_KEY={_shellquote(wandb_key)}
export WANDB_PROJECT={_shellquote(wandb_project)}
export WANDB_ENTITY={_shellquote(wandb_entity)}
export HF_TOKEN={_shellquote(hf_token)}
export PYTHONIOENCODING=utf-8
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
cd {base}
uv run python scripts/04_train.py --tiers {tier_args}
"""


def _quote(s: str) -> str:
    return "'" + s.replace("'", "'\\''") + "'"


def _shellquote(s: str) -> str:
    if not s:
        return "''"
    return "'" + s.replace("'", "'\\''") + "'"


if __name__ == "__main__":
    sys.exit(main())
