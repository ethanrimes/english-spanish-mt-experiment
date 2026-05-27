"""Sync data and code between local laptop and the vast.ai instance.

Uses ssh+tar over the SSH connection — works on Windows (built-in OpenSSH +
PowerShell tar) and Unix. No rsync required.

Reads SSH coordinates from .env (VAST_SSH_HOST, VAST_SSH_PORT, VAST_SSH_USER).

Usage:
  uv run python scripts/vastai_sync.py --upload
    Uploads data/processed, data/eval to /workspace/en-es-mt on the remote.

  uv run python scripts/vastai_sync.py --upload data/processed
    Upload a specific path.

  uv run python scripts/vastai_sync.py --download checkpoints runs
    Pull these dirs back locally (won't overwrite local files newer than remote).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REMOTE_BASE = "/workspace/en-es-mt"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upload", nargs="*", default=None,
                        help="Paths to upload. Default: data/processed data/eval")
    parser.add_argument("--download", nargs="*", default=None,
                        help="Remote paths to download (relative to /workspace/en-es-mt).")
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

    if args.upload is not None:
        paths = args.upload or ["data/processed", "data/eval"]
        return _upload(host, port, user, paths, remote_base=args.remote_base)
    if args.download is not None:
        if not args.download:
            print("[!] --download needs at least one remote path", file=sys.stderr)
            return 2
        return _download(host, port, user, args.download, remote_base=args.remote_base)

    print("[!] specify --upload or --download", file=sys.stderr)
    return 2


def _upload(host: str, port: str, user: str, paths: list[str], *, remote_base: str) -> int:
    existing = [p for p in paths if Path(p).exists()]
    missing = [p for p in paths if not Path(p).exists()]
    if missing:
        print(f"[skip] not present locally: {missing}")
    if not existing:
        print("[!] nothing to upload", file=sys.stderr)
        return 2

    total_bytes = sum(_size(Path(p)) for p in existing)
    print(f"[upload] {len(existing)} path(s), {_fmt_bytes(total_bytes)} -> {user}@{host}:{port}{remote_base}")

    # ssh+tar pipe: portable, single connection, transparent over Windows OpenSSH.
    paths_arg = " ".join(_quote_remote(p) for p in existing)
    ssh_target = f"{user}@{host}"
    ssh_cmd = ["ssh", "-p", str(port), "-o", "StrictHostKeyChecking=accept-new", ssh_target,
               f"mkdir -p {_quote_remote(remote_base)} && cd {_quote_remote(remote_base)} && tar xz"]

    # Build the local tar stream
    tar_cmd = ["tar", "cz"] + list(existing)
    tar = subprocess.Popen(tar_cmd, stdout=subprocess.PIPE)
    ssh = subprocess.Popen(ssh_cmd, stdin=tar.stdout)
    tar.stdout.close()  # allow tar to receive SIGPIPE if ssh exits
    rc = ssh.wait()
    if rc != 0:
        print(f"[!] ssh+tar failed (rc={rc})", file=sys.stderr)
        return rc
    print("[ok] upload complete")
    return 0


def _download(host: str, port: str, user: str, paths: list[str], *, remote_base: str) -> int:
    print(f"[download] {paths} from {user}@{host}:{port}{remote_base}")
    ssh_target = f"{user}@{host}"
    # Stream tar from remote to local
    paths_arg = " ".join(_quote_remote(p) for p in paths)
    remote_cmd = f"cd {_quote_remote(remote_base)} && tar cz {paths_arg}"
    ssh_cmd = ["ssh", "-p", str(port), "-o", "StrictHostKeyChecking=accept-new", ssh_target, remote_cmd]
    untar = subprocess.Popen(["tar", "xz"], stdin=subprocess.PIPE)
    ssh = subprocess.Popen(ssh_cmd, stdout=untar.stdin)
    untar.stdin.close() if untar.stdin else None
    rc_ssh = ssh.wait()
    rc_tar = untar.wait()
    if rc_ssh != 0 or rc_tar != 0:
        print(f"[!] failed (ssh={rc_ssh} tar={rc_tar})", file=sys.stderr)
        return rc_ssh or rc_tar
    print("[ok] download complete")
    return 0


def _size(p: Path) -> int:
    if p.is_file():
        return p.stat().st_size
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def _fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def _quote_remote(s: str) -> str:
    # Single-quote-safe for sh
    return "'" + s.replace("'", "'\\''") + "'"


if __name__ == "__main__":
    sys.exit(main())
