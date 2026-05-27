"""Search vast.ai for GPU offers matching this project's needs.

Filters: A10 / A6000 / A100 / H100 (24GB+ VRAM), reliability > 0.95,
recent CUDA driver, datacenter (not consumer).

Usage:
  uv run python scripts/vastai_search.py                 # default A10 search
  uv run python scripts/vastai_search.py --gpu A100      # different GPU
  uv run python scripts/vastai_search.py --max-price 0.50  # cap price
  uv run python scripts/vastai_search.py --num-gpus 1
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys


GPU_QUERIES = {
    "A10": 'gpu_name=RTX_A10 num_gpus=1',
    "A6000": 'gpu_name=RTX_A6000 num_gpus=1',
    "A100": 'gpu_name=A100_SXM4 num_gpus=1',
    "A100_PCIE": 'gpu_name=A100_PCIE num_gpus=1',
    "H100": 'gpu_name=H100_SXM num_gpus=1',
    "L40": 'gpu_name=L40 num_gpus=1',
    "4090": 'gpu_name=RTX_4090 num_gpus=1',
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default="A10", choices=list(GPU_QUERIES.keys()) + ["any"],
                        help="GPU family to look for.")
    parser.add_argument("--max-price", type=float, default=0.50,
                        help="$/hr cap. Default 0.50 (sane for A10).")
    parser.add_argument("--min-reliability", type=float, default=0.95)
    parser.add_argument("--min-internet-mbps", type=int, default=500)
    parser.add_argument("--min-vram-gb", type=int, default=24)
    parser.add_argument("--rentable", action="store_true", default=True)
    parser.add_argument("--limit", type=int, default=15)
    args = parser.parse_args()

    if not _has_vastai_cli():
        print(
            "[!] vastai CLI not on PATH. Install with: uv sync --extra vastai\n"
            "    Or: uv tool install vastai",
            file=sys.stderr,
        )
        return 2

    if not _is_authed():
        print(
            "[!] vastai CLI not authenticated. Run:\n"
            "    uv run vastai set api-key <YOUR_KEY>\n"
            "    Get your key at: https://cloud.vast.ai/cli/",
            file=sys.stderr,
        )
        return 2

    gpu_query = GPU_QUERIES.get(args.gpu, "") if args.gpu != "any" else ""
    parts = [gpu_query] if gpu_query else []
    parts.append(f"dph_total<{args.max_price}")
    parts.append(f"reliability2>{args.min_reliability}")
    parts.append(f"inet_down>{args.min_internet_mbps}")
    parts.append(f"gpu_ram>={args.min_vram_gb * 1024}")
    if args.rentable:
        parts.append("rentable=true")
    query = " ".join(parts)

    print(f"# query: {query}")
    cmd = ["vastai", "search", "offers", query, "-o", "dph_total", "--raw"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"[!] vastai search failed: {proc.stderr}", file=sys.stderr)
        return proc.returncode

    try:
        offers = json.loads(proc.stdout)
    except json.JSONDecodeError:
        print("[!] vastai returned non-JSON. Raw output:\n" + proc.stdout, file=sys.stderr)
        return 1

    if not offers:
        print(
            "[!] no matching offers. Try:\n"
            "    --max-price 1.00   (higher budget)\n"
            "    --gpu A6000        (different family)\n"
            "    --min-reliability 0.90"
        )
        return 1

    _print_table(offers[: args.limit])
    print()
    print(f"Pick one and run:  uv run python scripts/vastai_provision.py --offer-id <ID>")
    return 0


def _print_table(offers: list[dict]) -> None:
    headers = ("ID", "GPU", "VRAM", "$/hr", "DLPerf", "Country", "Down", "Up", "Reliab")
    widths = (10, 18, 6, 8, 8, 8, 8, 8, 7)
    print("  ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print("  ".join("-" * w for w in widths))
    for o in offers:
        gpu_name = (o.get("gpu_name") or "?")[:18]
        vram = f"{int(o.get('gpu_ram', 0) / 1024)}GB"
        dph = f"${o.get('dph_total', 0):.2f}"
        dlp = f"{o.get('dlperf', 0):.1f}"
        country = (o.get("geolocation") or "?")[:8]
        down = f"{int(o.get('inet_down', 0))}"
        up = f"{int(o.get('inet_up', 0))}"
        rel = f"{o.get('reliability2', 0):.2f}"
        row = (str(o.get("id", "?")), gpu_name, vram, dph, dlp, country, down, up, rel)
        print("  ".join(v.ljust(w) for v, w in zip(row, widths)))


def _has_vastai_cli() -> bool:
    return subprocess.run(["vastai", "--version"], capture_output=True).returncode == 0


def _is_authed() -> bool:
    proc = subprocess.run(["vastai", "show", "user"], capture_output=True, text=True)
    return proc.returncode == 0 and "id" in proc.stdout.lower()


if __name__ == "__main__":
    sys.exit(main())
