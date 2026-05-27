"""Source registry. Loads data sources defined in configs/data.yaml and
provides typed accessors. Each source has a name, loader-kind, URL, and
download caps for local vs Azure environments.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from ..utils.io import load_yaml, repo_root


@dataclass(frozen=True)
class Source:
    name: str
    loader: Literal["opus_tmx_zip", "hf_dataset"]
    url: str | None = None
    license: str = "unknown"
    domain: str = "unknown"
    enabled: bool = True
    max_pairs_local: int | None = None
    max_pairs_azure: int | None = None
    hf_id: str | None = None
    hf_config: str | None = None
    hf_split: str | None = None
    extra: dict = field(default_factory=dict)


@dataclass(frozen=True)
class DataConfig:
    sources: list[Source]
    eval: list[Source]
    clean: dict
    tiers: dict
    paths: dict


def load_data_config(path: str | Path | None = None) -> DataConfig:
    if path is None:
        path = repo_root() / "configs" / "data.yaml"
    raw = load_yaml(path)
    sources = [_to_source(s) for s in raw.get("sources", [])]
    eval_sources = [_to_source(s) for s in raw.get("eval", [])]
    return DataConfig(
        sources=sources,
        eval=eval_sources,
        clean=raw.get("clean", {}),
        tiers=raw.get("tiers", {}),
        paths=raw.get("paths", {}),
    )


def _to_source(d: dict) -> Source:
    known = {
        "name", "loader", "url", "license", "domain", "enabled",
        "max_pairs_local", "max_pairs_azure",
        "hf_id", "config", "split",
    }
    extra = {k: v for k, v in d.items() if k not in known}
    return Source(
        name=d["name"],
        loader=d["loader"],
        url=d.get("url"),
        license=d.get("license", "unknown"),
        domain=d.get("domain", "unknown"),
        enabled=d.get("enabled", True),
        max_pairs_local=d.get("max_pairs_local"),
        max_pairs_azure=d.get("max_pairs_azure"),
        hf_id=d.get("hf_id"),
        hf_config=d.get("config"),
        hf_split=d.get("split"),
        extra=extra,
    )


def active_sources(cfg: DataConfig, *, environment: Literal["local", "azure"]) -> list[Source]:
    """Return enabled sources. Sources with max_pairs_<env> == 0 are skipped."""
    out = []
    for s in cfg.sources:
        if not s.enabled:
            continue
        cap = s.max_pairs_local if environment == "local" else s.max_pairs_azure
        if cap == 0:
            continue
        out.append(s)
    return out
