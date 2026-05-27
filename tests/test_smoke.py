"""Quick import + structural smoke tests. No GPU, no network, no model weights.

These exist so a single `uv run pytest tests/test_smoke.py` after installing
deps fails fast if any module is broken — before we burn time on data
downloads or training.
"""
from __future__ import annotations


def test_package_imports():
    """All public modules import without error."""
    import en_es_mt  # noqa: F401
    from en_es_mt.data import clean, download, format, sources, tiers  # noqa: F401
    from en_es_mt.eval import metrics, translate  # noqa: F401
    from en_es_mt.model import loader, tokenizer_probe  # noqa: F401
    from en_es_mt.obs import tracking  # noqa: F401
    from en_es_mt.train import callbacks, collator, eta, prompt, trainer  # noqa: F401
    from en_es_mt.utils import io, logging, seed  # noqa: F401


def test_load_data_config():
    """data.yaml parses + has all six tiers."""
    from en_es_mt.data.sources import load_data_config

    cfg = load_data_config()
    assert cfg.tiers["sizes"] == [10000, 50000, 100000, 500000, 1000000, 5000000]
    assert {s.name for s in cfg.sources}.issuperset(
        {"tatoeba", "ted2020", "news_commentary", "europarl"}
    )


def test_load_train_config():
    """train.yaml parses + every tier has overrides."""
    from en_es_mt.utils.io import load_yaml, repo_root

    cfg = load_yaml(repo_root() / "configs" / "train.yaml")
    for tier in (10000, 50000, 100000, 500000, 1000000, 5000000):
        assert tier in cfg["tiers"], f"missing tier {tier}"


def test_load_model_config():
    """model.yaml points at SmolLM2-360M."""
    from en_es_mt.utils.io import load_yaml, repo_root

    cfg = load_yaml(repo_root() / "configs" / "model.yaml")
    assert "SmolLM2-360M" in cfg["model"]["hf_id"]


def test_prompt_formatter_masks_prefix():
    """Prompt prefix tokens get -100 labels; only target tokens contribute loss."""
    from transformers import AutoTokenizer
    import random

    from en_es_mt.data.format import PromptFormatter, make_training_example

    # GPT-2 is small and ships with the test environment; we don't load
    # weights, just the tokenizer.
    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    fmt = PromptFormatter(
        src_lang_names={"en": "English", "es": "Spanish"},
        template="placeholder",
        direction_sampling="en2es",
    )
    ex = make_training_example(
        {"en": "Hello.", "es": "Hola."},
        tokenizer=tok, formatter=fmt, max_seq_len=64,
        rng=random.Random(0), mask_loss_on_prompt=True,
    )
    # At least one position must be unmasked (the target) and at least one masked.
    assert any(label == -100 for label in ex["labels"])
    assert any(label != -100 for label in ex["labels"])
    assert len(ex["input_ids"]) == len(ex["labels"]) == len(ex["attention_mask"])


def test_eta_registry_roundtrip(tmp_path, monkeypatch):
    """Append + load + estimate round-trip on a temp registry."""
    import en_es_mt.train.eta as eta_mod

    fake_runs = tmp_path / "runs"
    fake_runs.mkdir()
    monkeypatch.setattr(eta_mod, "registry_path", lambda: fake_runs / "registry.jsonl")

    rec = eta_mod.RunRecord(
        tier=10000, started_at_utc="2026-05-26T00:00:00+00:00",
        ended_at_utc="2026-05-26T00:30:00+00:00", duration_sec=1800,
        status="completed", model="test", examples_per_sec=50.0,
        total_examples=80000, epochs=8,
    )
    eta_mod.append_run(rec)
    assert len(eta_mod.load_registry()) == 1

    f = eta_mod.estimate_tier_duration(target_tier=50000, target_epochs=5.0, eval_passes=10)
    assert f.confidence in ("measured", "extrapolated")
    assert f.est_duration_sec is not None
    assert f.est_duration_sec > 0
