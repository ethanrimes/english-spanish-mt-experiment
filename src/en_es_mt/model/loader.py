"""Base-model loader.

Wraps HuggingFace AutoModelForCausalLM + AutoTokenizer with our project
conventions:
- bfloat16 on CUDA, fp32 on CPU
- pad_token defaulted to eos_token if missing (SmolLM2 ships with eos)
- gradient checkpointing toggle
- SDPA attention by default (efficient on PyTorch 2.x)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class ModelBundle:
    model: Any
    tokenizer: Any
    config_dict: dict


def load_model_and_tokenizer(model_cfg: dict, *, for_training: bool = True) -> ModelBundle:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    mcfg = model_cfg["model"]
    hf_id = mcfg["hf_id"]
    revision = mcfg.get("revision", "main")
    trust = mcfg.get("trust_remote_code", False)

    log.info("loading tokenizer %s @ %s", hf_id, revision)
    tokenizer = AutoTokenizer.from_pretrained(hf_id, revision=revision, trust_remote_code=trust)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        log.info("set pad_token = eos_token (%r)", tokenizer.eos_token)

    dtype_str = mcfg.get("dtype", "bfloat16")
    torch_dtype = _resolve_dtype(dtype_str)
    attn_impl = mcfg.get("attn_implementation", "sdpa")

    log.info("loading model %s (dtype=%s, attn=%s)", hf_id, torch_dtype, attn_impl)
    model = AutoModelForCausalLM.from_pretrained(
        hf_id, revision=revision, trust_remote_code=trust,
        torch_dtype=torch_dtype, attn_implementation=attn_impl,
    )

    if for_training and mcfg.get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
        # Required when GC is on for causal LM:
        if hasattr(model, "config"):
            model.config.use_cache = False

    if torch.cuda.is_available():
        # Trainer moves the model to GPU; this is just a sanity log.
        log.info("CUDA available: %s", torch.cuda.get_device_name(0))
    else:
        log.warning("CUDA not available — running on CPU (training will be slow)")

    n_params = sum(p.numel() for p in model.parameters())
    log.info("model loaded: %d parameters", n_params)

    return ModelBundle(model=model, tokenizer=tokenizer, config_dict=model_cfg)


def _resolve_dtype(s: str):
    import torch

    return {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }[s]
