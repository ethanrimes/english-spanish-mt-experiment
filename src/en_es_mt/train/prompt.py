"""Thin re-export so training code can `from ...train.prompt import ...`
without reaching into the data package."""
from ..data.format import (
    Direction,
    PromptFormatter,
    make_inference_prompt,
    make_training_example,
)

__all__ = ["Direction", "PromptFormatter", "make_inference_prompt", "make_training_example"]
