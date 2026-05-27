"""BLEU / chrF / COMET evaluation.

SacreBLEU and chrF are computed on raw strings (no detokenization needed).
COMET requires the model weights (~1.5GB) and a GPU for fast scoring —
the function tolerates the model not being available and returns None.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass
class TranslationScores:
    bleu: float
    chrf: float
    chrf_pp: float
    comet: float | None
    n: int


def score_translations(
    hyps: list[str],
    refs: list[str],
    *,
    sources: list[str] | None = None,
    compute_comet: bool = False,
    comet_model: str = "Unbabel/wmt22-comet-da",
) -> TranslationScores:
    """`refs` are reference translations; `sources` are source-language
    sentences, only required if `compute_comet=True`."""
    import sacrebleu

    bleu_obj = sacrebleu.corpus_bleu(hyps, [refs])
    chrf_obj = sacrebleu.corpus_chrf(hyps, [refs])
    chrf_pp_obj = sacrebleu.corpus_chrf(hyps, [refs], word_order=2)

    comet_score = None
    if compute_comet:
        if sources is None:
            log.warning("compute_comet=True but no sources provided — skipping COMET")
        else:
            comet_score = _comet(sources, hyps, refs, model_id=comet_model)

    return TranslationScores(
        bleu=bleu_obj.score,
        chrf=chrf_obj.score,
        chrf_pp=chrf_pp_obj.score,
        comet=comet_score,
        n=len(hyps),
    )


def _comet(sources: list[str], hyps: list[str], refs: list[str], *, model_id: str) -> float | None:
    """Compute COMET; returns None if the package or model is unavailable."""
    try:
        from comet import download_model, load_from_checkpoint
    except Exception as e:
        log.warning("COMET unavailable (%s) — skipping", e)
        return None
    try:
        model_path = download_model(model_id)
        model = load_from_checkpoint(model_path)
        data = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(sources, hyps, refs)]
        out = model.predict(data, batch_size=16, gpus=1 if _has_cuda() else 0, progress_bar=False)
        return float(out.system_score)
    except Exception as e:
        log.warning("COMET scoring failed: %s — skipping", e)
        return None


def _has_cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False
