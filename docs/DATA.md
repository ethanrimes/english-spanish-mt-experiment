# Data pipeline

## Pipeline overview

```
[OPUS / HuggingFace]
       │
       │  scripts/01_download_data.py
       ▼
data/raw/{source}.jsonl       ← per-source raw pairs
       │
       │  scripts/02_prepare_data.py  (clean + filter)
       ▼
data/interim/{source}.jsonl   ← per-source cleaned pairs
       │
       │  scripts/02_prepare_data.py  (tier sampling)
       ▼
data/processed/T{N}/{train,val}.jsonl  + manifests/T{N}.json
```

Everything in `data/` is gitignored except for the small per-tier manifests in `data/processed/manifests/`. The manifests are committed so the repository alone is enough to verify which exact subset each run used.

## Source corpora

| Source           | Domain      | Approx pairs | Notes                                                     | License        |
| ---------------- | ----------- | ------------ | --------------------------------------------------------- | -------------- |
| Tatoeba          | general     | ~150k        | Crowd-sourced sentence pairs; very clean; often short.    | CC-BY-2.0      |
| TED2020          | talks       | ~410k        | TED talk subtitles; conversational + lecture register.    | CC-BY-NC-4.0   |
| News-Commentary  | news        | ~430k        | Formal journalistic; well-edited; multi-domain content.   | CC-BY-NC-SA-4.0 |
| Europarl-v8      | parliament  | ~2M          | EU parliament proceedings; formal; long sentences.        | OPUS Europarl  |
| OpenSubtitles-v2024 | subtitles | ~50M       | Movie/TV subtitles; conversational; noisy alignment.      | OPUS  |
| ParaCrawl-v9     | web         | ~78M         | Web-mined; very diverse; noisy.                          | CC0 |
| CCMatrix-v1      | web-mined   | ~470M        | Multilingual aligned; very noisy. **Disabled by default.** | OPUS |

All sources retain their upstream licenses. Aggregated training data is for research only.

### Per-source download caps

OpenSubtitles, ParaCrawl, and CCMatrix are huge (multi-GB). The data config has two caps per source:
- `max_pairs_local`: applied when running with `--env local` (laptop). Default 200k each.
- `max_pairs_azure`: applied when running with `--env azure` (cloud). Default 5M each.

This keeps local prep cheap while still allowing the 5M tier to be populated on Azure.

## Cleaning rules

Applied per source, in this order (see `src/en_es_mt/data/clean.py`):

1. **Whitespace normalize** + optional HTML-tag strip.
2. **Length filter** — drop pairs where either side is < `min_chars` (3) or > `max_chars` (500).
3. **Length-ratio filter** — drop pairs where `max(len_en, len_es) / min(len_en, len_es) > max_length_ratio` (3.0). Catches misaligned pairs from auto-extracted corpora.
4. **Identical filter** — drop pairs where the EN side equals the ES side (case-folded). Common noise.
5. **Language-ID filter** — fastText `lid.176` scores each side; drop if confidence in the expected language < `langid_min_prob` (0.5). Skipped with `--no-langid` for fast iteration.
6. **Within-source dedupe** — exact `(en, es)` match, case-folded.

Drop statistics per source are written to `data/processed/manifests/clean_stats.json`.

## Tier construction

Tiers: **10k / 50k / 100k / 500k / 1M / 5M**.

Key properties:

- **Strict nesting**: T_i ⊂ T_{i+1}. Sample the largest tier first; subsample for smaller ones. Smaller-tier examples are guaranteed inside larger-tier ones, so quality-vs-data comparisons aren't confounded by accidentally easier examples in the smaller set.
- **Source stratification**: each tier has its own source-weight mix in `configs/data.yaml :: tiers.source_weights_per_tier`. Small tiers favor clean sources (Tatoeba, News-Commentary); large tiers admit web-mined data (OpenSubtitles, ParaCrawl).
- **Length stratification**: within each source's quota, examples are distributed evenly across EN-character-length buckets (0–40, 40–80, 80–160, 160–320, 320–500) so no tier is dominated by short subtitles or long Europarl speeches.
- **Global dedupe before sampling**: identical `(en, es)` pairs across sources are dropped, keeping the first occurrence (deterministic by stable ID).
- **Validation split**: 2% of each tier reserved as a per-tier val set. Separate from FLORES devtest, which is the cross-run benchmark.

Determinism: seed = `20260526`. Identical re-runs of `02_prepare_data.py` produce byte-identical tier files; the SHA-256 of each split is captured in the per-tier manifest.

## Held-out evaluation

We use **FLORES-200** for evaluation:

- `flores200_dev` (997 sentences) — used for training-time generation eval on a 200-sentence subsample.
- `flores200_devtest` (1012 sentences) — used for the final per-run BLEU/chrF/COMET.

FLORES-200 is the standard MT benchmark (Costa-jussà et al. 2022). Sentences are translated by professional translators across hundreds of languages from the same English source, so the EN-ES split is parallel and high-quality.

## English forgetting probe

For catastrophic-forgetting detection, we run mean-NLL on a held-out English-only set every eval cycle. This set is the English side of FLORES-200 dev (sentences identical across runs). Source: `data/eval/english_holdout.jsonl`, generated automatically by `scripts/02_prepare_data.py` from `flores200_dev.jsonl`. Each record is `{"text": "<english sentence>"}`.

## Reproducing the pipeline

```powershell
# Local laptop — small slice for sanity-checking
uv run python scripts/01_download_data.py --env local --include-eval
uv run python scripts/02_prepare_data.py --tiers 10000 50000

# Azure VM — full pull
uv run python scripts/01_download_data.py --env azure --include-eval
uv run python scripts/02_prepare_data.py --tiers 10000 50000 100000 500000 1000000 5000000
```

Per-tier manifests will land in `data/processed/manifests/T{N}.json` with counts, source distribution, and split SHA-256s.
