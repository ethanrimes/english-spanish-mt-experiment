# English ↔ Spanish MT Experiment

> Fine-tuning a predominantly-English small LM to translate English ↔ Spanish, and measuring how translation quality scales with parallel-data volume.

## Motivation

How much Spanish can an "English-only" small LM (~360M parameters) learn purely through supervised fine-tuning on parallel sentence pairs? We answer this empirically by training the same base model on four nested data tiers (10k, 50k, 100k, 500k pairs) and plotting BLEU / chrF / COMET against data size.

## Base model

**[SmolLM2-360M](https://huggingface.co/HuggingFaceTB/SmolLM2-360M)** (HuggingFace, 360M parameters).

Chosen because its training corpus (FineWeb-Edu + SmolLM-Corpus) was aggressively English-filtered with a classifier, giving the lowest realistic Spanish exposure among well-known small base models. We make no claim of *zero* Spanish — no public LLM trained on web data is truly monolingual — but it is the closest practical choice. See [`docs/MODEL_CHOICE.md`](docs/MODEL_CHOICE.md) for the full rationale and a tokenizer-level probe of Spanish handling.

## Experimental design: six runs, multi-checkpoint each

We train **six independent models** — one per data tier. Each training **run** produces many **checkpoint snapshots** through its lifetime (every N optimizer steps + a "best so far" by FLORES-dev BLEU). The snapshot trail is the rollback path if a run starts catastrophically forgetting English or regresses on Spanish.

| Run name | Training pairs | Notes                                              |
| -------- | -------------- | -------------------------------------------------- |
| T10k     | 10,000         | Sample-efficiency floor; clean sources only        |
| T50k     | 50,000         | Mid-low; still mostly clean sources                |
| T100k    | 100,000        | Mid; lightly includes OpenSubtitles                |
| T500k    | 500,000        | Target operating point; ParaCrawl introduced       |
| T1M      | 1,000,000      | Scale check; ParaCrawl + OpenSubtitles heavy       |
| T5M      | 5,000,000      | Ceiling run; full diverse web-mined coverage       |

Every run keeps a rolling window of `save_total_limit` checkpoints plus the best-by-eval-BLEU snapshot and the final-step snapshot. Each tier's training set is a stratified random subsample drawn from the same cleaned union pool, balanced across source corpus and sentence length. Tiers are **strictly nested** (T10k ⊂ T50k ⊂ ... ⊂ T5M) so the smaller-tier examples are guaranteed inside the larger ones — makes quality-vs-data comparisons fair.

Held-out evaluation uses **FLORES-200** devtest (identical across all runs).

Source corpora (English↔Spanish): Tatoeba, TED2020, News-Commentary, Europarl, OpenSubtitles (sampled), ParaCrawl (sampled). See [`docs/DATA.md`](docs/DATA.md).

## Layout

```
.
├── configs/                YAML configs for data, model, training, Azure
├── docs/                   Project documentation (model choice, data, Azure)
├── src/en_es_mt/           Library code (data, model, train, eval, obs)
├── scripts/                Numbered entrypoints (download → prepare → train → eval)
├── azure/                  Azure ML job specs + submission helper
├── tests/                  Pytest suite
├── data/                   (gitignored) raw + processed parallel corpora
├── checkpoints/            (gitignored) training checkpoints
└── runs/                   (gitignored) W&B / MLflow / TensorBoard local cache
```

## Quickstart

```powershell
# 1. Install (CPU laptop for prep / dry runs)
uv sync --extra cpu --extra dev

# 2. Download + prepare data (writes to data/raw, data/processed)
uv run python scripts/01_download_data.py
uv run python scripts/02_prepare_data.py --tiers 10000 50000 100000 500000

# 3. Verify the base model + probe how it tokenizes Spanish
uv run python scripts/03_verify_model.py

# 4. Train a single tier locally (CPU smoke test) — full runs go on Azure
uv run python scripts/04_train.py --config configs/train.yaml --tier 10000

# 5. Evaluate against FLORES-200 devtest
uv run python scripts/05_evaluate.py --checkpoint checkpoints/T10k/best
```

## Running on Azure ML

```powershell
# Configure once
copy .env.example .env   # fill in subscription, RG, workspace, etc.

# Submit a job per tier
uv run python azure/submit_job.py --tier 10000
uv run python azure/submit_job.py --tier 50000
uv run python azure/submit_job.py --tier 100000
uv run python azure/submit_job.py --tier 500000
```

See [`docs/AZURE_SETUP.md`](docs/AZURE_SETUP.md) for one-time Azure resource setup.

## Observability

Every training run logs to **both** Weights & Biases and MLflow (Azure ML auto-attaches the latter):

- Step-level: train loss, LR, grad norm, throughput
- Eval-level: BLEU / chrF / COMET on FLORES dev for both directions
- Artifacts: best checkpoint + last-N rolling checkpoints, sample translations every N steps
- Catastrophic-forgetting safeguard: English-only perplexity is logged each eval cycle on a held-out English set, so any regression is immediately visible

See [`docs/OBSERVABILITY.md`](docs/OBSERVABILITY.md).

## License

MIT. Data corpora retain their original licenses — see [`docs/DATA.md`](docs/DATA.md).
