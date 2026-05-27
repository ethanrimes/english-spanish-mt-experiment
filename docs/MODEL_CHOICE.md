# Base model: SmolLM2-360M

## Selection

We use [`HuggingFaceTB/SmolLM2-360M`](https://huggingface.co/HuggingFaceTB/SmolLM2-360M) as the base model. 360M parameters; decoder-only Transformer (LlamaForCausalLM); trained by the HuggingFace SmolLM team and released November 2024.

## Why this model

The study's central question — *how much Spanish can a predominantly-English LM learn from supervised fine-tuning?* — only makes sense if the base model has minimal pre-existing Spanish exposure. We evaluated several candidates against three criteria:

1. **Low Spanish contamination in pretraining data** — most important.
2. **Strong English representations** — the model must have already learned syntax, semantics, and world knowledge so fine-tuning only needs to teach the lexical/morphological mapping to Spanish.
3. **Parameter count below ~1B** — per the project brief; SmolLM2-360M is below the target with headroom for the fine-tuned weights to absorb new lexical knowledge.

| Candidate                 | Params | Pretraining data         | Spanish exposure                 | Tokenizer English-bias |
| ------------------------- | ------ | ------------------------ | -------------------------------- | ---------------------- |
| **SmolLM2-360M** *(chosen)* | 360M | FineWeb-Edu + SmolLM-Corpus, English-filtered via classifier | Lowest among well-known options | High |
| Pythia-410M               | 410M   | The Pile                 | Low (~0.2% of Pile is Spanish)    | High |
| Pythia-1B                 | 1B     | The Pile                 | Same                              | High |
| TinyLlama-1.1B            | 1.1B   | SlimPajama + StarCoder   | Moderate (CommonCrawl-based)      | Medium |
| OLMo-1B                   | 1B     | Dolma                    | Moderate                          | Medium |
| OPT-350M                  | 350M   | Books + CommonCrawl etc. | Moderate                          | Medium |

SmolLM2-360M's pretraining corpus was deliberately English-filtered using a fastText classifier on FineWeb. We don't have access to a per-language audit of the training data, so this is "predominantly English" — not "zero Spanish." Pythia-410M is a close second on data transparency (The Pile is fully documented and ~99% English), and we recommend running it as an ablation if you want a published-baseline check.

## Honest caveat: no model has *zero* Spanish exposure

Every public LLM trained on web data has seen some Spanish, by accident or otherwise. We empirically quantify the Spanish disadvantage by probing the tokenizer (see `scripts/03_verify_model.py` and the generated `docs/MODEL_PROBE.md`). The metric to watch:

- **Fertility ratio** = mean tokens-per-word(Spanish) / mean tokens-per-word(English).
  - Ratio > 1.0 means Spanish text costs more tokens, which is the standard signal of an English-biased tokenizer.
  - Expected value for SmolLM2-360M: ~1.4–1.7 (typical for English-trained BPE tokenizers when scoring Spanish FLORES text).

A high fertility ratio matters because:
- Training is more expensive per Spanish sentence (more tokens per example).
- The model has to learn correct *composition* of subword pieces into Spanish words — harder than directly producing whole-word tokens.
- Generation is slower per Spanish output token.

## Tokenizer decision: keep original, do not extend

We deliberately **do not** extend the tokenizer with Spanish-specific tokens. Rationale:

- Extending the vocab would mean adding randomly-initialized embeddings to the model. Their gradient signal during fine-tuning is weak, especially at small data tiers (10k–100k pairs). The model often fails to integrate them well.
- The whole point of the study is to see how well the base model can learn Spanish *as a foreign language* through subword decomposition. Pre-supplying Spanish vocab partially circumvents that.
- If we eventually decide vocab extension is worth it, it should be a *separate* ablation (e.g., extend + continue pretraining on Spanish before fine-tuning).

If you ever want to add it: see `model/loader.py` — the `extend_tokenizer_for_spanish` flag in `configs/model.yaml` is the wiring point.

## Prompt template

All training and inference uses:

```
English: <text>
Spanish: <text>
```

Direction is sampled 50/50 per training example. Loss is computed only on the target side (tokens after the second `"Lang: "` marker), so the model learns to *generate* the translation rather than memorize the prompt prefix.

See `src/en_es_mt/data/format.py` for the implementation.

## Architecture notes

- **Decoder-only** vs encoder-decoder. Most published MT baselines (MarianMT, mT5, NLLB) are encoder-decoder. Decoder-only translation works well in practice but is generally a few BLEU lower than equally-sized seq2seq at the same data — this is built into our benchmark targets in `docs/BENCHMARKS.md`.
- **SDPA attention.** We use `attn_implementation="sdpa"` (PyTorch's native efficient attention) instead of FlashAttention to keep the dependency footprint clean. FlashAttention can be swapped in via `attn_implementation="flash_attention_2"` if/when we want it.
- **bfloat16.** Default dtype on A100. Falls back to fp32 on CPU smoke tests.
