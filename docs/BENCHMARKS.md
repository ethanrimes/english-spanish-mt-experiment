# EN-ES Machine Translation Benchmarks: Realistic Targets by Data Scale

**Scope.** Targets for fine-tuning SmolLM2-360M (decoder-only, English-heavy pretraining) on parallel EN-ES at six scales. Eval: FLORES-200 devtest, BLEU/chrF/COMET. Numbers below are **realistic targets**, not SOTA ceilings. EN-ES is a high-resource, typologically close pair on a Wikipedia-flavoured test set — numbers land well above what you'd see for, e.g., EN-Swahili at the same data scale.

---

## 1. Per-tier target table (FLORES-200 devtest)

Ranges assume vanilla causal-LM fine-tune with a translation prompt, no back-translation, basic dedup, full FT. **Low end** = sloppy hyperparams or noisy mined corpus; **high end** = well-tuned, clean corpus (CCMatrix/Europarl-grade). BLEU is sacrebleu BLEU; chrF is sacrebleu default.

| Pairs    | EN→ES BLEU | EN→ES chrF | ES→EN BLEU | ES→EN chrF | Citations |
|----------|------------|------------|------------|------------|-----------|
| 10k      | 4 – 10     | 25 – 38    | 5 – 12     | 28 – 40    | Sennrich & Zhang 2019 [Revisiting Low-Resource NMT](https://aclanthology.org/P19-1021/); Araabi & Monz 2020 [Optimizing Transformer for Low-Resource NMT](https://arxiv.org/abs/2011.02266) |
| 50k      | 10 – 18    | 35 – 48    | 12 – 20    | 38 – 50    | Araabi & Monz 2020 [arXiv:2011.02266](https://arxiv.org/abs/2011.02266); Sennrich & Zhang 2019 [ACL P19-1021](https://aclanthology.org/P19-1021/) |
| 100k     | 14 – 22    | 40 – 52    | 16 – 24    | 42 – 53    | Koehn & Knowles 2017 [Six Challenges for NMT](https://aclanthology.org/W17-3204/); Gowda & May 2020 [Finding the Optimal Vocabulary Size for NMT](https://arxiv.org/abs/2004.02334) |
| 500k     | 20 – 28    | 48 – 57    | 22 – 30    | 50 – 58    | Tatoeba Challenge [Tiedemann 2020](https://arxiv.org/abs/2010.06354); OPUS-MT [Tiedemann et al. 2022](https://arxiv.org/abs/2212.01936) |
| 1M       | 24 – 31    | 51 – 60    | 26 – 33    | 53 – 61    | OPUS-MT [arXiv:2212.01936](https://arxiv.org/abs/2212.01936); Bapna et al. 2022 [Building MT for the Next Thousand Languages](https://arxiv.org/abs/2205.03983) |
| 5M       | 27 – 33    | 54 – 62    | 29 – 35    | 56 – 63    | NLLB Team 2022 [arXiv:2207.04672](https://arxiv.org/abs/2207.04672); OPUS-MT [arXiv:2212.01936](https://arxiv.org/abs/2212.01936); Ghorbani et al. 2021 [Scaling Laws for NMT](https://arxiv.org/abs/2109.07740) |

**COMET (wmt22-comet-da, 0-100 scale).** Add the following targets independently — COMET correlates with quality but not linearly with BLEU; gains compress at the top. Both directions roughly:

| Pairs | COMET-22 target |
|-------|-----------------|
| 10k   | 55 – 68         |
| 50k   | 68 – 76         |
| 100k  | 73 – 80         |
| 500k  | 80 – 85         |
| 1M    | 82 – 86         |
| 5M    | 84 – 88         |

Anchor: NLLB-200-3.3B ~87 COMET-22 on EN-ES FLORES; Google Translate ~88; GPT-4 ~87–89. A 360M decoder-only hitting 85 at 5M pairs would be respectable.

**Confidence.** The 10k / 50k / 100k rows extrapolate from seq2seq encoder-decoder literature; decoder-only at this scale is under-studied and may underperform by 2–5 BLEU at the low end. The 500k / 1M / 5M rows are more grounded — architecture matters less than data at that scale.

---

## 2. Caveats (how your setup differs from cited papers)

Cited ranges mostly come from encoder-decoder seq2seq (Transformer-base, MarianMT, mT5). Your setup diverges in four ways:

1. **Decoder-only vs. enc-dec.** Causal LMs are 2–5 BLEU weaker than equivalently-sized enc-dec Transformers for supervised MT at the same scale ([ALMA, Xu et al. 2023](https://arxiv.org/abs/2309.11674); [Alves et al. 2024](https://arxiv.org/abs/2406.09140)) — they spend capacity modelling the source as a prefix. Working assumption: push the low end of each range down ~2 BLEU at small scales.

2. **English-monolingual base.** SmolLM2's tokenizer has poor Spanish coverage (fertility ~1.4–1.7× English). Hurts every scale but worst at 10k–100k. Expect 1–3 BLEU below a multilingual-pretrained 360M baseline (e.g. mT5-small). At 5M the gap narrows as the model repairs its own tokenization.

3. **Full FT vs. LoRA.** Full FT is the right call at 360M — LoRA at small scales trails by ~2 BLEU for translation specifically ([LoRA-FT LLaMA-3, IEEE 2024](https://ieeexplore.ieee.org/document/11200663/)). Don't downgrade targets.

4. **FLORES is Wikipedia-flavoured.** Europarl/OpenSubtitles-heavy training data lands 3–5 BLEU below CCMatrix-trained equivalents — worst at small scales. Mixing CCMatrix/NLLB-mined data is the highest-leverage data lever.

**Net effect:** table is accurate for a well-tuned setup; expect the *first* run at each scale to land 3–5 BLEU below the midpoint. Budget for one round of HP tuning before declaring a tier underperforming.

---

## 3. Headline reference points (FLORES-200 devtest, EN→ES unless noted)

"What good looks like" anchors. **Not all numbers are sacrebleu-comparable** — NLLB papers report spBLEU (~3–5 points higher than sacrebleu BLEU on Latin-script). Flagged below.

| System | Params | EN→ES BLEU | EN→ES chrF | Notes / source |
|--------|--------|-----------|-----------|----------------|
| NLLB-200 distilled 600M | 600M enc-dec MoE-distilled | ~28–29 (spBLEU ~30–32) | ~57 (chrF++) | [NLLB paper](https://arxiv.org/abs/2207.04672); HF model card lists FLORES-200 devtest metrics linked from the model card |
| NLLB-200 3.3B | 3.3B dense | ~30–32 (spBLEU ~33–35) | ~59–60 | [NLLB Team 2022](https://arxiv.org/abs/2207.04672); strongest of the open NLLB checkpoints |
| M2M-100 12B | 12B enc-dec | ~28 | ~57 | [Fan et al. 2020](https://arxiv.org/abs/2010.11125); pre-FLORES-200 architecture, beaten by NLLB |
| MarianMT / OPUS-MT en-es | ~75M enc-dec | ~27–28 on FLORES; **54.9 BLEU / 0.721 chrF on Tatoeba** | ~55 on FLORES | [HF model card](https://huggingface.co/Helsinki-NLP/opus-mt-en-es); Tatoeba ≠ FLORES — Tatoeba has shorter, easier sentences and the 54.9 number is *not* directly comparable |
| OPUS-MT es-en | ~75M enc-dec | — | — | **59.6 BLEU / 0.739 chrF on Tatoeba** ([HF model card](https://huggingface.co/Helsinki-NLP/opus-mt-es-en)); FLORES would be ~6–10 BLEU lower |
| Google Translate (API, 2023–24) | unknown, very large | ~31–33 | ~60 | Reported in [WMT24 findings](https://aclanthology.org/2024.wmt-1.1/) and [GPT-4 vs NMT comparisons](https://arxiv.org/abs/2301.08745) |
| GPT-4 (zero-shot) | ~1.7T MoE (rumoured) | ~28–30 | ~60 | [Jiao et al. 2023](https://arxiv.org/abs/2301.08745) — surprisingly close to Google despite zero-shot |
| ALMA-7B (LLaMA-2 finetuned) | 7B decoder-only | ~28 (WMT'22 test, not FLORES) | — | [Xu et al. 2023, arXiv:2309.11674](https://arxiv.org/abs/2309.11674); proves decoder-only LLMs *can* close the gap at 7B+ |
| TowerLLM 7B (Unbabel) | 7B decoder-only | ~30+ COMET-22 leader | — | [Tower announcement](https://unbabel.com/announcing-tower-an-open-multilingual-llm-for-translation-related-tasks/); only COMET reported publicly |

**Calibration.** 360M decoder-only at 5M pairs hitting ~30 BLEU / ~60 chrF would be OPUS-MT–tier and ~2–3 BLEU behind NLLB-distilled-600M — realistic ceiling. At 1M, target ~25–28 BLEU; at 100k, ~16–20 BLEU; at 10k, double-digit BLEU at all is the success criterion. Anything substantially above the top end of a row likely means a data leak — sanity-check by deduping FLORES devtest against training data.

---

## 4. Operational notes

- **sacrebleu `--tokenize 13a`** for cross-paper BLEU comparability. spBLEU (NLLB) reports 3–5 pts higher — don't mix.
- **chrF (not chrF++)** is the de facto FLORES standard since FLORES-101.
- **`wmt22-comet-da`** (or reference-free `wmt23-cometkiwi-da`) — pick one and use across all six runs.
- **Report BOTH directions.** ES→EN typically beats EN→ES by 1–3 BLEU (English easier to generate, tokenizer bias). Gap will be larger here because of the English-monolingual base.

