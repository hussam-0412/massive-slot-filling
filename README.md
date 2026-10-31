# Project 2 — Cross-Lingual Slot Filling Under Morphological Richness

COE025 Natural Language Processing · Token Classification · Group **Coders**

Slot filling on **MASSIVE** (Amazon Science) across Turkish, Finnish and
English: 55 slot types (111 BIO labels), ~19.5k utterances per language.

**Why this is not NER.** Slots are *intent-relative argument roles*, not entity
types. The same surface string takes a different label under a different
intent — `"eight"` is `time` under `alarm_set` but `quantity` under
`lists_createoradd`. The research question is how agglutinative morphology
degrades span boundaries, tested on Turkish and Finnish against analytic
English, which puts BERT2D's two-dimensional positional hypothesis on a
token-level task it was never evaluated on.

**Not a saturated benchmark.** Published slot F1 on MASSIVE: XLM-R Base
averages **73.6** (range 63.3–83.5), mT5 **76.8**. Zero-shot cross-lingual
falls to roughly **50**.

| # | Method | Idea | Needs |
|---|--------|------|-------|
| 1 | Most-frequent-tag baseline | memorise each word's commonest slot, no context | stdlib only |
| 2 | HMM + Viterbi | joint `P(labels, words)`, decoded exactly | stdlib only |
| 3 | Linear-chain CRF | conditional model over engineered morphological features | `sklearn-crfsuite` |
| 4 | BERTurk / XLM-R fine-tune | contextual encoder + token classification head | `torch`, `transformers`, GPU |

## Quick start

```bash
# 1. Smoke-test everything offline — no downloads, no dependencies
python data/make_sample_massive.py --out data/sample_massive --n 1500
python data/load_massive.py --jsonl-dir data/sample_massive --out data/massive
python src/test_pipeline.py                       # 37 checks
python src/dataset_stats.py --data data/massive
python src/crosslingual.py --data data/massive --method hmm

# 2. Get the real corpus
pip install -r requirements.txt
python data/load_massive.py --locales tr-TR en-US --out data/massive

# 3. Classical methods, per language
python src/run_all.py --data data/massive/tr-TR --methods baseline,hmm,crf

# 4. The headline result: cross-lingual transfer matrix
python src/crosslingual.py --data data/massive --method crf

# 5. Neural (GPU — Colab is fine)
python src/bert_finetune.py --data data/massive/tr-TR --model dbmdz/bert-base-turkish-cased
python src/bert_finetune.py --data data/massive/tr-TR --model xlm-roberta-base

# 6. Error analysis for the presentation
python src/error_analysis.py --pred results/pred_crf_dev.json

# 7. Is the difference between two methods real?
python src/significance.py --pred results/pred_crf_dev.json \
                           --compare results/pred_hmm_dev.json

# 8. Where does the difference come from?
python src/stratified.py --pred results/pred_crf_dev.json \
                         --train data/massive/tr-TR/train.conll
```

`--final-test` scores the held-out test split. Use it **once**, at the end.

## Dataset

[AmazonScience/massive](https://huggingface.co/datasets/AmazonScience/massive)
· [paper](https://arxiv.org/abs/2204.08582)

1M human-localized virtual-assistant utterances, 51 languages, 60 intents,
**55 slot types**, 18 domains; ~19,521 utterances/language
(587k train / 104k dev / 152k test). Human-translated and human-annotated,
not synthetic.

MASSIVE stores slots inline rather than as columns:

```
utt       : wake me up at five am on friday
annot_utt : wake me up at [time : five am] on [date : friday]
```

`data/load_massive.py` converts that bracket notation to aligned BIO and
verifies the stripped annotation reproduces `utt`. Utterances where the two
disagree are **counted and reported**, not silently dropped — `--strict`
excludes them.

> The trap this parser is tested against: two adjacent slots of the same type,
> `[date : friday] [date : saturday]`, must produce `B-date B-date`, not
> `B-date I-date`. Getting that wrong merges two spans into one and corrupts
> every F1 number downstream without raising an error.

## Evaluation

`src/evaluate.py` implements span-level exact-match scoring from scratch — no
`seqeval` dependency, so the core pipeline runs on a bare Python install.

A prediction is a true positive only if **both boundaries and the slot type**
match. Predicting `York` when the gold span is `New York` is one false positive
*and* one false negative.

```
Precision = TP / (TP + FP)      Recall = TP / (TP + FN)
F1 = 2PR / (P + R) = 2TP / (2TP + FP + FN)      (micro-averaged)
```

Verified against the worked example from the W6 lecture (gold `Ada` +
`New York`, predicted `Ada` + `York` → F1 exactly 0.50).

**Micro and macro are both reported.** Micro weights every span equally; macro
weights every *class* equally. With 55 slot types a method can score well on
micro while failing most of the inventory, so the micro–macro gap is the most
informative number on the results slide.

### Is a difference real? (`src/significance.py`)

A single F1 is not a result. On a dev split of a few hundred sentences, the
gap between 0.71 and 0.73 is usually noise.

- **Bootstrap 95% CI** — resamples sentences with replacement, because spans
  within a sentence are not independent.
- **Paired permutation test** (approximate randomisation) — both methods are
  scored on the same sentences, so the comparison is paired. This is the
  standard test for span metrics, because F1 is a corpus-level ratio rather
  than a mean of per-sentence scores, so a t-test would be testing the wrong
  quantity.

Observed on a deliberately undertrained run: baseline 0.8309 vs HMM 0.7682
looks decisive, but comes out **p = 0.066 — not significant**. Report the test,
not just the gap.

### Where does the difference come from? (`src/stratified.py`)

Three breakdowns that turn the morphology claim into a measurement:

- **By OOV status** — spans containing a token unseen in training vs. spans
  fully covered. *A larger OOV penalty in Turkish than in English is the
  evidence that agglutination drives the errors.* This is the central
  hypothesis, stated as a number.
- **By span length** — longer spans need two correct boundaries, so F1 should
  fall with length; a method that falls faster is modelling boundaries worse.
- **By class frequency tier** — head/mid/tail thirds of the inventory, which is
  where the micro–macro gap comes from.

## Measuring morphological richness

`src/dataset_stats.py` quantifies the thesis instead of asserting it:

- **type/token ratio** — agglutination inflates the vocabulary, so this rises
  with morphological complexity
- **test OOV rate** — share of test tokens never seen in training; the direct
  cause of degraded span boundaries
- **tokens per span** — agglutinative languages pack meaning into fewer, longer
  wordforms, so spans shorten in token terms
- plus class count, support per slot, and how many slots are rare enough to
  wreck macro-F1

## Turkish-specific features (method 3)

`src/features.py`:

- **Apostrophe features.** Turkish separates a proper noun from its case suffix
  with an apostrophe — `Ankara'dan`, `Türkiye'nin`. Exposed as
  `has_apostrophe` and `apos_root`, so `Ankara`, `Ankara'dan` and
  `Ankara'daki` share evidence instead of being three unrelated strings.
- **Long suffixes (up to 6 chars)** for agglutinative morphology
  (`ev → evlerimizden`). The 2–3 char window that suffices for English
  truncates the signal.
- **Turkish-aware casing.** Naive `.lower()` maps `İ` to `i` + combining dot
  and corrupts the token; `tr_lower()` handles dotted/dotless i.

## Subword alignment (method 4)

First-subword policy, per W6: a word's label goes on its **first** subword;
continuation pieces and special tokens get `-100`, which the loss ignores.
`-100` is **not** an extra class — `O` remains a real, modelled label.
`id2label` is saved with the model so spans decode correctly later.

## Protocol

- Split assignment is identical for every method.
- Dev is for tuning. Test is scored once and never enters selection.
- The HMM's vocabulary and emission counts come from training data only.

## Layout

```
├── data/
│   ├── load_massive.py        # annot_utt -> BIO CoNLL, with alignment checks
│   ├── make_sample_massive.py # synthetic MASSIVE-shaped JSONL, offline tests
│   └── make_sample.py         # generic BIO sample
├── src/
│   ├── conll.py               # CoNLL I/O, splits, corpus stats
│   ├── evaluate.py            # span decoding + exact-match P/R/F1
│   ├── features.py            # morphological feature engineering
│   ├── baseline.py            # method 1
│   ├── hmm_viterbi.py         # method 2
│   ├── crf_model.py           # method 3
│   ├── bert_finetune.py       # method 4
│   ├── crosslingual.py        # transfer matrix across locales
│   ├── dataset_stats.py       # corpus + morphology statistics
│   ├── error_analysis.py      # BOUNDARY / TYPE / SPURIOUS / MISSED
│   ├── run_all.py             # train + evaluate, write metrics.json
│   └── test_pipeline.py       # 37 checks, no pytest needed
└── results/
```

## Notes

Numbers from `data/sample_massive` are **not results** — it is
template-generated with a closed vocabulary, so every model saturates at 1.00.
Report real MASSIVE figures only.
