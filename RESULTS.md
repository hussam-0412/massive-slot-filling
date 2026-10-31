# Project 2 — Results

Cross-lingual slot filling on MASSIVE · Group Coders
All model scores below use the **development (dev) split**. No final test-set model scores are included in the supplied results. Test vocabulary statistics were inspected, so the test split should not be described as completely untouched.

The saved predictions and metrics are the basis of this report. See [README.md](README.md) for preprocessing, model settings, file descriptions and runnable commands.

---

## 1. Dataset

| | tr-TR | en-US |
|---|---|---|
| train / dev / test sentences | 11,514 / 2,033 / 2,974 | 11,514 / 2,033 / 2,974 |
| train tokens | 63,263 | 79,732 |
| distinct training wordforms | **10,032** | **5,302** |
| type/token ratio | **0.159** | **0.066** |
| test token OOV rate (relative to training vocabulary) | **9.6%** | **3.5%** |
| slot classes | 55 | 55 |
| BIO labels observed in training | 109 | 107 |
| training spans | 11,257 | 11,344 |
| classes with <20 train spans | 11 | 11 |
| skipped utterances in conversion (reported in the group summary) | **0** | **0** |

The two locales share the task and slot inventory, but use localized utterances. Turkish has approximately **1.9× as many distinct training wordforms** under the supplied whitespace tokenization. This is consistent with a role for morphology; lexical choices and tokenization also differ, so the comparison does not isolate agglutination as the cause.

The OOV rates above describe **test tokens**. The span-level OOV analysis in Section 5 instead uses **dev spans**. These are different populations. A reported zero skip count also does not establish that every annotation aligned: the converter's default mode retains alignment mismatches, while `--strict` excludes them.

Smallest classes: `music_album` and `game_type` have **1** training span each.

---

## 2. Main results — Turkish

| method | micro P | micro R | micro F1 | macro F1 | token accuracy | invalid BIO transitions |
|---|---|---|---|---|---|---|
| most-frequent-tag | 0.4921 | 0.4877 | 0.4899 | 0.3747 | 0.8312 | 440 |
| HMM + Viterbi | 0.5957 | 0.5203 | 0.5555 | 0.3558 | 0.8336 | 1 |
| CRF | 0.7800 | 0.6660 | 0.7185 | 0.5252 | 0.8830 | 1 |
| **BERTurk** | 0.7641 | 0.8152 | **0.7888** | **0.6124** | **0.9143** | 53 |

Span matches require the same boundaries and slot type after the evaluator's BIO-to-span conversion. Macro F1 averages the types appearing in either gold or predicted spans, rather than a fixed inventory of 55 types. The Turkish CRF evaluation includes 53 types; the other Turkish evaluations include 54. This difference should be considered when comparing macro scores.

## 3. Main results — English

| method | micro F1 | macro F1 | token accuracy | invalid BIO transitions |
|---|---|---|---|---|
| most-frequent-tag | 0.5491 | 0.4179 | 0.8735 | 462 |
| HMM + Viterbi | 0.6630 | 0.4383 | 0.8853 | 1 |
| CRF | 0.7876 | 0.6328 | 0.9231 | 1 |

The three classical models score lower on Turkish than English in these runs. English macro scores use 53 observed evaluation types. **BERTurk was evaluated on Turkish only**; no English transformer result is supplied.

---

## 4. Significance (Turkish dev, 2,033 sentences)

Sentence-level 95% bootstrap confidence intervals, 1,000 resamples:

| method | F1 | 95% CI |
|---|---|---|
| baseline | 0.4899 | [0.4675, 0.5132] |
| HMM | 0.5555 | [0.5331, 0.5793] |
| CRF | 0.7185 | [0.6977, 0.7384] |
| BERTurk | 0.7888 | [0.7716, 0.8061] |

Two-sided paired permutation tests, 10,000 permutations:

| comparison | difference | p-value | verdict |
|---|---|---|---|
| CRF vs baseline | +0.2286 | 0.0001 | highly significant |
| CRF vs HMM | +0.1630 | 0.0001 | highly significant |
| BERTurk vs CRF | +0.0703 | 0.0001 | highly significant |

The listed paired tests support differences between these saved prediction sets. The displayed p-values are rounded to four decimal places; the implementation uses an add-one correction. These tests measure uncertainty from the dev sentences, not variation across independent training seeds, and do not replace a final test-set evaluation.

---

## 5. Unseen wordforms — BERTurk has a smaller dev OOV gap

F1 grouped by whether a span contains any wordform absent from the Turkish training vocabulary. Vocabulary membership is case-sensitive. Gold and predicted spans are assigned to buckets separately, and each bucket's F1 is computed from its own TP, FP and FN counts:

| | CRF | BERTurk |
|---|---|---|
| seen spans | 0.7809 | 0.8145 |
| spans with an OOV token | 0.4201 | **0.6915** |
| **F1 drop from seen to OOV spans** | **0.3608** | **0.1230** |
| recall on OOV spans | 0.3190 | **0.7071** |

On *seen* spans BERTurk beats the CRF by only **+0.034**.
On *OOV* spans it beats it by **+0.271**.

Of the **1,991 gold Turkish dev spans**, 420 (21.1%) contain an OOV wordform. BERTurk's F1 improvement is larger in this bucket than in the seen-wordform bucket.

This supports better handling of unseen wordforms in this run. It does **not** establish that subword tokenization alone causes the improvement: BERTurk also differs in pretraining, architecture and contextual representation. The CRF already uses character suffix and other hand-built features. A controlled ablation would be needed to separate these effects.

Bucket F1 values cannot be weighted by the share of gold spans to attribute a fraction of the total F1 gain, because F1 depends on both gold and predicted counts. No percentage of the overall gain is assigned to OOV spans here.

### Within-language CRF comparison

| Dev span group | Turkish CRF F1 | English CRF F1 |
|---|---|---|
| Seen wordforms | 0.7809 | 0.8133 |
| Contains an OOV wordform | 0.4201 | 0.5459 |
| F1 drop | 0.3608 | 0.2674 |

The Turkish CRF has a larger dev OOV gap. This is a descriptive comparison across localized datasets, not a causal estimate of the cost of morphology or of each additional OOV token.

---

## 6. Stratified analysis (Turkish)

### By span length

| span length | CRF | BERTurk | English CRF |
|---|---|---|---|
| 1 token | 0.7696 | 0.8164 | 0.8383 |
| 2 tokens | 0.6621 | 0.7710 | 0.7271 |
| 3 tokens | 0.6641 | 0.7303 | 0.7044 |
| 4+ tokens | **0.3596** | **0.4667** | **0.5000** |

Four-or-more-token spans have the lowest F1 in each column. There are only **49** such gold spans in Turkish dev and **69** in English dev, so these estimates have limited support. Exact-span errors may involve boundaries, slot types or both; this table alone does not identify their cause.

### By training class frequency

| tier | CRF | BERTurk | English CRF |
|---|---|---|---|
| head (top 18 classes) | 0.7487 | 0.8253 | 0.8188 |
| mid (next 18 classes) | 0.5986 | 0.6628 | 0.6623 |
| tail (remaining 19 classes) | 0.4235 | 0.4915 | 0.5657 |

Tiers are assigned separately for each locale by training span counts. Values are micro F1 within each tier, not averages of the class F1 scores. The Turkish dev tail contains only 61 gold spans.

Seven classes **with gold Turkish dev examples** have BERTurk F1 of 0: `alarm_type`, `audiobook_author`, `cooking_type`, `game_type`, `ingredient`, `movie_name` and `transport_name`. Each has fewer than five dev examples. `transport_descriptor` also has F1 of 0 because it is predicted once despite having no gold dev examples.

These sparse classes make individual per-class estimates unstable. Their scores do not show that the classes are inherently unlearnable.

---

## 7. Structural validity

| method | micro F1 | invalid BIO transitions |
|---|---|---|
| baseline | 0.4899 | **440** |
| HMM + Viterbi | 0.5555 | **1** |
| CRF | 0.7185 | **1** |
| BERTurk | 0.7888 | **53** |

BERTurk has the highest F1 but more invalid transitions than the HMM or CRF. The baseline has the most invalid transitions overall.

The counter measures invalid `I-` transitions, **not sentences or whole sequences**. For example, `I-date` immediately after `O` is counted as invalid. The evaluator permissively treats a stray `I-` tag as the beginning of a new span and reports transition violations separately.

BERTurk uses contextual token representations with a token-classification head and no structured BIO decoder. The HMM and CRF learn transition scores, but these implementations do not hard-forbid all illegal BIO transitions; each produces one violation here.

A constrained decoder or a BERT–CRF extension is a possible follow-up experiment. Neither was evaluated in the supplied results.

---

## 8. Cross-lingual transfer (CRF)

|  | → tr-TR | → en-US |
|---|---|---|
| **train tr-TR** | 0.7185 | 0.0673 |
| **train en-US** | 0.0261 | 0.7876 |

For a model trained on Turkish, moving from Turkish dev to English dev lowers F1 by **0.6512**. For a model trained on English, moving to Turkish dev lowers F1 by **0.7615**.

Direct transfer is weak for this CRF feature set. These runs show limited transfer of the learned lexical, orthographic and contextual feature weights. A multilingual encoder is a possible future comparison; no multilingual neural transfer experiment is included.

---

## 9. BERTurk training history

The following epoch values and training-loss endpoints were reported in the supplied group summary. The saved `metrics_bert.json` contains the selected model's dev metrics; the package does not include a separate per-epoch training log.

| epoch | eval loss | eval F1 |
|---|---|---|
| 1 | 0.3734 | 0.7375 |
| 2 | 0.3717 | 0.7661 |
| 3 | **0.3627** | 0.7842 |
| 4 | 0.4414 | 0.7768 |
| 5 | 0.4511 | **0.7888** |
| 6 | 0.4874 | 0.7797 |
| 7 | 0.5393 | 0.7842 |
| 8 | 0.5688 | 0.7809 |
| 9 | 0.5991 | 0.7864 |
| 10 | 0.6050 | 0.7829 |

Reported training loss fell from 1.514 to 0.0098. The lowest dev loss occurred at epoch 3, while the highest dev F1 occurred at epoch 5. The script selects the checkpoint with the best dev F1, consistent with the saved F1 of 0.7888.

Rising dev loss with little further F1 improvement is consistent with overfitting or increasing confidence in errors. Early stopping is worth testing, but one training run does not establish an optimal epoch count. A dev-selected checkpoint still needs independent final evaluation.

---

## 10. Cost

| method | train | predict | notes |
|---|---|---|---|
| baseline | 0.05 s | 0.00 s | Saved Turkish timing; prediction time rounded |
| HMM | 0.06 s | 6.41 s | Saved Turkish timing |
| CRF | 168.08 s | 0.34 s | Saved Turkish timing |
| BERTurk | 579 s | 2.1 s | Reported in the group summary; T4 GPU reported |

The saved Turkish timings show HMM prediction taking approximately **19×** as long as CRF prediction. HMM Viterbi decoding has O(T·|Y|²) complexity in this implementation, whereas the CRF uses a compiled library. These are run-specific measurements, not hardware-normalized benchmarks. A recorded 0.00 seconds means below the stored timing precision, not zero computational work. BERTurk timings are reported by the group and are not stored in `metrics_bert.json`.

---

## Reproducing the analysis

Run the following from the repository root to analyze the existing predictions without retraining:

```bash
python src/significance.py --pred results/tr-TR/pred_crf_dev.json --compare results/tr-TR/pred_hmm_dev.json
python src/significance.py --pred results/tr-TR/pred_crf_dev.json --compare results/tr-TR/pred_baseline_dev.json
python src/significance.py --pred results/bert/pred_bert_dev.json --compare results/tr-TR/pred_crf_dev.json
python src/stratified.py --pred results/tr-TR/pred_crf_dev.json --train data/massive/tr-TR/train.conll
python src/stratified.py --pred results/en-US/pred_crf_dev.json --train data/massive/en-US/train.conll
python src/stratified.py --pred results/bert/pred_bert_dev.json --train data/massive/tr-TR/train.conll
```

Evidence files: `results/dataset_stats.json`, the language-specific `metrics.json` and `stratified.json` files, `results/bert/metrics_bert.json`, all `pred_*_dev.json` files, `results/crosslingual.json`, and `results/analysis_log.txt`. The saved analysis log covers selected classical-model analyses, not the BERTurk training history.

See [README.md](README.md) for full setup and training commands, including Colab use from an iPad. The `.bat` wrappers are for Windows and can recreate data or overwrite experiment outputs; they are not needed to inspect this report. No new training or final test evaluation was performed for this documentation update.
