# Project 2 — Results

Cross-lingual slot filling on MASSIVE · Group Coders
All figures are the **dev** split. Test is untouched, for a final single run.

---

## 1. Dataset

| | tr-TR | en-US |
|---|---|---|
| train / dev / test sentences | 11,514 / 2,033 / 2,974 | 11,514 / 2,033 / 2,974 |
| train tokens | 63,263 | 79,732 |
| distinct wordforms | **10,032** | **5,302** |
| type/token ratio | **0.159** | **0.066** |
| test OOV rate | **9.6%** | **3.5%** |
| slot classes | 55 | 55 |
| BIO labels | 109 | 107 |
| training spans | 11,257 | 11,344 |
| classes with <20 train spans | 11 | 11 |
| skipped utterances in conversion | **0** | **0** |

Same sentences, same annotations, **1.9× the vocabulary** in Turkish. That is
agglutination, measured before any model is trained.

Smallest classes: `music_album` and `game_type` have **1** training span each.

---

## 2. Main results — Turkish

| method | micro P | micro R | micro F1 | macro F1 | token acc | invalid BIO | train | predict |
|---|---|---|---|---|---|---|---|---|
| most-frequent-tag | 0.4921 | 0.4877 | 0.4899 | 0.3747 | 0.8312 | 440 | 0.1 s | 0.0 s |
| HMM + Viterbi | 0.5957 | 0.5203 | 0.5555 | 0.3558 | 0.8336 | 1 | 0.1 s | 6.4 s |
| CRF | 0.7800 | 0.6660 | 0.7185 | 0.5252 | 0.8830 | 1 | 168 s | 0.3 s |
| **BERTurk** | 0.7641 | 0.8152 | **0.7888** | **0.6124** | **0.9143** | 53 | 579 s | 2.1 s |

## 3. Main results — English

| method | micro F1 | macro F1 | token acc | invalid BIO |
|---|---|---|---|---|
| most-frequent-tag | 0.5491 | 0.4179 | 0.8735 | 462 |
| HMM + Viterbi | 0.6630 | 0.4383 | 0.8853 | 1 |
| CRF | 0.7876 | 0.6328 | 0.9231 | 1 |

Turkish is below English at every rung: 0.49/0.56/0.72 vs 0.55/0.66/0.79.
The gap is systematic, not an artifact of one method.

---

## 4. Significance (Turkish dev, 2,033 sentences)

95% bootstrap confidence intervals, 1,000 resamples:

| method | F1 | 95% CI |
|---|---|---|
| baseline | 0.4899 | [0.4675, 0.5132] |
| HMM | 0.5555 | [0.5331, 0.5793] |
| CRF | 0.7185 | [0.6977, 0.7384] |
| BERTurk | 0.7888 | [0.7716, 0.8061] |

Paired permutation tests, 10,000 permutations:

| comparison | difference | p-value | verdict |
|---|---|---|---|
| CRF vs baseline | +0.2286 | 0.0001 | highly significant |
| CRF vs HMM | +0.1630 | 0.0001 | highly significant |
| BERTurk vs CRF | +0.0703 | 0.0001 | highly significant |

No CIs overlap. Every step up the ladder is real.

---

## 5. THE HEADLINE — subword tokenization solves the morphology problem

F1 split by whether a span contains a token unseen in training:

| | CRF | BERTurk |
|---|---|---|
| seen spans | 0.7809 | 0.8145 |
| spans with an OOV token | 0.4201 | **0.6915** |
| **OOV penalty** | **−0.3608** | **−0.1230** |
| recall on OOV spans | 0.3190 | **0.7071** |

On *seen* spans BERTurk beats the CRF by only **+0.034**.
On *OOV* spans it beats it by **+0.271**.

OOV spans are 21% of the data but account for roughly **two-thirds** of
BERTurk's total advantage. Almost all of its win is about wordforms it has
never seen.

**Mechanism:** `Ankara'dan` is one unknown token to a word-level CRF.
WordPiece splits it into `Ankara` + `'` + `##dan`, pieces the model already
knows — so an unseen *word* is not unseen at the *subword* level. The hand-built
Turkish features (apostrophe clitics, 6-char suffixes) approximate this and get
about a third of the way there.

This is BERT2D's thesis about subwords in morphologically rich languages,
tested on a token-level task the paper did not evaluate.

### Cross-lingual confirmation (CRF)

| | OOV rate | OOV penalty |
|---|---|---|
| Turkish | 9.6% | −0.3608 |
| English | 3.5% | −0.2674 |

Turkish is hit twice: 2.7× more OOV tokens, and each costs 1.35× more F1.

---

## 6. Stratified analysis (Turkish)

### By span length — boundaries are the weak point

| span length | CRF | BERTurk | English CRF |
|---|---|---|---|
| 1 token | 0.7696 | 0.8164 | 0.8383 |
| 2 tokens | 0.6621 | 0.7710 | 0.7271 |
| 3 tokens | 0.6641 | 0.7303 | 0.7044 |
| 4+ tokens | **0.3596** | **0.4667** | **0.5000** |

Long spans need two correct boundaries. Both models collapse there, and Turkish
collapses harder than English.

### By class frequency tier — where the micro/macro gap lives

| tier | CRF | BERTurk | English CRF |
|---|---|---|---|
| head (top 18 classes) | 0.7487 | 0.8253 | 0.8188 |
| mid | 0.5986 | 0.6628 | 0.6623 |
| tail (bottom 18) | 0.4235 | 0.4915 | 0.5657 |

Seven classes score exactly 0.0000 even for BERTurk: `alarm_type`,
`audiobook_author`, `cooking_type`, `game_type`, `ingredient`, `movie_name`,
`transport_name`. All have fewer than 5 dev examples.

**55 classes is the label inventory, not the learnable inventory.**

---

## 7. Structural validity — the accuracy/well-formedness trade-off

| method | micro F1 | invalid BIO sequences |
|---|---|---|
| baseline | 0.4899 | **440** |
| HMM + Viterbi | 0.5555 | **1** |
| CRF | 0.7185 | **1** |
| BERTurk | 0.7888 | **53** |

The most accurate model is the least well-formed. BERTurk classifies each token
independently with no transition modelling, so it emits sequences like `I-date`
after `O`, which cannot decode to a valid span. The HMM and CRF forbid this by
construction through transition scores.

This is the empirical case for putting a CRF layer on top of BERT.

---

## 8. Cross-lingual transfer (CRF)

|  | → tr-TR | → en-US |
|---|---|---|
| **train tr-TR** | 0.7185 | 0.0673 |
| **train en-US** | 0.0261 | 0.7876 |

Transfer gap: Turkish 0.6512, English 0.7615.

Zero-shot transfer collapses to near zero. A CRF is a lexical model and the two
languages share almost no surface vocabulary. This is the motivation for a
multilingual encoder, not a failure of the experiment.

---

## 9. Training behaviour — BERTurk overfits after epoch 3

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

Training loss fell 1.514 → 0.0098 while eval loss **rose** from epoch 3 onward.
Classic overfitting. `load_best_model_at_end` selected epoch 5.

Ten epochs (the hyperparameter from the BERT2D paper) is too many for this
dataset. 4–5 would do, at half the compute.

---

## 10. Cost

| method | train | predict | notes |
|---|---|---|---|
| baseline | 0.1 s | 0.0 s | CPU |
| HMM | 0.1 s | **6.4 s** | Viterbi is O(T·\|Y\|²), 109 labels |
| CRF | 168 s | 0.3 s | CPU |
| BERTurk | 579 s | 2.1 s | T4 GPU |

The HMM trains instantly but predicts **20× slower than the CRF** — Viterbi does
~12,000 operations per token at 109 labels. The CRF trains 1,700× slower than
the HMM but predicts 20× faster. Accuracy is not the only axis.

---

## Reproducing

```
RUN_ME.bat            # data + baseline/HMM/CRF, Turkish
RUN_ANALYSIS.bat      # English, significance, stratified, transfer matrix
```
BERTurk needs a GPU; see README for the Colab steps.
