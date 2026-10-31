# Project 2: Turkish–English Slot Filling on MASSIVE

**COE025 Natural Language Processing · Group Coders**

This project labels the words in virtual-assistant requests with semantic slots such as `time`, `date`, `person` and `place_name`. It compares a word-frequency baseline, an HMM, a linear-chain CRF and fine-tuned BERTurk.

The experiments cover **Turkish (`tr-TR`) and English (`en-US`)**. BERTurk was run on Turkish only. The cross-language transfer experiment uses the CRF. Finnish and XLM-R are not part of the reported experiments.

**All model scores below are validation/dev scores. The supplied results contain no final test-set model evaluation.**

## Dataset and preprocessing

Source: [MASSIVE 1.1, Amazon Science](https://huggingface.co/datasets/AmazonScience/massive). The download script uses the [official MASSIVE release](https://github.com/alexa/massive#accessing-and-processing-the-data).

The included processed files use the original train/dev/test partition assignments:

| Property | Turkish | English |
|---|---:|---:|
| Training utterances | 11,514 | 11,514 |
| Validation utterances | 2,033 | 2,033 |
| Test utterances | 2,974 | 2,974 |
| Total utterances | 16,521 | 16,521 |
| Training tokens | 63,263 | 79,732 |
| Distinct training wordforms | 10,032 | 5,302 |
| Slot types observed in training | 55 | 55 |
| BIO labels observed in training | 109 | 107 |

BIO means beginning (`B-`), inside (`I-`) or outside (`O`) a slot. The observed label counts differ because some slot types have no inside-token examples in a language's training split. They are not 111 labels observed in both languages.

`data/load_massive.py` transforms the raw inline annotations into token–label pairs. For example:

```text
wake me up at [time : five am] on [date : friday]
```

becomes:

```text
wake    O
me      O
up      O
at      O
five    B-time
am      I-time
on      O
friday  B-date
```

The script:

1. Loads the locale JSONL files and their existing partition fields.
2. Splits annotated text on whitespace, retaining spelling, case and punctuation.
3. Assigns a new `B-` label at the start of every annotated slot, including adjacent slots of the same type.
4. Skips missing or empty annotations and reports skipped-row counts.
5. Checks annotation/plain-text alignment. With `--strict`, mismatches are excluded. The default retains the tokens derived from the annotation and does not separately report a mismatch count.
6. Writes `train.conll`, `dev.conll` and `test.conll`, with one token and BIO label per line and a blank line between utterances.

This is conversion and validation of an existing annotated corpus. The group did not collect or manually annotate the source dataset. The CoNLL files are processed data; they are not the original JSONL source files.

## Methods

| Method | Implementation and settings |
|---|---|
| Most-frequent-tag baseline | Assigns each word its most frequent training label, without sentence context |
| HMM + Viterbi | Learns emission and transition probabilities; smoothing `alpha=0.1`, rare-word threshold 2 |
| Linear-chain CRF | Uses word/context features, prefixes and suffixes, casing and apostrophe-root features; L-BFGS, `c1=0.1`, `c2=0.1`, maximum 150 iterations |
| BERTurk | Fine-tunes `dbmdz/bert-base-turkish-cased` with a token-classification head; default seed 42, batch size 16, learning rate 5e-5, maximum length 512, 10 epochs, weight decay 0.01 |

For BERTurk, each word's label is assigned to its first subword. Continuation subwords and special tokens receive `-100` and are ignored by the loss. The `O` label is still a normal output class. The script selects the checkpoint with the highest dev span F1.

The CRF feature extractor includes Turkish-aware handling of dotted/dotless I, suffixes up to six characters and roots before an apostrophe. The same feature implementation is used for both languages.

## Evaluation and results

The primary metric is **micro-averaged exact-match slot-span F1**. A predicted span must match both the boundaries and slot type. Token accuracy alone can hide missed slots and boundary errors, particularly when many tokens have the `O` label.

Macro F1 gives equal weight to each slot type represented in the gold or predicted spans of that evaluation. The implementation does not average over a fixed list of all 55 types when some types are absent. This should be considered when comparing macro scores.

| Method | Turkish micro F1 | Turkish macro F1 | English micro F1 | English macro F1 |
|---|---:|---:|---:|---:|
| Most-frequent-tag | 0.4899 | 0.3747 | 0.5491 | 0.4179 |
| HMM + Viterbi | 0.5555 | 0.3558 | 0.6630 | 0.4383 |
| CRF | 0.7185 | 0.5252 | 0.7876 | 0.6328 |
| BERTurk | **0.7888** | **0.6124** | Not run | Not run |

These values come from the saved predictions and metrics in `results/`; they are the group's runs, not borrowed benchmark scores. F1 values use a 0–1 scale: 0.7888 is 78.88% F1, not 78.88% token accuracy.

The CRF transfer experiment trains on one language's training split and evaluates on each language's dev split:

| Training language | Turkish dev F1 | English dev F1 |
|---|---:|---:|
| Turkish | 0.7185 | 0.0673 |
| English | 0.0261 | 0.7876 |

BERTurk has the strongest Turkish dev F1 among the tested models. The English CRF exceeds the Turkish CRF, while direct cross-language transfer of the lexical CRF performs poorly.

### Interpretation and limits

- The analysis groups errors by unseen wordforms, span length and training frequency. These observations are consistent with a role for morphology, but they do not isolate morphology or subword tokenization as the cause. The models also differ in pretraining and architecture.
- `invalid_bio_transitions` counts invalid `I-` transitions, not the number of sentences containing an error. Turkish counts are 440 for the baseline, 1 for HMM, 1 for CRF and 53 for BERTurk. Learned transition scores do not guarantee perfectly valid BIO sequences.
- The evaluator permissively starts a new span for a stray `I-` label, while separately counting the invalid transition.
- Dev data are used for checkpoint selection and analysis. These are development findings, not an unbiased final test benchmark. Descriptive test-vocabulary statistics have been computed, although test model scores are not included.
- Bootstrap confidence intervals and paired permutation tests operate on the saved dev predictions. They do not measure variation across independent training seeds.
- The dependency file specifies minimum versions rather than an exact recorded training environment. Identical neural-training scores across environments are not guaranteed.

The detailed experiment report is in [RESULTS.md](RESULTS.md). The JSON predictions and metrics provide the underlying evidence.

## Repository contents

| Path | Purpose |
|---|---|
| `README.md` | Project description, evaluation protocol and commands |
| `RESULTS.md` | Detailed supplied experiment report |
| `requirements.txt` | Python dependency requirements |
| `RUN_ME.bat` | Windows environment setup, tests, data download and Turkish classical models |
| `RUN_ANALYSIS.bat` | Windows English training, significance tests, stratified analysis and CRF transfer |
| `RUN_FINAL_TEST.bat` | Optional final classical-model evaluation after model choices are fixed |
| `data/load_massive.py` | Raw MASSIVE acquisition and inline-annotation conversion |
| `data/make_sample_massive.py` | Synthetic examples for testing the parser; not experimental data |
| `data/massive/tr-TR/` | Processed Turkish train/dev/test CoNLL files |
| `data/massive/en-US/` | Processed English train/dev/test CoNLL files |
| `src/baseline.py` | Word-frequency tagger |
| `src/hmm_viterbi.py` | HMM estimation and Viterbi decoding |
| `src/crf_model.py` | Linear-chain CRF wrapper |
| `src/features.py` | Token/context and morphological features |
| `src/bert_finetune.py` | BERTurk training, subword alignment and prediction export |
| `src/conll.py` | CoNLL reading/writing and split utilities |
| `src/evaluate.py` | Span decoding, metrics and invalid-transition counts |
| `src/run_all.py` | Classical training/evaluation for one language |
| `src/crosslingual.py` | Train/evaluate CRF or other classical methods across locales |
| `src/dataset_stats.py` | Split sizes, vocabulary, label counts and descriptive statistics |
| `src/error_analysis.py` | Boundary, type, missed and spurious error examples |
| `src/significance.py` | Sentence bootstrap intervals and paired permutation tests |
| `src/stratified.py` | Error breakdowns by vocabulary coverage, span length and class frequency |
| `src/test_pipeline.py` | 48 offline checks covering parsing, scoring and core utilities |
| `results/tr-TR/` | Turkish classical-model metrics, predictions and stratified counts |
| `results/en-US/` | English classical-model metrics, predictions and stratified counts |
| `results/bert/` | Turkish BERTurk dev metrics and predictions |
| `results/crosslingual.json` | CRF transfer matrix |
| `results/dataset_stats.json` | Statistics for both language datasets |
| `results/analysis_log.txt` | Saved console output for English classical models, selected significance comparisons, stratified analyses and transfer |

Model weights and training checkpoints are not included in the submitted experiment package.

## Running the project

Use **Python 3.10 or later**. Run commands from the repository root. On systems where the command is named `python3`, substitute it for `python`.

### Inspect and validate without training

The saved result files can be read without installing machine-learning libraries. The existing offline checks use the Python standard library:

```bash
python src/test_pipeline.py
```

Expected output ends with `48 checks passed.` This checks the core utilities; it is not a fresh reproduction of neural training.

### Classical models using the included data

Install the CRF dependency, then run each language. The commands write new outputs under `runs/` to preserve the submitted evidence in `results/`.

```bash
python -m pip install sklearn-crfsuite
python src/run_all.py --data data/massive/tr-TR --methods baseline,hmm,crf --out runs/tr-TR
python src/run_all.py --data data/massive/en-US --methods baseline,hmm,crf --out runs/en-US
python src/crosslingual.py --data data/massive --method crf --split dev --out runs/crosslingual.json
```

If CRF is unavailable, the baseline and HMM can run with `--methods baseline,hmm`.

### Analyse the saved predictions

```bash
python src/error_analysis.py --pred results/tr-TR/pred_crf_dev.json
python src/significance.py --pred results/tr-TR/pred_crf_dev.json --compare results/tr-TR/pred_hmm_dev.json
python src/significance.py --pred results/bert/pred_bert_dev.json --compare results/tr-TR/pred_crf_dev.json
python src/stratified.py --pred results/tr-TR/pred_crf_dev.json --train data/massive/tr-TR/train.conll --out runs/stratified_crf_tr.json
python src/stratified.py --pred results/bert/pred_bert_dev.json --train data/massive/tr-TR/train.conll --out runs/stratified_bert_tr.json
python src/dataset_stats.py --data data/massive --out runs/dataset_stats.json
```

### Regenerate processed data from the source release

The loader downloads MASSIVE 1.1 and extracts the raw JSONL files under `data/raw/`. The raw corpus requires additional download and disk space. Regenerated output goes to a separate directory:

```bash
python data/load_massive.py --locales tr-TR en-US --source raw --raw-dir data/raw --out runs/rebuilt_massive
```

For already downloaded raw JSONL files:

```bash
python data/load_massive.py --locales tr-TR en-US --source jsonl --jsonl-dir PATH_TO_JSONL_FOLDER --out runs/rebuilt_massive
```

Replace `PATH_TO_JSONL_FOLDER` with the directory containing `tr-TR.jsonl` and `en-US.jsonl`. The loader clears each requested locale's existing output directory before processing. Choose the output path deliberately. Adding `--strict` changes mismatch handling and may change the resulting data.

### BERTurk on a GPU or Google Colab

Install the dependencies and use the Turkish data:

```bash
python -m pip install -r requirements.txt
python src/bert_finetune.py --data data/massive/tr-TR --model dbmdz/bert-base-turkish-cased --out runs/bert --seed 42 --epochs 10 --batch-size 16 --lr 5e-5 --max-length 512
```

In Colab, enable a GPU runtime, make the repository files available in the runtime, change to the repository root and prefix shell commands with `!`. The `.bat` files are Windows launchers; use the Python commands in Colab.

### Synthetic parser checks

Synthetic examples belong in separate directories. They are not used in the reported benchmark:

```bash
python data/make_sample_massive.py --out runs/sample_jsonl --locales tr-TR en-US --n 1500
python data/load_massive.py --source jsonl --jsonl-dir runs/sample_jsonl --locales tr-TR en-US --out runs/sample_conll
```

### Final test protocol

Freeze model and hyperparameter choices using the training/dev splits before evaluating the test split. The classical and BERTurk scripts offer `--final-test`; this is optional and is not included in the commands above. `RUN_FINAL_TEST.bat` covers the classical methods only. If final test results are produced, report them separately from these dev scores.

The Windows launchers are convenience wrappers. They write to `results/`; `RUN_ME.bat` also regenerates `data/massive/`, and `RUN_ANALYSIS.bat` moves Turkish outputs into their locale folder. Use a separate working copy when running these wrappers to preserve the submitted files.

## Dataset attribution and references

MASSIVE is copyright Amazon.com Inc. or its affiliates and is distributed under [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/), as stated in its [dataset card](https://huggingface.co/datasets/AmazonScience/massive#licensing-information). The included CoNLL data are a format conversion of the Turkish and English portions. The transformation is described above. This dataset license does not establish a license for the group's own source code.

- FitzGerald et al. (2022), [MASSIVE: A 1M-Example Multilingual Natural Language Understanding Dataset with 51 Typologically-Diverse Languages](https://arxiv.org/abs/2204.08582).
- Bastianelli et al. (2020), [SLURP: A Spoken Language Understanding Resource Package](https://aclanthology.org/2020.emnlp-main.588/). MASSIVE uses English SLURP utterances as its starting point.
- [BERTurk model card](https://huggingface.co/dbmdz/bert-base-turkish-cased), dbmdz. This project fine-tunes the existing pretrained model.

