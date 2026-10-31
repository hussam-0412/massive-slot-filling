"""Train and evaluate every classical method on one fixed split.

    python src/run_all.py --data data/sample
    python src/run_all.py --data data/wikiner --methods baseline,hmm,crf

Design rule enforced here (W6 / W3 evaluation protocol):
  * the split assignment is identical for every method;
  * dev is used for tuning and reporting during development;
  * test is scored once, at the end, and never used for selection.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from conll import describe, entity_types, read_conll, split_sentences  # noqa: E402
from evaluate import evaluate, format_report  # noqa: E402


def load_splits(data_dir: str):
    train_p = os.path.join(data_dir, "train.conll")
    dev_p = os.path.join(data_dir, "dev.conll")
    test_p = os.path.join(data_dir, "test.conll")

    if os.path.exists(train_p) and os.path.exists(test_p):
        train = read_conll(train_p)
        dev = read_conll(dev_p) if os.path.exists(dev_p) else []
        test = read_conll(test_p)
        if not dev:
            train, dev, _ = split_sentences(train, dev_frac=0.1, test_frac=0.0)
        return train, dev, test

    # Single-file corpus: build a fixed split ourselves.
    candidates = [
        f for f in os.listdir(data_dir) if f.endswith((".conll", ".txt", ".iob"))
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No CoNLL files found in {data_dir}. "
            "Run data/download.py first, or data/make_sample.py for a smoke test."
        )
    all_sents = read_conll(os.path.join(data_dir, sorted(candidates)[0]))
    return split_sentences(all_sents)


def build(method: str):
    if method == "baseline":
        from baseline import MostFrequentTagTagger

        return MostFrequentTagTagger()
    if method == "hmm":
        from hmm_viterbi import HMMTagger

        return HMMTagger(alpha=0.1, rare_threshold=2)
    if method == "crf":
        from crf_model import CRFTagger

        return CRFTagger(c1=0.1, c2=0.1, max_iterations=150)
    raise ValueError(f"unknown method: {method}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/sample")
    ap.add_argument("--methods", default="baseline,hmm,crf")
    ap.add_argument("--out", default="results")
    ap.add_argument(
        "--final-test",
        action="store_true",
        help="Also score the test split. Use ONCE, after all tuning is done.",
    )
    args = ap.parse_args()

    train, dev, test = load_splits(args.data)
    print(describe("train", train))
    print(describe("dev", dev))
    print(describe("test", test))
    print("entity types:", ", ".join(entity_types(train)) or "(none)")
    print()

    os.makedirs(args.out, exist_ok=True)
    summary = {}

    for method in [m.strip() for m in args.methods.split(",") if m.strip()]:
        print(f"--- {method} ---")
        try:
            model = build(method)
        except ImportError as exc:
            print(f"  SKIPPED: {exc}\n")
            continue

        t0 = time.time()
        model.fit(train)
        train_time = time.time() - t0

        t0 = time.time()
        dev_pred = model.predict(dev)
        predict_time = time.time() - t0

        dev_gold = [list(labels) for _, labels in dev]
        dev_scores = evaluate(dev_gold, dev_pred)
        print(format_report(f"{method} / dev", dev_scores))
        print(f"  train {train_time:.1f}s   predict {predict_time:.1f}s")

        entry = {
            "dev": {k: v for k, v in dev_scores.items()},
            "train_seconds": round(train_time, 2),
            "predict_seconds": round(predict_time, 2),
        }

        if args.final_test:
            test_pred = model.predict(test)
            test_gold = [list(labels) for _, labels in test]
            test_scores = evaluate(test_gold, test_pred)
            print(format_report(f"{method} / TEST", test_scores))
            entry["test"] = {k: v for k, v in test_scores.items()}
            with open(
                os.path.join(args.out, f"pred_{method}_test.json"),
                "w",
                encoding="utf-8",
            ) as fh:
                json.dump(
                    [
                        {"tokens": t, "gold": g, "pred": p}
                        for (t, g), p in zip(test, test_pred)
                    ],
                    fh,
                    ensure_ascii=False,
                )

        # Always dump dev predictions for error analysis.
        with open(
            os.path.join(args.out, f"pred_{method}_dev.json"), "w", encoding="utf-8"
        ) as fh:
            json.dump(
                [
                    {"tokens": t, "gold": g, "pred": p}
                    for (t, g), p in zip(dev, dev_pred)
                ],
                fh,
                ensure_ascii=False,
            )

        summary[method] = entry
        print()

    with open(os.path.join(args.out, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    # Compact comparison table for the slide.
    width = 80
    print("=" * width)
    print(
        f"{'method':<22}{'micro P':>9}{'micro R':>9}{'micro F1':>10}"
        f"{'macro F1':>10}{'tok acc':>9}{'bad BIO':>9}"
    )
    print("-" * width)
    for method, entry in summary.items():
        d = entry["dev"]
        print(
            f"{method:<22}{d['precision']:>9.4f}{d['recall']:>9.4f}"
            f"{d['f1']:>10.4f}{d['macro_f1']:>10.4f}"
            f"{d['token_accuracy']:>9.4f}{d['invalid_bio_transitions']:>9}"
        )
    print("=" * width)
    print(
        "\nmicro weights every span equally; macro weights every CLASS equally.\n"
        "A large micro-macro gap means the method is carried by frequent slots.\n"
        "Next: src/significance.py (is a difference real?), "
        "src/stratified.py (where does it come from?)"
    )
    print(f"\nwrote {os.path.join(args.out, 'metrics.json')}")


if __name__ == "__main__":
    main()
