"""Corpus statistics across locales.

Two jobs:

1. Evidence for the grading criteria - class count, dataset size, support
   distribution, and how far the task is from trivial.
2. Quantifying morphological richness, which is the project's actual thesis.
   Turkish and Finnish are agglutinative; English is not. Three measures make
   that visible rather than asserted:

     type/token ratio   - distinct wordforms per token. Agglutination inflates
                          the vocabulary, so this rises with morphology.
     test OOV rate      - share of test tokens never seen in training. The
                          direct cause of degraded span boundaries.
     tokens per span    - agglutinative languages pack into fewer, longer
                          wordforms, so spans get shorter in token terms.

    python src/dataset_stats.py --data data/massive --locales tr-TR en-US
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from conll import Sentence, read_conll  # noqa: E402
from evaluate import bio_to_spans  # noqa: E402


def load_locale(data_dir: str, locale: str) -> dict[str, list[Sentence]]:
    base = os.path.join(data_dir, locale)
    splits = {}
    for name in ("train", "dev", "test"):
        path = os.path.join(base, f"{name}.conll")
        if os.path.exists(path):
            splits[name] = read_conll(path)
    return splits


def stats_for(splits: dict[str, list[Sentence]]) -> dict:
    train = splits.get("train", [])
    test = splits.get("test", []) or splits.get("dev", [])

    train_vocab = {tok for toks, _ in train for tok in toks}
    train_tokens = sum(len(t) for t, _ in train)

    test_tokens = [tok for toks, _ in test for tok in toks]
    oov = sum(1 for tok in test_tokens if tok not in train_vocab)

    label_counts: Counter = Counter()
    span_lengths: list[int] = []
    n_spans = 0
    for _, labels in train:
        for lab in labels:
            label_counts[lab] += 1
        for s in bio_to_spans(labels):
            span_lengths.append(s[1] - s[0])
            n_spans += 1

    slot_support: Counter = Counter()
    for _, labels in train:
        for s in bio_to_spans(labels):
            slot_support[s[2]] += 1

    o_count = label_counts.get("O", 0)
    total_labels = sum(label_counts.values())

    return {
        "sentences": {k: len(v) for k, v in splits.items()},
        "train_tokens": train_tokens,
        "train_types": len(train_vocab),
        "type_token_ratio": len(train_vocab) / train_tokens if train_tokens else 0.0,
        "test_oov_rate": oov / len(test_tokens) if test_tokens else 0.0,
        "slot_classes": len(slot_support),
        "bio_labels": len(label_counts),
        "train_spans": n_spans,
        "mean_tokens_per_span": (
            sum(span_lengths) / len(span_lengths) if span_lengths else 0.0
        ),
        "o_label_share": o_count / total_labels if total_labels else 0.0,
        "slot_support": dict(slot_support.most_common()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/massive")
    ap.add_argument("--locales", nargs="+", default=["tr-TR", "en-US"])
    ap.add_argument("--out", default="results/dataset_stats.json")
    ap.add_argument("--rare-threshold", type=int, default=20)
    args = ap.parse_args()

    all_stats = {}
    for locale in args.locales:
        splits = load_locale(args.data, locale)
        if not splits:
            print(f"{locale}: no data found under {args.data}/{locale}/ - skipping")
            continue
        all_stats[locale] = stats_for(splits)

    if not all_stats:
        print("No locales loaded. Run data/load_massive.py first.")
        return

    # --- headline table -------------------------------------------------
    print(f"{'locale':<10}{'train':>9}{'dev':>7}{'test':>7}"
          f"{'tokens':>10}{'types':>9}{'TTR':>8}{'OOV':>8}")
    print("-" * 68)
    for locale, s in all_stats.items():
        n = s["sentences"]
        print(
            f"{locale:<10}{n.get('train', 0):>9}{n.get('dev', 0):>7}"
            f"{n.get('test', 0):>7}{s['train_tokens']:>10}{s['train_types']:>9}"
            f"{s['type_token_ratio']:>8.3f}{s['test_oov_rate']:>8.3f}"
        )

    print()
    print(f"{'locale':<10}{'slots':>8}{'BIO':>7}{'spans':>9}"
          f"{'tok/span':>10}{'O share':>10}{'rare':>7}")
    print("-" * 61)
    for locale, s in all_stats.items():
        rare = sum(
            1 for c in s["slot_support"].values() if c < args.rare_threshold
        )
        print(
            f"{locale:<10}{s['slot_classes']:>8}{s['bio_labels']:>7}"
            f"{s['train_spans']:>9}{s['mean_tokens_per_span']:>10.2f}"
            f"{s['o_label_share']:>10.3f}{rare:>7}"
        )
    print(f"\n('rare' = slot classes with fewer than {args.rare_threshold} "
          "training spans; these are where the macro-F1 damage happens)")

    # --- morphology contrast --------------------------------------------
    if len(all_stats) > 1:
        print("\nmorphological richness contrast")
        print("-" * 34)
        ranked = sorted(
            all_stats.items(), key=lambda kv: kv[1]["type_token_ratio"], reverse=True
        )
        for locale, s in ranked:
            print(
                f"  {locale:<8} TTR {s['type_token_ratio']:.3f}   "
                f"test OOV {s['test_oov_rate']:.1%}"
            )

    # --- least supported slots, per locale -------------------------------
    for locale, s in all_stats.items():
        tail = list(s["slot_support"].items())[-8:]
        if tail:
            print(f"\n{locale} least-supported slots:")
            for slot, c in tail:
                print(f"    {slot:<28}{c:>6}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(all_stats, fh, indent=2, ensure_ascii=False)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
