"""F1 broken down by the factors the project claims matter.

A single corpus F1 cannot support a claim about morphology. Three breakdowns
can, and together they form the argument:

  BY OOV STATUS      Spans containing a token never seen in training versus
                     spans fully covered by the training vocabulary. If
                     agglutination is what hurts Turkish, the OOV gap should
                     be larger in Turkish than in English. This is the
                     project's central hypothesis, stated as a measurement.

  BY SPAN LENGTH     1-token, 2-token, 3-token, 4+ spans. Longer spans need
                     two correct boundaries, so F1 should fall with length.
                     If it falls faster for one method, that method is
                     modelling boundaries worse, not entities worse.

  BY CLASS FREQUENCY Head / mid / tail thirds of the slot inventory by
                     training support. With 55 classes this is where the
                     micro/macro gap comes from, and it says whether a method
                     is genuinely better or just better at frequent slots.

    python src/stratified.py --pred results/pred_crf_dev.json \
                             --train data/massive/tr-TR/train.conll
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from conll import read_conll  # noqa: E402
from evaluate import bio_to_spans  # noqa: E402


def f1(tp: int, fp: int, fn: int) -> float:
    denom = 2 * tp + fp + fn
    return (2 * tp / denom) if denom else 0.0


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    return p, r, f1(tp, fp, fn)


def bucket_table(buckets: dict, title: str, label: str) -> None:
    print(f"\n{title}")
    print("-" * 64)
    print(f"{label:<20}{'P':>9}{'R':>9}{'F1':>9}{'gold':>8}{'pred':>8}")
    for key in sorted(buckets):
        c = buckets[key]
        p, r, f = prf(c["tp"], c["fp"], c["fn"])
        print(
            f"{str(key):<20}{p:>9.4f}{r:>9.4f}{f:>9.4f}"
            f"{c['tp'] + c['fn']:>8}{c['tp'] + c['fp']:>8}"
        )


def new_bucket() -> dict:
    return {"tp": 0, "fp": 0, "fn": 0}


def length_key(span) -> str:
    n = span[1] - span[0]
    return f"{n} token" if n < 4 else "4+ tokens"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--train", required=True, help="training CoNLL, for vocabulary")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    train = read_conll(args.train)
    train_vocab = {tok for toks, _ in train for tok in toks}

    support: Counter = Counter()
    for _, labels in train:
        for s in bio_to_spans(labels):
            support[s[2]] += 1

    # Head / mid / tail thirds of the class inventory by training support.
    ranked = [c for c, _ in support.most_common()]
    third = max(len(ranked) // 3, 1)
    tier = {}
    for i, cls in enumerate(ranked):
        if i < third:
            tier[cls] = "1-head"
        elif i < 2 * third:
            tier[cls] = "2-mid"
        else:
            tier[cls] = "3-tail"

    with open(args.pred, encoding="utf-8") as fh:
        rows = json.load(fh)

    by_oov: dict = {"seen": new_bucket(), "has OOV token": new_bucket()}
    by_len: dict = {}
    by_tier: dict = {}

    def oov_key(tokens, span) -> str:
        toks = tokens[span[0] : span[1]]
        return "seen" if all(t in train_vocab for t in toks) else "has OOV token"

    for row in rows:
        tokens, gold, pred = row["tokens"], row["gold"], row["pred"]
        gold_spans = set(bio_to_spans(gold))
        pred_spans = set(bio_to_spans(pred))

        for span in gold_spans | pred_spans:
            is_tp = span in gold_spans and span in pred_spans
            is_fp = span in pred_spans and span not in gold_spans
            is_fn = span in gold_spans and span not in pred_spans
            field = "tp" if is_tp else ("fp" if is_fp else "fn")

            by_oov[oov_key(tokens, span)][field] += 1
            by_len.setdefault(length_key(span), new_bucket())[field] += 1
            by_tier.setdefault(
                tier.get(span[2], "3-tail"), new_bucket()
            )[field] += 1

    print(f"stratified analysis: {os.path.basename(args.pred)}")
    print(f"training vocabulary: {len(train_vocab)} types, "
          f"{len(support)} slot classes")

    bucket_table(by_oov, "by OOV status (the morphology hypothesis)", "bucket")
    seen_f1 = f1(**by_oov["seen"])
    oov_f1 = f1(**by_oov["has OOV token"])
    print(f"\n  OOV penalty: {seen_f1 - oov_f1:+.4f} F1 "
          f"({seen_f1:.4f} seen -> {oov_f1:.4f} with OOV)")
    print("  Compare this gap across locales: a larger gap in Turkish than in")
    print("  English is the evidence that agglutination drives the errors.")

    bucket_table(by_len, "by span length (boundary modelling)", "span length")
    bucket_table(by_tier, "by class frequency tier (micro vs macro gap)", "tier")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "by_oov": by_oov,
                    "by_length": by_len,
                    "by_tier": by_tier,
                    "oov_penalty": seen_f1 - oov_f1,
                },
                fh,
                indent=2,
            )
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
