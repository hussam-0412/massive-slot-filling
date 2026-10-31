"""Confidence intervals and significance tests for span F1.

A single F1 number is not a result: on a 150-sentence dev split the difference
between 0.71 and 0.73 is noise. Two tools fix that, and both resample at the
SENTENCE level because spans within a sentence are not independent.

1. Bootstrap confidence interval
   Resample sentences with replacement B times, recompute F1 each time, and
   report the 2.5th/97.5th percentiles. Answers "how precisely do we know
   this method's F1?"

2. Paired permutation test (approximate randomisation)
   Both methods are scored on the SAME sentences, so the comparison is
   paired. Under the null hypothesis the two systems are interchangeable, so
   swapping their predictions on a random subset of sentences should not
   change the F1 difference. Swap repeatedly, count how often the shuffled
   difference is at least as large as the observed one, and that proportion
   is the p-value. Answers "is method A actually better than method B?"

Approximate randomisation is the standard test for NLP span metrics because
F1 is a corpus-level ratio, not a mean of per-sentence scores, so a t-test
over per-sentence F1 would be testing the wrong quantity.

    python src/significance.py --pred results/pred_crf_dev.json
    python src/significance.py --pred results/pred_crf_dev.json \
                               --compare results/pred_hmm_dev.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluate import bio_to_spans  # noqa: E402


def sentence_counts(gold: list[str], pred: list[str]) -> tuple[int, int, int]:
    """(tp, fp, fn) for one sentence, the unit we resample."""
    g = set(bio_to_spans(gold))
    p = set(bio_to_spans(pred))
    tp = len(g & p)
    return tp, len(p - g), len(g - p)


def f1_from_counts(counts: list[tuple[int, int, int]]) -> float:
    tp = sum(c[0] for c in counts)
    fp = sum(c[1] for c in counts)
    fn = sum(c[2] for c in counts)
    denom = 2 * tp + fp + fn
    return (2 * tp / denom) if denom else 0.0


def bootstrap_ci(
    counts: list[tuple[int, int, int]],
    n_samples: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Return (point estimate, lower bound, upper bound)."""
    rng = random.Random(seed)
    n = len(counts)
    point = f1_from_counts(counts)

    scores = []
    for _ in range(n_samples):
        sample = [counts[rng.randrange(n)] for _ in range(n)]
        scores.append(f1_from_counts(sample))
    scores.sort()

    lo = scores[int((alpha / 2) * n_samples)]
    hi = scores[min(int((1 - alpha / 2) * n_samples), n_samples - 1)]
    return point, lo, hi


def paired_permutation_test(
    counts_a: list[tuple[int, int, int]],
    counts_b: list[tuple[int, int, int]],
    n_samples: int = 10000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Return (F1_a, F1_b, p-value) for the two-sided paired test."""
    if len(counts_a) != len(counts_b):
        raise ValueError("the two systems must be scored on the same sentences")

    rng = random.Random(seed)
    f1_a = f1_from_counts(counts_a)
    f1_b = f1_from_counts(counts_b)
    observed = abs(f1_a - f1_b)

    n = len(counts_a)
    at_least_as_extreme = 0
    for _ in range(n_samples):
        swapped_a, swapped_b = [], []
        for i in range(n):
            if rng.random() < 0.5:
                swapped_a.append(counts_b[i])
                swapped_b.append(counts_a[i])
            else:
                swapped_a.append(counts_a[i])
                swapped_b.append(counts_b[i])
        if abs(f1_from_counts(swapped_a) - f1_from_counts(swapped_b)) >= observed:
            at_least_as_extreme += 1

    # Add-one correction: a p-value of exactly 0 is not supportable from a
    # finite number of resamples.
    p = (at_least_as_extreme + 1) / (n_samples + 1)
    return f1_a, f1_b, p


def load_counts(path: str) -> tuple[list[tuple[int, int, int]], list[dict]]:
    with open(path, encoding="utf-8") as fh:
        rows = json.load(fh)
    return [sentence_counts(r["gold"], r["pred"]) for r in rows], rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="results/pred_<method>_dev.json")
    ap.add_argument("--compare", help="a second prediction file to test against")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--permutations", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    counts_a, rows_a = load_counts(args.pred)
    name_a = os.path.basename(args.pred)

    point, lo, hi = bootstrap_ci(
        counts_a, n_samples=args.bootstrap, seed=args.seed
    )
    print(f"{name_a}")
    print(f"  sentences        {len(counts_a)}")
    print(f"  span F1          {point:.4f}")
    print(f"  95% CI           [{lo:.4f}, {hi:.4f}]   "
          f"(+/-{(hi - lo) / 2:.4f}, {args.bootstrap} bootstrap samples)")

    if not args.compare:
        print(
            "\nPass --compare <other pred file> to test whether a difference "
            "between two methods is significant."
        )
        return

    counts_b, rows_b = load_counts(args.compare)
    name_b = os.path.basename(args.compare)

    if len(rows_a) != len(rows_b):
        print(
            f"\nERROR: {name_a} has {len(rows_a)} sentences but {name_b} has "
            f"{len(rows_b)}. Both methods must be evaluated on the same split."
        )
        return

    point_b, lo_b, hi_b = bootstrap_ci(
        counts_b, n_samples=args.bootstrap, seed=args.seed
    )
    print(f"\n{name_b}")
    print(f"  span F1          {point_b:.4f}")
    print(f"  95% CI           [{lo_b:.4f}, {hi_b:.4f}]")

    f1_a, f1_b, p = paired_permutation_test(
        counts_a, counts_b, n_samples=args.permutations, seed=args.seed
    )
    better = name_a if f1_a > f1_b else name_b
    print("\npaired permutation test (approximate randomisation)")
    print(f"  difference       {abs(f1_a - f1_b):.4f}  in favour of {better}")
    print(f"  p-value          {p:.4f}   ({args.permutations} permutations)")
    if p < 0.001:
        verdict = "highly significant (p < 0.001)"
    elif p < 0.01:
        verdict = "significant (p < 0.01)"
    elif p < 0.05:
        verdict = "significant (p < 0.05)"
    else:
        verdict = "NOT significant - do not claim this method is better"
    print(f"  verdict          {verdict}")


if __name__ == "__main__":
    main()
