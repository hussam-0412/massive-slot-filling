"""Error analysis for the presentation's error slide.

Breaks every mistake into the categories the W6 lecture distinguished, which
is far more informative on a slide than a single F1 number:

  BOUNDARY  - right type, wrong extent  (predicted "York", gold "New York")
  TYPE      - right extent, wrong type  (ORG labelled as LOC)
  SPURIOUS  - predicted an entity where gold has none
  MISSED    - gold entity with no overlapping prediction

It also reports how many invalid BIO transitions each method emits, which is
the concrete argument for joint decoding over independent token classification.

    python src/error_analysis.py --pred results/pred_crf_dev.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluate import bio_to_spans, count_invalid_transitions  # noqa: E402


def overlaps(a: tuple[int, int, str], b: tuple[int, int, str]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def categorise(gold_labels: list[str], pred_labels: list[str]):
    gold_spans = bio_to_spans(gold_labels)
    pred_spans = bio_to_spans(pred_labels)
    gold_set, pred_set = set(gold_spans), set(pred_spans)

    errors = []
    for p in pred_spans:
        if p in gold_set:
            continue
        same_extent = [g for g in gold_spans if g[0] == p[0] and g[1] == p[1]]
        if same_extent:
            errors.append(("TYPE", p, same_extent[0]))
            continue
        overlapping = [g for g in gold_spans if overlaps(p, g)]
        if overlapping:
            same_type = [g for g in overlapping if g[2] == p[2]]
            errors.append(
                ("BOUNDARY", p, (same_type or overlapping)[0])
            )
            continue
        errors.append(("SPURIOUS", p, None))

    for g in gold_spans:
        if g in pred_set:
            continue
        if not any(overlaps(g, p) for p in pred_spans):
            errors.append(("MISSED", None, g))
    return errors


def text(tokens: list[str], span) -> str:
    if span is None:
        return "-"
    return " ".join(tokens[span[0] : span[1]])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="results/pred_<method>_dev.json")
    ap.add_argument("--examples", type=int, default=12)
    args = ap.parse_args()

    with open(args.pred, encoding="utf-8") as fh:
        rows = json.load(fh)

    kinds: Counter = Counter()
    confusion: Counter = Counter()
    by_type: Counter = Counter()
    invalid = 0
    samples: dict[str, list[str]] = {}

    for row in rows:
        tokens, gold, pred = row["tokens"], row["gold"], row["pred"]
        invalid += count_invalid_transitions(pred)
        for kind, p_span, g_span in categorise(gold, pred):
            kinds[kind] += 1
            ref = g_span or p_span
            by_type[ref[2]] += 1
            if kind == "TYPE":
                confusion[(g_span[2], p_span[2])] += 1
            bucket = samples.setdefault(kind, [])
            if len(bucket) < args.examples:
                bucket.append(
                    f'    "{text(tokens, p_span)}" [{p_span[2] if p_span else "-"}]'
                    f'  <-  gold "{text(tokens, g_span)}" '
                    f'[{g_span[2] if g_span else "-"}]'
                )

    total = sum(kinds.values())
    print(f"total errors: {total}")
    print(f"invalid BIO transitions in predictions: {invalid}\n")

    print(f"{'error kind':<12}{'count':>8}{'share':>9}")
    print("-" * 29)
    for kind, c in kinds.most_common():
        print(f"{kind:<12}{c:>8}{c / total:>8.1%}")

    print(f"\n{'entity type':<16}{'errors':>8}")
    print("-" * 24)
    for typ, c in by_type.most_common():
        print(f"{typ:<16}{c:>8}")

    if confusion:
        print(f"\n{'gold -> predicted':<34}{'count':>7}")
        print("-" * 41)
        for (g, p), c in confusion.most_common(12):
            print(f"{g + ' -> ' + p:<34}{c:>7}")

    for kind in ("BOUNDARY", "TYPE", "SPURIOUS", "MISSED"):
        if kind in samples:
            print(f"\n{kind} examples:")
            print("\n".join(samples[kind]))


if __name__ == "__main__":
    main()
