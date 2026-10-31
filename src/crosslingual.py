"""Cross-lingual transfer matrix for slot filling.

Trains on each source locale and evaluates on every target locale, producing
the matrix that is the project's headline result:

                 -> tr-TR   -> en-US
    tr-TR          0.74       0.21        (diagonal = in-language)
    en-US          0.18       0.81        (off-diagonal = zero-shot)

Finnish (fi-FI) is still supported - pass --locales tr-TR en-US fi-FI to add
a third agglutinative language if you have the time for it.

The diagonal answers "how hard is this language?"; the off-diagonal answers
"how much does morphology block transfer?". For the classical methods the
off-diagonal will be near zero - they are lexical models and share no
vocabulary across languages - which is itself the point: it motivates the
multilingual encoder in method 4.

    python src/crosslingual.py --data data/massive --method crf
    python src/crosslingual.py --data data/massive --method hmm --split test
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from conll import Sentence, read_conll  # noqa: E402
from evaluate import evaluate  # noqa: E402
from run_all import build  # noqa: E402


def load_split(data_dir: str, locale: str, split: str) -> list[Sentence]:
    path = os.path.join(data_dir, locale, f"{split}.conll")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return read_conll(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/massive")
    ap.add_argument("--locales", nargs="+", default=["tr-TR", "en-US"])
    ap.add_argument("--method", default="crf", choices=["baseline", "hmm", "crf"])
    ap.add_argument(
        "--split",
        default="dev",
        choices=["dev", "test"],
        help="Evaluate on dev while developing; use test once, at the end.",
    )
    ap.add_argument("--out", default="results/crosslingual.json")
    args = ap.parse_args()

    train_sets, eval_sets = {}, {}
    for locale in args.locales:
        try:
            train_sets[locale] = load_split(args.data, locale, "train")
            eval_sets[locale] = load_split(args.data, locale, args.split)
        except FileNotFoundError as exc:
            print(f"{locale}: missing {exc} - skipping")

    locales = [loc for loc in args.locales if loc in train_sets]
    if not locales:
        print("No locales loaded. Run data/load_massive.py first.")
        return

    matrix: dict[str, dict[str, float]] = {}

    for source in locales:
        try:
            model = build(args.method)
        except ImportError as exc:
            print(f"SKIPPED: {exc}")
            return

        print(f"training {args.method} on {source} "
              f"({len(train_sets[source])} sentences) ...")
        model.fit(train_sets[source])

        matrix[source] = {}
        for target in locales:
            data = eval_sets[target]
            gold = [list(labels) for _, labels in data]
            pred = model.predict(data)
            matrix[source][target] = evaluate(gold, pred)["f1"]

    # --- the matrix ------------------------------------------------------
    width = max(len(loc) for loc in locales) + 2
    print(f"\ncross-lingual span F1 ({args.method}, {args.split} split)")
    header = " " * (width + 8) + "".join(f"-> {loc:<9}" for loc in locales)
    print(header)
    print("-" * len(header))
    for source in locales:
        row = f"train {source:<{width}}"
        for target in locales:
            value = matrix[source][target]
            marker = "*" if source == target else " "
            row += f"{value:>8.4f}{marker}  "
        print(row)
    print("\n* = in-language (diagonal). Off-diagonal cells are zero-shot.")

    # --- transfer gap ----------------------------------------------------
    print("\ntransfer gap (in-language F1 minus mean zero-shot F1)")
    print("-" * 52)
    for source in locales:
        in_lang = matrix[source][source]
        others = [matrix[source][t] for t in locales if t != source]
        if not others:
            continue
        mean_zs = sum(others) / len(others)
        print(
            f"  {source:<8} in-language {in_lang:.4f}   "
            f"zero-shot {mean_zs:.4f}   gap {in_lang - mean_zs:.4f}"
        )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(
            {"method": args.method, "split": args.split, "matrix": matrix},
            fh,
            indent=2,
        )
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
