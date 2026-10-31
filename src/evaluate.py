"""Span-level exact-match evaluation for BIO-tagged sequences.

Implemented from scratch (no seqeval dependency) following the CoNLL
convention taught in W6:

  * A predicted entity counts as a true positive only if BOTH its boundaries
    and its type match a gold entity exactly.
  * Precision = TP / (TP + FP), Recall = TP / (TP + FN),
    F1 = 2PR / (P + R) = 2TP / (2TP + FP + FN)   (micro-averaged)

Token-level accuracy is also reported, but it is NOT the headline metric:
a high token score does not repair a boundary error.
"""

from __future__ import annotations

from collections import defaultdict

Span = tuple[int, int, str]  # (start, end_exclusive, type)


def bio_to_spans(labels: list[str]) -> list[Span]:
    """Decode a BIO label sequence into entity spans.

    Follows conlleval's permissive decoding: an ``I-X`` that does not continue
    an open ``X`` span opens a new one rather than being dropped. Independent
    token predictions can produce such sequences, so the decoder must not
    crash on them.
    """
    spans: list[Span] = []
    start: int | None = None
    cur_type: str | None = None

    for i, lab in enumerate(labels):
        if lab == "O" or lab is None:
            if start is not None:
                spans.append((start, i, cur_type))
                start, cur_type = None, None
            continue

        if "-" in lab:
            prefix, typ = lab.split("-", 1)
        else:  # malformed tag, treat the whole thing as a type with B
            prefix, typ = "B", lab

        if prefix == "B":
            if start is not None:
                spans.append((start, i, cur_type))
            start, cur_type = i, typ
        elif prefix == "I":
            if start is None or typ != cur_type:
                # Invalid continuation -> start a new span.
                if start is not None:
                    spans.append((start, i, cur_type))
                start, cur_type = i, typ
        else:  # unknown prefix
            if start is not None:
                spans.append((start, i, cur_type))
            start, cur_type = i, typ

    if start is not None:
        spans.append((start, len(labels), cur_type))
    return spans


def count_invalid_transitions(labels: list[str]) -> int:
    """Number of ``I-X`` tags that do not continue an open ``X`` span.

    Useful for the error-analysis slide: it quantifies how much structure a
    method violates when it labels each token independently.
    """
    invalid = 0
    cur_type: str | None = None
    for lab in labels:
        if lab == "O":
            cur_type = None
            continue
        if "-" not in lab:
            cur_type = None
            continue
        prefix, typ = lab.split("-", 1)
        if prefix == "B":
            cur_type = typ
        elif prefix == "I":
            if cur_type != typ:
                invalid += 1
            cur_type = typ
    return invalid


class Scores(dict):
    """Plain dict with a readable __str__ so results print nicely."""

    def __str__(self) -> str:  # pragma: no cover - presentation only
        return (
            f"micro P={self['precision']:.4f}  R={self['recall']:.4f}  "
            f"F1={self['f1']:.4f}\n"
            f"macro P={self['macro_precision']:.4f}  "
            f"R={self['macro_recall']:.4f}  F1={self['macro_f1']:.4f}  "
            f"(weighted F1={self['weighted_f1']:.4f}, "
            f"{self['n_classes']} classes)\n"
            f"TP={self['tp']} FP={self['fp']} FN={self['fn']}  "
            f"token_acc={self['token_accuracy']:.4f}  "
            f"invalid_BIO={self['invalid_bio_transitions']}"
        )


def evaluate(
    gold_sentences: list[list[str]],
    pred_sentences: list[list[str]],
) -> Scores:
    """Micro-averaged exact-match span scores plus per-type breakdown."""
    if len(gold_sentences) != len(pred_sentences):
        raise ValueError(
            f"sentence count mismatch: {len(gold_sentences)} gold vs "
            f"{len(pred_sentences)} predicted"
        )

    tp = fp = fn = 0
    tok_correct = tok_total = 0
    invalid = 0
    per_type: dict[str, dict[str, int]] = defaultdict(
        lambda: {"tp": 0, "fp": 0, "fn": 0}
    )

    for gold, pred in zip(gold_sentences, pred_sentences):
        if len(gold) != len(pred):
            raise ValueError(
                f"token count mismatch in a sentence: {len(gold)} vs {len(pred)}"
            )

        tok_total += len(gold)
        tok_correct += sum(g == p for g, p in zip(gold, pred))
        invalid += count_invalid_transitions(pred)

        gold_spans = set(bio_to_spans(gold))
        pred_spans = set(bio_to_spans(pred))

        matched = gold_spans & pred_spans
        tp += len(matched)
        fp += len(pred_spans - gold_spans)
        fn += len(gold_spans - pred_spans)

        for span in matched:
            per_type[span[2]]["tp"] += 1
        for span in pred_spans - gold_spans:
            per_type[span[2]]["fp"] += 1
        for span in gold_spans - pred_spans:
            per_type[span[2]]["fn"] += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 0.0

    by_type = {}
    for typ, c in sorted(per_type.items()):
        p = c["tp"] / (c["tp"] + c["fp"]) if (c["tp"] + c["fp"]) else 0.0
        r = c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) else 0.0
        f = (2 * p * r / (p + r)) if (p + r) else 0.0
        by_type[typ] = {
            "precision": p,
            "recall": r,
            "f1": f,
            "support": c["tp"] + c["fn"],
            **c,
        }

    # Macro-F1 averages the per-class F1 equally, so a method that nails the
    # three frequent slots and fails the long tail cannot hide behind micro.
    # With 55 classes the micro/macro gap is the single most informative
    # number on the results slide.
    macro_f1 = (
        sum(s["f1"] for s in by_type.values()) / len(by_type) if by_type else 0.0
    )
    macro_precision = (
        sum(s["precision"] for s in by_type.values()) / len(by_type)
        if by_type
        else 0.0
    )
    macro_recall = (
        sum(s["recall"] for s in by_type.values()) / len(by_type) if by_type else 0.0
    )
    total_support = sum(s["support"] for s in by_type.values())
    weighted_f1 = (
        sum(s["f1"] * s["support"] for s in by_type.values()) / total_support
        if total_support
        else 0.0
    )

    return Scores(
        precision=precision,
        recall=recall,
        f1=f1,
        macro_f1=macro_f1,
        macro_precision=macro_precision,
        macro_recall=macro_recall,
        weighted_f1=weighted_f1,
        n_classes=len(by_type),
        tp=tp,
        fp=fp,
        fn=fn,
        token_accuracy=tok_correct / tok_total if tok_total else 0.0,
        invalid_bio_transitions=invalid,
        by_type=by_type,
    )


def format_report(name: str, scores: Scores) -> str:
    lines = [f"=== {name} ===", str(scores), ""]
    lines.append(f"{'type':<14}{'P':>8}{'R':>8}{'F1':>8}{'support':>9}")
    for typ, s in scores["by_type"].items():
        lines.append(
            f"{typ:<14}{s['precision']:>8.4f}{s['recall']:>8.4f}"
            f"{s['f1']:>8.4f}{s['support']:>9}"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Self-test reproducing the worked example from the W6 lecture:
#   tokens: Ada | visited | New | York | .
#   gold:   Ada (PER), New York (LOC)
#   pred:   Ada (PER), York (LOC)
#   => TP=1, FP=1, FN=1, F1 = 2/(2+1+1) = 0.50
# --------------------------------------------------------------------------
if __name__ == "__main__":
    gold = [["B-PER", "O", "B-LOC", "I-LOC", "O"]]
    pred = [["B-PER", "O", "O", "B-LOC", "O"]]
    s = evaluate(gold, pred)
    assert s["tp"] == 1 and s["fp"] == 1 and s["fn"] == 1, s
    assert abs(s["f1"] - 0.5) < 1e-9, s
    print("lecture example reproduced:", s)

    # Adjacent same-type entities must stay separate (two B-PER in a row).
    assert bio_to_spans(["B-PER", "B-PER"]) == [(0, 1, "PER"), (1, 2, "PER")]
    # Invalid continuation opens a new span rather than crashing.
    assert bio_to_spans(["O", "I-LOC"]) == [(1, 2, "LOC")]
    print("all span-decoding assertions passed")
