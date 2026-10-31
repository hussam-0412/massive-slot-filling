"""Self-contained tests. Run with:  python src/test_pipeline.py

No pytest required, so every teammate can run it on a bare Python install.
"""

from __future__ import annotations

import os
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "data"))

from baseline import MostFrequentTagTagger
from conll import read_conll, write_conll
from evaluate import bio_to_spans, count_invalid_transitions, evaluate
from features import apostrophe_root, is_title, token_features, tr_lower, word_shape
from hmm_viterbi import HMMTagger

PASSED = 0


def check(label: str, condition: bool, detail: str = "") -> None:
    global PASSED
    if condition:
        PASSED += 1
        print(f"  PASS  {label}")
    else:
        raise AssertionError(f"FAIL  {label}  {detail}")


def test_span_decoding() -> None:
    print("span decoding")
    check(
        "simple span",
        bio_to_spans(["B-PER", "I-PER", "O"]) == [(0, 2, "PER")],
    )
    check(
        "adjacent same-type entities stay separate",
        bio_to_spans(["B-PER", "B-PER"]) == [(0, 1, "PER"), (1, 2, "PER")],
    )
    check(
        "stray I- opens a new span",
        bio_to_spans(["O", "I-LOC"]) == [(1, 2, "LOC")],
    )
    check(
        "type change inside a span splits it",
        bio_to_spans(["B-PER", "I-LOC"]) == [(0, 1, "PER"), (1, 2, "LOC")],
    )
    check("empty sequence", bio_to_spans([]) == [])
    check(
        "invalid transitions counted",
        count_invalid_transitions(["O", "I-LOC", "I-PER"]) == 2,
    )


def test_lecture_example() -> None:
    """W6 worked example: F1 must be exactly 0.50."""
    print("lecture worked example")
    gold = [["B-PER", "O", "B-LOC", "I-LOC", "O"]]
    pred = [["B-PER", "O", "O", "B-LOC", "O"]]
    s = evaluate(gold, pred)
    check("TP=1 FP=1 FN=1", (s["tp"], s["fp"], s["fn"]) == (1, 1, 1), str(s))
    check("F1 == 0.50", abs(s["f1"] - 0.5) < 1e-9, str(s))


def test_perfect_and_empty() -> None:
    print("evaluation edge cases")
    gold = [["B-ORG", "I-ORG", "O"]]
    check("perfect prediction -> F1 1.0", evaluate(gold, gold)["f1"] == 1.0)
    all_o = [["O", "O", "O"]]
    s = evaluate(gold, all_o)
    check("all-O prediction -> F1 0.0", s["f1"] == 0.0 and s["fn"] == 1)
    s2 = evaluate(all_o, all_o)
    check("no entities anywhere -> F1 0.0, no crash", s2["f1"] == 0.0)


def test_mismatch_raises() -> None:
    print("guards")
    try:
        evaluate([["O"]], [["O"], ["O"]])
        check("sentence count mismatch raises", False)
    except ValueError:
        check("sentence count mismatch raises", True)
    try:
        evaluate([["O", "O"]], [["O"]])
        check("token count mismatch raises", False)
    except ValueError:
        check("token count mismatch raises", True)


def test_turkish_features() -> None:
    print("Turkish features")
    check("dotted I folds correctly", tr_lower("İstanbul") == "istanbul")
    check("dotless I folds correctly", tr_lower("ISPARTA") == "ısparta")
    check("no combining dot leaks", "i̇" not in tr_lower("İzmir"))
    check("title detection", is_title("İzmir") and not is_title("ev"))
    check("apostrophe root", apostrophe_root("Ankara'dan") == "Ankara")
    check("no apostrophe", apostrophe_root("evlerimizden") is None)
    check("word shape", word_shape("Ankara'dan") == "Xx'x")

    f = token_features(["Ankara'dan", "geldi", "."], 0)
    check("apostrophe feature set", f["word.has_apostrophe"] is True)
    check("apostrophe root feature", f["word.apos_root"] == "ankara")
    check("right context", f["+1:word.lower"] == "geldi")
    check("BOS marker", f["-1:boundary"] == "BOS")
    check("long suffixes present", "suffix6" in f)


def test_conll_roundtrip() -> None:
    print("CoNLL round-trip")
    sents = [(["Ankara'dan", "geldi"], ["B-LOC", "O"]), (["Ali"], ["B-PER"])]
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "x.conll")
        write_conll(p, sents)
        back = read_conll(p)
    check("round-trip preserves content", back == sents, str(back))


def test_models_learn() -> None:
    print("models learn a trivial pattern")
    train = [(["Ali", "geldi"], ["B-PER", "O"])] * 40
    train += [(["Ankara", "güzel"], ["B-LOC", "O"])] * 40
    dev = [(["Ali", "geldi"], ["B-PER", "O"]), (["Ankara", "güzel"], ["B-LOC", "O"])]
    gold = [labs for _, labs in dev]

    base = MostFrequentTagTagger().fit(train)
    check("baseline memorises", evaluate(gold, base.predict(dev))["f1"] == 1.0)

    hmm = HMMTagger(alpha=0.01, rare_threshold=1).fit(train)
    s = evaluate(gold, hmm.predict(dev))
    check("HMM learns the pattern", s["f1"] == 1.0, str(s))


def test_hmm_beats_baseline_on_context() -> None:
    """An ambiguous word that only context can disambiguate.

    "Ay" is a person name after "Sayın" and a plain noun otherwise. The
    baseline must pick one tag for the word; the HMM can use the previous
    label to choose. This is the whole argument for sequence models.
    """
    print("HMM uses context where the baseline cannot")
    train = []
    train += [(["Sayın", "Ay", "geldi"], ["O", "B-PER", "O"])] * 30
    train += [(["Ay", "parlak", "bugün"], ["O", "O", "O"])] * 60
    dev = [(["Sayın", "Ay", "geldi"], ["O", "B-PER", "O"])]
    gold = [labs for _, labs in dev]

    base_f1 = evaluate(gold, MostFrequentTagTagger().fit(train).predict(dev))["f1"]
    hmm_f1 = evaluate(
        gold, HMMTagger(alpha=0.01, rare_threshold=0).fit(train).predict(dev)
    )["f1"]
    check(
        f"HMM ({hmm_f1:.2f}) > baseline ({base_f1:.2f}) on ambiguous token",
        hmm_f1 > base_f1,
        f"hmm={hmm_f1} base={base_f1}",
    )


def test_massive_parser() -> None:
    """MASSIVE's annot_utt bracket notation -> aligned BIO labels."""
    print("MASSIVE annot_utt parsing")
    from load_massive import check_alignment, convert_rows, parse_annot_utt

    toks, labs = parse_annot_utt(
        "wake me up at [time : five am] on [date : friday]"
    )
    check(
        "multiword slot gets B- then I-",
        labs == ["O", "O", "O", "O", "B-time", "I-time", "O", "B-date"],
        str(labs),
    )
    check(
        "tokens match the plain utterance",
        check_alignment("wake me up at five am on friday", toks),
    )

    # The trap: two adjacent slots of the SAME type must stay two spans.
    # Emitting B-date I-date here would silently merge them and corrupt F1.
    _, labs = parse_annot_utt("[date : friday] [date : saturday]")
    check("adjacent same-type slots stay separate", labs == ["B-date", "B-date"])

    toks, labs = parse_annot_utt("no slots at all")
    check("slotless utterance is all O", labs == ["O"] * 4 and len(toks) == 4)

    # An empty slot value must not produce a zero-width span.
    toks, labs = parse_annot_utt("hello [time : ] world")
    check(
        "empty slot value is dropped",
        toks == ["hello", "world"] and labs == ["O", "O"],
        str((toks, labs)),
    )

    # Turkish apostrophe clitics must stay inside the slot value.
    toks, labs = parse_annot_utt("[place_name : ankara'dan] geldim")
    check(
        "Turkish clitic stays in the span",
        toks == ["ankara'dan", "geldim"] and labs == ["B-place_name", "O"],
        str((toks, labs)),
    )

    rows = [
        {"utt": "call john", "annot_utt": "call [person : john]"},
        {"utt": "", "annot_utt": ""},  # malformed row must be skipped
    ]
    sents, skipped = convert_rows(rows)
    check("malformed rows are skipped, not crashed on", len(sents) == 1 and skipped == 1)

    # Round-trip: parsed labels must decode back to the right spans.
    _, labs = parse_annot_utt("at [time : half past eight] in [place_name : london]")
    spans = bio_to_spans(labs)
    check(
        "decoded spans match the annotation",
        spans == [(1, 4, "time"), (5, 6, "place_name")],
        str(spans),
    )


def test_macro_vs_micro() -> None:
    """Macro-F1 must expose long-tail failure that micro-F1 hides."""
    print("macro vs micro F1")
    # 9 frequent spans predicted perfectly, 1 rare-class span missed entirely.
    gold = [["B-time"] for _ in range(9)] + [["B-rare"]]
    pred = [["B-time"] for _ in range(9)] + [["O"]]
    s = evaluate(gold, pred)
    check("micro F1 stays high", abs(s["f1"] - (2 * 9) / (2 * 9 + 0 + 1)) < 1e-9)
    # time: F1 = 1.0, rare: F1 = 0.0  -> macro = 0.5
    check("macro F1 exposes the tail", abs(s["macro_f1"] - 0.5) < 1e-9, str(s))
    check("macro is far below micro", s["macro_f1"] < s["f1"] - 0.3)
    check("class count reported", s["n_classes"] == 2)


def test_significance() -> None:
    print("significance testing")
    from significance import (
        bootstrap_ci,
        f1_from_counts,
        paired_permutation_test,
        sentence_counts,
    )

    # f1_from_counts must agree with the main scorer.
    counts = [(2, 1, 1), (3, 0, 2)]
    check(
        "corpus F1 from counts",
        abs(f1_from_counts(counts) - (2 * 5) / (2 * 5 + 1 + 3)) < 1e-9,
    )

    # Per-sentence counts must match the lecture example.
    tp, fp, fn = sentence_counts(
        ["B-PER", "O", "B-LOC", "I-LOC", "O"], ["B-PER", "O", "O", "B-LOC", "O"]
    )
    check("sentence counts match lecture example", (tp, fp, fn) == (1, 1, 1))

    # A bootstrap interval must bracket its own point estimate.
    sample = [(2, 0, 1)] * 40 + [(0, 2, 1)] * 10
    point, lo, hi = bootstrap_ci(sample, n_samples=300, seed=1)
    check("CI brackets the point estimate", lo <= point <= hi, f"{lo} {point} {hi}")
    check("CI has non-zero width", hi > lo)

    # Identical systems cannot be significantly different.
    a = [(2, 1, 1)] * 60
    _, _, p_same = paired_permutation_test(a, list(a), n_samples=500, seed=1)
    check("identical systems give p ~ 1.0", p_same > 0.9, f"p={p_same}")

    # A large consistent gap must come out significant.
    strong = [(5, 0, 0)] * 60
    weak = [(0, 5, 5)] * 60
    _, _, p_diff = paired_permutation_test(strong, weak, n_samples=500, seed=1)
    check("large gap is significant", p_diff < 0.05, f"p={p_diff}")

    # p-values are probabilities.
    check("p-value in (0, 1]", 0 < p_diff <= 1 and 0 < p_same <= 1)


def main() -> None:
    for fn in (
        test_span_decoding,
        test_lecture_example,
        test_perfect_and_empty,
        test_mismatch_raises,
        test_turkish_features,
        test_conll_roundtrip,
        test_models_learn,
        test_hmm_beats_baseline_on_context,
        test_massive_parser,
        test_macro_vs_micro,
        test_significance,
    ):
        fn()
    print(f"\n{PASSED} checks passed.")


if __name__ == "__main__":
    main()
