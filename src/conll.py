"""CoNLL-format reading/writing for token classification.

A sentence is a pair (tokens, labels) of equal-length lists of str.
The reader is deliberately tolerant: it accepts tab- or space-separated
columns, ignores -DOCSTART- lines, and takes the FIRST column as the token
and the LAST column as the label (so 2-column and 4-column CoNLL both work).
"""

from __future__ import annotations

import os
import random
from typing import Iterator

Sentence = tuple[list[str], list[str]]


def read_conll(path: str) -> list[Sentence]:
    """Read a CoNLL file into a list of (tokens, labels)."""
    sentences: list[Sentence] = []
    tokens: list[str] = []
    labels: list[str] = []

    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.rstrip("\n").rstrip("\r")
            if not line.strip():
                if tokens:
                    sentences.append((tokens, labels))
                    tokens, labels = [], []
                continue
            if line.startswith("-DOCSTART-"):
                continue

            parts = line.split("\t") if "\t" in line else line.split()
            if len(parts) < 2:
                # A token with no label: treat as outside.
                tokens.append(parts[0])
                labels.append("O")
                continue
            tokens.append(parts[0])
            labels.append(parts[-1])

    if tokens:
        sentences.append((tokens, labels))
    return sentences


def write_conll(path: str, sentences: list[Sentence]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for tokens, labels in sentences:
            for tok, lab in zip(tokens, labels):
                fh.write(f"{tok}\t{lab}\n")
            fh.write("\n")


def label_set(sentences: list[Sentence]) -> list[str]:
    """Sorted label inventory, with O first so it gets index 0."""
    labels = {lab for _, labs in sentences for lab in labs}
    labels.discard("O")
    return ["O"] + sorted(labels)


def entity_types(sentences: list[Sentence]) -> list[str]:
    types = set()
    for _, labs in sentences:
        for lab in labs:
            if lab != "O" and "-" in lab:
                types.add(lab.split("-", 1)[1])
    return sorted(types)


def split_sentences(
    sentences: list[Sentence],
    dev_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
) -> tuple[list[Sentence], list[Sentence], list[Sentence]]:
    """Deterministic split, used ONLY when the corpus ships without splits.

    The split assignment is fixed by `seed` and must stay identical across
    every method compared, otherwise the comparison is meaningless.
    """
    shuffled = list(sentences)
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    n_test = int(n * test_frac)
    n_dev = int(n * dev_frac)
    test = shuffled[:n_test]
    dev = shuffled[n_test : n_test + n_dev]
    train = shuffled[n_test + n_dev :]
    return train, dev, test


def describe(name: str, sentences: list[Sentence]) -> str:
    n_tok = sum(len(t) for t, _ in sentences)
    n_ent = sum(
        1 for _, labs in sentences for lab in labs if lab.startswith("B-")
    )
    return (
        f"{name:<6} sentences={len(sentences):>6}  tokens={n_tok:>8}  "
        f"entities={n_ent:>7}"
    )


def iter_tokens(sentences: list[Sentence]) -> Iterator[tuple[str, str]]:
    for tokens, labels in sentences:
        yield from zip(tokens, labels)
