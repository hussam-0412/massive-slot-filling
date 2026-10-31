"""Method 1 - most-frequent-tag baseline.

For each word, memorise the tag it carried most often in training and emit
that tag at test time. Unknown words fall back to "O".

This model has NO notion of context, which is exactly the point: it sets the
floor that every sequence model in this project has to beat, and it makes the
gain from context legible on the results slide.
"""

from __future__ import annotations

from collections import Counter, defaultdict

from conll import Sentence


class MostFrequentTagTagger:
    name = "baseline-most-frequent-tag"

    def __init__(self, lowercase_backoff: bool = True):
        self.lowercase_backoff = lowercase_backoff
        self.word2tag: dict[str, str] = {}
        self.lower2tag: dict[str, str] = {}
        self.default_tag = "O"

    def fit(self, train: list[Sentence]) -> "MostFrequentTagTagger":
        counts: dict[str, Counter] = defaultdict(Counter)
        lower_counts: dict[str, Counter] = defaultdict(Counter)
        tag_counts: Counter = Counter()

        for tokens, labels in train:
            for tok, lab in zip(tokens, labels):
                counts[tok][lab] += 1
                lower_counts[tok.lower()][lab] += 1
                tag_counts[lab] += 1

        self.word2tag = {w: c.most_common(1)[0][0] for w, c in counts.items()}
        self.lower2tag = {
            w: c.most_common(1)[0][0] for w, c in lower_counts.items()
        }
        self.default_tag = tag_counts.most_common(1)[0][0] if tag_counts else "O"
        return self

    def predict_sentence(self, tokens: list[str]) -> list[str]:
        out = []
        for tok in tokens:
            if tok in self.word2tag:
                out.append(self.word2tag[tok])
            elif self.lowercase_backoff and tok.lower() in self.lower2tag:
                out.append(self.lower2tag[tok.lower()])
            else:
                out.append("O")
        return out

    def predict(self, sentences: list[Sentence]) -> list[list[str]]:
        return [self.predict_sentence(tokens) for tokens, _ in sentences]
