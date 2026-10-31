"""Method 3 - linear-chain CRF over hand-engineered Turkish features.

Unlike the HMM, a CRF scores the whole label sequence *conditionally* on the
input, so it can use arbitrary overlapping features of the sentence (suffixes,
casing, neighbouring words) without having to model how those features were
generated. That is the reason it normally beats the HMM on NER.

Requires: pip install sklearn-crfsuite
"""

from __future__ import annotations

from conll import Sentence
from features import sentence_features

try:
    import sklearn_crfsuite

    HAVE_CRFSUITE = True
except ImportError:  # pragma: no cover
    HAVE_CRFSUITE = False


class CRFTagger:
    name = "crf"

    def __init__(
        self,
        c1: float = 0.1,
        c2: float = 0.1,
        max_iterations: int = 150,
        all_possible_transitions: bool = True,
    ):
        if not HAVE_CRFSUITE:
            raise ImportError(
                "sklearn-crfsuite is not installed.\n"
                "    pip install sklearn-crfsuite"
            )
        self.model = sklearn_crfsuite.CRF(
            algorithm="lbfgs",
            c1=c1,                 # L1 -> feature selection, keeps the model sparse
            c2=c2,                 # L2 -> smoothness
            max_iterations=max_iterations,
            all_possible_transitions=all_possible_transitions,
        )

    def fit(self, train: list[Sentence]) -> "CRFTagger":
        X = [sentence_features(tokens) for tokens, _ in train]
        y = [list(labels) for _, labels in train]
        self.model.fit(X, y)
        return self

    def predict(self, sentences: list[Sentence]) -> list[list[str]]:
        X = [sentence_features(tokens) for tokens, _ in sentences]
        return [list(seq) for seq in self.model.predict(X)]

    def predict_sentence(self, tokens: list[str]) -> list[str]:
        return list(self.model.predict([sentence_features(tokens)])[0])

    # -- interpretability: useful material for the presentation -------------
    def top_transitions(self, n: int = 15) -> list[tuple[str, str, float]]:
        items = sorted(
            self.model.transition_features_.items(),
            key=lambda kv: kv[1],
            reverse=True,
        )[:n]
        return [(a, b, w) for (a, b), w in items]

    def top_features(self, n: int = 20) -> list[tuple[str, str, float]]:
        items = sorted(
            self.model.state_features_.items(),
            key=lambda kv: kv[1],
            reverse=True,
        )[:n]
        return [(attr, label, w) for (attr, label), w in items]
