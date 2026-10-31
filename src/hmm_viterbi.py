"""Method 2 - Hidden Markov Model decoded with Viterbi.

The HMM assigns a joint probability to the label sequence and the observed
words:

    P(y, x) = P(y_1 | START) * prod_t P(x_t | y_t) * P(y_t | y_{t-1})

Training is pure counting with add-k smoothing. Decoding is Viterbi in log
space, which finds the single highest-scoring label sequence in O(T * |Y|^2).

Unknown-word handling matters a great deal in Turkish: the language is
agglutinative, so the test set is full of surface forms never seen in
training (Ankara / Ankara'dan / Ankara'daki ...). Instead of giving every
unknown word a uniform emission, we back off to a *suffix signature*
estimated from rare training words, which keeps morphological evidence alive.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict

from conll import Sentence

NEG_INF = float("-inf")
START = "<START>"


def _shape(token: str) -> str:
    """Coarse orthographic signature used for unknown words."""
    if token.isdigit():
        return "<NUM>"
    if any(ch.isdigit() for ch in token):
        return "<ALNUM>"
    if token.isupper() and len(token) > 1:
        return "<ALLCAPS>"
    if "'" in token or "’" in token:
        # Turkish writes a proper noun's case suffix after an apostrophe:
        # Ankara'dan, Türkiye'nin. A very strong entity signal.
        return "<APOS_TITLE>" if token[:1].isupper() else "<APOS>"
    if token[:1].isupper():
        return "<TITLE>"
    if not any(ch.isalnum() for ch in token):
        return "<PUNCT>"
    return "<LOWER>"


def _suffix_key(token: str, n: int = 3) -> str:
    return f"{_shape(token)}|{token[-n:].lower()}"


class HMMTagger:
    name = "hmm-viterbi"

    def __init__(self, alpha: float = 0.1, rare_threshold: int = 2, suffix_n: int = 3):
        self.alpha = alpha            # add-k smoothing constant
        self.rare_threshold = rare_threshold
        self.suffix_n = suffix_n

        self.tags: list[str] = []
        self.tag_index: dict[str, int] = {}
        self.log_trans: list[list[float]] = []
        self.log_start: list[float] = []
        self.known_vocab: set[str] = set()
        self.log_emit: dict[str, dict[str, float]] = {}
        self.log_emit_default: dict[str, float] = {}
        self.log_unk: dict[str, dict[str, float]] = {}
        self.log_unk_default: dict[str, float] = {}

    # -- training -----------------------------------------------------------
    def fit(self, train: list[Sentence]) -> "HMMTagger":
        tag_counts: Counter = Counter()
        start_counts: Counter = Counter()
        trans_counts: dict[str, Counter] = defaultdict(Counter)
        emit_counts: dict[str, Counter] = defaultdict(Counter)
        word_counts: Counter = Counter()

        for tokens, labels in train:
            for tok in tokens:
                word_counts[tok] += 1

        for tokens, labels in train:
            if not labels:
                continue
            start_counts[labels[0]] += 1
            for i, (tok, lab) in enumerate(zip(tokens, labels)):
                tag_counts[lab] += 1
                emit_counts[lab][tok] += 1
                if i > 0:
                    trans_counts[labels[i - 1]][lab] += 1

        self.tags = sorted(tag_counts)
        self.tag_index = {t: i for i, t in enumerate(self.tags)}
        n_tags = len(self.tags)
        a = self.alpha

        # Start distribution
        total_start = sum(start_counts.values())
        self.log_start = [
            math.log((start_counts[t] + a) / (total_start + a * n_tags))
            for t in self.tags
        ]

        # Transitions
        self.log_trans = []
        for prev in self.tags:
            row_total = sum(trans_counts[prev].values())
            self.log_trans.append(
                [
                    math.log((trans_counts[prev][nxt] + a) / (row_total + a * n_tags))
                    for nxt in self.tags
                ]
            )

        # A word is "known" if it is frequent enough that its own emission
        # counts are worth trusting; everything else is routed to the
        # unknown-word model below.
        self.known_vocab = {
            w for w, c in word_counts.items() if c > self.rare_threshold
        }

        # Counts split into the known and unknown halves, per tag.
        known_counts: dict[str, Counter] = defaultdict(Counter)
        unk_counts: dict[str, Counter] = defaultdict(Counter)
        known_total: Counter = Counter()
        unk_total: Counter = Counter()

        for tokens, labels in train:
            for tok, lab in zip(tokens, labels):
                if tok in self.known_vocab:
                    known_counts[lab][tok] += 1
                    known_total[lab] += 1
                else:
                    unk_counts[lab][_suffix_key(tok, self.suffix_n)] += 1
                    unk_total[lab] += 1

        v_known = max(len(self.known_vocab), 1)
        n_keys = max(len({k for c in unk_counts.values() for k in c}), 1)

        # P(w | tag) is a two-part mixture so that the two halves stay on a
        # comparable scale:
        #
        #   known word   : (1 - p_unk(tag)) * P_known(w | tag)
        #   unknown word :      p_unk(tag)  * P_sig(signature(w) | tag)
        #
        # Without this split the suffix-signature distribution (a few hundred
        # outcomes) dominates the per-word distribution (tens of thousands of
        # outcomes) and the decoder prefers the backoff over real evidence.
        self.log_emit = {}
        self.log_emit_default = {}
        self.log_unk = {}
        self.log_unk_default = {}

        for tag in self.tags:
            total = max(tag_counts[tag], 1)
            # Reserve mass for unknown words, with a floor so they are never
            # impossible and a ceiling so known words are never impossible.
            p_unk = min(max(unk_total[tag] / total, 1e-5), 0.5)
            log_known_mass = math.log(1.0 - p_unk)
            log_unk_mass = math.log(p_unk)

            kt = known_total[tag]
            denom_known = kt + a * v_known
            self.log_emit[tag] = {
                w: log_known_mass + math.log((c + a) / denom_known)
                for w, c in known_counts[tag].items()
            }
            # Known word never observed with this tag.
            self.log_emit_default[tag] = log_known_mass + math.log(a / denom_known)

            ut = unk_total[tag]
            denom_unk = ut + a * n_keys
            self.log_unk[tag] = {
                key: log_unk_mass + math.log((c + a) / denom_unk)
                for key, c in unk_counts[tag].items()
            }
            self.log_unk_default[tag] = log_unk_mass + math.log(a / denom_unk)
        return self

    # -- emission lookup ----------------------------------------------------
    def _emission(self, token: str, tag: str) -> float:
        if token in self.known_vocab:
            table = self.log_emit.get(tag, {})
            if token in table:
                return table[token]
            return self.log_emit_default.get(tag, -25.0)
        key = _suffix_key(token, self.suffix_n)
        return self.log_unk.get(tag, {}).get(key, self.log_unk_default.get(tag, -25.0))

    # -- decoding -----------------------------------------------------------
    def predict_sentence(self, tokens: list[str]) -> list[str]:
        if not tokens:
            return []
        n_tags = len(self.tags)
        if n_tags == 0:
            return ["O"] * len(tokens)

        # viterbi[t][j] = best log score of a path ending in tag j at time t
        viterbi = [[NEG_INF] * n_tags for _ in tokens]
        backptr = [[0] * n_tags for _ in tokens]

        for j, tag in enumerate(self.tags):
            viterbi[0][j] = self.log_start[j] + self._emission(tokens[0], tag)

        for t in range(1, len(tokens)):
            token = tokens[t]
            for j, tag in enumerate(self.tags):
                emit = self._emission(token, tag)
                best_score, best_i = NEG_INF, 0
                for i in range(n_tags):
                    prev = viterbi[t - 1][i]
                    if prev == NEG_INF:
                        continue
                    score = prev + self.log_trans[i][j]
                    if score > best_score:
                        best_score, best_i = score, i
                viterbi[t][j] = best_score + emit
                backptr[t][j] = best_i

        # Backtrace from the best final state
        last = max(range(n_tags), key=lambda j: viterbi[-1][j])
        path = [last]
        for t in range(len(tokens) - 1, 0, -1):
            last = backptr[t][last]
            path.append(last)
        path.reverse()
        return [self.tags[j] for j in path]

    def predict(self, sentences: list[Sentence]) -> list[list[str]]:
        return [self.predict_sentence(tokens) for tokens, _ in sentences]
