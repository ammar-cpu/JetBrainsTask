from collections import Counter

import numpy as np


class Vocabulary:

    def __init__(self, min_count: int = 5, subsample_t: float = 1e-5) -> None:
        self.min_count = min_count
        self.subsample_t = subsample_t

        self.word2idx: dict[str, int] = {}
        self.idx2word: list[str] = []
        self.counts: np.ndarray = np.array([])
        self._noise_dist: np.ndarray = np.array([])

    @property
    def size(self) -> int:
        return len(self.idx2word)

    def build(self, tokens: list[str], rng: np.random.Generator) -> list[int]:
        counter = Counter(tokens)
        kept = {w: c for w, c in counter.items() if c >= self.min_count}

        self.idx2word = sorted(kept.keys())
        self.word2idx = {w: i for i, w in enumerate(self.idx2word)}
        self.counts = np.array([kept[w] for w in self.idx2word], dtype=np.float64)

        total = self.counts.sum()
        freqs = self.counts / total

        # P_keep(w) = sqrt(t / f(w)) — drops frequent words like "the"
        keep_prob = np.minimum(np.sqrt(self.subsample_t / freqs), 1.0)

        # noise distribution for negative sampling: f(w)^0.75 / Z
        powered = self.counts ** 0.75
        self._noise_dist = powered / powered.sum()

        corpus: list[int] = []
        for token in tokens:
            idx = self.word2idx.get(token)
            if idx is not None and rng.random() < keep_prob[idx]:
                corpus.append(idx)

        return corpus

    def sample_negatives(self, count: int, rng: np.random.Generator) -> np.ndarray:
        return rng.choice(self.size, size=count, p=self._noise_dist)