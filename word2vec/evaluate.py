import numpy as np

from word2vec.model import Word2Vec
from word2vec.vocabulary import Vocabulary


def most_similar(word: str, model: Word2Vec, vocab: Vocabulary, top_k: int = 10):
    if word not in vocab.word2idx:
        return []

    idx = vocab.word2idx[word]
    vec = model.W_center[idx]

    norms = np.linalg.norm(model.W_center, axis=1).clip(min=1e-10)
    sims = (model.W_center @ vec) / (norms * np.linalg.norm(vec))
    sims[idx] = -np.inf

    top = np.argsort(sims)[::-1][:top_k]
    return [(vocab.idx2word[i], float(sims[i])) for i in top]


def analogy(a: str, b: str, c: str, model: Word2Vec, vocab: Vocabulary, top_k: int = 5):
    for w in (a, b, c):
        if w not in vocab.word2idx:
            return []

    vec = (
        model.W_center[vocab.word2idx[b]]
        - model.W_center[vocab.word2idx[a]]
        + model.W_center[vocab.word2idx[c]]
    )

    norms = np.linalg.norm(model.W_center, axis=1).clip(min=1e-10)
    sims = (model.W_center @ vec) / (norms * np.linalg.norm(vec))

    for w in (a, b, c):
        sims[vocab.word2idx[w]] = -np.inf

    top = np.argsort(sims)[::-1][:top_k]
    return [(vocab.idx2word[i], float(sims[i])) for i in top]


def print_report(model: Word2Vec, vocab: Vocabulary) -> None:
    probe_words = ["king", "computer", "university", "good", "water"]

    print("\n  Nearest neighbours:")
    for word in probe_words:
        neighbours = most_similar(word, model, vocab, top_k=5)
        if not neighbours:
            continue
        nns = ", ".join(f"{w} ({s:.3f})" for w, s in neighbours)
        print(f"    {word:>12s} -> {nns}")

    analogy_tests = [
        ("king", "man", "queen"),
        ("paris", "france", "berlin"),
        ("big", "bigger", "small"),
    ]

    print("\n  Analogies (a - b + c = ?):")
    for a, b, c in analogy_tests:
        results = analogy(a, b, c, model, vocab, top_k=3)
        if results:
            res = ", ".join(f"{w} ({s:.3f})" for w, s in results)
            print(f"    {a} - {b} + {c} = {res}")
    print()