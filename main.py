import argparse

import numpy as np

from word2vec import Config, Word2Vec, Vocabulary, Trainer
from word2vec.data import download_text8, load_tokens


def main():
    p = argparse.ArgumentParser(description="Train word2vec (SGNS) in pure NumPy")
    p.add_argument("--corpus", type=str, default=None)
    p.add_argument("--max-tokens", type=int, default=None)
    p.add_argument("--dim", type=int, default=100)
    p.add_argument("--window", type=int, default=5)
    p.add_argument("--neg", type=int, default=5)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=0.025)
    p.add_argument("--min-count", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=str, default="embeddings.txt")
    args = p.parse_args()

    config = Config(
        embedding_dim=args.dim,
        window_size=args.window,
        num_negatives=args.neg,
        epochs=args.epochs,
        initial_lr=args.lr,
        min_count=args.min_count,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )

    corpus_path = args.corpus or download_text8()
    raw_tokens = load_tokens(corpus_path, config.max_tokens)

    rng = np.random.default_rng(config.seed)
    vocab = Vocabulary(min_count=config.min_count, subsample_t=config.subsample_t)
    corpus = vocab.build(raw_tokens, rng)
    print(f"[vocab] {vocab.size:,} words | {len(corpus):,} tokens after filtering")

    model = Word2Vec(vocab.size, config.embedding_dim, seed=config.seed)

    trainer = Trainer(model, vocab, corpus, config)
    trainer.fit()

    model.save(args.output, vocab.idx2word)


if __name__ == "__main__":
    main()