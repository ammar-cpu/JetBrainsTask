import time

import numpy as np

from word2vec.config import Config
from word2vec.data import generate_skipgram_pairs
from word2vec.evaluate import print_report
from word2vec.model import Word2Vec
from word2vec.vocabulary import Vocabulary


class Trainer:

    def __init__(self, model: Word2Vec, vocab: Vocabulary, corpus: list[int], config: Config):
        self.model = model
        self.vocab = vocab
        self.corpus = corpus
        self.config = config
        self.rng = np.random.default_rng(config.seed)

    def fit(self) -> None:
        cfg = self.config

        est_pairs_per_epoch = len(self.corpus) * cfg.window_size
        total_steps = est_pairs_per_epoch * cfg.epochs
        global_step = 0

        for epoch in range(1, cfg.epochs + 1):
            t0 = time.time()

            pairs = generate_skipgram_pairs(self.corpus, cfg.window_size, self.rng)
            self.rng.shuffle(pairs)

            epoch_loss = 0.0
            n_pairs = len(pairs)

            for i, (center, context) in enumerate(pairs):
                progress = global_step / max(total_steps, 1)
                lr = max(cfg.initial_lr * (1.0 - progress), cfg.min_lr)

                neg = self.vocab.sample_negatives(cfg.num_negatives, self.rng)
                loss = self.model.train_step(center, context, neg, lr)
                epoch_loss += loss
                global_step += 1

                if (i + 1) % cfg.log_every == 0:
                    avg = epoch_loss / (i + 1)
                    print(f"  epoch {epoch} | {i+1:>10,}/{n_pairs:,} pairs | "
                          f"loss={avg:.4f} | lr={lr:.6f}")

            elapsed = time.time() - t0
            print(f"[epoch {epoch}/{cfg.epochs}]  "
                  f"loss={epoch_loss / n_pairs:.4f}  pairs={n_pairs:,}  time={elapsed:.1f}s")
            print_report(self.model, self.vocab)