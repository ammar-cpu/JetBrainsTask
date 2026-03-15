import time

import numpy as np

from word2vec.config import Config
from word2vec.data import generate_skipgram_pairs
from word2vec.evaluate import print_report
from word2vec.model import Word2Vec
from word2vec.vocabulary import Vocabulary

BATCH_SIZE = 512


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
            pairs_arr = np.array(pairs)

            epoch_loss = 0.0
            n_pairs = len(pairs_arr)

            for start in range(0, n_pairs, BATCH_SIZE):
                end = min(start + BATCH_SIZE, n_pairs)
                batch = pairs_arr[start:end]
                B = len(batch)

                progress = global_step / max(total_steps, 1)
                lr = max(cfg.initial_lr * (1.0 - progress), cfg.min_lr)

                centers = batch[:, 0]
                contexts = batch[:, 1]
                negatives = np.column_stack([
                    self.vocab.sample_negatives(B, self.rng)
                    for _ in range(cfg.num_negatives)
                ])

                loss = self.model.train_batch(centers, contexts, negatives, lr)
                epoch_loss += loss
                global_step += B

                pairs_done = start + B
                if pairs_done % (cfg.log_every - cfg.log_every % BATCH_SIZE) < BATCH_SIZE:
                    avg = epoch_loss / pairs_done
                    print(f"  epoch {epoch} | {pairs_done:>10,}/{n_pairs:,} pairs | "
                          f"loss={avg:.4f} | lr={lr:.6f}")

            elapsed = time.time() - t0
            print(f"[epoch {epoch}/{cfg.epochs}]  "
                  f"loss={epoch_loss / n_pairs:.4f}  pairs={n_pairs:,}  time={elapsed:.1f}s")
            print_report(self.model, self.vocab)