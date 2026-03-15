"""
Skip-Gram with Negative Sampling (SGNS)

Loss for one (center, context+) pair with K negative samples:

    L = -log sig(u_o . v_c) - sum_k log sig(-u_k . v_c)

Gradients:
    dL/dv_c  = (sig(u_o . v_c) - 1) * u_o  +  sum_k sig(u_k . v_c) * u_k
    dL/du_o  = (sig(u_o . v_c) - 1) * v_c
    dL/du_k  = sig(u_k . v_c) * v_c
"""

import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    # split into two cases to avoid overflow in exp()
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )


class Word2Vec:

    def __init__(self, vocab_size: int, embedding_dim: int, seed: int = 42) -> None:
        self.V = vocab_size
        self.D = embedding_dim
        rng = np.random.default_rng(seed)

        scale = 0.5 / embedding_dim
        self.W_center = rng.uniform(-scale, scale, (vocab_size, embedding_dim))
        self.W_context = np.zeros((vocab_size, embedding_dim))

    def train_step(
        self, center_idx: int, context_idx: int, neg_indices: np.ndarray, lr: float,
    ) -> float:
        # .copy() because numpy indexing with a scalar returns a view,
        # and we don't want the SGD updates below to corrupt values
        # we still need for other gradients
        v_c = self.W_center[center_idx].copy()
        u_o = self.W_context[context_idx].copy()
        u_neg = self.W_context[neg_indices].copy()

        # forward
        sig_pos = _sigmoid(u_o @ v_c)
        sig_neg = _sigmoid(u_neg @ v_c)

        # loss
        eps = 1e-7
        loss = -np.log(sig_pos + eps) - np.sum(np.log(1.0 - sig_neg + eps))

        # gradients (see module docstring for derivation)
        grad_vc = (sig_pos - 1.0) * u_o + (sig_neg[:, None] * u_neg).sum(axis=0)
        grad_uo = (sig_pos - 1.0) * v_c
        grad_uneg = sig_neg[:, None] * v_c[None, :]

        # SGD
        self.W_center[center_idx] -= lr * grad_vc
        self.W_context[context_idx] -= lr * grad_uo
        self.W_context[neg_indices] -= lr * grad_uneg

        return float(loss)

    def compute_loss(
        self, center_idx: int, context_idx: int, neg_indices: np.ndarray,
    ) -> float:
        """Forward pass only, no update. Used by gradient_check.py."""
        v_c = self.W_center[center_idx]
        u_o = self.W_context[context_idx]
        u_neg = self.W_context[neg_indices]

        eps = 1e-7
        sig_pos = _sigmoid(u_o @ v_c)
        sig_neg = _sigmoid(u_neg @ v_c)
        return float(-np.log(sig_pos + eps) - np.sum(np.log(1.0 - sig_neg + eps)))

    def get_embedding(self, idx: int) -> np.ndarray:
        return self.W_center[idx]

    def save(self, path: str, idx2word: list[str]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"{self.V} {self.D}\n")
            for i, word in enumerate(idx2word):
                vec = " ".join(f"{x:.6f}" for x in self.W_center[i])
                f.write(f"{word} {vec}\n")
        print(f"[model] Saved to {path}")