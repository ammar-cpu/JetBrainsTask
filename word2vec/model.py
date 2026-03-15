"""
Skip-Gram with Negative Sampling (SGNS)

Loss:  L = -log sig(u_o . v_c) - sum_k log sig(-u_k . v_c)

Gradients:
    dL/dv_c  = (sig(u_o . v_c) - 1) * u_o  +  sum_k sig(u_k . v_c) * u_k
    dL/du_o  = (sig(u_o . v_c) - 1) * v_c
    dL/du_k  = sig(u_k . v_c) * v_c
"""

import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
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
        self.W_context = rng.uniform(-scale, scale, (vocab_size, embedding_dim))

    def train_batch(self, centers: np.ndarray, contexts: np.ndarray,
                    negatives: np.ndarray, lr: float) -> float:
        B, K = negatives.shape

        v_c = self.W_center[centers]
        u_o = self.W_context[contexts]
        u_neg = self.W_context[negatives]

        scores_pos = np.sum(u_o * v_c, axis=1)
        scores_neg = np.einsum("bkd,bd->bk", u_neg, v_c)

        sig_pos = _sigmoid(scores_pos)
        sig_neg = _sigmoid(scores_neg)

        eps = 1e-7
        loss = -np.sum(np.log(sig_pos + eps)) - np.sum(np.log(1.0 - sig_neg + eps))

        coeff_pos = (sig_pos - 1.0)[:, None]
        coeff_neg = sig_neg[:, :, None]
        grad_vc = coeff_pos * u_o + (coeff_neg * u_neg).sum(axis=1)
        grad_uo = coeff_pos * v_c
        grad_uneg = coeff_neg * v_c[:, None, :]

        # np.add.at handles duplicate indices correctly (unlike -=)
        np.add.at(self.W_center, centers, -lr * grad_vc)
        np.add.at(self.W_context, contexts, -lr * grad_uo)
        np.add.at(self.W_context, negatives, -lr * grad_uneg)

        return float(loss)

    def train_step(self, center_idx: int, context_idx: int,
                   neg_indices: np.ndarray, lr: float) -> float:
        v_c = self.W_center[center_idx].copy()
        u_o = self.W_context[context_idx].copy()
        u_neg = self.W_context[neg_indices].copy()

        sig_pos = _sigmoid(u_o @ v_c)
        sig_neg = _sigmoid(u_neg @ v_c)

        eps = 1e-7
        loss = -np.log(sig_pos + eps) - np.sum(np.log(1.0 - sig_neg + eps))

        grad_vc = (sig_pos - 1.0) * u_o + (sig_neg[:, None] * u_neg).sum(axis=0)
        grad_uo = (sig_pos - 1.0) * v_c
        grad_uneg = sig_neg[:, None] * v_c[None, :]

        self.W_center[center_idx] -= lr * grad_vc
        self.W_context[context_idx] -= lr * grad_uo
        self.W_context[neg_indices] -= lr * grad_uneg

        return float(loss)

    def compute_loss(self, center_idx: int, context_idx: int,
                     neg_indices: np.ndarray) -> float:
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