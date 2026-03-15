"""
Verifies analytical gradients against numerical (finite-difference) gradients.
For each parameter theta_i: dL/dtheta_i ~ [L(theta_i + h) - L(theta_i - h)] / 2h
Relative errors below 1e-5 mean the gradients are correct.
"""

import numpy as np
from word2vec.model import Word2Vec, _sigmoid


def numerical_grad_row(model, param_matrix, row_idx, center, context, negs, h=1e-5):
    D = param_matrix.shape[1]
    grad = np.zeros(D)
    for d in range(D):
        old = param_matrix[row_idx, d]

        param_matrix[row_idx, d] = old + h
        loss_plus = model.compute_loss(center, context, negs)

        param_matrix[row_idx, d] = old - h
        loss_minus = model.compute_loss(center, context, negs)

        param_matrix[row_idx, d] = old
        grad[d] = (loss_plus - loss_minus) / (2 * h)
    return grad


def rel_error(a, n):
    return float(np.max(np.abs(a - n) / np.maximum(np.abs(a) + np.abs(n), 1e-8)))


def main():
    np.random.seed(42)

    model = Word2Vec(50, 10, seed=42)
    center, context = 3, 12
    negs = np.array([0, 7, 22, 35, 41])

    # analytical gradients
    v_c = model.W_center[center].copy()
    u_o = model.W_context[context].copy()
    u_neg = model.W_context[negs].copy()

    sig_pos = _sigmoid(u_o @ v_c)
    sig_neg = _sigmoid(u_neg @ v_c)

    a_vc = (sig_pos - 1.0) * u_o + (sig_neg[:, None] * u_neg).sum(axis=0)
    a_uo = (sig_pos - 1.0) * v_c
    a_uneg = sig_neg[:, None] * v_c[None, :]

    # numerical gradients
    n_vc = numerical_grad_row(model, model.W_center, center, center, context, negs)
    n_uo = numerical_grad_row(model, model.W_context, context, center, context, negs)
    n_uneg = np.zeros_like(a_uneg)
    for k, neg_idx in enumerate(negs):
        n_uneg[k] = numerical_grad_row(model, model.W_context, neg_idx, center, context, negs)

    err_vc = rel_error(a_vc, n_vc)
    err_uo = rel_error(a_uo, n_uo)
    err_uneg = rel_error(a_uneg, n_uneg)

    print("Gradient check")
    print(f"  dL/dv_c   error: {err_vc:.2e}")
    print(f"  dL/du_o   error: {err_uo:.2e}")
    print(f"  dL/du_neg error: {err_uneg:.2e}")

    threshold = 1e-5
    if all(e < threshold for e in (err_vc, err_uo, err_uneg)):
        print(f"PASSED (all < {threshold:.0e})")
    else:
        print("FAILED")
        exit(1)


if __name__ == "__main__":
    main()