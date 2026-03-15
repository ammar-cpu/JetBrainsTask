# word2vec-numpy

Word2Vec (skip-gram + negative sampling) from scratch using only NumPy.

Trained on the text8 dataset (~17M tokens of cleaned Wikipedia).

## Setup


python3 -m venv venv
source venv/bin/activate
pip install -e .


## Running


python3 main.py --max-tokens 1000000 --epochs 3     # quick test
python3 main.py                                       # full training
python3 main.py --corpus my_text.txt --dim 200        # custom corpus
python3 gradient_check.py                              # verify gradients


## Gradient derivation

Loss for one (center, context) pair with K negative samples:

```
L = -log sig(u_o . v_c) - sum_k log sig(-u_k . v_c)
```

Gradients:


dL/dv_c  = (sig(u_o . v_c) - 1) * u_o  +  sum_k sig(u_k . v_c) * u_k
dL/du_o  = (sig(u_o . v_c) - 1) * v_c
dL/du_k  = sig(u_k . v_c) * v_c


These come from applying the chain rule through the sigmoid, using `1 - sig(-x) = sig(x)`. The positive term pulls center and context vectors closer, the negative terms push them apart. Verified numerically in `gradient_check.py` (all errors < 1e-7).

## References

- Mikolov et al., "Efficient Estimation of Word Representations in Vector Space" (2013)
- Mikolov et al., "Distributed Representations of Words and Phrases and their Compositionality" (2013)
- Rong, "word2vec Parameter Learning Explained" (2014)
