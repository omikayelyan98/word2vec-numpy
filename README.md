# Word2Vec from Scratch in NumPy

A clean, fully documented implementation of **Word2Vec (Skip-Gram with Negative Sampling)** in pure NumPy — no PyTorch, no TensorFlow, no ML frameworks of any kind.

---

## What This Is

This project implements the core training loop of Word2Vec as described in the original paper by Mikolov et al. (2013). Every component — the forward pass, the loss function, the gradient derivations, and the parameter updates — is written from scratch using only NumPy.

The goal is not just a working implementation, but a fully understandable one. Every design decision follows the original paper or the original C implementation, and the code is commented to explain the reasoning behind each step.

---

## How It Works

### The Big Picture

Word2Vec learns word embeddings by training a shallow neural network on a self-supervised task: given a center word, predict the words that appear around it in a sliding window (skip-gram). Words that appear in similar contexts end up with similar vector representations.

### Skip-Gram with Negative Sampling (SGNS)

Instead of computing a full softmax over the entire vocabulary (which would be extremely slow), Negative Sampling turns the problem into a binary classification task:

- For each **(center, context)** pair, train the model to output a high score
- For each **negative sample** (a randomly drawn word that is *not* the context), train the model to output a low score

The objective to maximise for one training pair is:

```
log σ(v_c · v̂_o)  +  Σ_{k=1}^{K} log σ(−v_c · v̂_k)
```

Where:
- `v_c` = embedding of the center word (from `W_in`)
- `v̂_o` = embedding of the true context word (from `W_out`)
- `v̂_k` = embedding of the k-th negative sample (from `W_out`)
- `σ` = sigmoid function

### Gradient Derivations

Taking the gradients of the loss with respect to each parameter:

```
∂L/∂v_c  = (σ(v_c · v̂_o) − 1) · v̂_o  +  Σ_k σ(v_c · v̂_k) · v̂_k
∂L/∂v̂_o = (σ(v_c · v̂_o) − 1) · v_c
∂L/∂v̂_k =  σ(v_c · v̂_k) · v_c   for each k
```

Parameters are updated with standard SGD after each training pair.

---

## Implementation Details

### Two Embedding Matrices

The model maintains two separate matrices:
- `W_in` — center word embeddings (used as the final word vectors)
- `W_out` — context word embeddings (used during training, discarded afterward)

This separation follows the original implementation. Center and context words play structurally different roles in the objective, so keeping them separate improves training stability.

### Subsampling Frequent Words

Very frequent words like "the" or "is" appear so often that they contribute little additional signal. Following Mikolov et al., each word is discarded during corpus preparation with probability:

```
P(discard) = 1 - sqrt(t / f(w))
```

Where `f(w)` is the word's relative frequency and `t` is a threshold (default `1e-4`). This speeds up training and improves the quality of embeddings for content words.

### Noise Distribution for Negative Sampling

Negative samples are drawn from a unigram distribution raised to the 3/4 power:

```
P(w) ∝ count(w)^0.75
```

This smooths the distribution relative to raw frequencies, giving rare words a higher chance of being sampled as negatives than they would have otherwise.

### Learning Rate Schedule

The learning rate decays linearly from `initial_lr` to `min_lr` over the total number of training steps, following the original word2vec C implementation.

---

## Project Structure

```
word2vec_numpy.py   — full implementation (data loading, model, training, evaluation)
README.md           — this file
```

---

## Getting Started

### Requirements

```
python >= 3.10
numpy
```

No other dependencies.

### Run

```bash
python word2vec_numpy.py
```

On first run, the script downloads the **text8** dataset (~100MB, first 1M characters used by default) from `mattmahoney.net`. Training one epoch on 1M characters takes a few minutes on a standard laptop.

### Expected Output

```
Downloading text8 dataset...
Loaded 173,829 tokens.
Vocabulary size: 3,721 (min_count=5)
After subsampling: 123,045 tokens remain.
Generating skip-gram pairs...
Total training pairs: 1,024,312

Training Word2Vec (Skip-Gram + Negative Sampling)
  vocab=3721  dim=100  window=5  negatives=5  epochs=1

  Epoch 1 | Step 100,000/1,024,312 | Avg loss: 3.1245 | LR: 0.022556
  ...
Training complete.

Similarity checks (using W_in embeddings):
  king         → prince (0.712), lord (0.698), emperor (0.681), ...
  computer     → software (0.743), system (0.731), digital (0.704), ...
```

---

## Hyperparameters

| Parameter | Default | Description |
|---|---|---|
| `EMBED_DIM` | 100 | Dimensionality of word vectors |
| `WINDOW_SIZE` | 5 | Max context window radius |
| `N_NEGATIVES` | 5 | Negative samples per positive pair |
| `MIN_COUNT` | 5 | Minimum word frequency for vocabulary |
| `SUBSAMPLE_T` | 1e-4 | Subsampling threshold |
| `N_EPOCHS` | 1 | Training epochs |
| `INITIAL_LR` | 0.025 | Starting learning rate |
| `MAX_CHARS` | 1,000,000 | Characters to use from text8 |

Increasing `MAX_CHARS` to 10M+ and `N_EPOCHS` to 3-5 will produce noticeably better embeddings.

---

## References

- Mikolov et al. (2013) — [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781)
- Mikolov et al. (2013) — [Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546)
- Original C implementation: [word2vec on Google Code](https://code.google.com/archive/p/word2vec/)
