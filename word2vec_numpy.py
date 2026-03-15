"""
Word2Vec: Skip-Gram with Negative Sampling
Implemented in pure NumPy (no PyTorch / TensorFlow / ML frameworks).

Dataset: Text8 excerpt (first 1M characters of Wikipedia dump),
         downloaded on the fly via urllib.
"""

import numpy as np
import urllib.request
import re
from collections import Counter

# ── Reproducibility ──────────────────────────────────────────────────────────
np.random.seed(42)


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA LOADING & PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def load_text8(max_chars: int = 1_000_000) -> list[str]:
    """Download and return the first `max_chars` characters of text8 as tokens."""
    url = "http://mattmahoney.net/dc/text8.zip"
    print("Downloading text8 dataset...")
    import zipfile, io
    with urllib.request.urlopen(url) as response:
        data = response.read()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        text = z.read("text8").decode("utf-8")[:max_chars]
    tokens = text.strip().split()
    print(f"Loaded {len(tokens):,} tokens.")
    return tokens


def build_vocab(tokens: list[str], min_count: int = 5):
    """
    Build vocabulary from token list.
    Words appearing fewer than min_count times are discarded.

    Returns
    -------
    word2idx : dict[str, int]
    idx2word : list[str]
    vocab_size : int
    """
    counts = Counter(tokens)
    vocab = [w for w, c in counts.items() if c >= min_count]
    vocab.sort()  # deterministic ordering
    word2idx = {w: i for i, w in enumerate(vocab)}
    idx2word = vocab
    print(f"Vocabulary size: {len(vocab):,} (min_count={min_count})")
    return word2idx, idx2word, len(vocab)


def subsample_tokens(tokens: list[str], word2idx: dict, counts: Counter,
                     t: float = 1e-4) -> list[int]:
    """
    Subsample frequent words using Mikolov et al.'s formula:
        P(discard) = 1 - sqrt(t / f(w))
    where f(w) is the relative frequency of word w.

    Returns a list of token indices (unknown words already removed).
    """
    total = sum(counts[w] for w in word2idx)
    freq = {w: counts[w] / total for w in word2idx}
    keep_prob = {w: min(1.0, (np.sqrt(freq[w] / t) + 1) * (t / freq[w]))
                 for w in word2idx}
    result = []
    for w in tokens:
        if w not in word2idx:
            continue
        if np.random.rand() < keep_prob[w]:
            result.append(word2idx[w])
    print(f"After subsampling: {len(result):,} tokens remain.")
    return result


def build_noise_distribution(counts: Counter, word2idx: dict,
                              power: float = 0.75) -> np.ndarray:
    """
    Unigram distribution raised to the 3/4 power, as in the original paper.
    Used for drawing negative samples.
    """
    freq = np.zeros(len(word2idx))
    for w, idx in word2idx.items():
        freq[idx] = counts[w] ** power
    freq /= freq.sum()
    return freq


# ─────────────────────────────────────────────────────────────────────────────
# 2. TRAINING DATA GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def generate_skipgram_pairs(token_ids: list[int],
                             window_size: int = 5) -> list[tuple[int, int]]:
    """
    Slide a window over the corpus and yield (center, context) pairs.
    Window size is sampled uniformly in [1, window_size] per center word,
    following the original word2vec implementation.
    """
    pairs = []
    n = len(token_ids)
    for i, center in enumerate(token_ids):
        w = np.random.randint(1, window_size + 1)
        start = max(0, i - w)
        end = min(n, i + w + 1)
        for j in range(start, end):
            if j != i:
                pairs.append((center, token_ids[j]))
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# 3. MODEL: EMBEDDINGS
# ─────────────────────────────────────────────────────────────────────────────

class Word2Vec:
    """
    Skip-Gram Word2Vec with Negative Sampling.

    Parameters
    ----------
    vocab_size  : number of unique words
    embed_dim   : dimensionality of word vectors
    noise_dist  : pre-computed unigram noise distribution
    n_negatives : number of negative samples per positive pair
    lr          : initial learning rate (linearly decayed during training)
    """

    def __init__(self, vocab_size: int, embed_dim: int,
                 noise_dist: np.ndarray, n_negatives: int = 5,
                 lr: float = 0.025):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.noise_dist = noise_dist
        self.n_negatives = n_negatives
        self.lr = lr

        # W_in  : "input"  embeddings  — one row per word (center vectors)
        # W_out : "output" embeddings  — one row per word (context vectors)
        # Initialise uniformly in [-0.5/d, 0.5/d] as in the original C code.
        scale = 0.5 / embed_dim
        self.W_in  = np.random.uniform(-scale, scale, (vocab_size, embed_dim))
        self.W_out = np.zeros((vocab_size, embed_dim))

    # ── Forward pass + loss ──────────────────────────────────────────────────

    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        # Numerically stable sigmoid
        return np.where(x >= 0,
                        1 / (1 + np.exp(-x)),
                        np.exp(x) / (1 + np.exp(x)))

    def negative_sampling_loss(self, center: int, context: int
                                ) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the Negative Sampling objective for one (center, context) pair.

        Objective (to maximise):
            log σ(v_c · v̂_o)  +  Σ_{k=1}^{K} log σ(−v_c · v̂_k)

        where v_c  = W_in[center]
              v̂_o = W_out[context]
              v̂_k = W_out[neg_k]

        Returns
        -------
        loss        : scalar loss (negated, so we minimise)
        grad_center : gradient w.r.t. W_in[center]
        grad_ctx    : gradient w.r.t. W_out[context]
        grad_neg    : gradient w.r.t. W_out[neg_samples]  shape (K, embed_dim)
        """
        # Draw K negative samples, avoiding the true context word
        neg_samples = np.random.choice(self.vocab_size,
                                       size=self.n_negatives,
                                       p=self.noise_dist)

        v_c   = self.W_in[center]          # (d,)
        v_ctx = self.W_out[context]        # (d,)
        V_neg = self.W_out[neg_samples]    # (K, d)

        # Scores
        score_pos = np.dot(v_c, v_ctx)                    # scalar
        scores_neg = V_neg @ v_c                          # (K,)

        # Sigmoid activations
        sig_pos = self.sigmoid(score_pos)                 # scalar  → target 1
        sig_neg = self.sigmoid(scores_neg)                # (K,)    → target 0

        # Loss  (we return the value for logging; optimisation uses gradients)
        loss = -(np.log(sig_pos + 1e-9) +
                 np.sum(np.log(1 - sig_neg + 1e-9)))

        # ── Gradients (derived from the objective above) ──────────────────
        #
        # ∂L/∂v_c   = (σ(v_c·v̂_o) − 1)·v̂_o  +  Σ_k σ(v_c·v̂_k)·v̂_k
        # ∂L/∂v̂_o  = (σ(v_c·v̂_o) − 1)·v_c
        # ∂L/∂v̂_k  =  σ(v_c·v̂_k)·v_c   for each k

        err_pos = sig_pos - 1.0                           # scalar
        err_neg = sig_neg                                 # (K,)

        grad_ctx    = err_pos * v_c                       # (d,)
        grad_neg    = np.outer(err_neg, v_c)              # (K, d)
        grad_center = err_pos * v_ctx + V_neg.T @ err_neg # (d,)

        return loss, grad_center, grad_ctx, grad_neg, neg_samples

    # ── Parameter update (SGD) ───────────────────────────────────────────────

    def update(self, center: int, context: int) -> float:
        """One SGD step for a single (center, context) pair."""
        loss, grad_c, grad_ctx, grad_neg, neg_samples = \
            self.negative_sampling_loss(center, context)

        self.W_in[center]   -= self.lr * grad_c
        self.W_out[context] -= self.lr * grad_ctx
        for k, neg_idx in enumerate(neg_samples):
            self.W_out[neg_idx] -= self.lr * grad_neg[k]

        return loss


# ─────────────────────────────────────────────────────────────────────────────
# 4. TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────

def train(model: Word2Vec, pairs: list[tuple[int, int]],
          n_epochs: int = 1, log_every: int = 100_000,
          initial_lr: float = 0.025, min_lr: float = 0.0001):
    """
    Standard SGD training loop with linear learning-rate decay.

    Learning rate is decayed linearly from initial_lr to min_lr over the
    total number of training steps, following the original word2vec schedule.
    """
    total_steps = n_epochs * len(pairs)
    step = 0
    total_loss = 0.0

    for epoch in range(1, n_epochs + 1):
        # Shuffle pairs each epoch
        idx = np.random.permutation(len(pairs))
        shuffled = [pairs[i] for i in idx]

        for center, context in shuffled:
            # Linear LR decay
            progress = step / total_steps
            model.lr = max(min_lr, initial_lr * (1 - progress))

            loss = model.update(center, context)
            total_loss += loss
            step += 1

            if step % log_every == 0:
                avg_loss = total_loss / log_every
                print(f"  Epoch {epoch} | Step {step:,}/{total_steps:,} "
                      f"| Avg loss: {avg_loss:.4f} | LR: {model.lr:.6f}")
                total_loss = 0.0

    print("Training complete.")


# ─────────────────────────────────────────────────────────────────────────────
# 5. EVALUATION: COSINE SIMILARITY
# ─────────────────────────────────────────────────────────────────────────────

def most_similar(word: str, word2idx: dict, idx2word: list,
                 W: np.ndarray, top_n: int = 10) -> list[tuple[str, float]]:
    """
    Return the top_n most similar words by cosine similarity
    using the input (W_in) embedding matrix.
    """
    if word not in word2idx:
        return []
    idx = word2idx[word]
    vec = W[idx]
    vec_norm = vec / (np.linalg.norm(vec) + 1e-9)

    # Normalise all vectors
    norms = np.linalg.norm(W, axis=1, keepdims=True) + 1e-9
    W_norm = W / norms

    sims = W_norm @ vec_norm
    sims[idx] = -1  # exclude the query word itself
    top_indices = np.argsort(sims)[::-1][:top_n]
    return [(idx2word[i], float(sims[i])) for i in top_indices]


# ─────────────────────────────────────────────────────────────────────────────
# 6. MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # Hyperparameters
    EMBED_DIM   = 100
    WINDOW_SIZE = 5
    N_NEGATIVES = 5
    MIN_COUNT   = 5
    SUBSAMPLE_T = 1e-4
    N_EPOCHS    = 1
    INITIAL_LR  = 0.025
    MAX_CHARS   = 1_000_000   # increase for better embeddings

    # 1. Load data
    tokens = load_text8(max_chars=MAX_CHARS)
    counts = Counter(tokens)

    # 2. Build vocabulary
    word2idx, idx2word, vocab_size = build_vocab(tokens, min_count=MIN_COUNT)

    # 3. Subsample frequent words
    token_ids = subsample_tokens(tokens, word2idx, counts, t=SUBSAMPLE_T)

    # 4. Generate skip-gram pairs
    print("Generating skip-gram pairs...")
    pairs = generate_skipgram_pairs(token_ids, window_size=WINDOW_SIZE)
    print(f"Total training pairs: {len(pairs):,}")

    # 5. Build noise distribution for negative sampling
    noise_dist = build_noise_distribution(counts, word2idx)

    # 6. Initialise model
    model = Word2Vec(vocab_size=vocab_size, embed_dim=EMBED_DIM,
                     noise_dist=noise_dist, n_negatives=N_NEGATIVES,
                     lr=INITIAL_LR)

    # 7. Train
    print(f"\nTraining Word2Vec (Skip-Gram + Negative Sampling)")
    print(f"  vocab={vocab_size:,}  dim={EMBED_DIM}  "
          f"window={WINDOW_SIZE}  negatives={N_NEGATIVES}  epochs={N_EPOCHS}\n")
    train(model, pairs, n_epochs=N_EPOCHS,
          initial_lr=INITIAL_LR, log_every=100_000)

    # 8. Quick similarity check
    print("\nSimilarity checks (using W_in embeddings):")
    for query in ["king", "computer", "france", "science"]:
        results = most_similar(query, word2idx, idx2word, model.W_in, top_n=5)
        if results:
            neighbours = ", ".join(f"{w} ({s:.3f})" for w, s in results)
            print(f"  {query:12s} → {neighbours}")
        else:
            print(f"  '{query}' not in vocabulary.")
