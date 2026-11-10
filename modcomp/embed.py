"""Lossy embedding compression: sectors + correction vectors, judged by search.

For embeddings, exact reconstruction does not matter. What matters is whether
nearest-neighbour search still finds the same neighbours. Every method here
is scored by

* bits per vector (fixed-width codes, and after lzma on the codes),
* mean cosine(x, x_hat) between each vector and its reconstruction,
* recall@10: the fraction of each query's true top-10 (by dot product) that
  appears in the top-10 over the reconstructed vectors. Queries are not
  compressed, which is the usual "asymmetric" setup.

Methods, from the original idea to its learned generalization:

``grid01-k``      the original scheme: map components to [0,1], cut each
                  axis into 10**k sectors, store the sector (lossy).
``scalar-b``      the same axis-aligned sectors, but fit to each dimension's
                  actual range, with 2**b sectors per axis.
``kmeans-C+res-b`` learned sectors: the nearest of C centroids (log2 C bits)
                  plus a correction vector (residual), quantized at b bits per
                  dim over the residual's range. The "strip a modulus, store a
                  correction" idea in a learned basis.
``pq-M``          product quantization: M sub-vectors, each snapped to the
                  nearest of 256 learned sub-centroids (8 bits each).
``kmeans-C+pq-M`` learned sector + PQ-coded correction (IVF-ADC style).
``binary``        1 bit per dim: the sign around the mean.
``float16``       reference.
"""

from __future__ import annotations

import lzma

import numpy as np


# --------------------------------------------------------------------------
# k-means (numpy Lloyd)
# --------------------------------------------------------------------------

def _sqdist_argmin(X, C, batch=4096):
    cn = (C * C).sum(1)
    out = np.empty(len(X), dtype=np.int64)
    for i in range(0, len(X), batch):
        xb = X[i:i + batch]
        out[i:i + batch] = np.argmin(cn[None, :] - 2.0 * xb @ C.T, axis=1)
    return out


def kmeans(X, k, iters=20, seed=0):
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=np.float32)
    k = min(k, len(X))
    C = X[rng.choice(len(X), k, replace=False)].copy()
    for _ in range(iters):
        a = _sqdist_argmin(X, C)
        sums = np.zeros_like(C)
        np.add.at(sums, a, X)
        cnt = np.bincount(a, minlength=k)
        empty = cnt == 0
        C[~empty] = sums[~empty] / cnt[~empty, None]
        if empty.any():  # reseed empty clusters
            C[empty] = X[rng.choice(len(X), int(empty.sum()), replace=False)]
    return C


# --------------------------------------------------------------------------
# Methods. Each has fit(X), encode(X) -> int codes, decode(codes) -> X_hat,
# bits (fixed bits per vector) and model_bytes (codebooks etc.).
# --------------------------------------------------------------------------

class Float16:
    name = "float16"

    def fit(self, X):
        self.d = X.shape[1]
        return self

    def encode(self, X):
        return X.astype(np.float16).view(np.uint16).astype(np.int64)

    def decode(self, codes):
        return codes.astype(np.uint16).view(np.float16).astype(np.float32)

    @property
    def bits(self):
        return 16 * self.d

    model_bytes = 0


class _AxisSectors:
    """Per-dimension uniform sectors over [lo, hi]; decode to sector centres."""

    def _setup(self, lo, hi, levels):
        self.lo = lo.astype(np.float32)
        self.step = ((hi - lo) / levels).astype(np.float32)
        self.step[self.step == 0] = 1e-12
        self.levels = levels

    def _q(self, X):
        return np.clip(np.floor((X - self.lo) / self.step), 0, self.levels - 1).astype(np.int64)

    def _dq(self, codes):
        return self.lo + (codes + 0.5) * self.step


class Grid01(_AxisSectors):
    """The original scheme: [-1,1] -> [0,1], 10**k fixed sectors per axis."""

    def __init__(self, digits):
        self.digits = digits
        self.name = f"grid01 (s=1e-{digits})"

    def fit(self, X):
        self.d = X.shape[1]
        K = 10 ** self.digits
        # Components of unit-norm embeddings lie in [-1, 1]; x01 = (x + 1) / 2.
        self._setup(np.full(self.d, -1.0), np.full(self.d, 1.0), K)
        return self

    def encode(self, X):
        return self._q(X)

    def decode(self, codes):
        return self._dq(codes)

    @property
    def bits(self):
        return self.d * int(10 ** self.digits - 1).bit_length()

    model_bytes = 0


class Scalar(_AxisSectors):
    """Axis sectors fit to each dimension's range (clipped quantiles)."""

    def __init__(self, b, clip=0.001):
        self.b, self.clip = b, clip
        self.name = f"scalar-{b}"

    def fit(self, X):
        self.d = X.shape[1]
        lo = np.quantile(X, self.clip, axis=0)
        hi = np.quantile(X, 1 - self.clip, axis=0)
        self._setup(lo, hi, 1 << self.b)
        return self

    def encode(self, X):
        return self._q(X)

    def decode(self, codes):
        return self._dq(codes)

    @property
    def bits(self):
        return self.d * self.b

    @property
    def model_bytes(self):
        return 8 * self.d


class PQ:
    def __init__(self, m, ksub=256, iters=15):
        self.m, self.ksub, self.iters = m, ksub, iters
        self.name = f"pq-{m}"

    def fit(self, X):
        d = X.shape[1]
        if d % self.m:
            raise ValueError(f"dimension {d} not divisible by m={self.m}")
        self.d, self.ds = d, d // self.m
        self.books = [kmeans(X[:, j * self.ds:(j + 1) * self.ds], self.ksub,
                             self.iters, seed=j) for j in range(self.m)]
        return self

    def encode(self, X):
        return np.stack([_sqdist_argmin(X[:, j * self.ds:(j + 1) * self.ds], B)
                         for j, B in enumerate(self.books)], axis=1)

    def decode(self, codes):
        return np.concatenate([B[codes[:, j]] for j, B in enumerate(self.books)], axis=1)

    @property
    def bits(self):
        return self.m * int(self.ksub - 1).bit_length()

    @property
    def model_bytes(self):
        return 4 * self.ksub * self.d


class CoarsePlus:
    """Learned sector (nearest centroid) + correction coded by ``inner``."""

    def __init__(self, n_centroids, inner, iters=20):
        self.k, self.inner, self.iters = n_centroids, inner, iters
        self.name = f"kmeans-{n_centroids}+{inner.name.replace('scalar-', 'res-')}"

    def fit(self, X):
        self.C = kmeans(X, self.k, self.iters)
        a = _sqdist_argmin(X, self.C)
        self.inner.fit(X - self.C[a])
        self.d = X.shape[1]
        return self

    def encode(self, X):
        a = _sqdist_argmin(X, self.C)
        return np.concatenate([a[:, None], self.inner.encode(X - self.C[a])], axis=1)

    def decode(self, codes):
        return self.C[codes[:, 0]] + self.inner.decode(codes[:, 1:])

    @property
    def bits(self):
        return int(self.k - 1).bit_length() + self.inner.bits

    @property
    def model_bytes(self):
        return 4 * self.k * self.d + self.inner.model_bytes


class Binary:
    name = "binary"

    def fit(self, X):
        self.d = X.shape[1]
        self.mu = X.mean(0)
        self.scale = np.abs(X - self.mu).mean(0)
        return self

    def encode(self, X):
        return (X > self.mu).astype(np.int64)

    def decode(self, codes):
        return self.mu + (2 * codes - 1) * self.scale

    @property
    def bits(self):
        return self.d

    @property
    def model_bytes(self):
        return 8 * self.d


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------

def topk(db, queries, k):
    scores = queries @ db.T
    idx = np.argpartition(-scores, k, axis=1)[:, :k]
    return idx


def recall_at_k(db, db_hat, queries, k=10):
    truth = topk(db, queries, k)
    found = topk(db_hat, queries, k)
    hits = sum(len(np.intersect1d(t, f)) for t, f in zip(truth, found))
    return hits / (k * len(queries))


def mean_cosine(X, X_hat):
    num = (X * X_hat).sum(1)
    den = np.linalg.norm(X, axis=1) * np.linalg.norm(X_hat, axis=1) + 1e-12
    return float((num / den).mean())


def lzma_bits_per_vector(codes, bits_per_code):
    """Measured size of the codes after lzma (codes stored as small ints)."""
    dt = np.uint8 if bits_per_code <= 8 else (np.uint16 if bits_per_code <= 16 else np.uint32)
    return 8 * len(lzma.compress(codes.astype(dt).tobytes())) / len(codes)


def evaluate(method, db, queries, k=10):
    method.fit(db)
    codes = method.encode(db)
    db_hat = method.decode(codes).astype(np.float32)
    per_code = max(1, int(codes.max()).bit_length()) if codes.size else 1
    return {
        "name": method.name,
        "bits": method.bits,
        "lzma_bits": lzma_bits_per_vector(codes, per_code),
        "ratio": 32 * db.shape[1] / method.bits,
        "cosine": mean_cosine(db, db_hat),
        "recall": recall_at_k(db, db_hat, queries, k),
        "model_kib": method.model_bytes / 1024,
    }


# --------------------------------------------------------------------------
# Synthetic embedding-like data
# --------------------------------------------------------------------------

def synthetic_embeddings(n=20_000, n_queries=500, d=384, topics=100, seed=0):
    """Unit-norm vectors that mimic common properties of text embeddings.

    * an anisotropic spectrum (power-law eigenvalues in a random basis),
    * topic clusters,
    * a shared non-zero mean direction (embeddings sit in a narrow cone),
    * a few high-magnitude "outlier" dimensions.

    Real embeddings should replace this (pass ``--input``).
    """
    rng = np.random.default_rng(seed)
    U, _ = np.linalg.qr(rng.standard_normal((d, d)))
    lam = 1.0 / np.arange(1, d + 1) ** 0.8

    def draw(m, scale):
        return (rng.standard_normal((m, d)) * np.sqrt(lam) * scale) @ U.T

    mean = rng.standard_normal(d)
    mean *= 2.0 / np.linalg.norm(mean)
    centres = draw(topics, 1.0)
    total = n + n_queries
    X = mean + centres[rng.integers(0, topics, total)] + draw(total, 0.6)
    X[:, :4] *= 4.0  # outlier dims
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    X = X.astype(np.float32)
    return X[:n], X[n:]
