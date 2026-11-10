"""Encode one embedding as a short, URL-safe query-string value.

Output alphabet is base64url (``A-Z a-z 0-9 - _``): unreserved in URLs, so no
percent-escaping, 6 bits per character, no padding. Bits are written directly
into characters, so a code of B bits takes exactly ceil(B / 6) characters.

Codecs (all decode to a unit-norm vector):

``RotScalar(d, bits)``
    No training. Rotate by a fixed seeded orthogonal matrix, then quantize
    every coordinate with a uniform quantizer tuned for Gaussian data. The
    rotation spreads energy evenly across coordinates (outlier dims stop
    dominating), so one step size fits all of them. Shared state: a seed.

``SectorRot(k, bits)``
    Sector + correction. Sector = nearest of ``k`` k-means centroids learned
    from a corpus (log2 k bits). Correction = residual, rotated and quantized
    as above, plus a float16 scale. Shared state: the centroid table.

``Progressive(k, [m1, m2, ...])``
    Geohash-like. Sector, then successive corrections, each a product-quantized
    residual of the previous stage. Any prefix that ends on a stage boundary
    still decodes, just more coarsely. Shared state: all codebooks.
"""

from __future__ import annotations

import numpy as np

from .embed import PQ, _sqdist_argmin, kmeans

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
_INDEX = {c: i for i, c in enumerate(ALPHABET)}


# Optimal uniform (midrise) quantizer step for a unit Gaussian, by bits (Max 1960).
# Fixed, arbitrary seed for the shared rotation (both sides must agree on it).
ROTATION_SEED = 0x51C0DE

GAUSS_STEP = {1: 1.596, 2: 0.9957, 3: 0.5860, 4: 0.3352, 5: 0.1881,
              6: 0.1041, 7: 0.0569, 8: 0.0308}


# --------------------------------------------------------------------------
# Bits <-> characters
# --------------------------------------------------------------------------

class BitWriter:
    def __init__(self):
        self.acc = 0
        self.n = 0

    def write(self, value: int, width: int) -> None:
        if width:
            self.acc |= (int(value) & ((1 << width) - 1)) << self.n
            self.n += width

    def write_array(self, values, width: int) -> None:
        for v in np.asarray(values).ravel():
            self.write(int(v), width)

    def to_str(self) -> str:
        return "".join(ALPHABET[(self.acc >> (6 * i)) & 63]
                       for i in range((self.n + 5) // 6))


class BitReader:
    def __init__(self, s: str):
        try:
            self.acc = sum(_INDEX[c] << (6 * i) for i, c in enumerate(s))
        except KeyError as e:
            raise ValueError(f"invalid character {e.args[0]!r}") from None
        self.n = 6 * len(s)
        self.pos = 0

    def remaining(self) -> int:
        return self.n - self.pos

    def read(self, width: int) -> int:
        if self.pos + width > self.n:
            raise ValueError("string too short")
        v = (self.acc >> self.pos) & ((1 << width) - 1)
        self.pos += width
        return v

    def read_array(self, count: int, width: int) -> np.ndarray:
        return np.array([self.read(width) for _ in range(count)], dtype=np.int64)


def _f16_bits(x: float) -> int:
    return int(np.array(x, dtype=np.float16).view(np.uint16))


def _f16_value(bits: int) -> float:
    return float(np.array(bits, dtype=np.uint16).view(np.float16))


def _unit(x):
    return x / (np.linalg.norm(x) + 1e-12)


def chars_for_bits(bits: int) -> int:
    return (bits + 5) // 6


# --------------------------------------------------------------------------
# Rotated scalar quantization
# --------------------------------------------------------------------------

def seeded_rotation(d: int, seed: int) -> np.ndarray:
    """Deterministic orthogonal matrix (QR of a seeded Gaussian, sign-fixed)."""
    A = np.random.default_rng(seed).standard_normal((d, d))
    Q, R = np.linalg.qr(A)
    return (Q * np.sign(np.diag(R))).astype(np.float64)


class _RotQuant:
    """Rotate, then b-bit uniform quantization with step tuned to sigma."""

    def __init__(self, d: int, bits: int, seed: int = ROTATION_SEED, rotate: bool = True):
        if bits not in GAUSS_STEP:
            raise ValueError(f"bits must be in {sorted(GAUSS_STEP)}")
        self.d, self.b = d, bits
        self.Q = seeded_rotation(d, seed) if rotate else np.eye(d)
        self.L = 1 << bits

    def quantize(self, x, sigma):
        y = np.asarray(x, dtype=np.float64) @ self.Q
        step = GAUSS_STEP[self.b] * sigma
        return np.clip(np.floor(y / step), -self.L // 2, self.L // 2 - 1).astype(np.int64) + self.L // 2

    def dequantize(self, q, sigma):
        step = GAUSS_STEP[self.b] * sigma
        return ((q - self.L // 2 + 0.5) * step) @ self.Q.T


class RotScalar:
    """Unit vectors: after rotation every coordinate has sigma = 1/sqrt(d),
    so no scale needs to be stored."""

    def __init__(self, d: int, bits: int, seed: int = ROTATION_SEED, rotate: bool = True):
        self.rq = _RotQuant(d, bits, seed, rotate)
        self.sigma = 1.0 / np.sqrt(d)
        self.name = f"{'rotated' if rotate else 'unrotated'} {bits}-bit"
        self.bits = d * bits
        self.model_bytes = 8 if rotate else 0  # just a seed

    def fit(self, X):
        return self

    def encode(self, x) -> str:
        w = BitWriter()
        w.write_array(self.rq.quantize(_unit(x), self.sigma), self.rq.b)
        return w.to_str()

    def decode(self, s: str) -> np.ndarray:
        r = BitReader(s)
        return _unit(self.rq.dequantize(r.read_array(self.rq.d, self.rq.b), self.sigma))


class SectorRot:
    """Sector (nearest learned centroid) + rotated, quantized correction."""

    def __init__(self, d: int, k: int, bits: int, seed: int = ROTATION_SEED):
        self.k, self.kbits = k, int(k - 1).bit_length()
        self.rq = _RotQuant(d, bits, seed)
        self.name = f"sector {k} + {bits}-bit correction"
        self.bits = self.kbits + 16 + d * bits

    def fit(self, X):
        self.C = kmeans(X, self.k, iters=15).astype(np.float64)
        self.model_bytes = 4 * self.C.size
        return self

    def encode(self, x) -> str:
        x = _unit(np.asarray(x, dtype=np.float64))
        a = int(_sqdist_argmin(x[None], self.C)[0])
        res = x - self.C[a]
        sigma = np.float16(np.sqrt((res ** 2).mean()) + 1e-12)
        w = BitWriter()
        w.write(a, self.kbits)
        w.write(_f16_bits(sigma), 16)
        w.write_array(self.rq.quantize(res, float(sigma)), self.rq.b)
        return w.to_str()

    def decode(self, s: str) -> np.ndarray:
        r = BitReader(s)
        a = r.read(self.kbits)
        sigma = _f16_value(r.read(16))
        q = r.read_array(self.rq.d, self.rq.b)
        return _unit(self.C[a] + self.rq.dequantize(q, sigma))


class Progressive:
    """Sector, then a chain of PQ-coded corrections. Prefixes decode."""

    def __init__(self, k: int, stages, ksub: int = 256):
        self.k, self.kbits = k, int(k - 1).bit_length()
        self.stage_ms = list(stages)
        self.ksub = ksub
        self.name = f"progressive {k} + pq{'+'.join(map(str, stages))}"
        self.stage_bits = [self.kbits] + [m * int(ksub - 1).bit_length() for m in stages]
        self.bits = sum(self.stage_bits)

    def fit(self, X):
        X = np.asarray(X, dtype=np.float32)
        self.C = kmeans(X, self.k, iters=15)
        R = X - self.C[_sqdist_argmin(X, self.C)]
        self.pqs = []
        for m in self.stage_ms:
            pq = PQ(m, self.ksub).fit(R)
            R = R - pq.decode(pq.encode(R))
            self.pqs.append(pq)
        self.model_bytes = 4 * (self.C.size + sum(pq.ksub * pq.d for pq in self.pqs))
        return self

    def encode(self, x) -> str:
        x = _unit(np.asarray(x, dtype=np.float32))[None]
        w = BitWriter()
        a = _sqdist_argmin(x, self.C)
        w.write(int(a[0]), self.kbits)
        r = x - self.C[a]
        for pq in self.pqs:
            codes = pq.encode(r)
            w.write_array(codes, int(pq.ksub - 1).bit_length())
            r = r - pq.decode(codes)
        return w.to_str()

    def prefix_chars(self):
        """String lengths at which each stage is complete."""
        return [chars_for_bits(int(b)) for b in np.cumsum(self.stage_bits)]

    def decode(self, s: str) -> np.ndarray:
        r = BitReader(s)
        a = r.read(self.kbits)
        x = self.C[a].astype(np.float64)
        for pq in self.pqs:
            w = int(pq.ksub - 1).bit_length()
            if r.remaining() < pq.m * w:
                break  # truncated string: stop at the last complete stage
            x = x + pq.decode(r.read_array(pq.m, w)[None])[0]
        return _unit(x)
