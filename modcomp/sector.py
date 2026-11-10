"""Sector + correction compression for vectors whose components lie in [0, 1].

Setup
-----
* Every vector has ``D`` components, each a magnitude in ``[0, 1]`` stored as
  an ``N``-bit unsigned integer ``v = round(x * (2**N - 1))``.
* A precision ``s = 1 / K`` (e.g. ``K = 10**k`` for ``s = 0.0…1``) cuts each
  axis into ``K`` sectors of width ``s``, so the unit cube becomes a grid of
  ``K**D`` sectors.
* Each vector is stored as the **sector** it falls in (the coarse part, a
  D-vector of ints in ``[0, K)``) plus a **correction vector** (its exact
  position inside that sector, in N-bit units).

Everything is done in integer arithmetic, so reconstruction of the N-bit
values is exact. With ``Q = 2**N - 1`` the sector boundaries are

    b(j) = ceil(j * Q / K)      sector(v) = min(floor(v * K / Q), K - 1)

and ``correction = v - b(sector(v))``. The value 1.0 (``v == Q``) belongs to the
last sector, so the range is inclusive at both ends.

Storage modes
-------------
``ordered``   per-vector sector stream (codec-compressed) + corrections.
              Decodes in the original order.
``grouped``   vectors are sorted by sector; each *occupied* sector is stored
              once with a count, then all corrections. Decodes in sector order
              (the array is treated as a set).
``grouped+perm``  grouped, plus the permutation needed to restore the
              original order.
``sectors``   lossy: sector stream only; decodes to sector centres
              (error <= s/2 per component).
"""

from __future__ import annotations

import numpy as np

from .core import (
    CODECS,
    _CODEC_NAMES,
    _Reader,
    _compress,
    _decompress,
    _put_varint,
    bitpack,
    bitunpack,
    varint_decode,
    varint_encode,
)

MAGIC = b"MSEC"
VERSION = 1
MODES = {"ordered": 0, "grouped": 1, "grouped+perm": 2, "sectors": 3}
_MODE_NAMES = {v: k for k, v in MODES.items()}
_FLAG_CORR_CODEC = 1  # corrections stored as codec(bytes) instead of bit-packed


# --------------------------------------------------------------------------
# Fixed-point conversion
# --------------------------------------------------------------------------

def to_fixed(X, bits: int) -> np.ndarray:
    """Floats in [0, 1] -> N-bit unsigned integers."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[:, None]
    if X.size and (X.min() < 0 or X.max() > 1):
        raise ValueError("components must lie in [0, 1]")
    return np.rint(X * ((1 << bits) - 1)).astype(np.int64)


def from_fixed(V, bits: int) -> np.ndarray:
    return np.asarray(V, dtype=np.float64) / ((1 << bits) - 1)


# --------------------------------------------------------------------------
# The grid
# --------------------------------------------------------------------------

class SectorGrid:
    """K sectors per axis over N-bit integers in [0, 2**N - 1]."""

    def __init__(self, bits: int, sectors: int):
        if not 1 <= bits <= 32:
            raise ValueError("bits must be in 1..32")
        if sectors < 1:
            raise ValueError("sectors must be >= 1")
        self.bits = bits
        self.K = int(sectors)
        self.Q = (1 << bits) - 1
        if self.K > self.Q + 1:
            raise ValueError("more sectors than representable values")
        if (self.Q * self.K).bit_length() > 62:
            raise ValueError("bits + log2(sectors) too large for int64 arithmetic")
        j = np.arange(self.K + 1, dtype=np.int64)
        self.bounds = (j * self.Q + self.K - 1) // self.K  # b(0..K); b(K) == Q
        widths = np.diff(self.bounds)
        widths[-1] += 1  # last sector also holds Q itself (x == 1.0)
        self.widths = widths  # number of N-bit values in each sector
        self.max_correction = int(widths.max()) - 1
        self.correction_bits = self.max_correction.bit_length()
        self.sector_bits = (self.K - 1).bit_length()

    @classmethod
    def decimal(cls, bits: int, digits: int) -> "SectorGrid":
        """Grid with precision 10**-digits (e.g. digits=2 -> 0.01 sectors)."""
        return cls(bits, 10 ** digits)

    @property
    def precision(self) -> float:
        return 1.0 / self.K

    def check(self, V) -> np.ndarray:
        V = np.asarray(V)
        if V.ndim == 1:
            V = V[:, None]
        if V.ndim != 2 or not np.issubdtype(V.dtype, np.integer):
            raise TypeError("expected an (n, D) integer array; use to_fixed() for floats")
        if V.size and (V.min() < 0 or V.max() > self.Q):
            raise ValueError(f"values must lie in [0, {self.Q}]")
        return V.astype(np.int64, copy=False)

    def split(self, V):
        """Return ``(S, C)``: sector indices and correction vectors."""
        V = self.check(V)
        S = np.minimum(V * self.K // self.Q, self.K - 1)
        C = V - self.bounds[S]
        return S, C

    def merge(self, S, C) -> np.ndarray:
        return self.bounds[np.asarray(S)] + np.asarray(C, dtype=np.int64)

    def centres(self, S) -> np.ndarray:
        """Lossy reconstruction: the middle of each sector, in N-bit units."""
        S = np.asarray(S)
        return self.bounds[S] + (self.widths[S] - 1) // 2


# --------------------------------------------------------------------------
# Byte helpers
# --------------------------------------------------------------------------

def _uint_dtype(max_value: int):
    for dt in (np.uint8, np.uint16, np.uint32, np.uint64):
        if max_value <= np.iinfo(dt).max:
            return dt
    raise ValueError("value too large")


def _put_block(out: bytearray, data: bytes) -> None:
    _put_varint(out, len(data))
    out += data


def _grouped_order(S: np.ndarray) -> np.ndarray:
    """Stable lexicographic order of the sector rows."""
    if S.shape[0] == 0:
        return np.zeros(0, dtype=np.int64)
    return np.lexsort(S.T[::-1])


def _encode_table(U: np.ndarray, K: int, codec: str) -> bytes:
    """Sorted unique sector rows: delta the first column, store row-major."""
    T = U.copy()
    if len(T):
        T[:, 0] = np.diff(T[:, 0], prepend=0)
    return _compress(T.astype(_uint_dtype(K - 1)).tobytes(), codec)


def _decode_table(buf: bytes, u: int, d: int, K: int, codec: str) -> np.ndarray:
    raw = _decompress(buf, codec)
    T = np.frombuffer(raw, dtype=_uint_dtype(K - 1)).astype(np.int64).reshape(u, d)
    if u:
        T[:, 0] = np.cumsum(T[:, 0])
    return T


def _encode_corrections(C: np.ndarray, grid: SectorGrid, codec: str):
    """Pick the smaller of bit-packed or codec(bytes). Returns (flag, data)."""
    widths = np.full(C.shape[1], grid.correction_bits, dtype=np.int64)
    packed = bitpack(C, widths)
    if codec == "none":
        return 0, packed
    coded = _compress(C.astype(_uint_dtype(grid.max_correction)).tobytes(), codec)
    if len(coded) < len(packed):
        return _FLAG_CORR_CODEC, coded
    return 0, packed


def _decode_corrections(buf: bytes, n: int, d: int, grid: SectorGrid,
                        codec: str, flags: int) -> np.ndarray:
    if flags & _FLAG_CORR_CODEC:
        raw = _decompress(buf, codec)
        dt = _uint_dtype(grid.max_correction)
        return np.frombuffer(raw, dtype=dt).astype(np.int64).reshape(n, d)
    widths = np.full(d, grid.correction_bits, dtype=np.int64)
    return bitunpack(buf, n, widths)


# --------------------------------------------------------------------------
# Container
# --------------------------------------------------------------------------

def encode(V, grid: SectorGrid, mode: str = "ordered", codec: str = "lzma"):
    """Encode N-bit vectors. Returns ``(blob, sizes)``.

    ``sizes`` has ``header``, ``sectors`` (sector stream or sector table +
    counts), ``corrections``, ``permutation`` and ``total`` byte counts.
    """
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; choose from {sorted(MODES)}")
    if codec not in CODECS:
        raise ValueError(f"unknown codec {codec!r}; choose from {sorted(CODECS)}")
    V = grid.check(V)
    n, d = V.shape
    S, C = grid.split(V)

    sizes = {"sectors": 0, "corrections": 0, "permutation": 0}
    body = bytearray()
    flags = 0
    extra = bytearray()

    if mode in ("ordered", "sectors"):
        sec = _compress(S.astype(_uint_dtype(grid.K - 1)).tobytes(), codec)
        _put_block(body, sec)
        sizes["sectors"] = len(sec)
        if mode == "ordered":
            flags, corr = _encode_corrections(C, grid, codec)
            _put_block(body, corr)
            sizes["corrections"] = len(corr)
    else:
        order = _grouped_order(S)
        Ss, Cs = S[order], C[order]
        if n:
            new = np.ones(n, dtype=bool)
            new[1:] = np.any(Ss[1:] != Ss[:-1], axis=1)
            starts = np.flatnonzero(new)
            U = Ss[starts]
            counts = np.diff(np.append(starts, n))
        else:
            U = np.zeros((0, d), dtype=np.int64)
            counts = np.zeros(0, dtype=np.int64)
        _put_varint(extra, len(U))
        table = _encode_table(U, grid.K, codec)
        cnt = _compress(varint_encode(counts.astype(np.uint64)), codec)
        _put_block(body, table)
        _put_block(body, cnt)
        sizes["sectors"] = len(table) + len(cnt)
        flags, corr = _encode_corrections(Cs, grid, codec)
        _put_block(body, corr)
        sizes["corrections"] = len(corr)
        if mode == "grouped+perm":
            w = np.array([max(n - 1, 0).bit_length()], dtype=np.int64)
            perm = bitpack(order[:, None], w)
            _put_block(body, perm)
            sizes["permutation"] = len(perm)

    head = bytearray(MAGIC)
    head += bytes([VERSION, MODES[mode], CODECS[codec], flags])
    for x in (n, d, grid.bits, grid.K):
        _put_varint(head, x)
    blob = bytes(head + extra + body)
    sizes["total"] = len(blob)
    sizes["header"] = len(blob) - sizes["sectors"] - sizes["corrections"] - sizes["permutation"]
    return blob, sizes


def decode(blob: bytes) -> np.ndarray:
    """Decode to N-bit integers.

    ``ordered`` and ``grouped+perm`` give the original order; ``grouped`` gives
    sector order; ``sectors`` gives sector centres (lossy).
    """
    r = _Reader(blob)
    if r.take(4) != MAGIC:
        raise ValueError("not a sector blob")
    version, mode_id, codec_id, flags = r.take(4)
    if version != VERSION or mode_id not in _MODE_NAMES or codec_id not in _CODEC_NAMES:
        raise ValueError("unsupported sector blob")
    mode, codec = _MODE_NAMES[mode_id], _CODEC_NAMES[codec_id]
    n, d, bits, K = (r.varint() for _ in range(4))
    grid = SectorGrid(bits, K)
    sdt = _uint_dtype(K - 1)

    if mode in ("ordered", "sectors"):
        raw = _decompress(r.take(r.varint()), codec)
        S = np.frombuffer(raw, dtype=sdt).astype(np.int64).reshape(n, d)
        if mode == "sectors":
            out = grid.centres(S)
        else:
            C = _decode_corrections(r.take(r.varint()), n, d, grid, codec, flags)
            out = grid.merge(S, C)
    else:
        u = r.varint()
        U = _decode_table(r.take(r.varint()), u, d, K, codec)
        counts = varint_decode(_decompress(r.take(r.varint()), codec), u).astype(np.int64)
        if counts.sum() != n:
            raise ValueError("sector counts do not add up")
        S = np.repeat(U, counts, axis=0)
        C = _decode_corrections(r.take(r.varint()), n, d, grid, codec, flags)
        out = grid.merge(S, C)
        if mode == "grouped+perm":
            w = np.array([max(n - 1, 0).bit_length()], dtype=np.int64)
            order = bitunpack(r.take(r.varint()), n, w)[:, 0]
            restored = np.empty_like(out)
            restored[order] = out
            out = restored
    if r.pos != len(blob):
        raise ValueError("trailing bytes")
    return out


# --------------------------------------------------------------------------
# Estimates
# --------------------------------------------------------------------------

def stats(V, grid: SectorGrid) -> dict:
    """Bit-level accounting for one grid (per vector, idealized)."""
    from .analysis import entropy_bits

    V = grid.check(V)
    n, d = V.shape
    S, C = grid.split(V)
    if n:
        _, cell_counts = np.unique(S, axis=0, return_counts=True)
    else:
        cell_counts = np.zeros(0, dtype=np.int64)
    p = cell_counts / max(n, 1)
    cell_entropy = float(-(cell_counts * np.log2(p)).sum()) if n else 0.0
    return {
        "occupied_sectors": int(len(cell_counts)),
        "vectors_per_sector": n / max(len(cell_counts), 1),
        "raw_bits": grid.bits * d,
        "sector_bits_fixed": grid.sector_bits * d,
        "sector_bits_ideal": d * float(np.log2(grid.K)),
        "correction_bits_fixed": grid.correction_bits * d,
        # Ideal per-vector cost of the sector if each whole cell is a symbol.
        "cell_entropy_per_vector": cell_entropy / max(n, 1),
        "per_dim_sector_entropy_per_vector":
            sum(entropy_bits(S[:, j]) for j in range(d)) / max(n, 1),
        "correction_entropy_per_vector":
            sum(entropy_bits(C[:, j]) for j in range(d)) / max(n, 1),
    }
