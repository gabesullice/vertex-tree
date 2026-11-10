"""Size estimates and parameter search for modulus-split compression.

Two useful limits frame the search:

* ``m == 1``: every residue is 0 bits and the correction *is* the data, so the
  scheme degenerates to plain delta coding.
* ``m > range``: the correction is constant, so the scheme degenerates to
  fixed-width raw storage.

Anything interesting lives between those two.
"""

from __future__ import annotations

import math

import numpy as np

from .core import _as_int_array, as_per_dim, residue_widths, split, zigzag


def entropy_bits(values) -> float:
    """Total zeroth-order empirical Shannon entropy of ``values``, in bits."""
    values = np.asarray(values).ravel()
    if values.size == 0:
        return 0.0
    _, counts = np.unique(values, return_counts=True)
    p = counts / values.size
    return float(-(counts * np.log2(p)).sum())


def _col_delta(col: np.ndarray) -> np.ndarray:
    return np.diff(col, prepend=0)


def transitions_by_offset(col: np.ndarray, m: int) -> np.ndarray:
    """For each offset c in [0, m), count i where the quotient of col changes.

    Uses a cyclic difference array, so it is exact and costs O(N + m).
    """
    col = np.asarray(col, dtype=np.int64)
    if len(col) < 2 or m == 1:
        always = int(np.count_nonzero(np.diff(col))) if m == 1 else 0
        return np.full(m, always, dtype=np.int64)
    a, b = col[:-1], col[1:]
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    span = hi - lo
    always = int(np.count_nonzero(span >= m))
    sel = (span > 0) & (span < m)
    s = (lo[sel] + 1) % m
    e = hi[sel] % m  # inclusive
    wrap = s > e
    # Each pair adds +1 on the cyclic offset range [s, e]; a wrapping range
    # is split into [s, m) and [0, e].
    plus = np.concatenate((s, np.zeros(int(wrap.sum()), dtype=np.int64)))
    minus = np.concatenate((np.where(wrap, m, e + 1), e[wrap] + 1))
    counts = (np.bincount(plus, minlength=m + 1)
              - np.bincount(minus, minlength=m + 1))
    return np.cumsum(counts[:m]) + always


def best_offset(col: np.ndarray, m: int, max_exact: int = 1 << 22) -> int:
    """Offset in [0, m) that minimizes quotient transitions along ``col``."""
    if m == 1:
        return 0
    if m <= max_exact:
        return int(np.argmin(transitions_by_offset(col, m)))
    # Very large modulus: sample candidate offsets.
    cand = np.linspace(0, m - 1, 4096).astype(np.int64)
    trans = [np.count_nonzero(np.diff((col - c) // m)) for c in cand]
    return int(cand[int(np.argmin(trans))])


def estimate(V, m, offset=0) -> dict:
    """Idealized bit counts for one parameter choice (no codec, no header).

    * ``residue_bits``      fixed-width storage, N * sum(ceil(log2 m))
    * ``residue_bits_ideal`` N * sum(log2 m) (mixed-radix packing bound)
    * ``correction_bits``   zeroth-order entropy of delta(Q), per dimension
    """
    V = _as_int_array(V)
    n, d = V.shape
    m = as_per_dim(m, d, "modulus")
    c = as_per_dim(offset, d, "offset")
    R, Q = split(V, m, c)
    widths = residue_widths(m)
    corr = sum(entropy_bits(_col_delta(Q[:, j])) for j in range(d))
    return {
        "modulus": m,
        "offset": c,
        "residue_bits": int(n * widths.sum()),
        "residue_bits_ideal": float(n * np.log2(m.astype(np.float64)).sum()),
        "correction_bits": corr,
        "correction_nonzero": int(np.count_nonzero(np.diff(Q, axis=0))),
        "total_bits": n * widths.sum() + corr,
    }


def baselines(V) -> dict:
    """Reference points to beat, in bits."""
    V = _as_int_array(V)
    n, d = V.shape
    ranges = V.max(axis=0) - V.min(axis=0) if n else np.zeros(d, dtype=np.int64)
    fixed = sum(int(r).bit_length() for r in ranges) * n
    return {
        "int32_bits": 32 * n * d,
        "fixed_width_bits": fixed,
        "delta_entropy_bits": sum(entropy_bits(_col_delta(V[:, j])) for j in range(d)),
        "raw_entropy_bits": sum(entropy_bits(V[:, j]) for j in range(d)),
    }


def power_of_two_candidates(V) -> list[int]:
    V = _as_int_array(V)
    span = int((V.max() - V.min()) if V.size else 0)
    return [1 << k for k in range(max(1, span.bit_length() + 1) + 1)]


def choose_parameters(V, candidates=None, optimize_offset: bool = True):
    """Pick modulus (and offset) per dimension to minimize estimated bits.

    Dimensions are independent under the per-dimension entropy model, so each
    is optimized on its own. Returns ``(modulus, offset, per_dim_table)`` where
    the table lists ``(m, offset, bits)`` for every candidate in every dim.

    Power-of-two candidates are the natural default: residues are stored at
    ceil(log2 m) bits, so the largest modulus for a given width is 2**w.
    """
    V = _as_int_array(V)
    n, d = V.shape
    if candidates is None:
        candidates = power_of_two_candidates(V)
    best_m = np.ones(d, dtype=np.int64)
    best_c = np.zeros(d, dtype=np.int64)
    table = []
    for j in range(d):
        col = V[:, j]
        rows = []
        for m in candidates:
            m = int(m)
            c = best_offset(col, m) if optimize_offset else 0
            q = (col - c) // m
            bits = n * int(m - 1).bit_length() + entropy_bits(_col_delta(q))
            rows.append((m, c, bits))
        m, c, _ = min(rows, key=lambda r: r[2])
        best_m[j], best_c[j] = m, c
        table.append(rows)
    return best_m, best_c, table


def fmt_bytes(bits: float) -> str:
    b = bits / 8
    if b < 1024:
        return f"{b:,.0f} B"
    return f"{b / 1024:,.1f} KiB"


def bits_per_vector(bits: float, n: int) -> float:
    return bits / n if n else math.nan
