"""Benchmark modulus-split compression against simple baselines.

    python -m modcomp.bench                       # all synthetic datasets
    python -m modcomp.bench --dataset walk --sweep
    python -m modcomp.bench --input points.npy --step 0.001 --sweep
"""

from __future__ import annotations

import argparse
import lzma
import sys
import zlib

import numpy as np

from . import analysis as A
from .core import decode, delta, encode, quantize


# --------------------------------------------------------------------------
# Synthetic data
# --------------------------------------------------------------------------

def ds_walk(rng, n=100_000):
    """3D random walk with small integer steps (highly coherent)."""
    return np.cumsum(rng.integers(-3, 4, size=(n, 3)), axis=0) + 50_000


def ds_surface(rng, side=300):
    """Vertices of a smooth heightfield, row-major, quantized to 1e-3."""
    x, y = np.meshgrid(np.linspace(0, 10, side), np.linspace(0, 10, side))
    z = np.sin(x) * np.cos(0.7 * y) + 0.05 * rng.standard_normal(x.shape)
    pts = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)
    return quantize(pts, 1e-3)


def ds_spiral(rng, n=100_000):
    """Points along a spiral on the unit sphere, quantized to 1e-5."""
    t = np.linspace(0, 1, n)
    theta = np.arccos(1 - 2 * t)
    phi = 400 * np.pi * t
    pts = np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi),
                    np.cos(theta)], axis=1)
    return quantize(pts + 2e-5 * rng.standard_normal(pts.shape), 1e-5)


def ds_jumpy(rng, n=100_000):
    """Coherent runs separated by occasional large jumps."""
    steps = rng.integers(-2, 3, size=(n, 3))
    jumps = rng.random(n) < 0.002
    steps[jumps] = rng.integers(-40_000, 40_000, size=(int(jumps.sum()), 3))
    return np.cumsum(steps, axis=0) + 1_000_000


def ds_noise(rng, n=100_000):
    """Uniform random 16-bit coordinates (incompressible worst case)."""
    return rng.integers(0, 1 << 16, size=(n, 3))


DATASETS = {
    "walk": ds_walk,
    "surface": ds_surface,
    "spiral": ds_spiral,
    "jumpy": ds_jumpy,
    "noise": ds_noise,
}


def load_input(path: str, step: float | None) -> np.ndarray:
    if path.endswith(".npy"):
        X = np.load(path)
    else:
        X = np.loadtxt(path, delimiter="," if path.endswith(".csv") else None)
    X = np.atleast_2d(X)
    if not np.issubdtype(X.dtype, np.integer):
        if step is None:
            sys.exit("float input needs --step to quantize (e.g. --step 0.001)")
        X = quantize(X, step)
    return X.astype(np.int64)


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def _roundtrip(V, m, c, codec):
    blob, sizes = encode(V, m, c, codec=codec)
    out = decode(blob)
    if not np.array_equal(out, V):
        raise AssertionError(f"round trip failed (m={m}, offset={c}, codec={codec})")
    return sizes


def _row(label, nbytes, n, note=""):
    print(f"  {label:<34} {nbytes:>12,} B  {8 * nbytes / n:>8.2f} bits/vec  {note}")


def report(name: str, V: np.ndarray, sweep: bool) -> None:
    n, d = V.shape
    print(f"\n== {name}: {n:,} vectors x {d} dims, "
          f"range per dim {(V.max(0) - V.min(0)).tolist()}")

    raw = V.astype(np.int32 if np.abs(V).max() < 2**31 else np.int64).tobytes()
    draw = delta(V).astype(np.int64).tobytes()
    print("  -- baselines")
    _row("raw int32", len(raw), n)
    _row("zlib(raw)", len(zlib.compress(raw, 9)), n)
    _row("lzma(raw)", len(lzma.compress(raw, preset=9)), n)
    _row("zlib(delta)", len(zlib.compress(draw, 9)), n)
    _row("lzma(delta)", len(lzma.compress(draw, preset=9)), n)

    m, c, _ = A.choose_parameters(V)
    print(f"  -- modulus split  (chosen per dim: m={m.tolist()}, offset={c.tolist()})")
    for codec in ("zlib", "lzma"):
        s = _roundtrip(V, m, c, codec)
        _row(f"modsplit/{codec}", s["total"], n,
             f"residues {s['residues']:,} + corrections {s['corrections']:,}")

    est = A.estimate(V, m, c)
    base = A.baselines(V)
    print("  -- entropy model (zeroth order, ideal coder)")
    print(f"  {'delta coding':<34} {A.bits_per_vector(base['delta_entropy_bits'], n):>24.2f} bits/vec")
    print(f"  {'modsplit residues + corrections':<34} "
          f"{A.bits_per_vector(est['total_bits'], n):>24.2f} bits/vec"
          f"  ({A.bits_per_vector(est['residue_bits'], n):.2f} + "
          f"{A.bits_per_vector(est['correction_bits'], n):.2f}; "
          f"{est['correction_nonzero']:,} nonzero correction deltas)")

    if sweep:
        print("  -- sweep: same modulus in every dim, best offset per dim, zlib")
        print(f"  {'m':>10} {'total B':>12} {'bits/vec':>9} {'resid B':>11} "
              f"{'corr B':>11} {'nz corr':>9}")
        best = None
        for mm in A.power_of_two_candidates(V):
            cc = np.array([A.best_offset(V[:, j], mm) for j in range(d)])
            s = _roundtrip(V, mm, cc, "zlib")
            nz = int(np.count_nonzero(np.diff((V - cc) // mm, axis=0)))
            print(f"  {mm:>10,} {s['total']:>12,} {8 * s['total'] / n:>9.2f} "
                  f"{s['residues']:>11,} {s['corrections']:>11,} {nz:>9,}")
            if best is None or s["total"] < best[1]:
                best = (mm, s["total"])
        print(f"  best in sweep: m={best[0]:,} at {best[1]:,} B "
              f"({8 * best[1] / n:.2f} bits/vec)")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=sorted(DATASETS), action="append",
                   help="synthetic dataset(s) to run (default: all)")
    p.add_argument("--input", help=".npy, .csv or whitespace-separated text file")
    p.add_argument("--step", type=float, help="quantization step for float input")
    p.add_argument("--sweep", action="store_true", help="also sweep the modulus")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    if args.input:
        report(args.input, load_input(args.input, args.step), args.sweep)
        return
    rng = np.random.default_rng(args.seed)
    for name in args.dataset or DATASETS:
        report(name, DATASETS[name](rng), args.sweep)


if __name__ == "__main__":
    main()
