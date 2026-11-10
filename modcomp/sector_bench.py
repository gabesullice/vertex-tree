"""Benchmark sector + correction compression of [0, 1] vectors.

    python3 -m modcomp.sector_bench                         # all datasets
    python3 -m modcomp.sector_bench --dataset clusters3 --bits 32 --digits 1 2 3 4
    python3 -m modcomp.sector_bench --input vecs.npy --bits 16
"""

from __future__ import annotations

import argparse
import lzma
import zlib

import numpy as np

from . import sector as S


# --------------------------------------------------------------------------
# Synthetic data, all components in [0, 1]
# --------------------------------------------------------------------------

def _blobs(rng, n, d, k, sigma):
    centres = rng.random((k, d))
    pts = centres[rng.integers(0, k, n)] + sigma * rng.standard_normal((n, d))
    return np.clip(pts, 0, 1)


DATASETS = {
    "uniform3": ("uniform in the unit cube, D=3",
                 lambda rng: rng.random((100_000, 3))),
    "clusters3": ("20 tight clusters (sigma 0.005), D=3",
                  lambda rng: _blobs(rng, 100_000, 3, 20, 0.005)),
    "clusters16": ("30 clusters (sigma 0.01), D=16",
                   lambda rng: _blobs(rng, 50_000, 16, 30, 0.01)),
    "embed128": ("embedding-like: clip(0.5 + 0.12 N(0,1)), D=128",
                 lambda rng: np.clip(0.5 + 0.12 * rng.standard_normal((10_000, 128)), 0, 1)),
    "decimals4": ("2-decimal readings + tiny noise, D=4",
                  lambda rng: np.clip(np.round(rng.random((100_000, 4)), 2)
                                      + 0.0003 * rng.standard_normal((100_000, 4)), 0, 1)),
}


def _bpv(nbytes, n):
    return 8 * nbytes / n


def report(name, desc, X, bits, digits_list, codec):
    V = S.to_fixed(X, bits)
    n, d = V.shape
    comp = lzma.compress if codec == "lzma" else (lambda b: zlib.compress(b, 9))
    raw_bytes = V.astype(S._uint_dtype((1 << bits) - 1)).tobytes()
    # Fair baseline for the order-free "grouped" mode: sort the raw vectors
    # (no sectors at all), delta the first column, compress.
    Vs = V[np.lexsort(V.T[::-1])].copy()
    Vs[:, 0] = np.diff(Vs[:, 0], prepend=0)
    sorted_bytes = Vs.astype(S._uint_dtype((1 << bits) - 1)).tobytes()
    print(f"\n== {name}: {desc}; {n:,} vectors, {bits}-bit components")
    print(f"   baselines  packed raw {bits * d:,.1f}   "
          f"{codec}(raw) {_bpv(len(comp(raw_bytes)), n):,.1f}   "
          f"{codec}(sorted raw, order discarded) "
          f"{_bpv(len(comp(sorted_bytes)), n):,.1f}   (bits/vec)")
    print(f"   {'precision':>9} {'occupied':>9} {'vec/sec':>8} "
          f"{'ordered':>9} {'grouped':>9} {'+perm':>9} {'lossy':>9}  "
          f"{'cell H':>7} {'corr bits':>9}   (bits/vec)")
    for k in digits_list:
        grid = S.SectorGrid.decimal(bits, k)
        st = S.stats(V, grid)
        row = {}
        for mode in S.MODES:
            blob, _ = S.encode(V, grid, mode, codec)
            out = S.decode(blob)
            if mode in ("ordered", "grouped+perm"):
                assert np.array_equal(out, V), (name, k, mode)
            elif mode == "grouped":
                assert np.array_equal(np.sort(out, axis=0) if d == 1 else
                                      out[np.lexsort(out.T[::-1])],
                                      V[np.lexsort(V.T[::-1])]), (name, k, mode)
            else:
                err = np.abs(S.from_fixed(out, bits) - X).max()
                assert err <= grid.precision / 2 + 2 / grid.Q, (name, k, err)
            row[mode] = _bpv(len(blob), n)
        print(f"   {grid.precision:>9g} {st['occupied_sectors']:>9,} "
              f"{st['vectors_per_sector']:>8.1f} {row['ordered']:>9.1f} "
              f"{row['grouped']:>9.1f} {row['grouped+perm']:>9.1f} "
              f"{row['sectors']:>9.1f}  {st['cell_entropy_per_vector']:>7.1f} "
              f"{st['correction_bits_fixed']:>9}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", action="append", choices=sorted(DATASETS))
    p.add_argument("--input", help=".npy or .csv of floats in [0, 1]")
    p.add_argument("--bits", type=int, default=16, help="N, bits per component")
    p.add_argument("--digits", type=int, nargs="+", default=[1, 2, 3],
                   help="precision 10**-digits for each run")
    p.add_argument("--codec", choices=["zlib", "lzma"], default="lzma")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)

    if a.input:
        X = np.load(a.input) if a.input.endswith(".npy") else np.loadtxt(a.input, delimiter=",")
        report(a.input, "user data", np.atleast_2d(X), a.bits, a.digits, a.codec)
        return
    rng = np.random.default_rng(a.seed)
    for name in a.dataset or DATASETS:
        desc, make = DATASETS[name]
        report(name, desc, make(rng), a.bits, a.digits, a.codec)


if __name__ == "__main__":
    main()
