"""Compare embedding compression methods by bits/vector vs. search quality.

    python3 -m modcomp.embed_bench                       # synthetic, D=384
    python3 -m modcomp.embed_bench --input emb.npy       # your embeddings (n, D)
    python3 -m modcomp.embed_bench --input emb.npy --no-normalize
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from . import embed as E


def methods(d: int):
    pq_ms = [m for m in (d // 16, d // 8, d // 4) if m and d % m == 0]
    out = [E.Float16(), E.Grid01(1), E.Grid01(2), E.Binary()]
    out += [E.Scalar(b) for b in (8, 4, 2)]
    out += [E.CoarsePlus(256, E.Scalar(b)) for b in (4, 2)]
    out += [E.PQ(m) for m in pq_ms]
    out += [E.CoarsePlus(256, E.PQ(m)) for m in pq_ms]
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", help=".npy file of shape (n, D)")
    p.add_argument("--queries", type=int, default=500,
                   help="rows held out as uncompressed queries (with --input)")
    p.add_argument("--no-normalize", action="store_true",
                   help="do not L2-normalize --input rows")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)

    if a.input:
        X = np.load(a.input).astype(np.float32)
        if not a.no_normalize:
            X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
        rng = np.random.default_rng(a.seed)
        perm = rng.permutation(len(X))
        queries, db = X[perm[:a.queries]], X[perm[a.queries:]]
        source = a.input
    else:
        db, queries = E.synthetic_embeddings(seed=a.seed)
        source = "synthetic embedding-like data"

    n, d = db.shape
    print(f"{source}: {n:,} database vectors, {len(queries)} queries, D={d}")
    print(f"component range [{db.min():.3f}, {db.max():.3f}], "
          f"std {db.std():.4f}  (float32 = {32 * d:,} bits/vec)\n")
    print(f"{'method':<22} {'bits/vec':>9} {'lzma':>8} {'x smaller':>9} "
          f"{'cos(x,x^)':>9} {f'recall@{a.k}':>10} {'model KiB':>10} {'sec':>5}")
    for m in methods(d):
        t = time.time()
        r = E.evaluate(m, db, queries, a.k)
        print(f"{r['name']:<22} {r['bits']:>9,} {r['lzma_bits']:>8,.0f} "
              f"{r['ratio']:>9.1f} {r['cosine']:>9.4f} {r['recall']:>10.3f} "
              f"{r['model_kib']:>10,.0f} {time.time() - t:>5.1f}")


if __name__ == "__main__":
    main()
