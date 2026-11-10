"""How precise a coordinate can you fit in N query-string characters?

    python3 -m modcomp.qs_bench                    # synthetic, D=384
    python3 -m modcomp.qs_bench --input emb.npy    # real embeddings (n, D)

Each held-out query vector is encoded to a string, decoded, and the decoded
vector is used to search the (uncompressed) database. Reported per codec:
characters, mean and worst-1% cosine to the original vector, and recall@10
against searching with the original vector.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from . import embed as E
from . import qs


def codecs(d):
    pq = [m for m in (d // 16, d // 8) if d % m == 0]
    out = [qs.RotScalar(d, b) for b in (8, 6, 4, 3, 2, 1)]
    out.append(qs.RotScalar(d, 4, rotate=False))
    out += [qs.SectorRot(d, 1024, b) for b in (4, 3, 2, 1)]
    if len(pq) == 2:
        out.append(qs.Progressive(1024, [pq[0], pq[0], pq[1]]))
    return out


def evaluate(codec, db, queries, k):
    truth = E.topk(db, queries, k)
    dec, lens = [], set()
    for q in queries:
        s = codec.encode(q)
        lens.add(len(s))
        dec.append(codec.decode(s))
    dec = np.array(dec, dtype=np.float32)
    cos = (dec * queries).sum(1) / np.linalg.norm(queries, axis=1)
    found = E.topk(db, dec, k)
    rec = np.mean([len(np.intersect1d(t, f)) / k for t, f in zip(truth, found)])
    return max(lens), cos.mean(), np.percentile(cos, 1), rec, dec


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input")
    p.add_argument("--queries", type=int, default=300)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)

    if a.input:
        X = np.load(a.input).astype(np.float32)
        X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
        perm = np.random.default_rng(a.seed).permutation(len(X))
        queries, db = X[perm[:a.queries]], X[perm[a.queries:]]
    else:
        db, queries = E.synthetic_embeddings(n_queries=a.queries, seed=a.seed)
    n, d = db.shape
    print(f"{n:,} database vectors, {len(queries)} held-out query vectors, D={d}")
    print(f"reference: float32 = {qs.chars_for_bits(32 * d):,} chars, "
          f"float16 = {qs.chars_for_bits(16 * d):,} chars\n")
    print(f"{'codec':<34} {'chars':>6} {'cos mean':>9} {'cos p1':>8} "
          f"{f'recall@{a.k}':>10} {'shared KiB':>10} {'sec':>5}")
    for c in codecs(d):
        t = time.time()
        c.fit(db)
        chars, cm, c1, rec, _ = evaluate(c, db, queries, a.k)
        print(f"{c.name:<34} {chars:>6,} {cm:>9.5f} {c1:>8.5f} {rec:>10.3f} "
              f"{c.model_bytes / 1024:>10,.0f} {time.time() - t:>5.1f}")
        if isinstance(c, qs.Progressive):
            print("  prefixes of the progressive string:")
            full = [c.encode(q) for q in queries]
            truth = E.topk(db, queries, a.k)
            for L in c.prefix_chars():
                dec = np.array([c.decode(s[:L]) for s in full], dtype=np.float32)
                cos = (dec * queries).sum(1)
                found = E.topk(db, dec, a.k)
                rec = np.mean([len(np.intersect1d(t, f)) / a.k for t, f in zip(truth, found)])
                print(f"  {'first ' + str(L) + ' chars':<32} {L:>6} {cos.mean():>9.5f} "
                      f"{np.percentile(cos, 1):>8.5f} {rec:>10.3f}")

    demo = qs.RotScalar(d, 2)
    print(f"\nexample (rotated 2-bit, {qs.chars_for_bits(demo.bits)} chars):")
    print(f"  https://example.com/search?v={demo.encode(queries[0])}")


if __name__ == "__main__":
    main()
