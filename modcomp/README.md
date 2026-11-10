# modcomp — sector + correction vector compression experiment

Self-contained experiment (Python + numpy only). It shares no code with the
rest of the repository.

## The idea

* Every vector has a known number of dimensions `D`, and each component is a
  magnitude in `[0, 1]`, stored as an `N`-bit integer.
* Pick a precision `s = 0.0…1` (here `s = 10^-k`, or in general `1/K`). It
  cuts each axis into `K` sectors, so the unit cube becomes a grid of `K^D`
  sectors, each `s` wide.
* Store each vector as the **sector** it falls in (D small ints in `[0, K)`)
  plus a **correction vector** (its exact offset inside the sector, in N-bit
  units).

All of it is integer arithmetic, so the N-bit values come back exactly. With
`Q = 2^N − 1`:

```
boundary b(j) = ceil(j·Q / K)
sector(v)     = min(floor(v·K / Q), K − 1)     # 1.0 goes in the last sector
correction    = v − b(sector(v))
```

Storage modes (`modcomp.sector.encode(V, grid, mode)`):

| mode | what is stored | decodes to |
|---|---|---|
| `ordered` | sector of every vector (lzma/zlib) + corrections | original order, exact |
| `grouped` | each **occupied** sector once + a count, then the corrections | exact values, in sector order (a set) |
| `grouped+perm` | `grouped` + the permutation back to the original order | original order, exact |
| `sectors` | sector stream only | sector centres (lossy, error ≤ s/2) |

Corrections are stored bit-packed at `ceil(log2(max sector width))` bits, or
compressed if that is smaller.

## Usage

```bash
python3 -m modcomp.sector_bench                       # all synthetic datasets, 16-bit
python3 -m modcomp.sector_bench --bits 32 --digits 1 2 3 4
python3 -m modcomp.sector_bench --input vecs.npy --bits 16   # floats in [0,1]
python3 -m unittest discover -s modcomp/tests -t .
```

```python
from modcomp import sector as S
V = S.to_fixed(X, bits=16)                 # floats in [0,1] -> 16-bit ints
grid = S.SectorGrid.decimal(16, digits=2)  # 0.01-wide sectors
blob, sizes = S.encode(V, grid, mode="ordered", codec="lzma")
assert (S.decode(blob) == V).all()
sec, corr = grid.split(V)                  # the two parts directly
```

## First results (`python3 -m modcomp.sector_bench`, 16-bit, lzma, seed 0)

Bits per vector. Compare **ordered** with `lzma(raw)`. Compare **grouped**
with `lzma(sorted raw)`: both discard the order, and discarding the order of n
items is worth about log2(n!) bits on its own, with or without sectors.

| dataset | raw | lzma(raw) | best **ordered** | lzma(sorted raw) | best **grouped** | lossy @0.01 |
|---|---:|---:|---:|---:|---:|---:|
| uniform, D=3 | 48 | 48.0 | 49.5 (s=0.1) | 35.8 | 36.5 | 20.3 |
| 20 tight clusters, D=3 | 48 | 47.3 | **40.7** (s=0.01) | 26.6 | **23.7** (s=0.001) | 10.7 |
| 30 clusters, D=16 | 256 | 256.0 | **220.1** (s=0.1) | 238.4 | **199.0** (s=0.01) | 60.1 |
| embedding-like, D=128 | 2048 | 1996.1 | 1994.9 | 1996.2 | 1997.8 | 736.9 |
| 2-decimal readings, D=4 | 64 | 58.8 | **55.8** (s=0.01) | 44.6 | **42.3** (s=0.01) | 27.1 |

What the run shows:

1. **Splitting by itself saves nothing.** Sector plus correction is the same
   information as the N-bit value. On uniform data, ordered mode is slightly
   *worse* than raw.
2. **Decimal sectors cost about 1 bit per component.** 16-bit values with
   0.01 sectors need 7 bits of sector + 10 bits of correction = 17 bits.
   `1/10^k` does not split a power of two evenly. Power-of-two sector counts
   (`SectorGrid(bits, 128)`) avoid this, at the cost of non-decimal precision.
3. **The saving is the sector entropy, and only when vectors cluster.** Per
   vector, the most you can save is `D·log2(K) − H(sector)`, where `H(sector)`
   (column `cell H` in the bench) is the entropy of which sector a vector
   lands in. Corrections inside a sector are close to uniform, so they do not
   compress. Clustered data saves 11–16% lossless (D=3 and D=16). Uniform and
   high-dimensional embedding-like data save nothing, because with D=128 every
   vector has its own sector even at s=0.1.
4. **The sector stream alone is a good lossy code.** At s=0.01 it costs 10.7
   bits per vector for the D=3 clusters instead of 48, with error ≤ 0.005 per
   component. That makes the correction vector a natural refinement layer:
   send sectors first, corrections later.
5. **At 32 bits the relative gain shrinks.** The extra bits all go into the
   correction, which does not compress: D=3 clusters go from 96 to 88.7
   ordered, and from 78.9 to 74.3 grouped.

## Next things to try

* Power-of-two sector counts, or mixed-radix packing of corrections, to win
  back the ~1 bit per component that decimal sectors cost.
* Hierarchical sectors (0.1 → 0.01 → 0.001, i.e. an octree / 2^D-tree):
  each level codes only the occupied child sectors.
* Per-dimension precision, for dimensions with different spreads.
* Sectors of rotated or learned axes (e.g. PCA, or k-means centres as the
  "sectors") for high-D data, where axis-aligned cells are all unique.

## Earlier interpretation (kept for comparison)

`core.py`, `analysis.py` and `bench.py` implement the first reading of the
request: integer coordinates split as `v = q·m + r` with a power-of-two
modulus, delta-coding `q` along the array order. Run it with
`python3 -m modcomp.bench --sweep`. It was mostly beaten by plain delta coding.

## Files

* `sector.py`: `SectorGrid`, fixed-point conversion, the `MSEC` container (4 modes), `stats`
* `sector_bench.py`: synthetic [0,1] datasets and the CLI report
* `core.py`, `analysis.py`, `bench.py`: earlier integer modulus-split scheme
* `tests/`: round trips, exhaustive split/merge checks, lossy error bound, edge cases
