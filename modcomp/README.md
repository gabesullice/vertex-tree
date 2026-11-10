# modcomp — sector + correction vector compression experiment

Self-contained experiment (Python + numpy only). It shares no code with the
rest of the repository.

## One embedding in a query string (`qs.py`, `qs_bench.py`)

Goal: fit one precise "coordinate" (an embedding) into a URL parameter.

* **Alphabet:** base64url (`A–Z a–z 0–9 - _`), which is unreserved in URLs, so no
  percent-escaping is needed. Bits are written straight into characters, so B
  bits take exactly `ceil(B/6)` characters, with no padding.
* **Shared state:** with a single vector there is nothing to share within the
  data. Any learned "sectors" must be known to both the encoder and the
  decoder, like a map both sides hold.

```python
from modcomp import qs
codec = qs.RotScalar(d=384, bits=4)       # no training, shared state = a seed
url = "https://example.com/search?v=" + codec.encode(vec)   # 256 chars
vec2 = codec.decode(param)                # unit vector, cos ≈ 0.994

codec = qs.SectorRot(384, k=1024, bits=2).fit(corpus)   # shared centroid table
codec = qs.Progressive(1024, [24, 24, 48]).fit(corpus)  # geohash-like
```

Codecs:

| codec | layout | shared state |
|---|---|---|
| `RotScalar` | fixed seeded rotation, then b bits per coordinate (Gaussian-optimal uniform step) | a seed |
| `SectorRot` | **sector** (nearest of k centroids) + float16 scale + rotated b-bit **correction** | k × D centroid table |
| `Progressive` | sector, then a chain of PQ-coded corrections; **any stage-boundary prefix decodes** | centroids + PQ codebooks |

Synthetic run (`python3 -m modcomp.qs_bench`, D=384; float32 = 2048 chars,
float16 = 1024 chars). Held-out vectors are encoded, decoded, and used as
search queries. **Real embeddings will differ.**

| chars | codec | cos mean | cos worst 1% | recall@10 |
|---:|---|---:|---:|---:|
| 512 | rotated 8-bit | 0.99996 | 0.99996 | 0.989 |
| 384 | rotated 6-bit | 0.99951 | 0.99890 | 0.971 |
| 256 | rotated 4-bit | 0.99439 | 0.99181 | 0.906 |
| 261 | sector 1024 + 4-bit correction | 0.99884 | 0.99776 | 0.952 |
| 197 | sector 1024 + 3-bit correction | 0.99624 | 0.99361 | 0.922 |
| 128 | rotated 2-bit | 0.93925 | 0.92835 | 0.713 |
| 133 | sector 1024 + 2-bit correction | 0.98796 | 0.98044 | 0.847 |
| 130 | progressive (full) | 0.98924 | 0.98223 | 0.860 |
| 66 | progressive (prefix) | 0.96808 | 0.95242 | 0.739 |
| 69 | sector 1024 + 1-bit correction | 0.96259 | 0.94111 | 0.678 |
| 64 | rotated 1-bit | 0.79909 | 0.77287 | 0.508 |
| 2 | progressive (sector only) | 0.89235 | 0.82184 | 0.277 |

**D = 1536** (`python3 -m modcomp.qs_bench --dim 1536`; float32 = 8192 chars,
float16 = 4096 chars). This is the size of e.g. OpenAI `text-embedding-3-small`.

| chars | codec | cos mean | cos worst 1% | recall@10 | shared state |
|---:|---|---:|---:|---:|---|
| 2048 | rotated 8-bit | 0.99996 | 0.99985 | 0.995 | seed |
| **1536** | **rotated 6-bit** | **0.99949** | **0.99906** | **0.976** | seed |
| 1029 | sector 1024 + 4-bit correction | 0.99875 | 0.99809 | 0.959 | 6 MiB centroids |
| 1024 | rotated 4-bit | 0.99425 | 0.99270 | 0.924 | seed |
| 773 | sector 1024 + 3-bit correction | 0.99591 | 0.99427 | 0.926 | 6 MiB centroids |
| 517 | sector 1024 + 2-bit correction | 0.98698 | 0.98236 | 0.861 | 6 MiB centroids |
| 514 | progressive (full) | 0.98777 | 0.98245 | 0.862 | 10.5 MiB codebooks |
| 512 | rotated 2-bit | 0.93876 | 0.93275 | 0.770 | seed |
| 258 | progressive (prefix) | 0.96357 | 0.95060 | 0.739 | 10.5 MiB codebooks |
| 256 | rotated 1-bit | 0.79766 | 0.78540 | 0.580 | seed |

Characters scale linearly with D, and quality per bit per dimension is about
the same as at D=384. Under a ~2000-char URL budget, rotated 6-bit (1536
chars, nothing to ship but a seed) is the precise option. Sector + 4-bit
correction gets nearly the same precision in 1029 chars if both sides can
hold a centroid table.

What the run shows:

1. **With no shared model, rotate first.** A fixed random rotation makes every
   coordinate look alike (no outlier dims), so one quantizer step fits all of
   them and no per-dimension ranges need shipping. At the same 256 chars,
   recall goes from 0.698 unrotated to 0.906 rotated.
2. **A shared codebook roughly halves the string for the same precision.**
   Sector + 3-bit correction (197 chars) beats rotated 4-bit (256 chars).
   Sector + 2-bit (133 chars) is far better than rotated 2-bit (128 chars).
3. **Progressive codes behave like a geohash.** The first 2 chars name the
   sector, and each further block of chars refines the position. At about
   130 chars it slightly beats sector + scalar correction, and shortened
   strings still decode.
4. The rotation must not line up with the data's own structure. A first run
   accidentally reused the synthetic generator's seed, which undid the
   rotation and cost a lot of precision, so the rotation now has its own fixed
   seed (`qs.ROTATION_SEED`).

Practical note: the rotation matrix comes from numpy's seeded RNG and QR. An
encoder or decoder in another language should ship the matrix itself, or use
a portable construction (e.g. a randomized Hadamard transform).

## Embeddings (`embed.py`, `embed_bench.py`)

The target data is embeddings, so the goal is **lossy**: keep nearest-neighbour
search working with as few bits as possible. Every method is scored by bits
per vector, mean `cos(x, x̂)`, and recall@10. Queries are uncompressed and
search the reconstructed database.

The sector + correction idea carries over directly if the sectors are
**learned** instead of a fixed grid. The sector becomes the nearest of C
k-means centroids (`log2 C` bits), and the correction is the residual
`x − centroid`, quantized. This is the IVF-ADC design used by vector search
libraries such as FAISS.

```bash
python3 -m modcomp.embed_bench                    # synthetic, D=384
python3 -m modcomp.embed_bench --input emb.npy    # real embeddings, shape (n, D)
```

Synthetic run (20k unit vectors, D=384, float32 = 12,288 bits). It mimics
anisotropic spectrum, topic clusters, a shared mean direction and outlier
dims. **Results on real embeddings will differ.**

| bits/vec | grid sectors (original idea) | axis sectors fit to data | product quant. | **learned sector + correction** |
|---:|---|---|---|---|
| ~1536 | grid s=0.1: 0.366 | scalar-4: 0.753 | — | **kmeans-256 + res-4: 0.851** |
| ~770 | — | scalar-2: 0.382 | pq-96: 0.480 | **kmeans-256 + pq-96: 0.678** |
| ~390 | binary: 0.323 | — | pq-48: 0.330 | **kmeans-256 + pq-48: 0.534** |
| ~200 | — | — | pq-24: 0.227 | **kmeans-256 + pq-24: 0.421** |

(recall@10; references: float16 0.999 at 6144 bits, scalar-8 0.956 at 3072 bits)

What the run shows:

1. **A fixed [0,1] grid wastes most of its sectors.** Unit-norm embedding
   components cluster near 0 (std ≈ 0.05 here), so a 0.1 grid puts nearly
   everything in a few sectors. Entropy coding shrinks it (1536 → 351 bits
   after lzma), but the precision is in the wrong place.
2. **Fitting the sectors to each dimension's range** (scalar quantization)
   roughly doubles recall at the same bits.
3. **Learned sectors + a correction vector win at every budget.** The
   centroid absorbs the shared mean direction and the topic structure, so
   the correction has a much smaller range and its bits buy more precision.
4. Applied to the axis-aligned case, "coarse b1 bits + correction b2 bits"
   is exactly b1+b2-bit scalar quantization. The split only helps when the
   coarse part is shared or learned.

Things to try next on real embeddings: a random or PCA rotation before scalar
quantization (spreads outlier dims), more centroids (1024–65536), re-ranking
the top candidates with a finer correction, and Matryoshka-style truncation
if the model supports it.

## Sector grid for [0, 1] vectors (`sector.py`)

### The idea

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

### Usage

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

### First results (`python3 -m modcomp.sector_bench`, 16-bit, lzma, seed 0)

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

### Next things to try

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

* `qs.py`, `qs_bench.py`: one embedding <-> URL-safe string (rotated scalar, sector + correction, progressive)
* `embed.py`, `embed_bench.py`: embedding methods (grid, scalar, k-means + residual, PQ), recall/cosine evaluation, synthetic embeddings
* `sector.py`: `SectorGrid`, fixed-point conversion, the `MSEC` container (4 modes), `stats`
* `sector_bench.py`: synthetic [0,1] datasets and the CLI report
* `core.py`, `analysis.py`, `bench.py`: earlier integer modulus-split scheme
* `tests/`: round trips, exhaustive split/merge checks, lossy error bound, edge cases
