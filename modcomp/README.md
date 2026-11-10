# modcomp — modulus-split vector compression experiment

Self-contained experiment (Python + numpy only). It shares no code with the
rest of the repository.

## The idea

Strip a modulus `m` from every component of every vector and keep a correction:

```
v = q·m + r + c        0 ≤ r < m
```

| part | what it is | how it is stored |
|---|---|---|
| `r` residue | the low part left after stripping the modulus | fixed width, `ceil(log2 m)` bits per component, bit-packed; **random access** |
| `q` correction vector | how many whole moduli were stripped | delta-coded along the array (`q[i] - q[i-1]`), zigzag + varint, then zlib/lzma |
| `c` offset | per-dimension shift of where the modulus boundaries fall | header (one int per dim) |

`m` and `c` are chosen per dimension. Reconstruction is exact (lossless). For
float data, quantize first (`modcomp.quantize(X, step)`).

The offset matters. A coordinate that jitters across a multiple of `m` makes
the correction flip every step. `analysis.best_offset` finds the exact offset
that minimizes correction changes, in O(N + m), using a cyclic difference
array.

The two ends of the modulus range bracket the scheme:

* `m = 1`: residues are 0 bits and the correction *is* the data, so this is
  plain delta coding.
* `m > range`: the correction is constant, so this is fixed-width raw storage.

## Usage

```bash
python3 -m modcomp.bench                          # all synthetic datasets
python3 -m modcomp.bench --dataset walk --sweep   # size vs. modulus table
python3 -m modcomp.bench --input pts.npy --step 0.001 --sweep   # your data
python3 -m unittest discover -s modcomp/tests -t .
```

```python
import modcomp
blob, sizes = modcomp.encode(V, m=[64, 64, 16], offset=[3, 0, 9], codec="zlib")
V2 = modcomp.decode(blob)                 # exact
r7 = modcomp.residue_at(blob, 7)          # residue of vector 7, no full decode
m, c, _ = modcomp.choose_parameters(V)    # entropy-model optimum per dim
```

## First results (`python3 -m modcomp.bench --sweep`, seed 0)

Bits per 3-D vector (raw int32 = 96):

| dataset | zlib(delta) | lzma(delta) | modsplit m=1 (zlib) | best modsplit (zlib) |
|---|---:|---:|---:|---:|
| walk (random walk, steps ±3) | 14.06 | 9.75 | **9.62** | m=1 |
| surface (heightfield, row-major) | 16.85 | 11.20 | 12.79 | m=1 |
| spiral (smooth curve + 2-unit noise) | 39.99 | **20.94** | 30.46 | m=32: 23.13 |
| jumpy (coherent runs + rare jumps) | 12.30 | 9.03 | **8.21** | m=1 |
| noise (uniform 16-bit) | 69.98 | 55.29 | 58.41 | m=65536: **48.03** |

(The m=1 column already beats `zlib(delta)` because varint packs small deltas
into one byte, while `zlib(delta)` compresses raw int64 deltas.)

What the run shows:

1. **With fixed-width residues, stripping a modulus rarely beats plain delta
   coding on coherent data.** The residues are paid at the full `log2 m` bits,
   but in coherent data the low bits are not random: consecutive residues are
   close together, and the fixed-width store throws that away. The correction
   stream does shrink fast as `m` grows (walk: 120 KB at m=1 down to 332 B at
   m=1024), but the residue cost grows faster.
2. **It wins when the low bits really are noise.** On `spiral`, the low ~5 bits
   are sensor-style noise, and m=32 under zlib beats both delta coding and
   zlib(delta). lzma(delta) still edges it out. On `noise`, m ≥ range is just
   fixed-width storage, which beats every general-purpose codec.
3. **The real benefit is structure, not size.** Every residue can be read at a
   known bit offset, and the correction stream is tiny and sparse at moderate
   `m`. For example, walk at m=256 stores 8 bits per component plus about 2 KB
   of corrections for 100k vectors. That supports random access, or a cheap
   coarse level of detail (corrections only, i.e. `q·m`), with near-zero
   correction overhead.

## Next things to try

* **Entropy-code the residues with a predictor** (`r[i] - r[i-1] mod m`),
  which brings back the coherence the fixed-width store loses. The cost is
  random access, unless it is done in blocks.
* **Block-local moduli/offsets**: pick `(m, c)` per block of k vectors, so the
  modulus follows local variation.
* **Correction as a separate LOD layer**: decode `q·m + c + m/2` alone as a
  coarse approximation, and measure its error against bytes.
* **Mixed-radix residue packing** for non-power-of-two `m`, reaching the
  `log2 m` bound instead of `ceil(log2 m)` (reported as `residue_bits_ideal`
  in `analysis.estimate`).
* **Order matters**: spatially sorting the vectors (e.g. Morton order) should
  make corrections far sparser.

## Files

* `core.py`: split/merge, zigzag/varint/bit-packing, the `MODC` container, `residue_at`
* `analysis.py`: entropy model, optimal offset, per-dimension modulus choice, baselines
* `bench.py`: synthetic datasets and CLI report
* `tests/test_modcomp.py`: round trips, edge cases, corruption, brute-force check of the offset search
