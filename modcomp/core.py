"""Modulus-split compression of integer vector arrays.

Every component of every vector is written as

    v = q * m + r + c        with  0 <= r < m

where ``m`` is the modulus (per dimension), ``c`` is an optional offset (per
dimension), ``r`` is the *residue* and ``q`` is the *correction*.

* Residues are stored at a fixed width of ceil(log2 m) bits per component. They
  are random access: you can decode residue i without touching anything else.
* Correction vectors are delta-coded along the array (q_i - q_{i-1}),
  zigzag/varint-encoded, and then passed through a general-purpose codec. For
  spatially coherent data most deltas are zero, so this stream is small.

Reconstruction is exact.
"""

from __future__ import annotations

import lzma
import zlib

import numpy as np

MAGIC = b"MODC"
VERSION = 1

CODECS = {"none": 0, "zlib": 1, "lzma": 2}
_CODEC_NAMES = {v: k for k, v in CODECS.items()}

_FLAG_COMPRESS_RESIDUES = 1

# Keep products/shifts comfortably inside int64.
MAX_MODULUS = 1 << 40


# --------------------------------------------------------------------------
# Split / merge
# --------------------------------------------------------------------------

def _as_int_array(V) -> np.ndarray:
    V = np.asarray(V)
    if V.ndim == 1:
        V = V[:, None]
    if V.ndim != 2:
        raise ValueError(f"expected an (N, D) array, got shape {V.shape}")
    if not np.issubdtype(V.dtype, np.integer):
        raise TypeError(
            f"expected integer data, got {V.dtype}; quantize floats first "
            "(see modcomp.quantize)"
        )
    return V.astype(np.int64, copy=False)


def as_per_dim(value, d: int, name: str) -> np.ndarray:
    """Broadcast a scalar or length-``d`` sequence to an int64 vector."""
    arr = np.asarray(value, dtype=np.int64)
    if arr.ndim == 0:
        arr = np.full(d, int(arr), dtype=np.int64)
    if arr.shape != (d,):
        raise ValueError(f"{name} must be a scalar or have length {d}, got {arr.shape}")
    return arr


def _check_modulus(m: np.ndarray) -> None:
    if np.any(m < 1):
        raise ValueError("modulus must be >= 1 in every dimension")
    if np.any(m > MAX_MODULUS):
        raise ValueError(f"modulus must be <= 2**40 in every dimension")


def split(V, m, offset=0):
    """Split ``V`` into ``(R, Q)`` with ``V == Q * m + R + offset``.

    ``R`` is in ``[0, m)`` per dimension; ``Q`` is the correction array.
    """
    V = _as_int_array(V)
    d = V.shape[1]
    m = as_per_dim(m, d, "modulus")
    c = as_per_dim(offset, d, "offset")
    _check_modulus(m)
    Q, R = np.divmod(V - c, m)
    return R, Q


def merge(R, Q, m, offset=0) -> np.ndarray:
    """Inverse of :func:`split`."""
    R = np.asarray(R, dtype=np.int64)
    Q = np.asarray(Q, dtype=np.int64)
    d = R.shape[1]
    return Q * as_per_dim(m, d, "modulus") + R + as_per_dim(offset, d, "offset")


def quantize(X, step: float) -> np.ndarray:
    """Quantize float data to integers on a uniform grid of spacing ``step``."""
    return np.rint(np.asarray(X, dtype=np.float64) / step).astype(np.int64)


def dequantize(V, step: float) -> np.ndarray:
    return np.asarray(V, dtype=np.float64) * step


# --------------------------------------------------------------------------
# Delta / zigzag / varint
# --------------------------------------------------------------------------

def delta(Q: np.ndarray) -> np.ndarray:
    """Row-wise first difference; row 0 is kept as-is."""
    if len(Q) == 0:
        return Q.copy()
    return np.diff(Q, axis=0, prepend=np.zeros((1, Q.shape[1]), dtype=Q.dtype))


def undelta(dQ: np.ndarray) -> np.ndarray:
    return np.cumsum(dQ, axis=0)


def zigzag(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.int64)
    return ((x << 1) ^ (x >> 63)).view(np.uint64)


def unzigzag(u: np.ndarray) -> np.ndarray:
    u = np.asarray(u, dtype=np.uint64)
    return ((u >> np.uint64(1)) ^ (np.uint64(0) - (u & np.uint64(1)))).view(np.int64)


def varint_encode(u: np.ndarray) -> bytes:
    """LEB128-encode an array of unsigned integers (vectorized)."""
    u = np.asarray(u, dtype=np.uint64).ravel()
    if u.size == 0:
        return b""
    nbytes = np.ones(u.size, dtype=np.int64)
    tmp = u >> np.uint64(7)
    while np.any(tmp):
        nbytes += tmp > 0
        tmp >>= np.uint64(7)
    starts = np.concatenate(([0], np.cumsum(nbytes)[:-1]))
    out = np.zeros(int(nbytes.sum()), dtype=np.uint8)
    for k in range(int(nbytes.max())):
        sel = nbytes > k
        b = ((u[sel] >> np.uint64(7 * k)) & np.uint64(0x7F)).astype(np.uint8)
        b |= (nbytes[sel] > k + 1).astype(np.uint8) << 7
        out[starts[sel] + k] = b
    return out.tobytes()


def varint_decode(buf: bytes, count: int) -> np.ndarray:
    """Decode exactly ``count`` LEB128 values from ``buf``."""
    if count == 0:
        if buf:
            raise ValueError("trailing bytes in varint stream")
        return np.zeros(0, dtype=np.uint64)
    b = np.frombuffer(buf, dtype=np.uint8)
    ends = np.flatnonzero(b < 0x80)
    if len(ends) != count or ends[-1] != len(b) - 1:
        raise ValueError("malformed varint stream")
    starts = np.concatenate(([0], ends[:-1] + 1))
    group = np.repeat(np.arange(count), ends - starts + 1)
    k = np.arange(len(b)) - starts[group]
    if k.max() > 9:
        raise ValueError("varint too long")
    parts = (b & 0x7F).astype(np.uint64) << (7 * k).astype(np.uint64)
    return np.add.reduceat(parts, starts)


def _put_varint(out: bytearray, n: int) -> None:
    out += varint_encode(np.array([n], dtype=np.uint64))


class _Reader:
    def __init__(self, buf: bytes):
        self.buf = memoryview(buf)
        self.pos = 0

    def take(self, n: int) -> bytes:
        if self.pos + n > len(self.buf):
            raise ValueError("truncated stream")
        chunk = bytes(self.buf[self.pos:self.pos + n])
        self.pos += n
        return chunk

    def varint(self) -> int:
        shift = result = 0
        while True:
            byte = self.take(1)[0]
            result |= (byte & 0x7F) << shift
            if byte < 0x80:
                return result
            shift += 7
            if shift > 63:
                raise ValueError("varint too long")

    def svarint(self) -> int:
        u = self.varint()
        return (u >> 1) ^ -(u & 1)


# --------------------------------------------------------------------------
# Bit packing for residues
# --------------------------------------------------------------------------

def residue_widths(m: np.ndarray) -> np.ndarray:
    """Bits per residue: ceil(log2 m), 0 when m == 1."""
    m = np.asarray(m, dtype=np.int64)
    return np.array([int(x - 1).bit_length() for x in m], dtype=np.int64)


def bitpack(R: np.ndarray, widths: np.ndarray) -> bytes:
    """Pack each column of R at its fixed width (column-major, LSB first)."""
    chunks = []
    for d, w in enumerate(widths):
        if w == 0:
            continue
        col = R[:, d].astype(np.uint64)
        bits = (col[:, None] >> np.arange(w, dtype=np.uint64)) & np.uint64(1)
        chunks.append(bits.astype(np.uint8).ravel())
    if not chunks:
        return b""
    return np.packbits(np.concatenate(chunks), bitorder="little").tobytes()


def bitunpack(buf: bytes, n: int, widths: np.ndarray) -> np.ndarray:
    total = int(n * widths.sum())
    bits = np.unpackbits(np.frombuffer(buf, dtype=np.uint8), bitorder="little")
    if len(bits) < total:
        raise ValueError("truncated residue block")
    R = np.zeros((n, len(widths)), dtype=np.int64)
    pos = 0
    for d, w in enumerate(widths):
        if w == 0:
            continue
        block = bits[pos:pos + n * w].reshape(n, w).astype(np.uint64)
        R[:, d] = (block << np.arange(w, dtype=np.uint64)).sum(axis=1).astype(np.int64)
        pos += n * w
    return R


def residue_at(blob: bytes, i: int) -> np.ndarray:
    """Random access: read residue vector ``i`` without decoding anything else.

    Only valid when residues were stored uncompressed (the default).
    """
    hdr, r = _read_header(blob)
    if hdr["flags"] & _FLAG_COMPRESS_RESIDUES:
        raise ValueError("residues are compressed; random access unavailable")
    n, widths = hdr["n"], residue_widths(hdr["modulus"])
    if not 0 <= i < n:
        raise IndexError(i)
    r.varint()  # residue block length
    base_bit = r.pos * 8
    out = np.zeros(len(widths), dtype=np.int64)
    col_start = 0
    data = r.buf
    for d, w in enumerate(widths):
        bit = base_bit + col_start + i * w
        val = 0
        for k in range(int(w)):
            byte = data[(bit + k) >> 3]
            val |= ((byte >> ((bit + k) & 7)) & 1) << k
        out[d] = val
        col_start += n * int(w)
    return out


# --------------------------------------------------------------------------
# Container
# --------------------------------------------------------------------------

def _compress(data: bytes, codec: str) -> bytes:
    if codec == "none":
        return data
    if codec == "zlib":
        return zlib.compress(data, 9)
    if codec == "lzma":
        return lzma.compress(data, preset=9 | lzma.PRESET_EXTREME)
    raise ValueError(f"unknown codec {codec!r}; choose from {sorted(CODECS)}")


def _decompress(data: bytes, codec: str) -> bytes:
    if codec == "none":
        return data
    if codec == "zlib":
        return zlib.decompress(data)
    if codec == "lzma":
        return lzma.decompress(data)
    raise ValueError(f"unknown codec {codec!r}")


def encode(V, m, offset=0, codec: str = "zlib", compress_residues: bool = False):
    """Encode ``V`` and return ``(blob, sizes)``.

    ``sizes`` breaks the blob into ``header``, ``residues`` and
    ``corrections`` byte counts.

    Layout::

        "MODC" u8:version u8:codec u8:flags
        varint:N varint:D  D x varint:modulus  D x zigzag-varint:offset
        varint:len  residue block   (bit-packed, column-major)
        varint:len  correction block (codec(varint(zigzag(delta(Q)))))
    """
    if codec not in CODECS:
        raise ValueError(f"unknown codec {codec!r}; choose from {sorted(CODECS)}")
    V = _as_int_array(V)
    n, d = V.shape
    m = as_per_dim(m, d, "modulus")
    c = as_per_dim(offset, d, "offset")
    R, Q = split(V, m, c)

    flags = _FLAG_COMPRESS_RESIDUES if compress_residues else 0
    head = bytearray(MAGIC)
    head += bytes([VERSION, CODECS[codec], flags])
    _put_varint(head, n)
    _put_varint(head, d)
    head += varint_encode(m.astype(np.uint64))
    head += varint_encode(zigzag(c))

    residues = bitpack(R, residue_widths(m))
    if compress_residues:
        residues = _compress(residues, codec)
    corrections = _compress(varint_encode(zigzag(delta(Q))), codec)

    out = bytearray(head)
    _put_varint(out, len(residues))
    out += residues
    _put_varint(out, len(corrections))
    out += corrections

    sizes = {
        # Length prefixes are counted as header overhead.
        "header": len(out) - len(residues) - len(corrections),
        "residues": len(residues),
        "corrections": len(corrections),
        "total": len(out),
    }
    return bytes(out), sizes


def _read_header(blob: bytes):
    r = _Reader(blob)
    if r.take(4) != MAGIC:
        raise ValueError("not a modcomp blob")
    version, codec_id, flags = r.take(3)
    if version != VERSION:
        raise ValueError(f"unsupported version {version}")
    if codec_id not in _CODEC_NAMES:
        raise ValueError(f"unknown codec id {codec_id}")
    n = r.varint()
    d = r.varint()
    m = np.array([r.varint() for _ in range(d)], dtype=np.int64)
    c = np.array([r.svarint() for _ in range(d)], dtype=np.int64)
    hdr = {"n": n, "d": d, "codec": _CODEC_NAMES[codec_id], "flags": flags,
           "modulus": m, "offset": c}
    return hdr, r


def decode(blob: bytes) -> np.ndarray:
    hdr, r = _read_header(blob)
    n, d, codec = hdr["n"], hdr["d"], hdr["codec"]
    m, c = hdr["modulus"], hdr["offset"]

    residues = r.take(r.varint())
    if hdr["flags"] & _FLAG_COMPRESS_RESIDUES:
        residues = _decompress(residues, codec)
    R = bitunpack(residues, n, residue_widths(m))

    corrections = _decompress(r.take(r.varint()), codec)
    dQ = unzigzag(varint_decode(corrections, n * d)).reshape(n, d)
    Q = undelta(dQ)
    if r.pos != len(blob):
        raise ValueError("trailing bytes after correction block")
    return merge(R, Q, m, c)
