"""Experiment: compress vector arrays by stripping a modulus from every vector
and storing a correction vector.

    v = q * m + r + offset,   0 <= r < m

``r`` (residue) is stored at fixed width; ``q`` (correction) is delta-coded and
entropy-compressed. See README.md.
"""

from .core import (
    decode,
    dequantize,
    encode,
    merge,
    quantize,
    residue_at,
    split,
)
from .analysis import baselines, choose_parameters, estimate

__all__ = [
    "baselines",
    "choose_parameters",
    "decode",
    "dequantize",
    "encode",
    "estimate",
    "merge",
    "quantize",
    "residue_at",
    "split",
]
