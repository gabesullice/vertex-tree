"""Experiment: compress vector arrays by stripping a modulus from every vector
and storing a correction vector. See README.md.

* ``modcomp.sector``: vectors with components in [0, 1]; the unit cube is cut
  into sectors of width 10**-k and each vector is stored as sector + in-sector
  correction.
* ``modcomp.core`` / ``modcomp.analysis``: an earlier integer variant,
  ``v = q * m + r + offset``.
"""

from . import sector

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
    "sector",
    "split",
]
