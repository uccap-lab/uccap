"""Percentile-based capability indices for non-normal data.

Source: Practical process capability indices workflows (Jiang & Yang, IJAMT 2026), Section 4.
P0.135 and P99.865 replace mu -/+ 3 sigma (99.73% coverage) and the median P50 replaces mu.
"""
from __future__ import annotations

import math

from .indices import Spec

LOW_Q, MID_Q, HIGH_Q = 0.00135, 0.5, 0.99865


def quantiles(dist) -> tuple[float, float, float]:
    return tuple(float(v) for v in dist.ppf([LOW_Q, MID_Q, HIGH_Q]))


def percentile_indices(dist, spec: Spec) -> dict[str, float]:
    lo, med, hi = quantiles(dist)
    out: dict[str, float] = {}
    parts = []
    if spec.usl is not None:
        out["CNpu"] = (spec.usl - med) / (hi - med)
        parts.append(out["CNpu"])
    if spec.lsl is not None:
        out["CNpl"] = (med - spec.lsl) / (med - lo)
        parts.append(out["CNpl"])
    out["CNpk"] = min(parts)
    out["CNp"] = (spec.usl - spec.lsl) / (hi - lo) if spec.is_bilateral else math.nan
    if spec.target is not None:
        tau = math.sqrt(((hi - lo) / 6) ** 2 + (med - spec.target) ** 2)
        if spec.is_bilateral:
            out["CNpm"] = (spec.usl - spec.lsl) / (6 * tau)
            out["CNpmk"] = min(spec.usl - med, med - spec.lsl) / (3 * tau)
        else:
            limit = spec.usl if spec.usl is not None else spec.lsl
            out["CNpm"] = abs(limit - spec.target) / (3 * tau)
            out["CNpmk"] = abs(limit - med) / (3 * tau)
    return out


class EmpiricalQuantiles:
    """Non-parametric quantiles of the sample (``numpy.quantile`` with ``method``).

    The 0.135% and 99.865% quantiles need roughly a thousand observations or
    more; with small samples they collapse onto the minimum and maximum.
    """

    name = "empirical"

    def __init__(self, x, method: str = "linear"):
        import numpy as np
        self._x = np.asarray(x, float)
        self.method = method

    def ppf(self, q):
        import numpy as np
        return np.quantile(self._x, q, method=self.method)
