"""Control-chart unbiasing constants used by the within-sigma estimators.

Source: Practical process capability indices workflows (Jiang & Yang, IJAMT 2026,
doi:10.1007/s00170-026-17782-7), Table 1, and standard SPC tables.
"""
from __future__ import annotations

import math

# Expected range of n standard-normal observations.
D2 = {2: 1.128379, 3: 1.692569, 4: 2.058751, 5: 2.325929, 6: 2.534413,
      7: 2.704357, 8: 2.847201, 9: 2.970026, 10: 3.077505}

# Median range of n standard-normal observations (median moving range): the root of
# n * integral phi(x) [Phi(x + r) - Phi(x)]^(n-1) dx = 1/2, computed numerically; agrees with
# Table 1 of the workflow article to the four decimals printed there.
D4 = {2: 0.953873, 3: 1.587788, 4: 1.978320, 5: 2.256882, 6: 2.471652,
      7: 2.645452, 8: 2.790841, 9: 2.915438, 10: 3.024202}


def c4(n: int) -> float:
    """Bias-correction constant for the sample standard deviation."""
    if n < 2:
        raise ValueError("c4 requires n >= 2")
    return math.sqrt(2.0 / (n - 1)) * math.exp(math.lgamma(n / 2) - math.lgamma((n - 1) / 2))


def d2(n: int) -> float:
    try:
        return D2[n]
    except KeyError:
        raise ValueError(f"d2 is tabulated for n in 2..10, got {n}") from None


def d4(n: int) -> float:
    try:
        return D4[n]
    except KeyError:
        raise ValueError(f"d4 is tabulated for n in 2..10, got {n}") from None
