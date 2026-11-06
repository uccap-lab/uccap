"""Quick start: from raw measurements to a risk-calibrated approval decision.

Run:  python examples/quickstart.py
"""
import numpy as np

from uccap import capability_study, risk_decision

# Example 1: one dimension, 32 measurements in time order, spec 4.62 +/- 0.10.
x = np.random.default_rng(42).normal(loc=4.625, scale=0.023, size=32)
study = capability_study(x, lsl=4.52, usl=4.72, target=4.62)
print(study.report())

# Approve if Cpk >= 1.33, where a false accept costs 19x a false reject.
decision = risk_decision(study, c0=1.33, cost_ratio=19)
print(decision.report())

# Example 1b: after confirming the flagged points are special causes (e.g. a
# measurement error), recompute without them. The report lists what was excluded.
clean = study.without_outliers()
print(clean.report())
print(risk_decision(clean, c0=1.33, cost_ratio=19).report())

# Example 2: a one-sided characteristic (e.g. flatness, USL only) with skewed data.
y = np.random.default_rng(7).lognormal(mean=np.log(0.02), sigma=0.5, size=60)
study2 = capability_study(y, usl=0.12)
print()
print(study2.report())
print(risk_decision(study2, c0=1.33, alpha=0.05, rng=0).report())
