"""Batch example: many dimensions, a spec table, coloured HTML and Excel output.

Run:  python examples/batch_example.py
Writes examples/batch_results.html and examples/batch_results.xlsx.
"""
from pathlib import Path

import numpy as np
import pandas as pd

import uccap

rng = np.random.default_rng(2026)
n = 32
out = Path(__file__).parent

# Wide layout: one column per dimension, rows in time order (simulated data).
data = pd.DataFrame({
    "D101": rng.normal(4.620, 0.020, n),                     # capable, centred
    "D102": rng.normal(4.655, 0.022, n),                     # off-centre: near the threshold
    "D103": rng.normal(12.000, 0.050, n),                    # incapable
    "D104": 0.005 + rng.lognormal(np.log(0.02), 0.5, n),     # flatness, skewed, USL only
    "D105": rng.normal(50.780, 0.015, n),                    # asymmetric tolerance
    "D106": np.round(rng.normal(8.00, 0.04, n), 1),          # coarse gauge resolution
    "D107": rng.normal(2.500, 0.010, n),                     # no spec row
})
data.loc[[3, 17], "D105"] = np.nan                           # two missing readings
data.loc[25, "D101"] = 4.71                                  # one suspicious reading

specs = pd.DataFrame({
    "Dimension": ["D101", "D102", "D103", "D104", "D105", "D106"],
    "Nominal":   [4.620, 4.620, 12.000, None, 50.780, 8.00],
    "LSL":       [4.520, 4.520, 11.900, None, 50.700, 7.80],
    "USL":       [4.720, 4.720, 12.100, 0.120, 50.880, 8.20],
})

results = uccap.batch_study(data, specs, c0=1.33, cost_ratio=19)

pd.set_option("display.width", 200)
print(results.table(["dimension", "yield", "normality", "shape", "best_fit", "method",
                     "capability", "p_fail", "threshold_decision", "decision"]).to_string(index=False))
print()
print(results.report("D102"))

results.to_html(out / "batch_results.html", title="Capability results (simulated data)")
results.to_excel(out / "batch_results.xlsx")
