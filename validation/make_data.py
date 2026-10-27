"""Generate the validation data sets (fixed seed) as CSV files in validation/data/."""
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
rng = np.random.default_rng(20261027)
n = 50
SETS = {
    "normal": rng.normal(10.0, 0.5, n),
    "lognormal": rng.lognormal(0.0, 0.4, n),
    "weibull": 1.0 * rng.weibull(2.0, n),
    "gamma": rng.gamma(3.0, 0.5, n),
    "exponential": rng.exponential(0.5, n),
    "lev": rng.gumbel(0.0, 1.0, n),
    "sev": -rng.gumbel(0.0, 1.0, n),
    "logistic": rng.logistic(5.0, 0.3, n),
}
for name, x in SETS.items():
    np.savetxt(HERE / "data" / f"{name}.csv", np.round(x, 6), fmt="%.6f")
print("written", len(SETS), "data sets")
