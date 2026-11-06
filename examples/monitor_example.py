"""Monitoring example: is the approved process still acceptable, subgroup by subgroup?

Run:  python examples/monitor_example.py
Writes examples/monitor_results.html (and monitor_overview.png, M102_monitor.png with matplotlib).

The data are simulated: 600 consecutive parts (120 subgroups of 5) for four dimensions with a
tolerance of 10.00 +/- 0.10, in the scenarios of the article's simulation study scaled to this tolerance.
"""
from pathlib import Path

import numpy as np
import pandas as pd

import uccap

rng = np.random.default_rng(2026)
out = Path(__file__).parent
t = np.repeat(np.arange(1, 121), 5)                   # subgroup number of each part
sd = 0.025                                            # (USL - LSL) / 8, as sigma = 1 with LSL = -4, USL = 4

data = pd.DataFrame({
    "M101": rng.normal(10.0, sd, t.size),                                     # stable
    "M102": rng.normal(10.0 + 0.0375 * (t > 60), sd),                         # shift to 1.5 sigma at t = 61
    "M103": rng.normal(10.0 + 0.030 * ((t >= 45) & (t < 65)), sd),            # transient 1.2 sigma
    "M104": rng.normal(10.0, sd * np.where(t > 60, 1 + 0.8 * (t - 60) / 60, 1)),  # spread grows to 1.8 sigma
})
specs = {d: {"lsl": 9.9, "usl": 10.1, "target": 10.0} for d in data}

result = uccap.batch_monitor(data, specs)        # q0 = 0.997, gamma_A = 0.95, gamma_R = 0.80, t_min = 10

pd.set_option("display.width", 200)
print(result.table(["dimension", "subgroups", "Ppk_normal", "pi_final", "pi_min", "first_intervention",
                    "zone", "category"]).to_string(index=False))
print()
print(result.report("M102"))

result.to_html(out / "monitor_results.html", title="Acceptability monitoring example")
print("\nwritten: monitor_results.html")

try:
    import matplotlib
    matplotlib.use("Agg")
    result.plot().savefig(out / "monitor_overview.png", dpi=150, bbox_inches="tight")
    result.results["M102"].plot().savefig(out / "M102_monitor.png", dpi=150, bbox_inches="tight")
    print("written: monitor_overview.png, M102_monitor.png")
except ImportError:
    pass
