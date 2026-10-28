"""Plots: overview of one study, the figures by category, a batch overview and monitoring.

Run:  pip install "uccap[plot]"; python examples/plots_example.py
Writes PNG files to docs/figures/.
"""
import pathlib

import matplotlib
import numpy as np

import uccap

matplotlib.use("Agg")
out = pathlib.Path(__file__).resolve().parent.parent / "docs" / "figures"
out.mkdir(parents=True, exist_ok=True)

# A normal dimension, spec 4.62 +/- 0.10: approved by the threshold rule, not by the risk rule.
x = np.random.default_rng(42).normal(loc=4.625, scale=0.023, size=32)
study = uccap.capability_study(x, lsl=4.52, usl=4.72, target=4.62)
decision = uccap.risk_decision(study, c0=1.33, cost_ratio=19)
study.plot(decision, title="D101: normal data").savefig(out / "overview_normal.png", dpi=120)

# The same study, one figure per question.
# The UC-Cap figure needs the trained model parameters, which are not included in this release.
kinds = ("outliers", "stability", "capability", "approval") + (("uccap",) if uccap.hybrid.model_available() else ())
for kind in kinds:
    study.plot(decision, kind=kind).savefig(out / f"{kind}_normal.png", dpi=120)
# The outlier screen with every value written on its point.
study.plot(kind="outliers", show_data=True).savefig(out / "outliers_values_normal.png", dpi=120)

# A skewed one-sided characteristic: lognormal fit, bootstrap failure probability.
y = np.random.default_rng(7).lognormal(mean=np.log(0.02), sigma=0.5, size=60)
study2 = uccap.capability_study(y, usl=0.12)
decision2 = uccap.risk_decision(study2, c0=1.33, alpha=0.05, rng=0)
study2.plot(decision2, title="Flatness: lognormal, USL only").savefig(out / "overview_lognormal.png", dpi=120)

# Many dimensions from the template.
template = pathlib.Path(__file__).resolve().parent.parent / "templates" / "multiple_dimensions.csv"
result = uccap.batch_study(template, c0=1.33, cost_ratio=19)
result.plot().figure.savefig(out / "batch_overview.png", dpi=120)

# Monitoring after approval: the simulation setting of the monitoring article, shift to 1.5 after subgroup 60.
mu = np.r_[np.zeros(60), np.full(60, 1.5)]
z = (mu[:, None] + np.random.default_rng(1).standard_normal((120, 5))).ravel()
mon = uccap.monitor_study(z, lsl=-4, usl=4, prior=(0, 1e-3, 2, 1))
mon.plot(title="Persistent shift after subgroup 60").savefig(out / "monitor_trajectory.png", dpi=120)
# Many dimensions: 40 simulated dimensions (tolerance 10 +/- 0.10), stable or with a persistent shift, a transient
# shift or a growing spread at a random subgroup.
t = np.repeat(np.arange(1, 121), 5)
g = np.random.default_rng(2026)
dims = {}
for i in range(40):
    kind = ("stable", "stable", "shift", "transient", "spread")[i % 5]
    sd, mean, start = g.uniform(0.018, 0.030), 10 + g.uniform(-0.015, 0.015), int(g.integers(30, 100))
    mu, s = np.full(t.size, mean), np.full(t.size, sd)
    if kind == "shift":
        mu = mu + g.uniform(0.02, 0.05) * (t > start)
    elif kind == "transient":
        mu = mu + g.uniform(0.02, 0.05) * ((t > start) & (t <= start + int(g.integers(8, 20))))
    elif kind == "spread":
        s = sd * np.where(t > start, 1 + 0.8 * (t - start) / (120 - start), 1)
    dims[f"D{101 + i}"] = g.normal(mu, s)
batch = uccap.batch_monitor(dims, {d: {"lsl": 9.9, "usl": 10.1, "target": 10.0} for d in dims})
batch.plot().savefig(out / "monitor_overview.png", dpi=120)
print("figures written to", out)
