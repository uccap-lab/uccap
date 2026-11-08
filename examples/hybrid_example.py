"""UC-Cap hybrid risk for one dimension (simulated measurements).

The trained UC-Cap parameters are not included in this release. Give
their location with uccap.hybrid.set_model_path(path) or the
environment variable UCCAP_MODEL. Without them this example shows the statistical part only.
"""
import math

import numpy as np

import uccap.hybrid
from uccap import capability_study, hybrid_risk, risk_decision
from uccap.hybrid import compute_features, mechanism

x = np.random.default_rng(11).normal(10.012, 0.011, 20)   # 20 parts, tolerance 10.00 +/- 0.05

# The statistical anchor and the mechanism signals need no model parameters.
f = compute_features(x, 9.95, 10.05, 10.0)
z = f["A_z_stat_133"]
print(f"Ppk {f['A_Ppk']:.4f}   SE(Ppk) {f['A_SE_Ppk']:.4f}   anchor z {z:.4f}   "
      f"anchor only P(Ppk < 1.33) {1 / (1 + math.exp(-z)):.4f}")
print("mechanism signals:", mechanism.signals(f).active or "none")

if uccap.hybrid.model_available():
    r = hybrid_risk(x, lsl=9.95, usl=10.05, nominal=10.0)
    print()
    print(r)
else:
    print("\nUC-Cap score skipped: the trained parameters are not included in this release.")

# compare with the risk-calibrated approval of the same data
study = capability_study(x, lsl=9.95, usl=10.05, target=10.0)
print()
print(risk_decision(study, alpha=0.05).report())
