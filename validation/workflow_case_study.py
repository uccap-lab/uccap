"""Reproduce the case study of the workflow article with uccap.

Jiang & Yang, Practical process capability indices workflows, IJAMT 143:3581-3599 (2026),
doi:10.1007/s00170-026-17782-7, Section 6, Tables 1 and 3-5.

    python validation/workflow_case_study.py

Dimension 105 of Table 2 (32 parts, target 23.580, tolerance +/-0.100) is used. For it the article
reports the overall
sigma and 18 within-sigma estimates (average moving range AMR and median moving range MMR, windows
2 to 10), and the resulting Pp, Ppk and Cp, Cpk for each estimator. Table 1 gives the constants
d2 and d4 (and c4) for subgroup sizes 2 to 10.

Result: all constants agree to the four printed decimals; all 19 sigma estimates agree to the four
printed decimals; Pp and Ppk agree exactly. The within-sigma Cp and Cpk agree within 0.7 %. Table 2
prints the readings to three decimals; the indices were presumably computed before that rounding,
which moves the within sigma in its fifth decimal (simulating the rounding gives a spread of about
0.5 % for the AMR estimate).
"""
from __future__ import annotations

import numpy as np

from uccap.workflow import Spec
from uccap.workflow.constants import c4, d2, d4
from uccap.workflow.sigma import sigma_amr, sigma_mmr, sigma_overall

TARGET, TOL = 23.580, 0.100
X105 = [23.632, 23.601, 23.605, 23.629, 23.614, 23.611, 23.641, 23.610, 23.625, 23.619, 23.599, 23.613,
        23.629, 23.625, 23.623, 23.620, 23.604, 23.608, 23.651, 23.617, 23.622, 23.640, 23.641, 23.604,
        23.591, 23.594, 23.616, 23.616, 23.606, 23.635, 23.588, 23.615]

# Table 1: w -> (d2, c4, d4)
TABLE1 = {2: (1.1284, 0.7979, 0.9539), 3: (1.6926, 0.8862, 1.5878), 4: (2.0588, 0.9213, 1.9783),
          5: (2.3259, 0.9400, 2.2569), 6: (2.5344, 0.9515, 2.4717), 7: (2.7044, 0.9594, 2.6455),
          8: (2.8472, 0.9650, 2.7908), 9: (2.9700, 0.9693, 2.9154), 10: (3.0775, 0.9727, 3.0242)}

# Tables 3-5, row 105: overall, then AMR w = 2..10, then MMR w = 2..10
ESTIMATORS = ["overall"] + [f"AMR{w}" for w in range(2, 11)] + [f"MMR{w}" for w in range(2, 11)]
SIGMA = [0.0153, 0.0151, 0.0156, 0.0152, 0.0153, 0.0155, 0.0156, 0.0155, 0.0154, 0.0154,
         0.0157, 0.0161, 0.0152, 0.0144, 0.0170, 0.0172, 0.0168, 0.0161, 0.0155]
CP = [2.172, 2.201, 2.130, 2.186, 2.172, 2.144, 2.130, 2.144, 2.158, 2.158,
      2.117, 2.064, 2.186, 2.308, 1.955, 1.932, 1.978, 2.064, 2.144]
CPK = [1.368, 1.386, 1.342, 1.377, 1.368, 1.350, 1.342, 1.350, 1.359, 1.359,
       1.333, 1.300, 1.377, 1.454, 1.231, 1.217, 1.246, 1.300, 1.350]


def sigmas(x=X105) -> list[float]:
    x = np.asarray(x, float)
    return ([sigma_overall(x)] + [sigma_amr(x, w) for w in range(2, 11)]
            + [sigma_mmr(x, w) for w in range(2, 11)])


def rows() -> list[tuple]:
    """(estimator, sigma, Cp, Cpk) from uccap next to the article's values."""
    spec = Spec(lsl=TARGET - TOL, usl=TARGET + TOL, target=TARGET)
    mean = float(np.mean(X105))
    out = []
    for name, s, s_a, cp_a, cpk_a in zip(ESTIMATORS, sigmas(), SIGMA, CP, CPK):
        cp = (spec.usl - spec.lsl) / (6 * s)
        cpk = min(spec.usl - mean, mean - spec.lsl) / (3 * s)
        out.append((name, s, s_a, cp, cp_a, cpk, cpk_a))
    return out


def constants() -> list[tuple]:
    return [(w, d2(w), a2, c4(w), a4c, d4(w), a4) for w, (a2, a4c, a4) in TABLE1.items()]


def main():
    print("Table 1: constants (article, uccap)")
    for w, u2, a2, uc, ac, u4, a4 in constants():
        print(f"  w={w:<2}  d2 {a2:.4f} {u2:.4f}   c4 {ac:.4f} {uc:.4f}   d4 {a4:.4f} {u4:.4f}")
    print("\nTables 3-5, dimension 105 (article, uccap)")
    print(f"  {'estimator':<9} {'sigma':>15} {'Cp / Pp':>15} {'Cpk / Ppk':>15}")
    worst = 0.0
    for name, s, sa, cp, cpa, cpk, cpka in rows():
        dev = max(abs(cp / cpa - 1), abs(cpk / cpka - 1))
        worst = max(worst, dev if name != "overall" else 0)
        flag = "" if round(s, 4) == sa else "  <- sigma differs"
        print(f"  {name:<9} {sa:.4f} {s:.4f}   {cpa:.3f} {cp:.3f}   {cpka:.3f} {cpk:.3f}   {100 * dev:4.2f} %{flag}")
    print(f"\nlargest relative difference in Cp / Cpk: {100 * worst:.2f} % (see the note at the top)")


if __name__ == "__main__":
    main()
