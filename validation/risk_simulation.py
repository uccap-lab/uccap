"""Reproduce the simulation results of the risk-calibration article with uccap.

Jiang & Yang, Risk-calibrated process capability approval with finite samples,
IJAMT 144:7189-7206 (2026), doi:10.1007/s00170-026-18284-2, Section 5.

    python validation/risk_simulation.py            # B = 1,000,000 replications, about 10 s
    python validation/risk_simulation.py --quick    # B = 100,000

Setting of Section 5.1: normal data, LSL = -T, USL = T with T = 4, centred process, sigma = T / (3 Cpk),
C0 = 1.33, n = 32. Table 1 gives the margin k for each cost ratio. Table 2 compares the expected loss of
the deterministic rule (accept if Cpk_hat >= C0) and the cost-sensitive rule (alpha = 1 / (1 + lambda))
for a process just below the requirement, where every acceptance is a false acceptance and costs lambda
(c_FR = 1): EL = lambda x P(accept). It is reproduced with true Cpk = 1.32, the near-threshold value of
the article's Figure 2. The article used B = 12,000 replications, so its values carry a Monte Carlo
standard error; the script reports the difference in standard errors.

The cost-sensitive rule is evaluated through uccap.risk: a sample is accepted when its estimate reaches
min_accepted(n, cost_ratio=lambda), which is the same as risk_decision(...).accept (checked below).
"""
from __future__ import annotations

import argparse
import math

import numpy as np

from uccap.risk import k_from_cost_ratio, min_accepted, risk_decision

T, C0, N = 4.0, 1.33, 32
TABLE1 = {1: 0.0, 4: 0.842, 9: 1.282, 19: 1.645, 99: 2.326}          # lambda -> k
# lambda -> (deterministic EL, risk-calibrated EL, reduction %), as printed in Table 2
TABLE2 = {2: (0.793, 0.483, 39.16), 5: (2.037, 0.565, 72.29), 10: (4.021, 0.553, 86.26),
          20: (8.052, 0.527, 93.46), 50: (20.038, 0.479, 97.61), 100: (40.567, 0.375, 99.08)}
B_ARTICLE = 12_000


def estimates(true_cpk: float, n: int, b: int, rng) -> np.ndarray:
    """Plug-in Cpk estimates of b normal samples of size n (centred process)."""
    sigma = T / (3 * true_cpk)
    out = np.empty(b)
    for i in range(0, b, 20_000):  # chunks keep memory small
        x = rng.normal(0.0, sigma, (min(20_000, b - i), n))
        m, s = x.mean(1), x.std(1, ddof=1)
        out[i:i + len(m)] = np.minimum(T - m, m + T) / (3 * s)
    return out


def table1() -> list[tuple]:
    return [(lam, k, round(k_from_cost_ratio(lam), 3)) for lam, k in TABLE1.items()]


def table2(b: int = 1_000_000, true_cpk: float = 1.32, seed: int = 2026) -> list[dict]:
    est = estimates(true_cpk, N, b, np.random.default_rng(seed))
    p_det = float(np.mean(est >= C0))
    rows = []
    for lam, (det, cal, red) in TABLE2.items():
        p_cal = float(np.mean(est >= min_accepted(N, c0=C0, cost_ratio=lam)))
        row = {"lambda": lam, "det": lam * p_det, "det_article": det, "cal": lam * p_cal, "cal_article": cal}
        for key, p in (("det", p_det), ("cal", p_cal)):
            se = lam * math.sqrt(p * (1 - p) / B_ARTICLE)  # Monte Carlo SE of the article's value
            row[key + "_z"] = (row[key + "_article"] - row[key]) / se
        row["reduction"] = 100 * (1 - row["cal"] / row["det"])
        row["reduction_article"] = red
        rows.append(row)
    return rows


def check_rule_equivalence(n_samples: int = 200, seed: int = 1) -> None:
    """min_accepted gives the same decisions as risk_decision."""
    rng = np.random.default_rng(seed)
    for est in rng.uniform(1.2, 2.0, n_samples):
        for lam in TABLE2:
            assert risk_decision(float(est), n=N, c0=C0, cost_ratio=lam).accept == \
                (est >= min_accepted(N, c0=C0, cost_ratio=lam) - 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    b = 100_000 if ap.parse_args().quick else 1_000_000
    check_rule_equivalence()
    print("Table 1: margin k = z_(1-alpha), alpha = 1/(1+lambda)")
    print(f"{'lambda':>8} {'article':>9} {'uccap':>9}")
    for lam, k, u in table1():
        print(f"{lam:>8} {k:>9.3f} {u:>9.3f}")
    print(f"\nTable 2: expected loss, n = {N}, true Cpk = 1.32, B = {b:,} (article: B = {B_ARTICLE:,})")
    print(f"{'lambda':>6} | {'deterministic':^24} | {'risk-calibrated':^24} | {'reduction %':^15}")
    print(f"{'':>6} | {'article':>7} {'uccap':>7} {'diff/SE':>8} | {'article':>7} {'uccap':>7} {'diff/SE':>8} |"
          f" {'article':>7} {'uccap':>7}")
    for r in table2(b):
        print(f"{r['lambda']:>6} | {r['det_article']:>7.3f} {r['det']:>7.3f} {r['det_z']:>8.2f} |"
              f" {r['cal_article']:>7.3f} {r['cal']:>7.3f} {r['cal_z']:>8.2f} |"
              f" {r['reduction_article']:>7.2f} {r['reduction']:>7.2f}")


if __name__ == "__main__":
    main()
