"""Reproduce the simulation results of the monitoring article with uccap.monitor.

Jiang & Yang, Bayesian operational acceptability monitoring for manufacturing process control,
IJAMT (accepted 2026), Sections 5.3 and 5.4.

    python validation/monitor_simulation.py            # B = 1000 replications per scenario, about 2 minutes
    python validation/monitor_simulation.py --quick    # B = 200

Setting of Section 5.3: T = 120 subgroups of n = 5, change point t0 = 60, in-control process mu = 0,
sigma = 1, LSL = -4, USL = 4, q0 = 0.997, gamma_A = 0.95, gamma_R = 0.80, t_min = 10, forgetting
factor 0.95, M = 500 posterior draws, prior (mu0, kappa0, alpha0, beta0) = (0, 1e-3, 2, 1).

Two tables are compared. Table 4 (change-detection convention): the probability of any intervention in
the stable scenario and, for the persistent shift to mu = 1.0, the probability of any intervention and
the median delay from t0. Table 11 (true acceptability state): the probability of any intervention while
the process stays acceptable (stable, shifts to 0.5 and 1.0, a 20-subgroup transient at 1.2), and the
detection probability and median delay from the first unacceptable subgroup for the unacceptable
scenarios (persistent shift to 1.5, 10-subgroup transient at 1.5, sigma rising from 1.0 to 1.8 after t0).
The gradual-drift scenario is not reproduced because its path is not specified in the article.

The article used B = 1000 replications, so its values carry a Monte Carlo standard error of up to
about 0.016 for a probability; the script reports both values side by side.
"""
from __future__ import annotations

import argparse

import numpy as np

from uccap.monitor import monitor_study

T, N, T0, LSL, USL = 120, 5, 60, -4.0, 4.0
SETTINGS = dict(subgroup_size=N, q0=0.997, gamma_a=0.95, gamma_r=0.80, forgetting=0.95, t_min=10, n_draws=500,
                prior=(0.0, 1e-3, 2.0, 1.0))
TRANSIENT_START = 45   # the representative transient path of Table 3 starts at subgroup 45


def path(scenario: str):
    """Mean and standard deviation of every subgroup t = 1..T, and the first unacceptable subgroup."""
    t = np.arange(1, T + 1)
    mu, sd = np.zeros(T), np.ones(T)
    first_bad = None
    if scenario.startswith("shift"):
        mu[t > T0] = float(scenario.split()[1])
    elif scenario == "transient 1.2":
        mu[(t >= TRANSIENT_START) & (t < TRANSIENT_START + 20)] = 1.2
    elif scenario == "transient 1.5":
        mu[(t > T0) & (t <= T0 + 10)] = 1.5
    elif scenario == "variance":
        sd = np.where(t > T0, 1.0 + 0.8 * (t - T0) / (T - T0), 1.0)
    from uccap.monitor import conformance
    bad = conformance(mu, sd, LSL, USL) < SETTINGS["q0"]
    if bad.any():
        first_bad = int(t[bad][0])
    return mu, sd, first_bad


def replicate(scenario: str, b: int, seed: int = 2026) -> dict:
    mu, sd, first_bad = path(scenario)
    rng = np.random.default_rng(seed)
    any_action, delays_t0, delays_bad, detected = 0, [], [], 0
    for _ in range(b):
        x = (mu[:, None] + sd[:, None] * rng.standard_normal((T, N))).ravel()
        r = monitor_study(x, LSL, USL, rng=rng, **SETTINGS)
        zone = r.frame["zone"].to_numpy()
        act = np.flatnonzero(zone == "intervene") + 1
        any_action += len(act) > 0
        after = act[act > T0]
        if len(after):
            delays_t0.append(after[0] - (T0 + 1))
        if first_bad is not None:
            hit = act[act >= first_bad]
            if len(hit):
                detected += 1
                delays_bad.append(hit[0] - first_bad)
    out = {"any": any_action / b, "delay_t0": float(np.median(delays_t0)) if delays_t0 else None}
    if first_bad is not None:
        out.update(detect=detected / b, delay=float(np.median(delays_bad)) if delays_bad else None, first_bad=first_bad)
    return out


# Article values: Table 4 (SBOAM rows) and Table 11 (SBOAM column).
TABLE4 = {"stable": {"any": 0.015}, "shift 1.0": {"any": 0.841, "delay_t0": 20}}
TABLE11_A = {"stable": 0.015, "shift 0.5": 0.017, "shift 1.0": 0.874, "transient 1.2": 0.867}
TABLE11_B = {"shift 1.5": (1.00, 6), "transient 1.5": (0.88, 6), "variance": (1.00, 9)}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--quick", action="store_true", help="B = 200 replications")
    p.add_argument("--b", type=int, help="replications per scenario (default 1000)")
    args = p.parse_args(argv)
    b = args.b or (200 if args.quick else 1000)
    print(f"B = {b} replications per scenario, M = {SETTINGS['n_draws']} draws\n")
    res = {s: replicate(s, b) for s in ("stable", "shift 0.5", "shift 1.0", "transient 1.2", "shift 1.5",
                                         "transient 1.5", "variance")}
    print("Table 4 (change-detection convention)")
    print(f"{'scenario':<16}{'P(any) uccap':>14}{'article':>9}{'delay uccap':>13}{'article':>9}")
    for s, ref in TABLE4.items():
        d = res[s]["delay_t0"]
        print(f"{s:<16}{res[s]['any']:>14.3f}{ref['any']:>9.3f}"
              + (f"{d:>13.0f}{ref['delay_t0']:>9d}" if "delay_t0" in ref and d is not None else ""))
    print("\nTable 11 (a): process stays acceptable, probability of any intervention")
    print(f"{'scenario':<16}{'uccap':>8}{'article':>9}")
    for s, ref in TABLE11_A.items():
        print(f"{s:<16}{res[s]['any']:>8.3f}{ref:>9.3f}")
    print("\nTable 11 (b): process becomes unacceptable, detection probability / median delay")
    print(f"{'scenario':<16}{'first bad t':>12}{'uccap':>12}{'article':>12}")
    for s, (det, delay) in TABLE11_B.items():
        r = res[s]
        mine = f"{r['detect']:.2f}/{r['delay']:.0f}" if r["delay"] is not None else f"{r['detect']:.2f}/-"
        print(f"{s:<16}{r['first_bad']:>12d}{mine:>12}{f'{det:.2f}/{delay}':>12}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
