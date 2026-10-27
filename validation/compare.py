"""Compare uccap with the independent R reference values in validation/reference.json.

    python validation/compare.py          # prints a Markdown report (used for docs/validation.md)
"""
import json
import pathlib

import numpy as np

from uccap.workflow.constants import c4, d2
from uccap.workflow.fitting import fit_distribution
from uccap.workflow.normality import anderson_darling, check_normality
from uccap.workflow.percentile import LOW_Q, MID_Q, HIGH_Q, EmpiricalQuantiles
from uccap.workflow.sigma import sigma_amr, sigma_overall

HERE = pathlib.Path(__file__).resolve().parent
Q = [LOW_Q, MID_Q, HIGH_Q]


def rel(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-12)))


def rows():
    """Yield (group, item, uccap value, R value, difference, tolerance, kind)."""
    ref = json.loads((HERE / "reference.json").read_text())
    k = ref["constants"]
    for n, v in zip(k["n"], k["d2"]):
        yield "constants", f"d2({n})", d2(n), v, abs(d2(n) - v), 5e-4, "abs"
    for n, v in zip(k["c4_n"], k["c4"]):
        yield "constants", f"c4({n})", c4(n), v, abs(c4(n) - v), 1e-9, "abs"
    for name, r in ref["sets"].items():
        x = np.loadtxt(HERE / "data" / f"{name}.csv")
        yield name, "mean", x.mean(), r["mean"], rel(x.mean(), r["mean"]), 1e-12, "rel"
        yield name, "sigma overall", sigma_overall(x), r["sd"], rel(sigma_overall(x), r["sd"]), 1e-12, "rel"
        yield name, "sigma within (AMR)", sigma_amr(x), r["sigma_amr"], rel(sigma_amr(x), r["sigma_amr"]), 5e-4, "rel"
        eq = EmpiricalQuantiles(x).ppf(Q)
        yield name, "empirical P0.135/P50/P99.865", eq, r["empirical_quantiles"], rel(eq, r["empirical_quantiles"]), 1e-12, "rel"
        a_stat, a_p = anderson_darling(x)
        yield name, "Anderson-Darling A²", a_stat, r["anderson_stat"], abs(a_stat - r["anderson_stat"]), 1e-9, "abs"
        yield name, "Anderson-Darling p", a_p, r["anderson_p"], abs(a_p - r["anderson_p"]), 1e-6, "abs"
        sp = check_normality(x, "shapiro").p_value
        yield name, "Shapiro-Wilk p", sp, r["shapiro_p"], abs(sp - r["shapiro_p"]), 1e-4, "abs"
        lp = check_normality(x, "ks").p_value
        yield name, "Lilliefors p (Monte Carlo vs Dallal-Wilkinson)", lp, r["lilliefors_p"], abs(lp - r["lilliefors_p"]), 0.05, "abs"
        for fam, f in r["fits"].items():
            fit = fit_distribution(x, fam)
            yield name, f"{fam}: log-likelihood", fit.loglik, f["loglik"], f["loglik"] - fit.loglik, 1e-3, "R − uccap"
            qs = fit.dist.ppf(Q)
            yield name, f"{fam}: fitted P0.135/P50/P99.865", qs, f["quantiles"], rel(qs, f["quantiles"]), 2e-3, "rel"


def _num(a):
    a = float(a)
    return f"{a:.4f}" if a == 0 or abs(a) >= 1e-3 else f"{a:.3e}"


def fmt(v):
    """At most 4 decimal places (scientific notation below 0.001)."""
    if np.ndim(v):
        return " / ".join(_num(a) for a in np.ravel(v))
    return _num(v)


def main():
    out = ["| Data | Quantity | uccap | R | Difference | Tolerance | Pass |", "|---|---|---|---|---|---|---|"]
    n_ok = n = 0
    for g, item, u, r, diff, tol, kind in rows():
        ok = diff <= tol
        n += 1; n_ok += ok
        out.append(f"| {g} | {item} | {fmt(u)} | {fmt(r)} | {diff:.2e} ({kind}) | {tol:g} | {'yes' if ok else '**no**'} |")
    print(f"{n_ok} of {n} comparisons within tolerance\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
