"""Model inputs of UC-Cap V4.2, computed from one sample of measurements.

Source: A hybrid statistical learning framework for capability-based decision support in
manufacturing (Jiang & Yang, IJAMT 2026, doi:10.1007/s00170-026-18833-9), Sections 6 and 7.5.

Every quantity uses the sample passed in (the "A side" in the paper). In the training data a
tolerance with an upper limit only (GD&T) is coded with LSL = 0 and nominal = 0, and its Ppk is the
upper index Ppu; Pp is not computed for it (NaN, filled with the training median by the model's
preprocessing, as for any missing input), and the distance to the nearest limit is the smaller of
the distances to LSL = 0 and to USL. uccap codes such tolerances the same way. A lower limit
alone is not supported. Pp and Ppk follow the rule of the training data: if the
Anderson-Darling test passes (p >= 0.05) they are the normal overall indices with the sample
standard deviation, otherwise empirical percentile indices from the 0.135%, 50% and 99.865%
sample quantiles (linear interpolation). The target does not enter.

A_Best_Fit is "normal" for a passing AD test. For a failing test uccap tries the named
distributions of the model inputs (lognormal, weibull, gamma, logistic; two parameters, the
first three with the location fixed at 0), takes the lowest AICc (as in the workflow article) and
keeps it if its own Anderson-Darling goodness-of-fit test is not rejected at 0.05, otherwise
A_Best_Fit is "empirical". The model inputs Pp and Ppk stay the empirical percentile indices, as in
the training data; the percentile indices from the 0.135%, 50% and 99.865% quantiles of the named
distribution are returned in addition as Fit_Pp and Fit_Ppk (not model inputs). The article
does not state how the training table chose between a named distribution and "empirical"; the
goodness-of-fit rule is uccap's. Critical values of the goodness-of-fit test for estimated
parameters (D'Agostino and Stephens, Goodness-of-Fit Techniques, 1986, Ch. 4): lognormal, the
normal test on log(x); weibull, the extreme-value test on log(x), A2 (1 + 0.2/sqrt(n)) < 0.757;
logistic, A2 (1 + 0.25/n) < 0.660; gamma, A2 below the value for the fitted shape (0.786 at shape 1
down to 0.752 for a large shape).

Skewness and kurtosis are the moment ratios m3/m2^1.5 and m4/m2^2 (kurtosis 3 for a normal
distribution). SE(Ppk) is the standard deviation of ``n_boot`` nonparametric bootstrap Ppk values,
each computed with the same rule (normal or empirical) as the original sample.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from ..workflow.fitting import fit_distribution
from ..workflow.normality import anderson_darling

C0 = 1.33

NUMERIC = ("nominal", "USL", "LSL", "Spec_Width", "A_Ppk", "A_Pp", "A_mean", "A_sd", "A_Skewness",
           "A_Kurtosis", "A_Dist_to_LSL", "A_Dist_to_USL", "A_Dist_to_nearest_spec", "A_P_value",
           "A_SE_Ppk", "A_margin_to_threshold")
CATEGORICAL = ("A_normality", "A_Best_Fit")
ANCHOR = "A_z_stat_133"

# Named distributions among the model's A_Best_Fit levels, tried for non-normal samples.
NAMED_FITS = ("lognormal", "weibull", "gamma", "logistic")

# Gamma, shape known only through its estimate: 5% critical values of A2 by shape
# (D'Agostino and Stephens 1986, Table 4.21).
_GAMMA_A2 = ((1, 0.786), (2, 0.768), (3, 0.762), (4, 0.759), (5, 0.758), (6, 0.757), (8, 0.755),
             (10, 0.754), (20, 0.753), (math.inf, 0.752))


def _indices(lo: float, mid: float, hi: float, lsl: float, usl: float,
             upper_only: bool = False) -> tuple[float, float]:
    """(Pp, Ppk) from the 0.135%, 50% and 99.865% quantiles."""
    if not hi > lo:
        return float("nan"), float("nan")
    sides = [(usl - mid) / (hi - mid) if hi > mid else np.nan,
             np.nan if upper_only else (mid - lsl) / (mid - lo) if mid > lo else np.nan]
    pp = float("nan") if upper_only else float((usl - lsl) / (hi - lo))
    return pp, float(np.nanmin(sides))


def pp_ppk(x: np.ndarray, lsl: float, usl: float, empirical: bool = False,
           dist: Optional[str] = None, upper_only: bool = False) -> tuple[float, float]:
    """(Pp, Ppk) for the tolerance [lsl, usl]: normal, empirical percentile, or percentile from the
    named distribution ``dist`` fitted to ``x``. ``upper_only``: Ppk is the upper index Ppu and Pp is
    not computed (NaN)."""
    if dist is not None:
        try:
            fit = fit_distribution(x, dist)
        except Exception:
            return float("nan"), float("nan")
        return _indices(*fit.dist.ppf([0.00135, 0.5, 0.99865]), lsl, usl, upper_only)
    if empirical:
        return _indices(*np.quantile(x, [0.00135, 0.5, 0.99865]), lsl, usl, upper_only)
    s = x.std(ddof=1)
    if not s > 0:
        return float("nan"), float("nan")
    mu = x.mean()
    ppu = (usl - mu) / (3 * s)
    if upper_only:  # no Pp for an upper limit alone, as in the training data
        return float("nan"), float(ppu)
    return float((usl - lsl) / (6 * s)), float(min(ppu, (mu - lsl) / (3 * s)))


def bootstrap_se_ppk(x, lsl: float, usl: float, empirical: bool = False, n_boot: int = 200, rng=0,
                     upper_only: bool = False) -> float:
    """Standard deviation of the nonparametric bootstrap distribution of Ppk."""
    x = np.asarray(x, float)
    rng = np.random.default_rng(rng)
    draws = np.array([pp_ppk(rng.choice(x, x.size, replace=True), lsl, usl, empirical, upper_only=upper_only)[1]
                      for _ in range(n_boot)])
    draws = draws[np.isfinite(draws)]
    return float(draws.std(ddof=1)) if draws.size > 1 else float("nan")


def _a2(u: np.ndarray) -> float:
    """Anderson-Darling statistic from the fitted CDF values of the sample."""
    u = np.clip(np.sort(u), 1e-300, 1 - 1e-16)
    n = u.size
    i = np.arange(1, n + 1)
    return float(-n - np.sum((2 * i - 1) * (np.log(u) + np.log1p(-u[::-1]))) / n)


def fit_adequate(x: np.ndarray, fit) -> bool:
    """Anderson-Darling goodness of fit at 0.05 for a fitted named distribution."""
    n = x.size
    if fit.name == "lognormal":
        return anderson_darling(np.log(x))[1] >= 0.05
    a2 = _a2(fit.dist.cdf(x))
    if fit.name == "weibull":
        return a2 * (1 + 0.2 / math.sqrt(n)) < 0.757
    if fit.name == "logistic":
        return a2 * (1 + 0.25 / n) < 0.660
    if fit.name == "gamma":
        shape = fit.dist.args[0] if fit.dist.args else fit.dist.kwds["a"]
        return a2 < next(c for k, c in _GAMMA_A2 if shape <= k)
    raise ValueError(f"no goodness-of-fit rule for {fit.name!r}")


def best_fit_label(x: np.ndarray, empirical: bool) -> str:
    """A_Best_Fit: "normal" when the AD test passes; otherwise the named distribution with the
    lowest AICc if its goodness-of-fit test passes, else "empirical"."""
    if not empirical:
        return "normal"
    fits = []
    for name in NAMED_FITS:
        try:
            fit = fit_distribution(x, name)
        except Exception:
            continue
        if math.isfinite(fit.loglik):
            fits.append(fit)
    if not fits:
        return "empirical"
    best = min(fits, key=lambda f: f.aicc)
    return best.name if fit_adequate(x, best) else "empirical"


def compute_features(x, lsl: float, usl: float, nominal: Optional[float] = None, *,
                     n_boot: int = 200, rng=0) -> dict:
    """UC-Cap V4.2 inputs for one sample with a bilateral tolerance [lsl, usl], or an upper limit only.

    ``nominal`` (the drawing nominal) defaults to the midpoint of the tolerance. A tolerance with
    an upper limit only (a GD&T characteristic such as flatness or position, bounded below by 0) is
    coded as in the training data: LSL = 0, nominal = 0 unless given, and Ppk = Ppu. LSL = 0 with
    nominal 0, the usual record of such a characteristic, is read the same way.
    """
    from ..workflow.pipeline import _coerce
    x, unreadable = _coerce(x)
    if unreadable:
        shown = ", ".join(f"obs. {i} ({v!r})" for i, v in unreadable[:5])
        raise ValueError(f"{len(unreadable)} entries are not numbers: {shown}")
    x = x[np.isfinite(x)]
    if x.size < 8:
        raise ValueError("at least 8 measurements are needed")
    if np.ptp(x) == 0:
        raise ValueError("all measurements are identical; check the measurement resolution or the data entry")
    # GD&T as recorded in the training data: nominal 0, LSL 0, USL > 0 (or LSL left empty).
    upper_only = usl is not None and (lsl is None or (lsl == 0 and nominal == 0 and usl > 0))
    if upper_only:
        if usl <= 0:
            raise ValueError("an upper limit only is coded with LSL = 0, so it needs USL > 0")
        lsl, nominal = 0.0, 0.0 if nominal is None else nominal
    if lsl is None or usl is None:
        raise ValueError("UC-Cap needs both limits, or an upper limit only (coded LSL = 0, Ppk = Ppu)")
    L, U = float(lsl), float(usl)
    if L >= U:
        raise ValueError("lsl must be below usl")
    if nominal is None:
        nominal = (L + U) / 2

    mu, sd = float(x.mean()), float(x.std(ddof=1))
    m2 = float(np.mean((x - mu) ** 2))
    skew = float(np.mean((x - mu) ** 3) / m2 ** 1.5) if m2 > 0 else float("nan")
    kurt = float(np.mean((x - mu) ** 4) / m2 ** 2) if m2 > 0 else float("nan")
    p_ad = anderson_darling(x)[1]
    empirical = p_ad < 0.05
    pp, ppk = pp_ppk(x, L, U, empirical, upper_only=upper_only)
    d_lsl, d_usl = mu - L, U - mu
    fit = best_fit_label(x, empirical)
    se = bootstrap_se_ppk(x, L, U, empirical, n_boot, rng, upper_only=upper_only)
    named = fit not in ("normal", "empirical")
    fit_pp, fit_ppk = pp_ppk(x, L, U, dist=fit, upper_only=upper_only) if named else (float("nan"), float("nan"))

    return {
        "nominal": float(nominal),
        "USL": U,
        "LSL": L,
        "Spec_Width": U - L,
        "A_Ppk": ppk,
        "A_Pp": pp,
        "A_mean": mu,
        "A_sd": sd,
        "A_Skewness": skew,
        "A_Kurtosis": kurt,
        "A_Dist_to_LSL": d_lsl,
        "A_Dist_to_USL": d_usl,
        "A_Dist_to_nearest_spec": min(d_lsl, d_usl),  # an upper limit alone: LSL coded as 0
        "A_P_value": p_ad,
        "A_SE_Ppk": se,
        "A_margin_to_threshold": ppk - C0,
        "A_normality": "Fail" if empirical else "Pass",
        "A_Best_Fit": fit,
        ANCHOR: (C0 - ppk) / se if se > 0 else float("nan"),
        # Not model inputs: percentile indices from the named best fit (NaN for normal / empirical).
        "Fit_Pp": fit_pp,
        "Fit_Ppk": fit_ppk,
        "Upper_only": upper_only,
    }
