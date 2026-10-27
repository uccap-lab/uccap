from __future__ import annotations

import numpy as np

from ..workflow.fitting import THRESHOLD_FAMILIES, fit_distribution, refit_fixed_threshold
from ..workflow.indices import normal_indices
from ..workflow.percentile import EmpiricalQuantiles, percentile_indices
from ..workflow.sigma import sigma_overall


def bootstrap_cpk(study, n_boot: int = 1000, rng=None) -> np.ndarray:
    """Nonparametric bootstrap distribution of the study's capability estimate.

    Normal studies recompute Ppk (overall sigma) on each resample. Non-normal studies refit
    the selected distribution family and recompute CNpk; three-parameter families
    keep the original threshold estimate (see ``refit_fixed_threshold``).
    """
    rng = np.random.default_rng(rng)
    x = study.data
    out = np.empty(n_boot)
    for b in range(n_boot):
        xb = rng.choice(x, size=x.size, replace=True)
        if study.method == "normal":
            s = sigma_overall(xb)
            out[b] = normal_indices(float(xb.mean()), s, s, study.spec)["Ppk"]
        elif study.percentile_method == "empirical":
            out[b] = percentile_indices(EmpiricalQuantiles(xb), study.spec)["CNpk"]
        else:
            if study.fit.name in THRESHOLD_FAMILIES:
                fit = refit_fixed_threshold(xb, study.fit)
            else:
                fit = fit_distribution(xb, study.fit.name)
            out[b] = percentile_indices(fit.dist, study.spec)["CNpk"]
    return out


def bootstrap_p_fail(study, c0: float, n_boot: int = 1000, rng=None) -> float:
    """Bootstrap estimate of P(Cpk_true < C0 | data): share of resamples below C0."""
    draws = bootstrap_cpk(study, n_boot, rng)
    return float(np.mean(draws[np.isfinite(draws)] < c0))
