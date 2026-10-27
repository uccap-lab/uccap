"""Best-fit distribution selection for non-normal data.

Source: Practical process capability indices workflows (Jiang & Yang, IJAMT 2026), Section 4.
Candidates are ranked by AICc (preferred for small samples), not by the largest
goodness-of-fit p-value. The chosen family should also be physically plausible
for the characteristic; pass ``candidates`` to restrict the search.

The candidate library covers the families offered by Minitab's Individual
Distribution Identification and JMP's nonnormal capability analysis. Only the
families marked * are tried by default; pass ``candidates=ALL_CANDIDATES`` or a
list of names to use the others.

=================  ===  ==================================================
name               k    notes
=================  ===  ==================================================
normal*            2
lognormal*         2    location fixed at 0 (positive data only)
lognormal3*        3    threshold estimated
exponential*       1    location fixed at 0 (positive data only)
exponential2       2    threshold estimated
weibull*           2    location fixed at 0 (positive data only)
weibull3*          3    threshold estimated
gamma*             2    location fixed at 0 (positive data only)
gamma3*            3    threshold estimated
sev*               2    smallest extreme value (Gumbel, left-skewed)
lev*               2    largest extreme value (Gumbel, right-skewed)
logistic           2
loglogistic        2    location fixed at 0 (positive data only)
loglogistic3       3    threshold estimated
johnson_su         4    unbounded Johnson family, maximum likelihood
johnson_sb         4    bounded Johnson family, maximum likelihood
shash              4    sinh-arcsinh (Jones & Pewsey, 2009), as in JMP
boxcox             3    normal after a Box-Cox transform (positive data only)
normal_mix2        5    mixture of 2 normals (EM), as in JMP
normal_mix3        8    mixture of 3 normals (EM), as in JMP
=================  ===  ==================================================

Box-Cox is compared on the original scale: its likelihood includes the Jacobian of
the transform, so its AICc is comparable with the other families. Minitab's Johnson
*transformation* (Chou's method) and JMP's quantile-matched Johnson fit are replaced
here by maximum-likelihood Johnson SU / SB fits.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy import optimize, stats

# Default set: covers stable dimensional data, skew in either direction and
# zero-bounded (GD&T) characteristics with at most 3 parameters.
DEFAULT_CANDIDATES = (
    "normal", "lognormal", "lognormal3", "weibull", "weibull3", "gamma", "gamma3",
    "exponential", "sev", "lev",
)

# Opt-in families. Not in the default set because, with typical capability sample
# sizes (n ~ 30-100), they either duplicate a default family (exponential2, logistic,
# log-logistic), are transformations rather than distributions (boxcox), or have
# 4+ parameters and fit almost anything while extrapolating the 0.135% / 99.865%
# tails poorly (Johnson, SHASH). A normal mixture signals two process streams or
# a shift, which should be investigated rather than capability-rated.
OPTIONAL_CANDIDATES = (
    "exponential2", "logistic", "loglogistic", "loglogistic3", "boxcox",
    "johnson_su", "johnson_sb", "shash", "normal_mix2", "normal_mix3",
)
ALL_CANDIDATES = DEFAULT_CANDIDATES + OPTIONAL_CANDIDATES


@dataclass(frozen=True)
class FitResult:
    name: str
    dist: object  # anything with logpdf, cdf and ppf (a frozen scipy distribution or similar)
    k: int
    n: int

    @property
    def loglik(self) -> float:
        return self._loglik

    @property
    def aic(self) -> float:
        return -2 * self.loglik + 2 * self.k

    @property
    def bic(self) -> float:
        return -2 * self.loglik + self.k * math.log(self.n)

    @property
    def aicc(self) -> float:
        if self.n - self.k - 1 <= 0:
            return math.inf
        return self.aic + 2 * self.k * (self.k + 1) / (self.n - self.k - 1)


def _result(name, dist, k, x) -> FitResult:
    res = FitResult(name, dist, k, x.size)
    object.__setattr__(res, "_loglik", float(np.sum(dist.logpdf(x))))
    return res


# ---------------------------------------------------------------- custom families

class _SHASHGen(stats.rv_continuous):
    """Sinh-arcsinh: sinh(delta * asinh(z) - eps) ~ N(0, 1), z = (x - loc) / scale."""

    def _argcheck(self, eps, delta):
        return np.isfinite(eps) & (delta > 0)

    def _pdf(self, z, eps, delta):
        s = np.sinh(delta * np.arcsinh(z) - eps)
        c = np.cosh(delta * np.arcsinh(z) - eps)
        return delta * c / np.sqrt(1 + z ** 2) * stats.norm.pdf(s)

    def _logpdf(self, z, eps, delta):
        a = delta * np.arcsinh(z) - eps
        return (np.log(delta) + np.log(np.cosh(a)) - 0.5 * np.log1p(z ** 2)
                + stats.norm.logpdf(np.sinh(a)))

    def _cdf(self, z, eps, delta):
        return stats.norm.cdf(np.sinh(delta * np.arcsinh(z) - eps))

    def _ppf(self, q, eps, delta):
        return np.sinh((np.arcsinh(stats.norm.ppf(q)) + eps) / delta)


shash = _SHASHGen(name="shash")


class BoxCoxNormal:
    """X such that boxcox(X, lam) ~ Normal(mu, sigma)."""

    def __init__(self, lam: float, mu: float, sigma: float):
        self.lam, self.mu, self.sigma = lam, mu, sigma

    def _t(self, x):
        x = np.asarray(x, float)
        return np.log(x) if abs(self.lam) < 1e-12 else (x ** self.lam - 1) / self.lam

    def logpdf(self, x):
        x = np.asarray(x, float)
        with np.errstate(invalid="ignore", divide="ignore"):
            out = stats.norm.logpdf(self._t(x), self.mu, self.sigma) + (self.lam - 1) * np.log(x)
        return np.where(x > 0, out, -np.inf)

    def cdf(self, x):
        x = np.asarray(x, float)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(x > 0, stats.norm.cdf(self._t(np.maximum(x, 1e-300)), self.mu, self.sigma), 0.0)

    def ppf(self, q):
        y = stats.norm.ppf(q, self.mu, self.sigma)
        if abs(self.lam) < 1e-12:
            return np.exp(y)
        base = self.lam * y + 1
        with np.errstate(invalid="ignore"):
            return np.where(base > 0, base ** (1 / self.lam), np.nan if self.lam > 0 else np.inf)


class NormalMixture:
    def __init__(self, weights, means, sds):
        self.w, self.m, self.s = (np.asarray(v, float) for v in (weights, means, sds))

    def logpdf(self, x):
        x = np.asarray(x, float)[..., None]
        return np.log(np.sum(self.w * stats.norm.pdf(x, self.m, self.s), axis=-1) + 1e-300)

    def cdf(self, x):
        x = np.asarray(x, float)[..., None]
        return np.sum(self.w * stats.norm.cdf(x, self.m, self.s), axis=-1)

    def ppf(self, q):
        q = np.atleast_1d(np.asarray(q, float))
        lo, hi = (self.m - 10 * self.s).min(), (self.m + 10 * self.s).max()
        out = np.array([optimize.brentq(lambda v, qi=qi: self.cdf(v) - qi, lo, hi) for qi in q])
        return out if out.size > 1 else out[0]


def _em_normal_mixture(x: np.ndarray, g: int, n_init: int = 5, max_iter: int = 500) -> NormalMixture:
    best, best_ll = None, -np.inf
    rng = np.random.default_rng(0)
    floor = 1e-3 * x.std()
    for init in range(n_init):
        qs = np.sort(rng.uniform(0.1, 0.9, g)) if init else (np.arange(g) + 0.5) / g
        m = np.quantile(x, qs)
        s = np.full(g, x.std() / g)
        w = np.full(g, 1 / g)
        ll_old = -np.inf
        for _ in range(max_iter):
            dens = w * stats.norm.pdf(x[:, None], m, s) + 1e-300
            ll = np.sum(np.log(dens.sum(axis=1)))
            r = dens / dens.sum(axis=1, keepdims=True)
            nk = r.sum(axis=0)
            w = nk / x.size
            m = (r * x[:, None]).sum(axis=0) / nk
            s = np.maximum(np.sqrt((r * (x[:, None] - m) ** 2).sum(axis=0) / nk), floor)
            if ll - ll_old < 1e-8:
                break
            ll_old = ll
        if ll > best_ll:
            best, best_ll = NormalMixture(w, m, s), ll
    return best


# ---------------------------------------------------------------- fitters

def _scipy(family, fix_loc: bool = False, positive: bool = False):
    def fit(x):
        if positive and x.min() <= 0:
            raise ValueError("requires positive data")
        params = family.fit(x, floc=0) if fix_loc else family.fit(x)
        k = len(params) - (1 if fix_loc else 0)
        return family(*params), k
    return fit


THRESHOLD_FAMILIES = {"lognormal3": stats.lognorm, "weibull3": stats.weibull_min,
                      "gamma3": stats.gamma, "loglogistic3": stats.fisk}


def refit_fixed_threshold(x, fit: "FitResult") -> "FitResult":
    """Refit shape and scale with the threshold held at ``fit``'s estimate.

    Used inside the bootstrap: resamples never fall below the original minimum, so
    the original threshold stays valid, and skipping the threshold search makes each
    refit about 30x faster. Threshold uncertainty is not propagated.
    """
    family = THRESHOLD_FAMILIES[fit.name]
    x = np.asarray(x, float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        params = family.fit(x, floc=fit.dist.kwds.get("loc", fit.dist.args[-2]))
    return _result(fit.name, family(*params), 3, x)


def _threshold(family):
    """Three-parameter fit with the threshold constrained below min(x).

    Unconstrained MLE often places the threshold above the smallest observation,
    which gives zero likelihood. Instead, profile the likelihood over the gap
    g = min(x) - threshold on a log scale and fit shape and scale for each gap.
    """
    def fit(x):
        lo, spread = x.min(), x.std()

        def profile(t):
            loc = lo - spread * math.exp(t)
            params = family.fit(x, floc=loc)
            nll = -np.sum(family.logpdf(x, *params))
            return (nll if np.isfinite(nll) else 1e300), params

        res = optimize.minimize_scalar(lambda t: profile(t)[0], bounds=(-12, 6), method="bounded")
        nll, params = profile(res.x)
        if nll >= 1e300:
            raise ValueError("threshold fit failed")
        return family(*params), 3
    return fit


def _fit_boxcox(x):
    if x.min() <= 0:
        raise ValueError("requires positive data")
    _, lam = stats.boxcox(x)
    y = stats.boxcox(x, lam)
    return BoxCoxNormal(float(lam), float(y.mean()), float(y.std(ddof=0))), 3


def _fit_shash(x):
    z = stats.skew(x)
    params = shash.fit(x, -0.5 * z, 1.0, loc=np.median(x), scale=x.std())
    return shash(*params), 4


def _mix(g):
    def fit(x):
        if x.size < 10 * g:
            raise ValueError("too few observations for a normal mixture")
        return _em_normal_mixture(x, g), 3 * g - 1
    return fit


FITTERS: dict[str, Callable] = {
    "normal": _scipy(stats.norm),
    "lognormal": _scipy(stats.lognorm, fix_loc=True, positive=True),
    "lognormal3": _threshold(stats.lognorm),
    "exponential": _scipy(stats.expon, fix_loc=True, positive=True),
    "exponential2": _scipy(stats.expon),
    "weibull": _scipy(stats.weibull_min, fix_loc=True, positive=True),
    "weibull3": _threshold(stats.weibull_min),
    "gamma": _scipy(stats.gamma, fix_loc=True, positive=True),
    "gamma3": _threshold(stats.gamma),
    "sev": _scipy(stats.gumbel_l),
    "lev": _scipy(stats.gumbel_r),
    "logistic": _scipy(stats.logistic),
    "loglogistic": _scipy(stats.fisk, fix_loc=True, positive=True),
    "loglogistic3": _threshold(stats.fisk),
    "johnson_su": _scipy(stats.johnsonsu),
    "johnson_sb": _scipy(stats.johnsonsb),
    "shash": _fit_shash,
    "boxcox": _fit_boxcox,
    "normal_mix2": _mix(2),
    "normal_mix3": _mix(3),
}


def fit_distribution(x, name: str) -> FitResult:
    x = np.asarray(x, float)
    if name not in FITTERS:
        raise ValueError(f"unknown distribution {name!r}; choose from {sorted(FITTERS)}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dist, k = FITTERS[name](x)
        return _result(name, dist, k, x)


def best_fit(x, candidates=DEFAULT_CANDIDATES, criterion: str = "aicc",
             failures: list | None = None) -> list[FitResult]:
    """Fit every candidate and return the successful fits sorted best first.

    ``candidates`` is a sequence of names, or "default" / "all". Families that
    cannot be fitted (e.g. positive-only families on data with negative values) are
    skipped; pass a list as ``failures`` to collect ``(name, reason)`` pairs.
    """
    if isinstance(candidates, str):
        candidates = {"default": DEFAULT_CANDIDATES, "all": ALL_CANDIDATES}[candidates]
    fits = []
    for name in candidates:
        try:
            fit = fit_distribution(x, name)
        except Exception as exc:
            if failures is not None:
                failures.append((name, str(exc) or type(exc).__name__))
            continue
        if math.isfinite(fit.loglik):
            fits.append(fit)
        elif failures is not None:
            failures.append((name, "zero likelihood at an observation"))
    if not fits:
        raise RuntimeError("no candidate distribution could be fitted")
    return sorted(fits, key=lambda f: getattr(f, criterion))
