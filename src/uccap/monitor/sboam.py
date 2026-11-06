"""Sequential Bayesian operational acceptability monitoring (SBOAM) of one characteristic.

After a process has been approved, every completed subgroup updates an exponentially discounted
Normal-Inverse-Gamma posterior of the process state theta_t = (mu_t, sigma_t), and the posterior
acceptability probability

    pi_t = P(q(theta_t) >= q0 | D_1:t),   q(theta) = Phi((USL - mu)/sigma) - Phi((LSL - mu)/sigma)

decides between continued operation (pi_t >= gamma_A), continued monitoring, and intervention
(pi_t <= gamma_R, only from subgroup t_min on). Equation, table and section numbers refer to the article.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy import stats

ZONES = ("continue", "monitor", "intervene")
CATEGORIES = ("persistent low-confidence", "recovered transient", "no intervention-zone entry", "borderline final")


@dataclass(frozen=True)
class MonitorPrior:
    """Normal-Inverse-Gamma prior (mu0, kappa0, alpha0, beta0); it is not discounted (Eq. 25)."""
    mu0: float
    kappa0: float
    alpha0: float
    beta0: float

    @classmethod
    def tolerance_aware(cls, lsl: float, usl: float, nominal: Optional[float] = None, kappa0: float = 1e-3,
                        alpha0: float = 2.0) -> "MonitorPrior":
        """Tolerance-aware prior of the industrial replay (Sec. 4.4): mu0 = nominal, s0 = (USL - LSL)/8,
        beta0 = max(s0^2 (alpha0 - 1), 1e-12). Without a nominal the middle of the tolerance is used;
        with kappa0 = 1e-3 the prior centre carries almost no weight."""
        s0 = (usl - lsl) / 8.0
        mu0 = float(nominal) if nominal is not None else (lsl + usl) / 2.0
        return cls(mu0, kappa0, alpha0, max(s0 * s0 * (alpha0 - 1.0), 1e-12))


def conformance(mu, sigma, lsl: float, usl: float):
    """Specification-conformance probability q(theta) of Eq. (18); an infinite limit is left out."""
    mu, sigma = np.asarray(mu, float), np.asarray(sigma, float)
    hi = stats.norm.cdf((usl - mu) / sigma) if math.isfinite(usl) else 1.0
    lo = stats.norm.cdf((lsl - mu) / sigma) if math.isfinite(lsl) else 0.0
    return hi - lo


def cpk_for_q0(q0: float) -> float:
    """Cpk of a centred normal process whose conformance is q0, Eq. (17): q0 = 0.997 gives about 0.99."""
    return float(stats.norm.ppf((1 + q0) / 2) / 3)


@dataclass
class MonitorResult:
    """Trajectory of one monitored characteristic, one row per subgroup in ``frame``.

    ``frame`` columns: t (subgroup number from 1), n, mean, sd (of the subgroup), mu_post and sigma_post
    (posterior means), weight (discounted evidence W_t), pi (pi_t), q_pred (mean of q over the posterior
    draws, for reference only) and zone (continue, monitor or intervene).
    """
    frame: pd.DataFrame
    lsl: float
    usl: float
    settings: dict
    prior: MonitorPrior
    name: str = ""
    notes: list = field(default_factory=list)

    @property
    def pi(self) -> np.ndarray:
        return self.frame["pi"].to_numpy()

    @property
    def zone(self) -> str:
        """Zone of the last subgroup."""
        return str(self.frame["zone"].iloc[-1])

    @property
    def pi_final(self) -> float:
        return float(self.frame["pi"].iloc[-1])

    @property
    def pi_min(self) -> float:
        """Lowest pi_t over the decision-eligible period t >= t_min (nan when the record is shorter)."""
        eligible = self.frame.loc[self.frame["t"] >= self.settings["t_min"], "pi"]
        return float(eligible.min()) if len(eligible) else float("nan")

    @property
    def intervention(self) -> Optional[int]:
        """First posterior intervention time tau = inf{t >= t_min: pi_t <= gamma_R}, Eq. (33); None if none."""
        hit = self.frame.index[self.frame["zone"] == "intervene"]
        return int(self.frame.loc[hit[0], "t"]) if len(hit) else None

    @property
    def category(self) -> str:
        """Monitoring category of Table 14, assigned in the order persistent, recovered, no entry, borderline."""
        s, pi_t, low = self.settings, self.pi_final, self.pi_min
        if pi_t <= s["gamma_r"]:
            return CATEGORIES[0]
        if low == low and low <= s["gamma_r"] and pi_t >= s["gamma_a"]:
            return CATEGORIES[1]
        if not (low <= s["gamma_r"]):
            return CATEGORIES[2]
        return CATEGORIES[3]

    def report(self) -> str:
        s, f = self.settings, self.frame
        tau = self.intervention
        lsl, usl = _fmt(self.lsl), _fmt(self.usl)
        lines = [
            f"Operational acceptability monitoring{': ' + self.name if self.name else ''}",
            f"{'Specification':>26}  LSL {lsl}, USL {usl}",
            f"{'Conformance target q0':>26}  {s['q0']:g} (Cpk about {cpk_for_q0(s['q0']):.4f} for a centred process)",
            f"{'Zones':>26}  continue if pi >= {s['gamma_a']:g}, intervene if pi <= {s['gamma_r']:g} "
            f"(t >= {s['t_min']})",
            f"{'Forgetting factor':>26}  {s['forgetting']:g} (half-life {_half_life(s['forgetting'])} subgroups)",
            f"{'Prior':>26}  mu0 {self.prior.mu0:.4g}, kappa0 {self.prior.kappa0:g}, alpha0 {self.prior.alpha0:g}, "
            f"beta0 {self.prior.beta0:.4g}",
            f"{'Subgroups':>26}  {len(f)} of size {s['subgroup_size']}, {s['n_draws']} posterior draws each",
            f"{'Final pi':>26}  {self.pi_final:.4f} -> {self.zone}",
            f"{'Lowest pi (t >= t_min)':>26}  " + ("none (fewer than t_min subgroups)" if self.pi_min != self.pi_min
                                                   else f"{self.pi_min:.4f}"),
            f"{'First intervention':>26}  {'none' if tau is None else f'subgroup {tau}'}",
            f"{'Category':>26}  {self.category}",
            f"{'Subgroups per zone':>26}  " + ", ".join(f"{z} {int((f['zone'] == z).sum())}" for z in ZONES),
        ]
        for n in self.notes:
            lines.append(f"{'Note':>26}  {n}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.report()

    def plot(self, title=None):
        """Subgroup means and the pi_t trajectory with its zones (needs matplotlib)."""
        from ..plot import monitor_figure
        return monitor_figure(self, title=title)


def _fmt(v) -> str:
    return "none" if v is None or not math.isfinite(v) else f"{v:g}"


def _half_life(lam: float) -> str:
    return "infinite" if lam >= 1 else f"{math.log(0.5) / math.log(lam):.1f}"


def _limit(v, default: float) -> float:
    if v is None:
        return default
    v = float(v)
    return default if math.isnan(v) else v


def _helmert(g: np.ndarray) -> np.ndarray:
    """Within-subgroup contrasts that are independent N(0, sigma^2) for normal data, whatever the mean."""
    c, k = np.cumsum(g), np.arange(1, len(g))
    return (c[:-1] - k * g[1:]) / np.sqrt(k * (k + 1))


def subgroups(data, subgroup_size: int = 5) -> list:
    """Subgroups in production order: consecutive blocks of a 1-D series (an incomplete last block is
    left out), or a list of arrays as given. Missing values are dropped inside each subgroup."""
    if isinstance(data, (list, tuple)) and data and np.ndim(data[0]) == 1:
        groups = [np.asarray(g, float) for g in data]
    else:
        x = np.asarray(pd.to_numeric(pd.Series(np.asarray(data, dtype=object).ravel()), errors="coerce"), float)
        x = x[np.isfinite(x)]
        k = len(x) // subgroup_size
        groups = [x[i * subgroup_size:(i + 1) * subgroup_size] for i in range(k)]
    groups = [g[np.isfinite(g)] for g in groups]
    return [g for g in groups if len(g)]


def monitor_study(data, lsl: Optional[float] = None, usl: Optional[float] = None, nominal: Optional[float] = None, *,
                  subgroup_size: int = 5, q0: float = 0.997, gamma_a: float = 0.95, gamma_r: float = 0.80,
                  forgetting: float = 0.95, t_min: int = 10, n_draws: int = 2000,
                  prior: Union[str, MonitorPrior, Sequence[float]] = "tolerance", gauge_sd: Optional[float] = None,
                  zero_lsl_gdt: bool = True, rng=0, name: str = "") -> MonitorResult:
    """Monitor one characteristic subgroup by subgroup (Algorithm 1 of the article).

    ``data`` holds the measurements in production order: a 1-D series cut into consecutive subgroups of
    ``subgroup_size`` (an incomplete last subgroup is left out), or a list of subgroups.

    The defaults are the validation configuration of the article (Table 2): q0 = 0.997, gamma_A = 0.95,
    gamma_R = 0.80, forgetting factor 0.95, t_min = 10, subgroups of 5. They illustrate the method; the
    article treats them as engineering and operational policy parameters to be set for the application.

    ``prior="tolerance"`` is the tolerance-aware prior of the industrial replay; a :class:`MonitorPrior`
    or a tuple (mu0, kappa0, alpha0, beta0) gives any other, for example (0, 1e-3, 2, 1) of the simulation
    study. ``n_draws`` is M in Eq. (27); the Monte Carlo standard error of pi_t is at most 1/(2 sqrt(M)), Eq. (28).

    ``gauge_sd`` (a known gauge standard deviation) evaluates q on sigma^2 - gauge_sd^2 for every posterior
    draw, the approximate gauge-corrected variant of Sec. 5.5 (a sensitivity check, not an errors-in-variables
    posterior).

    With one specification limit, q uses that limit only. A GD&T characteristic recorded as nominal 0,
    LSL 0 and USL > 0 is read as an upper limit only when ``zero_lsl_gdt`` is true (as in
    :func:`uccap.batch_study`). An upper limit only is coded with LSL = 0 for the tolerance-aware prior,
    whose scale is then USL / 8, as uccap codes GD&T characteristics for the UC-Cap score; this is a uccap
    convention, the article's replay used two-sided tolerances. A lower limit only needs an explicit prior.
    """
    lsl, usl = _limit(lsl, -math.inf), _limit(usl, math.inf)
    nominal = None if nominal is None or (isinstance(nominal, float) and math.isnan(nominal)) else float(nominal)
    notes = []
    if not (math.isfinite(lsl) or math.isfinite(usl)):
        raise ValueError("no specification limits")
    if math.isfinite(lsl) and math.isfinite(usl) and usl <= lsl:
        raise ValueError("USL must be greater than LSL")
    if not 0 < gamma_r < gamma_a < 1:
        raise ValueError("thresholds must satisfy 0 < gamma_r < gamma_a < 1")
    if not 0 < q0 < 1:
        raise ValueError("q0 must be in (0, 1)")
    if not 0 < forgetting <= 1:
        raise ValueError("forgetting must be in (0, 1]")
    if subgroup_size < 1 or n_draws < 1 or t_min < 1:
        raise ValueError("subgroup_size, n_draws and t_min must be at least 1")
    gdt = zero_lsl_gdt and lsl == 0 and nominal == 0 and math.isfinite(usl) and usl > 0
    prior_lsl = lsl
    if gdt:  # 0 is a physical bound, not a limit
        lsl, prior_lsl = -math.inf, 0.0
        notes.append("LSL = 0 read as a GD&T bound (upper limit only); prior scale USL / 8")
    elif not math.isfinite(lsl) and usl > 0 and isinstance(prior, str):
        prior_lsl = 0.0  # upper limit only: coded with LSL = 0 as for GD&T characteristics
        notes.append("upper limit only: prior scale USL / 8 (LSL coded 0)")
    if isinstance(prior, MonitorPrior):
        pr = prior
    elif isinstance(prior, str):
        if prior != "tolerance":
            raise ValueError('prior must be "tolerance", a MonitorPrior or (mu0, kappa0, alpha0, beta0)')
        if not (math.isfinite(prior_lsl) and math.isfinite(usl)):
            raise ValueError("the tolerance-aware prior needs both limits; give prior=(mu0, kappa0, alpha0, beta0)")
        pr = MonitorPrior.tolerance_aware(prior_lsl, usl, nominal)
    else:
        pr = MonitorPrior(*map(float, prior))
    if pr.kappa0 <= 0 or pr.alpha0 <= 0 or pr.beta0 <= 0:
        raise ValueError("kappa0, alpha0 and beta0 must be positive")
    groups = subgroups(data, subgroup_size)
    if not groups:
        raise ValueError(f"fewer than {subgroup_size} values: no complete subgroup")
    gen = np.random.default_rng(rng)
    g2 = float(gauge_sd) ** 2 if gauge_sd else 0.0
    W = xbar = Q = 0.0
    rows = []
    for t, g in enumerate(groups, 1):
        n_t, m = len(g), float(g.mean())
        S = float(((g - m) ** 2).sum())
        Wm, Qm = forgetting * W, forgetting * Q           # Algorithm 1, step 2: discount the evidence
        W = Wm + n_t
        delta = m - xbar                                   # step 3: pooled update, Eq. (24)
        xbar = xbar + n_t * delta / W
        Q = Qm + S + Wm * n_t * delta * delta / W
        kappa = pr.kappa0 + W                              # Eq. (25)
        mu_p = (pr.kappa0 * pr.mu0 + W * xbar) / kappa
        alpha = pr.alpha0 + W / 2
        beta = pr.beta0 + Q / 2 + pr.kappa0 * W * (xbar - pr.mu0) ** 2 / (2 * kappa)
        s2 = beta / gen.gamma(alpha, 1.0, n_draws)         # Eq. (26): sigma^2 ~ Inv-Gamma(alpha, beta)
        mu = mu_p + np.sqrt(s2 / kappa) * gen.standard_normal(n_draws)
        s2q = np.maximum(s2 - g2, 1e-12 * s2) if g2 else s2
        q = conformance(mu, np.sqrt(s2q), lsl, usl)
        pi = float(np.mean(q >= q0))                       # Eq. (27)
        if t >= t_min and pi <= gamma_r:                   # Eq. (20)-(21)
            zone = "intervene"
        elif pi >= gamma_a:
            zone = "continue"
        else:
            zone = "monitor"
        rows.append({"t": t, "n": n_t, "mean": m, "sd": float(np.sqrt(S / (n_t - 1))) if n_t > 1 else np.nan,
                     "mu_post": mu_p, "sigma_post": float(math.sqrt(beta / (alpha - 1))) if alpha > 1 else np.nan,
                     "weight": W, "pi": pi, "q_pred": float(q.mean()), "zone": zone})
    if math.isfinite(lsl) != math.isfinite(usl) and not gdt and math.isfinite(lsl):
        notes.append("lower limit only: q uses that limit only")
    if gauge_sd:
        notes.append(f"gauge-corrected: q evaluated on sigma^2 - {float(gauge_sd):g}^2 (Sec. 5.5)")
    resid = np.concatenate([_helmert(g) for g in groups if len(g) > 1] or [np.empty(0)])
    if len(resid) >= 8 and np.ptp(resid) > 0:
        from ..workflow.normality import anderson_darling
        p = anderson_darling(resid)[1]
        if p < 0.05:
            notes.append(f"the within-subgroup variation fails the Anderson-Darling normality test (p = {p:.4f}); "
                         "the Gaussian model is a working approximation, read pi with care near the boundary")
    if len(groups) < t_min:
        notes.append(f"only {len(groups)} subgroups: intervention is not allowed before t_min = {t_min}")
    settings = dict(subgroup_size=subgroup_size, q0=q0, gamma_a=gamma_a, gamma_r=gamma_r, forgetting=forgetting,
                    t_min=t_min, n_draws=n_draws, gauge_sd=gauge_sd, rng=rng)
    return MonitorResult(pd.DataFrame(rows), lsl, usl, settings, pr, name, notes)
