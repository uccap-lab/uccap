"""Normal-based process capability indices.

Source: Practical process capability indices workflows (Jiang & Yang, IJAMT 2026), Section 3.
Use sigma_within for Cp/Cpk/Cpm/Cpmk and sigma_overall for Pp/Ppk.

Bilateral asymmetric tolerances (target T not at the midpoint) use the starred
indices Cp*, Cpk*, Cpm*, Cpmk* of the workflow paper, Eqs. (3), (5), (9) and (13)
(``Spec.asymmetric = "paper"``, default), or the Pearn–Chen generalizations with
d* and A* (``"pearn-chen"``). Unilateral tolerances
(only USL or only LSL) reduce Cpk to Cpu or Cpl; Cp and Pp are undefined.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


ASYMMETRIC_FORMS = {"paper", "pearn-chen"}


@dataclass(frozen=True)
class Spec:
    lsl: Optional[float] = None
    usl: Optional[float] = None
    target: Optional[float] = None
    asymmetric: str = "paper"

    def __post_init__(self):
        if self.asymmetric not in ASYMMETRIC_FORMS:
            raise ValueError(f"asymmetric must be one of {sorted(ASYMMETRIC_FORMS)}")
        if self.lsl is None and self.usl is None:
            raise ValueError("at least one of lsl or usl is required")
        if self.lsl is not None and self.usl is not None and self.lsl >= self.usl:
            raise ValueError("lsl must be below usl")
        if self.target is not None and self.is_bilateral and not self.lsl < self.target < self.usl:
            raise ValueError("target must lie inside the specification limits "
                             "(for a GD&T characteristic with nominal 0 and LSL 0, leave LSL empty)")

    @property
    def is_bilateral(self) -> bool:
        return self.lsl is not None and self.usl is not None

    @property
    def midpoint(self) -> Optional[float]:
        return (self.lsl + self.usl) / 2 if self.is_bilateral else None

    @property
    def is_asymmetric(self) -> bool:
        return (self.is_bilateral and self.target is not None
                and not math.isclose(self.target, self.midpoint, rel_tol=0, abs_tol=1e-12 * (self.usl - self.lsl)))


def cpu(mu: float, sigma: float, usl: float) -> float:
    return (usl - mu) / (3 * sigma)


def cpl(mu: float, sigma: float, lsl: float) -> float:
    return (mu - lsl) / (3 * sigma)


def cp(sigma: float, spec: Spec) -> float:
    if not spec.is_bilateral:
        return math.nan
    if spec.is_asymmetric:
        d_star = min(spec.usl - spec.target, spec.target - spec.lsl)
        return d_star / (3 * sigma)
    return (spec.usl - spec.lsl) / (6 * sigma)


def _asym_shift(mu: float, spec: Spec) -> tuple[float, float]:
    """Return (d*, A*) for Pearn–Chen asymmetric indices."""
    t = spec.target
    du, dl = spec.usl - t, t - spec.lsl
    d_star = min(du, dl)
    a_star = max(d_star * (mu - t) / du, d_star * (t - mu) / dl)
    return d_star, a_star


def _paper_sides(mu: float, sigma: float, spec: Spec) -> tuple[float, float]:
    """(Cpl*, Cpu*) of the workflow paper, Eq. (5): zero once |T - mu| exceeds that side's tolerance."""
    t = spec.target
    shift = abs(t - mu)
    dl, du = t - spec.lsl, spec.usl - t
    lower = 0.0 if shift > dl else (dl - shift) / (3 * sigma)
    upper = 0.0 if shift > du else (du - shift) / (3 * sigma)
    return lower, upper


def cpk(mu: float, sigma: float, spec: Spec) -> float:
    if spec.is_asymmetric:
        if spec.asymmetric == "paper":
            return min(_paper_sides(mu, sigma, spec))
        d_star, a_star = _asym_shift(mu, spec)
        return (d_star - a_star) / (3 * sigma)
    values = []
    if spec.usl is not None:
        values.append(cpu(mu, sigma, spec.usl))
    if spec.lsl is not None:
        values.append(cpl(mu, sigma, spec.lsl))
    return min(values)


def cpm(mu: float, sigma: float, spec: Spec) -> float:
    """Taguchi index; needs a target. Unilateral form uses the distance from target to the limit."""
    t = spec.target
    if t is None:
        return math.nan
    if spec.is_asymmetric:
        if spec.asymmetric == "paper":
            d_star = min(spec.usl - t, t - spec.lsl)
            return d_star / (3 * math.sqrt(sigma ** 2 + (mu - t) ** 2))
        d_star, a_star = _asym_shift(mu, spec)
        return d_star / (3 * math.sqrt(sigma ** 2 + a_star ** 2))
    tau = math.sqrt(sigma ** 2 + (mu - t) ** 2)
    if spec.is_bilateral:
        return (spec.usl - spec.lsl) / (6 * tau)
    limit_dist = spec.usl - t if spec.usl is not None else t - spec.lsl
    return limit_dist / (3 * tau)


def cpmk(mu: float, sigma: float, spec: Spec) -> float:
    t = spec.target
    if t is None:
        return math.nan
    if spec.is_asymmetric:
        if spec.asymmetric == "paper":
            return min(_paper_sides(mu, sigma, spec)) / math.sqrt(1 + ((mu - t) / sigma) ** 2)
        d_star, a_star = _asym_shift(mu, spec)
        return (d_star - a_star) / (3 * math.sqrt(sigma ** 2 + a_star ** 2))
    tau = math.sqrt(sigma ** 2 + (mu - t) ** 2)
    dists = []
    if spec.usl is not None:
        dists.append(spec.usl - mu)
    if spec.lsl is not None:
        dists.append(mu - spec.lsl)
    return min(dists) / (3 * tau)


def normal_indices(mu: float, sigma_within: float, sigma_overall: float, spec: Spec) -> dict[str, float]:
    out = {
        "Cp": cp(sigma_within, spec),
        "Cpk": cpk(mu, sigma_within, spec),
        "Pp": cp(sigma_overall, spec),
        "Ppk": cpk(mu, sigma_overall, spec),
        "Cpm": cpm(mu, sigma_within, spec),
        "Cpmk": cpmk(mu, sigma_within, spec),
        "Ppm": cpm(mu, sigma_overall, spec),
        "Ppmk": cpmk(mu, sigma_overall, spec),
    }
    if spec.usl is not None:
        out["Cpu"] = cpu(mu, sigma_within, spec.usl)
        out["Ppu"] = cpu(mu, sigma_overall, spec.usl)
    if spec.lsl is not None:
        out["Cpl"] = cpl(mu, sigma_within, spec.lsl)
        out["Ppl"] = cpl(mu, sigma_overall, spec.lsl)
    return out
