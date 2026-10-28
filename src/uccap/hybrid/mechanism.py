"""Decision layer of UC-Cap: reason and action (score -> level -> reason -> action).

Source: A hybrid statistical learning framework for capability-based decision support in
manufacturing (Jiang & Yang, IJAMT 2026), Sections 5.1-5.3 and Table 1. The layer is
deterministic and never changes the predicted probability or the score. Following the paper it

* names the mechanism behind the risk: a specification-side mean shift, spread, a distributional
  deviation, several of these together (mixed mechanism), or none of them although the model
  indicates risk (latent model risk);
* raises a Low level to Medium only for a compound mechanism (two or more signals together);
  otherwise a Low level is Acceptable / Accept, as for D003 and D018 in Table 1, and a single
  signal is mentioned in the expanded text only;
* recommends re-centering for a mean shift, a combined response for a mixed mechanism, and review
  rather than an automatic adjustment when no clear statistical failure mode is found.

Each reason and action comes twice: a compact label in the style of the paper's Table 1, and an
expanded explanation with the statistics behind it.

The paper does not give numerical criteria for the signals. uccap uses quantities already in the
model inputs (the statistics of the workflow and risk articles), chosen so that the labels of the
consistent rows of Table 1 are reproduced:

  distribution   Anderson-Darling normality test rejected at alpha = 0.05
  spread         Pp < 1.00: "not capable" in the workflow article (Section 3.1); Pp between 1.00
                 and 1.33 ("marginal") is not a spread signal (D020 in Table 1: Latent model risk)
  mean shift     Pp - Ppk > SE(Ppk): the loss from off-centring exceeds the sampling uncertainty

For a tolerance with an upper limit only (GD&T), Pp is not computed, as in the training data, so
the spread and mean-shift signals do not apply; only the distribution signal can be raised.

These rules illustrate the decision layer. Like the probability bands of the paper's illustrative
configuration (Section 5.1, where deployment-specific thresholds may be selected instead), they
show how the method works and are not a general standard for every application.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

DRIVERS = ("Acceptable", "Mean shift to USL", "Mean shift to LSL", "Excessive variation", "Skewed",
           "Mixed mechanism", "Latent model risk")

ACTIONS = ("Accept", "Monitor", "Re-center", "Reduce sd", "Review distribution", "Investigate latent risk")

SPREAD_LIMIT = 1.00   # workflow article, Section 3.1: Cp < 1.00 means not capable

_ORDER = ("spread", "shift", "distribution")
_ACTION = {"spread": "Reduce sd", "shift": "Re-center", "distribution": "Review distribution"}
_LEVEL_ACTION = {"Low": "Accept or continue routine monitoring", "Medium": "Review or collect additional evidence",
                 "High": "Reject, hold or initiate corrective action"}


@dataclass(frozen=True)
class Signals:
    shift: bool          # Pp - Ppk > SE(Ppk)
    spread: bool         # Pp < SPREAD_LIMIT
    distribution: bool   # normality rejected
    toward_usl: bool     # mean above the middle of the tolerance
    pp: float = float("nan")
    ppk: float = float("nan")
    se: float = float("nan")
    p_normal: float = float("nan")
    skewness: float = float("nan")
    kurtosis: float = float("nan")

    @property
    def active(self) -> tuple:
        return tuple(k for k in _ORDER if getattr(self, k))

    @property
    def compound(self) -> bool:
        return len(self.active) >= 2


def signals(features: dict) -> Signals:
    """Signals from the UC-Cap inputs (:func:`uccap.hybrid.compute_features`)."""
    pp, ppk, se = features["A_Pp"], features["A_Ppk"], features["A_SE_Ppk"]
    mid = (features["LSL"] + features["USL"]) / 2
    return Signals(shift=bool(pp - ppk > se), spread=bool(pp < SPREAD_LIMIT),
                   distribution=features["A_normality"] == "Fail",
                   toward_usl=bool(features["A_mean"] > mid), pp=pp, ppk=ppk, se=se,
                   p_normal=features["A_P_value"], skewness=features["A_Skewness"],
                   kurtosis=features["A_Kurtosis"])


def displayed_level(level: str, s: Signals) -> str:
    """Conservative layer: Low becomes Medium for a compound mechanism; Medium and High are kept."""
    return "Medium" if level == "Low" and s.compound else level


def primary_driver(level: str, s: Signals) -> str:
    """Compact reason label. A Low level without a compound mechanism is Acceptable (Table 1)."""
    if s.compound:
        return "Mixed mechanism"
    if level == "Low":
        return "Acceptable"
    if s.shift:
        return "Mean shift to USL" if s.toward_usl else "Mean shift to LSL"
    if s.spread:
        return "Excessive variation"
    if s.distribution:
        return "Skewed"
    return "Acceptable" if level == "Low" else "Latent model risk"


def action(driver: str, level: str, s: Signals) -> str:
    """Compact action label; a Low level is accepted (the paper's 'accept or continue routine monitoring')."""
    if level == "Low":
        return "Accept"
    if driver == "Latent model risk":
        return "Investigate latent risk"
    parts = [_ACTION[k] for k in s.active]
    return " + ".join([parts[0]] + [a[0].lower() + a[1:] for a in parts[1:]])


def _side(s: Signals) -> str:
    return "USL" if s.toward_usl else "LSL"


def _why(key: str, s: Signals) -> str:
    if key == "shift":
        return (f"the mean is shifted toward {_side(s)}: off-centring lowers Ppk to {s.ppk:.3f} from Pp = {s.pp:.3f}, "
                f"a loss larger than the standard error {s.se:.3f}")
    if key == "spread":
        return f"the spread is too large: Pp = {s.pp:.3f} is below {SPREAD_LIMIT:.2f} (not capable even if centred)"
    return (f"the Anderson-Darling normality test is rejected (p = {s.p_normal:.3f}; skewness {s.skewness:.2f}, "
            f"kurtosis {s.kurtosis:.2f}, 3 for a normal distribution), so normal-based indices may misstate the tails")


def _todo(key: str, s: Signals) -> str:
    if key == "shift":
        return f"re-center the process toward the target (away from {_side(s)}) and confirm with new parts"
    if key == "spread":
        return "reduce variation by finding and removing the main sources of spread"
    return ("review the distribution: check for special causes, mixed streams or measurement problems, and assess "
            "capability with the fitted distribution (percentile indices)")


def _cap(text: str) -> str:
    return text[0].upper() + text[1:]


def reason_detail(driver: str, level: str, s: Signals, p: float) -> str:
    """Expanded reason: the statistics behind the compact label."""
    if driver == "Acceptable":
        if s.active:
            return (f"Predicted risk {p:.2f}, Low band, no compound mechanism. Noted: "
                    + _why(s.active[0], s).replace(": ", "; ", 1) + ".")
        if math.isnan(s.pp):
            return (f"No mechanism signal: the tolerance is an upper limit only, so Pp is not computed and the mean-shift "
                    f"and spread signals do not apply; the normality test passes; the predicted risk {p:.2f} is in the "
                    f"Low band.")
        return (f"No mechanism signal: the mean is centred within the sampling uncertainty, Pp = {s.pp:.3f} is at least "
                f"{SPREAD_LIMIT:.2f} and the normality test passes; the predicted risk {p:.2f} is in the Low band.")
    if driver == "Latent model risk":
        state = ("upper limit only, so no mean-shift or spread signal" if math.isnan(s.pp)
                 else f"centred, Pp = {s.pp:.3f} at least {SPREAD_LIMIT:.2f}")
        return (f"No clear statistical failure mode ({state}, normality not "
                f"rejected), but the model puts the risk at {p:.2f}, in the {level} band: a risk beyond the classical indices.")
    if driver == "Mixed mechanism":
        return "Several mechanisms together: " + "; ".join(_why(k, s) for k in s.active) + "."
    return _cap(_why(s.active[0], s)) + "."


def action_detail(driver: str, level: str, s: Signals) -> str:
    """Expanded action: the level's action (paper Eq. 26) and the follow-up for the mechanism."""
    head = _LEVEL_ACTION[level]
    if driver == "Acceptable" and not s.active:
        return head + "."
    if driver == "Latent model risk":
        return (head + ": collect additional evidence (more parts, remeasurement, specification check) before any "
                "process adjustment.")
    steps = [_todo(k, s) for k in s.active]
    if level == "Low" or driver == "Acceptable":
        names = {"shift": f"the mean shift toward {_side(s)}", "spread": "the spread",
                 "distribution": "the distribution shape"}
        return head + "; watch " + " and ".join(names[k] for k in s.active) + " in later samples."
    lead = "combined response: " if len(steps) > 1 else ""
    return head + "; " + lead + "; ".join(steps) + "."
