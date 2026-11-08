"""Trained UC-Cap V4.2 network, evaluated with numpy.

pi = sigmoid(z + f(x)),  z = (C0 - Ppk) / SE(Ppk)  (fixed anchor),
f = MLP(24 -> 64 -> 64 -> 1, ReLU) on the standardised residual features.

The weights and preprocessing parameters (training medians, means, scales and category
levels) are those of the model reported in Table 10 of the paper. They are not included in this
release. Give their location with
:func:`set_model_path` or the environment variable ``UCCAP_MODEL``. Numeric NaN inputs are replaced by the training median, unknown categories are
encoded as all zeros, as in the scikit-learn pipeline used for training.
"""
from __future__ import annotations

import json
import math
import os
import textwrap
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

from .features import ANCHOR, C0, CATEGORICAL, NUMERIC

MODEL_FILE = Path(__file__).with_name("uccap_v4_2.json")   # used when present next to this module
MODEL_ENV = "UCCAP_MODEL"
_model_path = None


class ModelNotAvailable(RuntimeError):
    """The trained UC-Cap parameters were not found."""


def set_model_path(path) -> None:
    """Use the trained UC-Cap parameters (the JSON file) at ``path``; None returns to the default lookup."""
    global _model_path
    _model_path = None if path is None else Path(path).expanduser()
    load_model.cache_clear()


def model_path():
    """Location of the trained parameters: set_model_path(), then $UCCAP_MODEL, then the package folder."""
    for cand in (_model_path, os.environ.get(MODEL_ENV) or None, MODEL_FILE):
        if cand is not None and Path(cand).expanduser().is_file():
            return Path(cand).expanduser()
    return None


def model_available() -> bool:
    """True when the trained UC-Cap parameters can be loaded."""
    return model_path() is not None
# Prespecified bands of the paper's illustrative configuration (Eq. 22); they are not
# plant-specific optima and can be replaced, e.g. by review_thresholds() from costs.
TAU_LOW, TAU_HIGH = 0.40, 0.70
# Action of each level, as in the minimum-loss policy of the paper (Eq. 26).
LEVEL_ACTION = {"Low": "accept or continue routine monitoring", "Medium": "review or collect additional evidence",
                "High": "reject, hold or initiate corrective action"}


def review_thresholds(c_fa: float, c_fr: float, c_review: float) -> tuple[float, float]:
    """Risk bands from costs, with an intermediate review action (paper Eqs. 23-25).

    ``c_fa``: cost of accepting an incapable process, ``c_fr``: cost of rejecting a capable
    one, ``c_review``: normalised cost of additional inspection, remeasurement or
    investigation. Returns (alpha_L, alpha_H) = (c_R/c_FA, 1 - c_R/c_FR). When
    c_R >= c_FA c_FR / (c_FA + c_FR) a review is never the cheapest action and both
    values equal the binary threshold c_FR / (c_FA + c_FR).
    """
    if min(c_fa, c_fr) <= 0 or c_review < 0:
        raise ValueError("costs must be positive")
    if c_review >= c_fa * c_fr / (c_fa + c_fr):
        a = c_fr / (c_fa + c_fr)
        return a, a
    return c_review / c_fa, 1 - c_review / c_fr


@lru_cache(maxsize=1)
def load_model() -> dict:
    path = model_path()
    if path is None:
        raise ModelNotAvailable(
            "the trained UC-Cap parameters are not included in this release; "
            f"give their location with uccap.hybrid.set_model_path(path) or ${MODEL_ENV}")
    b = json.loads(path.read_text())
    assert tuple(b["numeric_features"]) == NUMERIC and tuple(b["categorical_features"]) == CATEGORICAL
    for k in ("numeric_median", "numeric_mean", "numeric_scale"):
        b[k] = np.asarray(b[k], float)
    b["layers"] = [(np.asarray(L["weight"], float), np.asarray(L["bias"], float)) for L in b["layers"]]
    return b


def _design(rows: list[dict], b: dict) -> np.ndarray:
    num = np.array([[float(r.get(c, np.nan)) for c in NUMERIC] for r in rows], float)
    num = np.where(np.isfinite(num), num, b["numeric_median"])
    num = (num - b["numeric_mean"]) / b["numeric_scale"]
    blocks = [num]
    for c, levels, default in zip(CATEGORICAL, b["categorical_levels"], b["categorical_most_frequent"]):
        vals = [r.get(c) for r in rows]
        vals = [default if v is None or v == "" or (isinstance(v, float) and math.isnan(v)) else str(v)
                for v in vals]
        blocks.append(np.array([[v == lv for lv in levels] for v in vals], float))
    return np.hstack(blocks)


def residual(rows: list[dict]) -> np.ndarray:
    """Learned correction f(x) on the logit scale, one value per row."""
    b = load_model()
    h = _design(rows, b)
    for i, (w, bias) in enumerate(b["layers"]):
        h = h @ w.T + bias
        if i < len(b["layers"]) - 1:
            h = np.maximum(h, 0.0)
    return h[:, 0]


def predict(rows) -> np.ndarray:
    """Risk pi for one feature dict or a list of them (or a DataFrame with the feature columns)."""
    if isinstance(rows, dict):
        rows = [rows]
    elif hasattr(rows, "to_dict"):
        rows = rows.to_dict("records")
    z = np.array([float(r.get(ANCHOR, np.nan)) for r in rows], float)
    logit = np.clip(z + residual(rows), -40, 40)
    return 1 / (1 + np.exp(-logit))


def risk_level(p: float, thresholds: tuple[float, float] = (TAU_LOW, TAU_HIGH)) -> str:
    """Low / Medium / High for a risk p, with the bands ``thresholds`` = (low, high)."""
    if not math.isfinite(p):
        return ""
    low, high = thresholds
    return "High" if p >= high else "Medium" if p >= low else "Low"


@dataclass(frozen=True)
class HybridRisk:
    p: float            # predicted probability that Ppk_true < C0
    level: str          # Low / Medium / High
    z: float            # statistical anchor (C0 - Ppk) / SE
    correction: float   # learned residual f(x) on the logit scale
    features: dict = field(repr=False)
    c0: float = C0
    thresholds: tuple = (TAU_LOW, TAU_HIGH)
    model_level: str = ""   # level from the bands alone, before the conservative layer
    driver: str = ""        # compact reason label (decision layer, as in the paper's Table 1)
    reason: str = ""        # expanded reason
    action: str = ""        # compact action label
    action_detail: str = ""  # expanded action

    @property
    def elevated(self) -> bool:
        """True when the conservative layer raised Low to Medium (score unchanged)."""
        return bool(self.model_level) and self.model_level != self.level

    @property
    def score(self) -> float:
        """Risk score 0-100 (100 x p), as in the paper."""
        return 100 * self.p

    @property
    def p_stat(self) -> float:
        """Anchor-only probability sigmoid(z), i.e. the model without the learned correction."""
        return 1 / (1 + math.exp(-max(min(self.z, 40), -40))) if math.isfinite(self.z) else float("nan")

    def report(self) -> str:
        f = self.features
        lines = [
            "UC-Cap hybrid risk (V4.2)",
            f"{('Ppu, ' if f.get('Upper_only') else 'Ppk, ') + ('normal' if f['A_normality'] == 'Pass' else 'empirical'):>22}  {f['A_Ppk']:.4f}",
            f"{'SE(Ppk), bootstrap':>22}  {f['A_SE_Ppk']:.4f}",
            f"{'best fit':>22}  {f['A_Best_Fit']}",
            f"{'anchor z':>22}  {self.z:.4f}",
            f"{'learned correction':>22}  {self.correction:+.4f}",
            f"{'P(Ppk < ' + format(self.c0, 'g') + ')':>22}  {self.p:.4f}",
            f"{'risk score':>22}  {self.score:.1f}",
            f"{'risk level':>22}  {self.level}" + ("   (raised from Low by the conservative layer)" if self.elevated else ""),
        ]
        if f.get("Upper_only"):
            lines.insert(1, f"{'tolerance':>22}  upper limit only: coded LSL = 0, nominal = 0, Ppk = Ppu, no Pp")
        if math.isfinite(f.get("Fit_Ppk", float("nan"))):
            lines.insert(4, f"{('Ppu, ' if f.get('Upper_only') else 'Ppk, ') + f['A_Best_Fit'] + ' fit':>22}  {f['Fit_Ppk']:.4f}   (percentile, not a model input)")
        if self.driver:
            def wrap(t):  # keep "Pp = 1.905" on one line
                t = textwrap.fill(t.replace(" = ", "\0"), 92, initial_indent=" " * 24, subsequent_indent=" " * 24)
                return t.replace("\0", " = ").lstrip()
            lines += [f"{'reason':>22}  {self.driver}",
                      f"{'action':>22}  {self.action}",
                      f"{'reason, in full':>22}  {wrap(self.reason)}",
                      f"{'action, in full':>22}  {wrap(self.action_detail)}"]
        return "\n".join(lines)

    def plot(self, title=None):
        """Contributions on the logit scale and the risk score (needs matplotlib)."""
        from ..plot import uccap_figure
        return uccap_figure(self, title=title)

    def __str__(self) -> str:
        return self.report()
