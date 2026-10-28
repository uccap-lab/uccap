"""Hybrid statistical-learning capability risk (UC-Cap).

Source: A hybrid statistical learning framework for capability-based decision support in
manufacturing (Jiang & Yang, IJAMT 2026, doi:10.1007/s00170-026-18833-9; arXiv:2604.13352).

    pi = sigmoid( (C0 - Ppk) / SE(Ppk) + f(x) ),   C0 = 1.33

The first term is the statistical anchor; f is a small neural network that corrects it using
distributional and specification features. The network is the paper's final model (V4.2,
Table 10). Its trained parameters are not included in this release; they are available from the
authors on request (see :func:`set_model_path`). Without them the features
(:func:`compute_features`) and the mechanism signals (:mod:`uccap.hybrid.mechanism`) are still
available. The model was trained for C0 = 1.33 on dimensional data with bilateral tolerances
(upper-limit-only GD&T characteristics coded with LSL = 0 and Ppk = Ppu), with inputs computed
from 20 measurements per characteristic (the A side of an observation-disjoint split, Sec. 7.5);
outside that setting treat the result as indicative.
"""
from __future__ import annotations

from typing import Optional

from .features import C0, compute_features
from . import mechanism
from .model import (LEVEL_ACTION, TAU_HIGH, TAU_LOW, HybridRisk, ModelNotAvailable, load_model, model_available,
                    model_path, predict, residual, review_thresholds, risk_level, set_model_path)


def hybrid_risk(data, lsl: Optional[float] = None, usl: Optional[float] = None,
                nominal: Optional[float] = None, *, n_boot: int = 200, rng=0,
                thresholds: Optional[tuple] = None, costs: Optional[tuple] = None) -> HybridRisk:
    """UC-Cap risk that the true Ppk is below 1.33, from raw measurements.

    ``data`` is a sequence of measurements, or a :class:`uccap.workflow.CapabilityStudy`
    (its data and limits are used and the other arguments ignored).

    The decision layer of the paper (Sections 5.1-5.3) then gives the primary risk driver, a
    reason and an action (see :mod:`uccap.hybrid.mechanism`); it may raise Low to Medium for a
    compound mechanism
    (``model_level`` keeps the level from the bands) and never changes ``p`` or the score.
    The level bands and the decision-layer rules illustrate the method, as in the paper; they are
    not a general standard for every application. The risk level uses the bands of the paper's
    illustrative configuration, 0.40 and
    0.70, unless ``thresholds=(low, high)`` is given, or ``costs=(c_FA, c_FR, c_review)``,
    from which the bands follow by :func:`review_thresholds`.
    """
    load_model()  # clear error before the bootstrap when the trained parameters are missing
    if costs is not None:
        thresholds = review_thresholds(*costs)
    thresholds = tuple(thresholds) if thresholds is not None else (TAU_LOW, TAU_HIGH)
    if hasattr(data, "spec") and hasattr(data, "data"):
        lsl, usl, nominal, data = data.spec.lsl, data.spec.usl, data.spec.target, data.data
    f = compute_features(data, lsl, usl, nominal, n_boot=n_boot, rng=rng)
    corr = float(residual([f])[0])
    p = float(predict(f)[0])
    model_level = risk_level(p, thresholds)
    sig = mechanism.signals(f)
    level = mechanism.displayed_level(model_level, sig)
    driver = mechanism.primary_driver(level, sig)
    return HybridRisk(p, level, f["A_z_stat_133"], corr, f, thresholds=thresholds, model_level=model_level,
                      driver=driver, reason=mechanism.reason_detail(driver, level, sig, p),
                      action=mechanism.action(driver, level, sig),
                      action_detail=mechanism.action_detail(driver, level, sig))


__all__ = ["hybrid_risk", "HybridRisk", "compute_features", "predict", "residual", "risk_level",
           "review_thresholds", "C0", "TAU_LOW", "TAU_HIGH", "LEVEL_ACTION", "ModelNotAvailable",
           "model_available", "model_path", "set_model_path"]
