"""Practical process capability indices workflow (Jiang & Yang, IJAMT 2026, doi:10.1007/s00170-026-17782-7)."""
from .fitting import best_fit, fit_distribution
from .indices import Spec, cp, cpk, cpl, cpm, cpmk, cpu, normal_indices
from .normality import check_normality, chi_square_gof
from .outliers import adjusted_boxplot_outliers, adjusted_fences, grubbs_outliers, iqr_fences, iqr_outliers
from .percentile import percentile_indices
from .pipeline import CapabilityStudy, capability_study
from .sigma import (sigma_amr, sigma_between, sigma_mmr, sigma_overall, sigma_pooled, sigma_rbar,
                    sigma_sbar, sigma_srmssd)

__all__ = [
    "Spec", "CapabilityStudy", "capability_study", "best_fit", "fit_distribution",
    "cp", "cpk", "cpl", "cpm", "cpmk", "cpu", "normal_indices", "percentile_indices",
    "check_normality", "chi_square_gof", "grubbs_outliers", "iqr_outliers", "adjusted_boxplot_outliers",
    "iqr_fences", "adjusted_fences", "sigma_amr", "sigma_between", "sigma_mmr",
    "sigma_overall", "sigma_pooled", "sigma_rbar", "sigma_sbar", "sigma_srmssd",
]
