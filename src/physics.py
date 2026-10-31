"""Physics helpers: short-encounter Pc approximation and unit conversions.

With only radial sigmas available, use an isotropic combined covariance:
    sigma^2 = sigma_c,r^2 + sigma_t,r^2
    Pc ~= (R^2 / 2 sigma^2) * exp(-d^2 / 2 sigma^2)
A rough prior used as a model feature, never as a tier input.
"""
import numpy as np

from src.tiering import RISK_FLOOR

HARD_BODY_RADIUS_M = 10.0


def combined_sigma(c_sigma_r, t_sigma_r):
    return np.sqrt(np.asarray(c_sigma_r, float) ** 2 + np.asarray(t_sigma_r, float) ** 2)


def log10_pc_approx(miss_m, c_sigma_r, t_sigma_r, radius_m=HARD_BODY_RADIUS_M):
    """log10 of the approximate Pc, computed in log space to avoid underflow."""
    s2 = np.maximum(combined_sigma(c_sigma_r, t_sigma_r) ** 2, 1e-6)
    d = np.asarray(miss_m, float)
    log_pc = np.log10(radius_m ** 2 / (2 * s2)) - d ** 2 / (2 * s2 * np.log(10))
    return np.clip(log_pc, RISK_FLOOR, 0.0)


def m_to_km(m):
    return np.asarray(m, float) / 1000.0
