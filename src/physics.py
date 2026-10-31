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


# --- maneuver suggestion ----------------------------------------------------

LEO_MEAN_MOTION = 2 * np.pi / (95 * 60)  # rad/s, ~95-minute orbit


def required_miss_m(risk_log10, miss_m, sigma_m, target_risk=-4.0, min_miss_m=1_000.0):
    """Smallest miss distance that clears both the Pc and the 1 km rule.

    Anchored to the CDM's own Pc: with the short-encounter shape, moving from
    miss d to d' changes log10 Pc by -(d'^2 - d^2) / (2 sigma^2 ln 10), so the
    Pc rule needs d'^2 >= d^2 + 2 sigma^2 ln10 (risk - target).
    """
    s2 = np.maximum(np.asarray(sigma_m, float) ** 2, 1e-6)
    extra_sq = np.maximum(2 * s2 * np.log(10) * (np.asarray(risk_log10, float) - target_risk), 0.0)
    return np.maximum(np.sqrt(np.asarray(miss_m, float) ** 2 + extra_sq), min_miss_m)


def radial_maneuver(risk_log10, miss_m, pos_r, pos_t, pos_n, sigma_m,
                    target_risk=-4.0, min_miss_m=1_000.0):
    """Radial offset at TCA (m) and approximate along-track burn (m/s).

    The offset is applied along the current radial separation, which is the
    best-determined direction. A tangential burn half an orbit before TCA
    shifts the radial position by about 4 dv / n, so dv ~= n |dr| / 4.
    Returns (dr_m, dv_mps); zero when the event already clears both rules.
    """
    miss = np.asarray(miss_m, float)
    d_req = required_miss_m(risk_log10, miss, sigma_m, target_risk, min_miss_m)
    r = np.asarray(pos_r, float)
    off_radial_sq = np.maximum(miss ** 2 - r ** 2, 0.0)  # t^2 + n^2, consistent with miss
    r_new = np.sqrt(np.maximum(d_req ** 2 - off_radial_sq, 0.0))
    dr = np.where(r >= 0, r_new - r, -r_new - r)
    dr = np.where(d_req <= miss, 0.0, dr)
    return dr, LEO_MEAN_MOTION * np.abs(dr) / 4
