"""Risk tier rules (NASA/SP-20205011318, ESA SDO practice).

Dataset units: risk is log10(Pc), miss distance is in metres.
Critical is checked first, then Elevated; anything left is Safe.
"""
import numpy as np

SAFE, ELEVATED, CRITICAL = "SAFE", "ELEVATED", "CRITICAL"
TIERS = [SAFE, ELEVATED, CRITICAL]  # ordered by severity
TIER_RANK = {t: i for i, t in enumerate(TIERS)}
TIER_COLORS = {SAFE: "#2e9e5b", ELEVATED: "#e8a33d", CRITICAL: "#d64545"}

RISK_CRITICAL = -4.0   # Pc >= 1e-4
RISK_ELEVATED = -6.0   # Pc >= 1e-6
MISS_CRITICAL_M = 1_000.0
MISS_ELEVATED_M = 5_000.0
RISK_FLOOR = -30.0     # dataset value for "Pc effectively zero"


def tier(risk_log10: float, miss_m: float) -> str:
    if risk_log10 >= RISK_CRITICAL or miss_m < MISS_CRITICAL_M:
        return CRITICAL
    if risk_log10 >= RISK_ELEVATED or miss_m <= MISS_ELEVATED_M:
        return ELEVATED
    return SAFE


def tier_array(risk_log10, miss_m) -> np.ndarray:
    """Vectorised tier() for pandas Series / numpy arrays."""
    risk = np.asarray(risk_log10, dtype=float)
    miss = np.asarray(miss_m, dtype=float)
    out = np.full(risk.shape, SAFE, dtype=object)
    out[(risk >= RISK_ELEVATED) | (miss <= MISS_ELEVATED_M)] = ELEVATED
    out[(risk >= RISK_CRITICAL) | (miss < MISS_CRITICAL_M)] = CRITICAL
    return out


def worse(a: str, b: str) -> str:
    return a if TIER_RANK[a] >= TIER_RANK[b] else b


def rule_fired(risk_log10: float, miss_m: float) -> str:
    """Short text naming which threshold put the event in its tier."""
    t = tier(risk_log10, miss_m)
    if t == CRITICAL:
        parts = []
        if risk_log10 >= RISK_CRITICAL:
            parts.append(f"Pc 1e{risk_log10:.1f} ≥ 1e-4")
        if miss_m < MISS_CRITICAL_M:
            parts.append(f"miss {miss_m / 1000:.2f} km < 1 km")
        return " and ".join(parts)
    if t == ELEVATED:
        parts = []
        if risk_log10 >= RISK_ELEVATED:
            parts.append(f"Pc 1e{risk_log10:.1f} ≥ 1e-6")
        if miss_m <= MISS_ELEVATED_M:
            parts.append(f"miss {miss_m / 1000:.2f} km ≤ 5 km")
        return " and ".join(parts)
    return f"Pc 1e{risk_log10:.1f} < 1e-6 and miss {miss_m / 1000:.2f} km > 5 km"
