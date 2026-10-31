"""Risk tier rules (NASA/SP-20205011318, ESA SDO practice).

Dataset units: risk is log10(Pc), miss distance is in metres.
Critical is checked first, then Elevated; anything left is Safe.
"""
import numpy as np

SAFE, ELEVATED, CRITICAL = "SAFE", "ELEVATED", "CRITICAL"
TIERS = [SAFE, ELEVATED, CRITICAL]  # ordered by severity
TIER_RANK = {t: i for i, t in enumerate(TIERS)}
# Tuned for the dashboard's near-black background (#101010): each has >= 5:1
# contrast, and Safe is a recessive steel blue (not green), so the worst tier
# pair keeps CIELAB dE >= 35 under deuteranopia and protanopia (Machado 2009).
# Elevated is orange, not yellow, so it stays distinct (dE 27) from the UI's
# yellow accent. Never show a tier by colour alone: always with its text label.
TIER_COLORS = {SAFE: "#4A86C5", ELEVATED: "#FF8A3D", CRITICAL: "#F2386F"}

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


# --- display formatting (shared with the dashboard) -------------------------

_SUPERSCRIPT = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def format_pc(risk_log10: float) -> str:
    """log10 Pc as '1.3×10⁻³'; the -30 floor reads '<10⁻³⁰' (effectively zero)."""
    if risk_log10 is None or np.isnan(risk_log10):
        return "n/a"
    if risk_log10 <= RISK_FLOOR:
        return "<10" + "-30".translate(_SUPERSCRIPT)
    exp = int(np.floor(risk_log10))
    mant = 10 ** (risk_log10 - exp)
    if round(mant, 1) >= 10:  # 9.96 rounds up to the next decade
        mant, exp = 1.0, exp + 1
    return f"{mant:.1f}×10" + str(exp).translate(_SUPERSCRIPT)


def format_miss(miss_m: float) -> str:
    """Whole metres below 10 km, so values near the 1 km / 5 km thresholds never
    round onto them; km with one decimal above."""
    if miss_m is None or np.isnan(miss_m):
        return "n/a"
    return f"{miss_m:,.0f} m" if miss_m < 10_000 else f"{miss_m / 1000:.1f} km"


NNBSP = " "  # narrow no-break space: '2 d 04 h' never wraps, and tight tracking can't swallow it


def format_ttca(days: float) -> str:
    """Time to TCA as '2 d 04 h' (narrow no-break spaces between number and unit)."""
    if days is None or np.isnan(days):
        return "n/a"
    hours = int(round(days * 24))
    s = NNBSP
    return f"{hours // 24}{s}d {hours % 24:02d}{s}h"


def format_prob(p: float) -> str:
    if p is None or np.isnan(p):
        return "n/a"
    return "<0.1%" if 0 < p < 0.001 else f"{p:.1%}"


def rule_fired(risk_log10: float, miss_m: float) -> str:
    """Short text naming which threshold put the event in its tier."""
    pc, miss = f"Pc {format_pc(risk_log10)}", f"miss {format_miss(miss_m)}"
    t = tier(risk_log10, miss_m)
    if t == CRITICAL:
        parts = []
        if risk_log10 >= RISK_CRITICAL:
            parts.append(f"{pc} ≥ 1.0×10⁻⁴")
        if miss_m < MISS_CRITICAL_M:
            parts.append(f"{miss} < 1,000 m")
        return " and ".join(parts)
    if t == ELEVATED:
        parts = []
        if risk_log10 >= RISK_ELEVATED:
            parts.append(f"{pc} ≥ 1.0×10⁻⁶")
        if miss_m <= MISS_ELEVATED_M:
            parts.append(f"{miss} ≤ 5,000 m")
        return " and ".join(parts)
    return f"{pc} < 1.0×10⁻⁶ and {miss} > 5,000 m"
