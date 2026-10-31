"""Debris cascade: if this conjunction became a collision, how much debris, and how much more likely are
collisions for the other missions in the same orbital shell (the first link of a Kessler-type chain)?

Two textbook models, both first-order and stated as such in the dashboard:

1. NASA Standard Breakup Model (Johnson et al. 2001, "NASA's new breakup model of EVOLVE 4.0").
   Specific energy E* = (1/2 m_small v^2) / m_large. At E* >= 40 J/g the collision is catastrophic and both
   bodies fragment: M = m_large + m_small (kg). Below it only the smaller body breaks up: M = m_small * v^2
   (v in km/s). Fragments with characteristic length >= Lc: N = 0.1 * M^0.75 * Lc^-1.71.
2. Particle-in-a-box flux (Kessler & Cour-Palais 1978). The fragments spread over a spherical shell around
   the collision altitude; their spatial density n = N / V_shell raises the collision rate of an object with
   cross-section A by n * v * A, so P(hit in T) = 1 - exp(-n v A T).

Masses come from size and drag area-to-mass (A / (A/m)), with type defaults when the data can't say.
"""
import numpy as np
import pandas as pd

R_EARTH_KM = 6378.137
LC_M = 0.10                  # count fragments of 10 cm and up: the size radar catalogues track
CATASTROPHIC_J_PER_G = 40.0
SHELL_HALF_WIDTH_KM = 50.0   # ponytail: fragments assumed spread evenly over +/-50 km within a year; a real cloud
                             # spreads unevenly by inclination and drag, so treat chain risk as an order of magnitude
V_REL_KM_S = 10.0            # typical LEO crossing speed for fragment-on-satellite impacts
YEAR_S = 365.25 * 86400
DEFAULT_MASS_KG = {"PAYLOAD": 1000.0, "ROCKET BODY": 1500.0, "DEBRIS": 10.0, "UNKNOWN": 10.0, "TBA": 10.0}

COLUMNS = ["mass_primary_kg", "mass_secondary_kg", "energy_j_per_g", "catastrophic", "frag_10cm", "shell_alt_km",
           "shell_missions", "chain_p1y_max", "chain_hits_1y", "debris_expected"]
NEEDS = ["relative_speed", "t_span", "c_span", "t_cd_area_over_mass", "c_cd_area_over_mass", "t_h_per", "t_h_apo",
         "mission_id"]


MASS_MAX_KG = 10_000.0  # ponytail: span x area-to-mass overestimates big solar arrays (up to 96 t in the test
                        # data); the heaviest LEO science satellites are ~8-11 t (Envisat, Hubble), so cap there


def mass_kg(span_m, area_over_mass, default_kg):
    """Cross-section pi (span/2)^2 over the drag area-to-mass ratio, capped at MASS_MAX_KG; the default where
    either input is unusable."""
    span = np.asarray(span_m, float)
    am = np.asarray(area_over_mass, float)
    m = np.pi * (span / 2) ** 2 / np.where(am > 1e-4, am, np.nan)
    m = np.where(np.isfinite(m) & (m > 0.01), np.minimum(m, MASS_MAX_KG), np.nan)
    return np.where(np.isnan(m), np.asarray(default_kg, float), m)


def breakup(m1_kg, m2_kg, v_m_s, lc_m: float = LC_M):
    """NASA SBM: (specific energy in J/g, catastrophic?, fragments >= lc_m)."""
    m1, m2, v = (np.asarray(x, float) for x in (m1_kg, m2_kg, v_m_s))
    small, large = np.minimum(m1, m2), np.maximum(m1, m2)
    energy = 0.5 * small * v ** 2 / large / 1000.0                   # J/kg -> J/g
    catastrophic = energy >= CATASTROPHIC_J_PER_G
    m_eff = np.where(catastrophic, large + small, small * (v / 1000.0) ** 2)
    return energy, catastrophic, 0.1 * m_eff ** 0.75 * lc_m ** -1.71


def shell_volume_km3(alt_km, half_width_km: float = SHELL_HALF_WIDTH_KM):
    r = R_EARTH_KM + np.asarray(alt_km, float)
    return 4 / 3 * np.pi * ((r + half_width_km) ** 3 - (r - half_width_km) ** 3)


def p_hit(fragments, alt_km, area_m2, years: float = 1.0):
    """Particle-in-a-box: chance an object of cross-section area_m2 in the shell is hit within `years`."""
    density = np.asarray(fragments, float) / shell_volume_km3(alt_km)    # per km^3
    rate = density * V_REL_KM_S * np.asarray(area_m2, float) / 1e6        # hits per second
    return 1 - np.exp(-rate * years * YEAR_S)


def missions(ev: pd.DataFrame) -> pd.DataFrame:
    """One row per mission (our primaries): mean altitude and cross-section, from the events it appears in."""
    alt = (ev.t_h_per + ev.t_h_apo) / 2
    return (ev.assign(alt_km=alt, area_m2=np.pi * (ev.t_span / 2) ** 2)
            .groupby("mission_id")[["alt_km", "area_m2"]].median())


def cascade(ev: pd.DataFrame, half_width_km: float = SHELL_HALF_WIDTH_KM) -> pd.DataFrame:
    """Cascade columns for each event (same index). NaN everywhere when the inputs aren't there."""
    if ev.empty or not set(NEEDS) <= set(ev.columns):
        return pd.DataFrame(np.nan, index=ev.index, columns=COLUMNS)
    m_t = mass_kg(ev.t_span, ev.t_cd_area_over_mass, DEFAULT_MASS_KG["PAYLOAD"])
    default_c = ev.get("c_object_type", pd.Series("UNKNOWN", index=ev.index)).map(DEFAULT_MASS_KG).fillna(10.0)
    m_c = mass_kg(ev.c_span, ev.c_cd_area_over_mass, default_c)
    energy, catastrophic, frags = breakup(m_t, m_c, ev.relative_speed)
    alt = ((ev.t_h_per + ev.t_h_apo) / 2).to_numpy()

    mis = missions(ev)
    others_alt, others_area, others_id = mis.alt_km.to_numpy(), mis.area_m2.to_numpy(), mis.index.to_numpy()
    # in the same shell, and not the mission that collided (it no longer exists)
    in_shell = ((np.abs(others_alt[None, :] - alt[:, None]) <= half_width_km)
                & (others_id[None, :] != ev.mission_id.to_numpy()[:, None]))
    p = p_hit(frags[:, None], alt[:, None], others_area[None, :]) * in_shell
    pc = 10 ** ev.get("risk_forecast", ev.get("risk_latest", pd.Series(np.nan, index=ev.index))).to_numpy(float)
    return pd.DataFrame({
        "mass_primary_kg": m_t, "mass_secondary_kg": m_c, "energy_j_per_g": energy, "catastrophic": catastrophic,
        "frag_10cm": frags, "shell_alt_km": alt, "shell_missions": in_shell.sum(axis=1),
        "chain_p1y_max": p.max(axis=1, initial=0.0), "chain_hits_1y": p.sum(axis=1),
        "debris_expected": pc * frags,     # fragments to expect from this event, weighted by its Pc
    }, index=ev.index)
