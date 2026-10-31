"""Verification loop for src/cascade.py: known answers for each formula, then the edge cases."""
import numpy as np
import pandas as pd
import pytest

from src import cascade as cz


def test_catastrophic_breakup_matches_the_sbm_formula():
    # 1,000 kg hit by 10 kg at 10 km/s: E* = 0.5*10*1e8/1000 J/kg = 500 J/g -> catastrophic, M = 1,010 kg
    energy, cat, n = cz.breakup(1000.0, 10.0, 10_000.0)
    assert energy == pytest.approx(500.0) and cat
    assert n == pytest.approx(0.1 * 1010 ** 0.75 * 0.1 ** -1.71)
    assert 900 < n < 1000   # same order as the ~2,000 catalogued pieces from Iridium-Cosmos (two ~600-900 kg bodies)


def test_non_catastrophic_breaks_only_the_small_body():
    # 1 kg at 1 km/s into 1,000 kg: E* = 0.5 J/g -> non-catastrophic, M = m_small * v^2 = 1 kg (km/s)^2
    energy, cat, n = cz.breakup(1000.0, 1.0, 1_000.0)
    assert energy == pytest.approx(0.5) and not cat
    assert n == pytest.approx(0.1 * 0.1 ** -1.71)


def test_exactly_40_j_per_g_is_catastrophic():
    v = np.sqrt(2 * 40_000 * 1000.0 / 10.0)   # makes E* exactly 40 J/g for 10 kg into 1,000 kg
    energy, cat, _ = cz.breakup(1000.0, 10.0, v)
    assert energy == pytest.approx(40.0) and cat


def test_breakup_is_symmetric_in_the_two_masses():
    assert np.allclose(cz.breakup(1000.0, 10.0, 9e3), cz.breakup(10.0, 1000.0, 9e3))


@pytest.mark.parametrize("am", [0.0, -0.1, np.nan, 1e-9])
def test_unusable_area_to_mass_falls_back_to_the_default(am):
    assert cz.mass_kg(2.0, am, 10.0) == 10.0


def test_mass_from_size_and_area_to_mass():
    assert cz.mass_kg(2.0, np.pi / 100, 10.0) == pytest.approx(100.0)   # area pi m^2 over pi/100 m^2/kg


def test_mass_is_capped_for_huge_spans():
    assert cz.mass_kg(27.7, 0.005, 1000.0) == cz.MASS_MAX_KG   # 120 t by the formula: a solar array, not mass


def test_flux_probability_grows_with_fragments_and_area_and_stays_below_one():
    p1, p2 = cz.p_hit(1000, 700, 20.0), cz.p_hit(2000, 700, 20.0)
    assert 0 < p1 < p2 < 1 and cz.p_hit(1000, 700, 40.0) > p1
    assert cz.p_hit(0, 700, 20.0) == 0
    assert cz.p_hit(1e15, 700, 20.0) == pytest.approx(1.0)   # saturates, never above 1


def event(event_id, mission, alt_km, **kw):
    row = dict(event_id=event_id, mission_id=mission, relative_speed=10_000.0, t_span=5.0, c_span=2.0,
               t_cd_area_over_mass=0.013, c_cd_area_over_mass=0.44, t_h_per=alt_km - 10, t_h_apo=alt_km + 10,
               c_object_type="DEBRIS", risk_forecast=-4.0)
    return {**row, **kw}


def test_chain_reaches_only_other_missions_in_the_shell():
    ev = pd.DataFrame([event(0, 1, 700), event(1, 2, 730), event(2, 3, 900)])
    out = cz.cascade(ev)
    # event 0 at 700 km: mission 2 (730 km) is in the +/-50 km shell, mission 3 (900 km) is not, mission 1 is itself
    assert out.shell_missions.tolist() == [1, 1, 0]
    assert out.chain_p1y_max[0] > 0 and out.chain_p1y_max[2] == 0 and out.chain_hits_1y[2] == 0
    assert out.debris_expected[0] == pytest.approx(1e-4 * out.frag_10cm[0])


def test_missing_inputs_give_nan_not_a_crash():
    ev = pd.DataFrame([event(0, 1, 700)]).drop(columns=["t_span"])
    out = cz.cascade(ev)
    assert list(out.columns) == cz.COLUMNS and out.isna().all().all()


def test_empty_input():
    out = cz.cascade(pd.DataFrame(columns=cz.NEEDS))
    assert out.empty and list(out.columns) == cz.COLUMNS


def test_unknown_object_type_uses_the_small_default():
    ev = pd.DataFrame([event(0, 1, 700, c_object_type="SOMETHING NEW", c_cd_area_over_mass=np.nan)])
    assert cz.cascade(ev).mass_secondary_kg[0] == 10.0


def test_on_real_test_data_if_present():
    from pathlib import Path
    path = Path(__file__).resolve().parent.parent / "data" / "test_data.csv"
    if not path.exists():
        pytest.skip("data/test_data.csv missing")
    raw = pd.read_csv(path, usecols=["event_id", "time_to_tca", "risk", *cz.NEEDS, "c_object_type"])
    latest = raw.loc[raw.groupby("event_id").time_to_tca.idxmin()].rename(columns={"risk": "risk_latest"})
    out = cz.cascade(latest)
    assert out.frag_10cm.notna().all() and (out.frag_10cm > 0).all()
    assert out.chain_p1y_max.between(0, 1).all()
    assert out.catastrophic.mean() > 0.5   # LEO crossing speeds of ~10 km/s are almost always catastrophic
