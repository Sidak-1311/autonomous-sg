import numpy as np

from src.physics import combined_sigma, log10_pc_approx, radial_maneuver


def test_pc_approx_floor_and_bounds():
    assert log10_pc_approx(1e6, 10.0, 10.0) == -30.0
    assert log10_pc_approx(0.0, 0.0, 0.0) <= 0.0   # zero sigma doesn't divide by zero


def test_combined_sigma():
    assert combined_sigma(3.0, 4.0) == 5.0


def test_maneuver_zero_when_already_safe():
    dr, dv = radial_maneuver(-8.0, 6_000.0, 100.0, 5_000.0, 3_000.0, 50.0)
    assert dr == 0 and dv == 0


def test_maneuver_reaches_one_km_for_close_low_pc_event():
    # Pc already tiny, so only the 1 km rule binds
    r, t, n = 100.0, 300.0, 200.0
    miss = np.sqrt(r**2 + t**2 + n**2)
    dr, dv = radial_maneuver(-12.0, miss, r, t, n, 10.0)
    new_miss = np.sqrt((r + dr) ** 2 + t**2 + n**2)
    assert np.isclose(new_miss, 1_000.0)
    assert dr > 0 and 0 < dv < 1.0  # sub-m/s burn for ~0.9 km shift


def test_maneuver_pushes_past_pc_threshold_with_large_sigma():
    # Pc 1e-2 with 2 km sigma: Pc rule needs more than 1 km
    r, t, n = -200.0, 100.0, 50.0
    miss = np.sqrt(r**2 + t**2 + n**2)
    sigma = 2_000.0
    dr, _ = radial_maneuver(-2.0, miss, r, t, n, sigma)
    new_miss = np.sqrt((r + dr) ** 2 + t**2 + n**2)
    new_risk = -2.0 - (new_miss**2 - miss**2) / (2 * sigma**2 * np.log(10))
    assert dr < 0                     # pushed further along the existing radial side
    assert new_miss > 1_000.0 and np.isclose(new_risk, -4.0)


def test_maneuver_on_exact_thresholds():
    # exactly 1 km and Pc exactly 1e-4: already clear, nothing to do
    dr, _ = radial_maneuver(-4.0, 1_000.0, 0.0, 1_000.0, 0.0, 100.0)
    assert dr == 0


def test_dilution_flags_huge_covariance_only():
    from src.physics import dilution
    diluted, pot = dilution(np.array([-6.0, -6.0]), np.array([100.0, 2_000.0]),
                            np.array([1_000.0, 50.0]))
    assert list(diluted) == [True, False]
    assert pot[0] > -6.0 and pot[0] <= 0.0   # sharper tracking raises Pc
    assert pot[1] == -6.0                    # not diluted: unchanged


def test_dilution_at_exact_peak_is_not_flagged():
    from src.physics import dilution
    d = 1_000.0
    diluted, pot = dilution(-5.0, d, d / np.sqrt(2))
    assert not diluted and pot == -5.0


def test_dilution_handles_zero_miss_and_floor():
    from src.physics import dilution
    diluted, pot = dilution(-30.0, 0.0, 10.0)
    assert np.isfinite(pot) and pot <= 0.0


def test_dilution_ignores_tiny_gains():
    from src.physics import dilution
    d = 1_000.0
    diluted, pot = dilution(-8.0, d, d / np.sqrt(2) * 1.05)  # just past the peak
    assert not diluted and pot > -8.0
