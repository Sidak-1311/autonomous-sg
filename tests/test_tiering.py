import numpy as np
import pytest

from src.tiering import CRITICAL, ELEVATED, SAFE, tier, tier_array, worse


@pytest.mark.parametrize(
    "risk, miss, expected",
    [
        (-30.0, 20_000, SAFE),
        (-6.01, 5_001, SAFE),
        (-6.0, 20_000, ELEVATED),      # Pc = 1e-6 is Elevated
        (-4.01, 20_000, ELEVATED),
        (-30.0, 5_000, ELEVATED),      # 5.0 km is Elevated
        (-30.0, 1_000, ELEVATED),      # 1.0 km is Elevated
        (-4.0, 20_000, CRITICAL),      # Pc = 1e-4 is Critical
        (-30.0, 999.9, CRITICAL),
        (-3.0, 7_000, CRITICAL),       # overlap: Critical wins over Safe miss distance
        (-5.0, 500, CRITICAL),
    ],
)
def test_tier_boundaries(risk, miss, expected):
    assert tier(risk, miss) == expected


def test_tier_array_matches_scalar():
    rng = np.random.default_rng(0)
    risk = rng.uniform(-30, -1, 2_000)
    miss = rng.uniform(0, 20_000, 2_000)
    assert list(tier_array(risk, miss)) == [tier(r, m) for r, m in zip(risk, miss)]


def test_worse():
    assert worse(SAFE, CRITICAL) == CRITICAL
    assert worse(ELEVATED, SAFE) == ELEVATED
    assert worse(ELEVATED, ELEVATED) == ELEVATED


# --- formatting --------------------------------------------------------------

from src.tiering import format_miss, format_pc, format_prob, format_ttca, rule_fired  # noqa: E402


@pytest.mark.parametrize("log10, text", [
    (-4.0, "1.0×10⁻⁴"), (-2.9, "1.3×10⁻³"), (-6.0, "1.0×10⁻⁶"),
    (-30.0, "<10⁻³⁰"), (-3.0018, "1.0×10⁻³"),   # 9.96e-4 rounds to the next decade
    (0.0, "1.0×10⁰"), (float("nan"), "n/a"),
])
def test_format_pc(log10, text):
    assert format_pc(log10) == text


def test_other_formats():
    assert format_miss(999.4) == "999 m" and format_miss(5_000) == "5,000 m"
    assert format_miss(12_345) == "12.3 km" and format_miss(float("nan")) == "n/a"
    assert format_ttca(2.17) == "2 d 04 h" and format_ttca(0.0) == "0 d 00 h"
    assert format_prob(0.0005) == "<0.1%" and format_prob(0.951) == "95.1%"
    assert format_prob(0.0) == "0.0%" and format_prob(float("nan")) == "n/a"


def test_rule_text_on_thresholds():
    assert rule_fired(-4.0, 20_000) == "Pc 1.0×10⁻⁴ ≥ 1.0×10⁻⁴"
    assert rule_fired(-30.0, 1_000) == "miss 1,000 m ≤ 5,000 m"
    assert rule_fired(-30.0, 999) == "miss 999 m < 1,000 m"
    assert rule_fired(-7.0, 6_000) == "Pc 1.0×10⁻⁷ < 1.0×10⁻⁶ and miss 6,000 m > 5,000 m"
