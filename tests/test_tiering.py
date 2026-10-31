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
