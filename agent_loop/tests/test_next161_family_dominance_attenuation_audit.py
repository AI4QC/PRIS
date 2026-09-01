from __future__ import annotations

import numpy as np

from src.next161_family_dominance_attenuation_audit import (
    ATTENUATIONS,
    FIXED_DIRECTIONS,
    family_attenuation_statistics,
)


def test_family_dominance_attenuation_uses_capped_family_means() -> None:
    term_ids = [
        "cov_demo",
        "scbv_demo",
        "cmvo_demo",
        "bvtbd_demo",
        "mhcr_demo",
    ]
    result = family_attenuation_statistics(
        np.array([[0.0, 1.0, 0.25, 0.4, 0.5]]), term_ids
    )
    expected = {
        "family_capmean_attenuation_0p1": 1.35,
        "family_capmean_attenuation_0p25": 1.275,
        "family_capmean_attenuation_0p5": 1.15,
        "family_capmean_attenuation_0p75": 1.025,
        "family_capmean_attenuation_1p0": 0.9,
    }
    assert set(result) == set(expected) == set(ATTENUATIONS) == set(FIXED_DIRECTIONS)
    for name, value in expected.items():
        np.testing.assert_allclose(result[name], [value])
        assert FIXED_DIRECTIONS[name] == -1
