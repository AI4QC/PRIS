import hashlib

import numpy as np

import src.next149_charge_order_spectrum_protection_search as n149


def test_apply_charge_order_protection_is_subtractive_and_fail_open() -> None:
    score, support = n149.apply_charge_order_spectrum_protection(
        base_score=np.array([3.0, 3.0, 3.0]),
        base_supported=np.array([True, True, False]),
        protection=np.array([0.5, np.nan, 0.5]),
        protection_active=np.array([True, False, True]),
        weight=2.0,
    )
    assert np.allclose(score, [2.0, 3.0, 0.0])
    assert support.tolist() == [True, True, False]


def test_frozen_weight_and_candidate_identities() -> None:
    assert n149.WEIGHTS == (0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)
    assert hashlib.sha256(
        "[0.0,1.0,2.0,4.0,8.0,16.0,32.0,64.0]".encode()
    ).hexdigest() == n149.EXPECTED_WEIGHT_GRID_SHA256
    assert n149.EXPECTED_CANDIDATE_KEY_SHA256 == (
        "6bb64825118b08264598350af117d9dcd971066cdcd3cdac004d6ab2d530e146"
    )
