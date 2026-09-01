import hashlib

import numpy as np

import src.next140_analytic_field_protection_search as n140


def test_apply_analytic_field_protection_is_subtractive_and_fail_open() -> None:
    score, support = n140.apply_analytic_field_protection(
        base_score=np.array([3.0, 3.0, 3.0]),
        base_supported=np.array([True, True, False]),
        protection=np.array([0.5, np.nan, 0.5]),
        protection_active=np.array([True, False, True]),
        weight=2.0,
    )

    assert np.allclose(score, [2.0, 3.0, 0.0])
    assert support.tolist() == [True, True, False]


def test_frozen_weight_and_candidate_identities() -> None:
    assert n140.WEIGHTS == (0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
    assert hashlib.sha256(
        "[0.0,0.1,0.25,0.5,1.0,2.0,4.0,8.0]".encode()
    ).hexdigest() == n140.EXPECTED_WEIGHT_GRID_SHA256
