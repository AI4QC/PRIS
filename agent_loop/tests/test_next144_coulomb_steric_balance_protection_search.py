import hashlib

import numpy as np

import src.next144_coulomb_steric_balance_protection_search as n144


def test_apply_coulomb_steric_balance_protection_is_subtractive_and_fail_open() -> None:
    score, support = n144.apply_coulomb_steric_balance_protection(
        base_score=np.array([3.0, 3.0, 3.0]),
        base_supported=np.array([True, True, False]),
        protection=np.array([0.5, np.nan, 0.5]),
        protection_active=np.array([True, False, True]),
        weight=2.0,
    )

    assert np.allclose(score, [2.0, 3.0, 0.0])
    assert support.tolist() == [True, True, False]


def test_frozen_weight_and_candidate_identities() -> None:
    assert n144.WEIGHTS == (0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
    assert hashlib.sha256(
        "[0.0,0.1,0.25,0.5,1.0,2.0,4.0,8.0]".encode()
    ).hexdigest() == n144.EXPECTED_WEIGHT_GRID_SHA256
    assert n144.EXPECTED_CANDIDATE_KEY_SHA256 == (
        "e80196eb8d646fd86c97c3f5a075d6e8acff5c8e8918d2ecdd361029050236ec"
    )
