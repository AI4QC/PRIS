import hashlib

import numpy as np

import src.next146_conditional_balance_exemption_search as n146


def test_apply_conditional_balance_exemption_is_sparse_and_fail_open() -> None:
    score, support, active = n146.apply_conditional_balance_exemption(
        base_score=np.array([3.0, 3.0, 3.0, 3.0]),
        base_supported=np.array([True, True, True, False]),
        residual=np.array([0.05, 0.2, np.nan, 0.05]),
        residual_supported=np.array([True, True, False, True]),
        cutoff=0.1,
        weight=0.5,
    )

    assert np.allclose(score, [2.5, 3.0, 3.0, 0.0])
    assert support.tolist() == [True, True, True, False]
    assert active.tolist() == [True, False, False, False]


def test_frozen_grids_and_candidate_identity() -> None:
    assert n146.CUTOFFS == (0.05, 0.1, 0.25, 0.5)
    assert n146.WEIGHTS == (0.1, 0.25, 0.5, 1.0, 2.0)
    assert hashlib.sha256("[0.05,0.1,0.25,0.5]".encode()).hexdigest() == (
        n146.EXPECTED_CUTOFF_GRID_SHA256
    )
    assert n146.EXPECTED_CANDIDATE_KEY_SHA256 == (
        "3ea4eb3c1817656f8e9ed24e35750cde30665255b48f223513e36fbb793727c4"
    )
