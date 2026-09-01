from __future__ import annotations

import numpy as np
import pytest

from src.next134_compactness_protection_search import (
    EXPECTED_CANDIDATE_COUNT,
    EXPECTED_CONFIGURATION_COUNT,
    build_compactness_configurations,
    compose_compactness_protection_score,
)


def test_frozen_configuration_and_candidate_counts_are_exact() -> None:
    configurations = build_compactness_configurations()
    assert len(configurations) == EXPECTED_CONFIGURATION_COUNT == 49
    assert EXPECTED_CANDIDATE_COUNT == 539
    assert sum(not item["term_ids"] for item in configurations) == 1
    assert sum(len(item["term_ids"]) == 1 for item in configurations) == 12
    assert sum(len(item["term_ids"]) == 2 for item in configurations) == 36


def test_compactness_terms_fail_open_independently_and_preserve_base_support() -> None:
    score, support = compose_compactness_protection_score(
        base_score=np.array([2.0, 1.0, 3.0]),
        base_supported=np.array([True, True, False]),
        protections=[np.array([0.5, 0.8, 0.2]), np.array([1.0, 1.0, 1.0])],
        active=[np.array([True, False, True]), np.array([False, True, True])],
        weights=[2.0, 2.0],
    )
    assert support.tolist() == [True, True, False]
    assert score[:2].tolist() == pytest.approx([1.0, 0.0])
    assert np.isnan(score[2])
