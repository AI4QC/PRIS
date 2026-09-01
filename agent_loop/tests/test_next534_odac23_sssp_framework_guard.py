from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.next534_odac23_sssp_framework_guard import (
    SSSP_CUTOFF,
    apply_sssp_framework_guard,
    search_two_partition_guard,
    sssp_deficit,
)


def _formula() -> dict[str, object]:
    return {
        "kind": "additive",
        "missing_policy": "KEEP",
        "domain_gate": {
            "periodic_dimension_max_min": 1.0,
            "periodic_framework_fraction_min": 0.5,
        },
        "terms": [
            {
                "feature": "risk",
                "direction": 1,
                "center": 0.0,
                "scale": 1.0,
                "weight": 1.0,
            }
        ],
        "threshold": 1.0,
    }


def _partition(role: str, n_each: int = 200) -> pd.DataFrame:
    protected = pd.DataFrame(
        {
            "material_id": [f"{role}-p-{index}" for index in range(n_each)],
            "partition_role": role,
            "combined_supported": True,
            "periodic_dimension_max": 3.0,
            "periodic_framework_fraction": 1.0,
            "risk": 0.0,
            "sssp_same_sign_shell_purity_q10": 0.8,
            "sssp_supported": True,
            "robust_aligned_framework_displacement_p95_median": 0.01,
            "defective": False,
            "open_metal_site": False,
        }
    )
    severe = pd.DataFrame(
        {
            "material_id": [f"{role}-s-{index}" for index in range(n_each)],
            "partition_role": role,
            "combined_supported": True,
            "periodic_dimension_max": 3.0,
            "periodic_framework_fraction": 1.0,
            "risk": 10.0,
            "sssp_same_sign_shell_purity_q10": 0.1,
            "sssp_supported": True,
            "robust_aligned_framework_displacement_p95_median": 0.3,
            "defective": False,
            "open_metal_site": False,
        }
    )
    return pd.concat([protected, severe], ignore_index=True)


def test_sssp_deficit_is_bounded_and_missing_is_exactly_zero() -> None:
    values = np.asarray([0.0, SSSP_CUTOFF / 2.0, SSSP_CUTOFF, 0.9, math.nan])
    supported = np.asarray([True, True, True, True, False])
    actual = sssp_deficit(values, supported)
    np.testing.assert_allclose(actual, [1.0, 0.5, 0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="support semantics differ"):
        sssp_deficit(np.asarray([math.nan]), np.asarray([True]))


def test_guard_adds_only_the_frozen_deficit_and_preserves_base_fail_open() -> None:
    frame = pd.DataFrame(
        {
            "combined_supported": [True, True, False],
            "periodic_dimension_max": [3.0, 3.0, 3.0],
            "periodic_framework_fraction": [1.0, 1.0, 1.0],
            "risk": [1.0, 2.0, 10.0],
            "sssp_same_sign_shell_purity_q10": [0.0, math.nan, 0.0],
            "sssp_supported": [True, False, True],
        }
    )
    score, supported, reject, deficit = apply_sssp_framework_guard(
        frame, _formula(), weight=0.5, threshold=1.4
    )
    np.testing.assert_allclose(score[:2], [1.5, 2.0])
    assert math.isnan(score[2])
    np.testing.assert_allclose(deficit, [1.0, 0.0, 1.0])
    assert supported.tolist() == [True, True, False]
    assert reject.tolist() == [True, True, False]


def test_two_partition_search_selects_an_eligible_shared_threshold() -> None:
    frame = pd.concat(
        [_partition("discovery"), _partition("internal_validation")],
        ignore_index=True,
    )
    result = search_two_partition_guard(frame, _formula(), weights=(0.0, 0.5))
    assert result["eligible_candidate_count"] >= 1
    assert result["passes_two_partition_readiness"] is True
    assert result["selected_formula"]["sssp_deficit_weight"] == 0.0
    assert result["selected_formula"]["threshold"] == 10.0
    for role in ("discovery", "internal_validation"):
        assert result["selected_metrics"][role]["passes_all_gates"] is True
    assert result["selected_metrics"]["combined"]["reject_precision_lower"] >= 0.80


def test_search_fails_closed_on_replication_or_a_partition_gate_failure() -> None:
    discovery = _partition("discovery")
    validation = _partition("internal_validation")
    validation["risk"] = 0.0
    stopped = search_two_partition_guard(
        pd.concat([discovery, validation], ignore_index=True),
        _formula(),
        weights=(0.0,),
    )
    assert stopped["passes_two_partition_readiness"] is False
    assert stopped["selected_formula"] is None

    leaked = discovery.copy()
    leaked["partition_role"] = "internal_replication"
    with pytest.raises(ValueError, match="development roles differ"):
        search_two_partition_guard(
            pd.concat([discovery, leaked], ignore_index=True),
            _formula(),
            weights=(0.0,),
        )
