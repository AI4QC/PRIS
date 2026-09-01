from __future__ import annotations

import inspect
import math

import numpy as np
import pandas as pd
import pytest

from src.next540_odac23_pbaaa_framework_guard import (
    apply_pbaaa_framework_guard,
    pbaaa_increment,
    run_two_partition_pbaaa_search,
    search_two_partition_pbaaa_guard,
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
            {"feature": "risk", "direction": 1, "center": 0.0, "scale": 1.0, "weight": 1.0}
        ],
        "threshold": 1.0,
    }


def _partition(role: str, n_each: int = 200) -> pd.DataFrame:
    rows = []
    for kind, risk, value, endpoint in (
        ("p", 0.0, 0.0, 0.01),
        ("s", 10.0, 1.0, 0.30),
    ):
        rows.extend(
            {
                "material_id": f"{role}-{kind}-{index}",
                "partition_role": role,
                "combined_supported": True,
                "periodic_dimension_max": 3.0,
                "periodic_framework_fraction": 1.0,
                "risk": risk,
                "pbaaa_periodic_bond_angle_affine_accommodation": value,
                "pbaaa_supported": True,
                "robust_aligned_framework_displacement_p95_median": endpoint,
                "defective": False,
                "open_metal_site": False,
            }
            for index in range(n_each)
        )
    return pd.DataFrame(rows)


def test_increment_is_bounded_and_missing_is_zero() -> None:
    actual = pbaaa_increment(
        np.asarray([0.0, 0.25, 1.0, math.nan]),
        np.asarray([True, True, True, False]),
    )
    np.testing.assert_allclose(actual, [0.0, 0.25, 1.0, 0.0])
    with pytest.raises(ValueError, match="support semantics differ"):
        pbaaa_increment(np.asarray([math.nan]), np.asarray([True]))


def test_apply_preserves_base_fail_open_and_missing_zero() -> None:
    frame = pd.DataFrame(
        {
            "combined_supported": [True, True, False],
            "periodic_dimension_max": [3.0, 3.0, 3.0],
            "periodic_framework_fraction": [1.0, 1.0, 1.0],
            "risk": [1.0, 2.0, 10.0],
            "pbaaa_periodic_bond_angle_affine_accommodation": [1.0, math.nan, 1.0],
            "pbaaa_supported": [True, False, True],
        }
    )
    score, supported, reject, increment = apply_pbaaa_framework_guard(
        frame, _formula(), weight=0.5, threshold=1.4
    )
    np.testing.assert_allclose(score[:2], [1.5, 2.0])
    assert math.isnan(score[2])
    assert supported.tolist() == [True, True, False]
    assert reject.tolist() == [True, True, False]
    np.testing.assert_allclose(increment, [1.0, 0.0, 1.0])


def test_search_requires_both_partitions_and_selects_shared_threshold() -> None:
    frame = pd.concat(
        [_partition("discovery"), _partition("internal_validation")], ignore_index=True
    )
    result = search_two_partition_pbaaa_guard(frame, _formula(), weights=(0.0, 0.5))
    assert result["passes_two_partition_readiness"] is True
    assert result["eligible_candidate_count"] >= 1
    assert result["selected_formula"]["pbaaa_weight"] == 0.0
    assert result["selected_formula"]["threshold"] == 10.0

    leaked = frame.copy()
    leaked.loc[leaked["partition_role"].eq("internal_validation"), "partition_role"] = (
        "internal_replication"
    )
    with pytest.raises(ValueError, match="development roles differ"):
        search_two_partition_pbaaa_guard(leaked, _formula(), weights=(0.0,))


def test_formal_runner_has_no_replication_endpoint_argument() -> None:
    parameters = set(inspect.signature(run_two_partition_pbaaa_search).parameters)
    assert parameters == {
        "framework_feature_dir",
        "pbaaa_feature_dir",
        "next79_dir",
        "endpoint_firewall_path",
        "discovery_dir",
        "validation_dir",
        "design_path",
        "output_dir",
        "require_formal_inputs",
    }
