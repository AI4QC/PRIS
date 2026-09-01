from __future__ import annotations

import inspect
import numpy as np
import pandas as pd
import pytest

import src.next422_p4bss_feature_audit as n


def test_hypothesis_universe_freezes_one_high_direction() -> None:
    assert n.HYPOTHESES == (("p4bss_bond_strength_pair_avoidance", "protected_high"),)
    assert n.QUANTILES == (1 / 16, 15 / 16)
    assert n.EXPECTED_INPUT_SHA256["design"] == n.n420.DESIGN_SHA256
    assert n.EXPECTED_INPUT_SHA256["next413_design"] != n.n420.DESIGN_SHA256


def test_bounded_mapping_supports_only_frozen_high_direction() -> None:
    values = np.asarray([0, 0.25, 0.5, 0.75, 1, np.nan])
    high = n.bounded_protection(values=values, direction="protected_high", q_lo=0.25, q_hi=0.75)
    np.testing.assert_allclose(high[:5], [0, 0, 0.5, 1, 1])
    assert np.isnan(high[5])
    with pytest.raises(ValueError, match="NEXT422 bounded"):
        n.bounded_protection(values=values, direction="protected_low", q_lo=0.25, q_hi=0.75)


def test_selection_uses_frozen_gates_and_reporting_rank() -> None:
    frame = pd.DataFrame({
        "hypothesis": ["z", "a", "b"],
        "passes_raw_gates": [True, True, False],
        "ranking_min_worst_fold_auc": [0.6, 0.6, 0.9],
        "ranking_min_aggregate_auc": [0.7, 0.7, 0.9],
        "ranking_mean_aggregate_auc": [0.8, 0.8, 0.9],
    })
    selected, leader = n.select_eligible_hypotheses(frame)
    assert selected.set_index("hypothesis")["eligible_for_search"].to_dict() == {"a": True, "b": False, "z": True}
    assert leader is not None and leader["hypothesis"] == "a"


def test_identity_alignment_adds_source_prefix() -> None:
    table = pd.DataFrame({"material_id": ["a", "b"], n.n420.FEATURE_NAMES[0]: [0.1, 0.2]})
    indexed = n._index_by_id(table=table, source="scigen", expected_material_ids=pd.Series(["scigen:a", "scigen:b"]))
    assert indexed.index.tolist() == ["scigen:a", "scigen:b"]
    with pytest.raises(ValueError, match="material identity differs"):
        n._index_by_id(table=table, source="scigen", expected_material_ids=pd.Series(["scigen:a", "scigen:c"]))


def test_strength_moments_allow_only_frozen_output_grid_roundoff() -> None:
    expected = np.asarray([4.0, 9.0])
    observed = np.asarray([3.0, 1.0])
    values = np.asarray([round(4 / 7, 10), 0.9])
    assert n._strength_moments_are_consistent(values, expected, observed).all()
    changed = values.copy(); changed[0] += 2e-10
    assert not n._strength_moments_are_consistent(changed, expected, observed)[0]


def test_audit_interface_excludes_validation_and_replication() -> None:
    parameters = tuple(inspect.signature(n.run_p4bss_feature_audit).parameters)
    assert "next412_dir" in parameters and "next421_dir" in parameters
    assert "scigen_discovery_endpoint_dir" in parameters
    assert not any(token in name for name in parameters for token in ("validation", "replication"))
