from __future__ import annotations

import inspect
import numpy as np
import pandas as pd
import pytest

import src.next437_pvtm_feature_audit as n


def test_hypothesis_universe_freezes_one_high_direction() -> None:
    assert n.HYPOTHESES == (("pvtm_positive_transport_margin", "protected_high"),)
    assert n.QUANTILES == (1 / 16, 15 / 16)
    assert n.EXPECTED_INPUT_SHA256["design"] == n.n435.DESIGN_SHA256


def test_bounded_mapping_supports_only_high_direction() -> None:
    values = np.asarray([0, .1, .2, .3, .4, np.nan])
    mapped = n.bounded_protection(values=values, direction="protected_high", q_lo=.1, q_hi=.3)
    np.testing.assert_allclose(mapped[:5], [0, 0, .5, 1, 1])
    assert np.isnan(mapped[5])
    with pytest.raises(ValueError, match="NEXT437 bounded"):
        n.bounded_protection(values=values, direction="protected_low", q_lo=.1, q_hi=.3)


def test_selection_uses_frozen_gates() -> None:
    frame = pd.DataFrame({
        "hypothesis": ["z", "a", "b"],
        "passes_raw_gates": [True, True, False],
        "ranking_min_worst_fold_auc": [.6, .6, .9],
        "ranking_min_aggregate_auc": [.7, .7, .9],
        "ranking_mean_aggregate_auc": [.8, .8, .9],
    })
    selected, leader = n.select_eligible_hypotheses(frame)
    assert selected.set_index("hypothesis")["eligible_for_search"].to_dict() == {"a": True, "b": False, "z": True}
    assert leader is not None and leader["hypothesis"] == "a"


def test_pvtm_rows_recompute_bounded_formula_and_zero_semantics() -> None:
    bounded = np.asarray([0.0, 0.5, 0.2])
    raw = np.asarray([0.0, 1.0, 0.25])
    feasible = np.asarray([False, True, True])
    assert n._pvtm_rows_are_consistent(bounded, raw, feasible).all()
    changed = bounded.copy(); changed[-1] += 2e-10
    assert not n._pvtm_rows_are_consistent(changed, raw, feasible)[-1]
    invalid_raw = raw.copy(); invalid_raw[0] = 0.1
    assert not n._pvtm_rows_are_consistent(bounded, invalid_raw, feasible)[0]


def test_audit_interface_excludes_validation_and_replication() -> None:
    parameters = tuple(inspect.signature(n.run_pvtm_feature_audit).parameters)
    assert "next412_dir" in parameters and "next436_dir" in parameters
    assert "scigen_discovery_endpoint_dir" in parameters
    assert not any(token in name for name in parameters for token in ("validation", "replication"))
