from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

import src.next507_mvch_feature_audit as n


def test_hypothesis_universe_freezes_one_high_direction() -> None:
    assert n.HYPOTHESES == (
        ("mvch_madelung_valence_class_homogeneity", "protected_high"),
    )
    assert n.QUANTILES == (1 / 16, 15 / 16)


def test_bounded_mapping_supports_only_frozen_high_direction() -> None:
    values = np.asarray([0, 0.25, 0.5, 0.75, 1, np.nan])
    mapped = n.bounded_protection(
        values=values, direction="protected_high", q_lo=0.25, q_hi=0.75
    )
    np.testing.assert_allclose(mapped[:5], [0, 0, 0.5, 1, 1])
    assert np.isnan(mapped[5])
    with pytest.raises(ValueError, match="NEXT507 bounded"):
        n.bounded_protection(
            values=values,
            direction="protected_low",
            q_lo=0.25,
            q_hi=0.75,
        )


def test_selection_uses_unchanged_frozen_gates_and_rank() -> None:
    frame = pd.DataFrame(
        {
            "hypothesis": ["z", "a", "b"],
            "passes_raw_gates": [True, True, False],
            "ranking_min_worst_fold_auc": [0.6, 0.6, 0.9],
            "ranking_min_aggregate_auc": [0.7, 0.7, 0.9],
            "ranking_mean_aggregate_auc": [0.8, 0.8, 0.9],
        }
    )
    selected, leader = n.select_eligible_hypotheses(frame)
    assert selected.set_index("hypothesis")["eligible_for_search"].to_dict() == {
        "a": True,
        "b": False,
        "z": True,
    }
    assert leader is not None and leader["hypothesis"] == "a"


def test_formal_rows_require_bounded_finite_anova_semantics() -> None:
    assert n._mvch_rows_are_consistent(
        values=np.asarray([0.0, 0.5, 1.0]),
        class_counts=np.asarray([1, 2, 3]),
        within=np.asarray([1.0, 0.5, 0.0]),
        total=np.asarray([1.0, 1.0, 0.0]),
    ).all()
    changed = n._mvch_rows_are_consistent(
        values=np.asarray([-0.1, 0.5, 1.1]),
        class_counts=np.asarray([1, 0, 3]),
        within=np.asarray([0.0, 2.0, 0.0]),
        total=np.asarray([1.0, 1.0, 1.0]),
    )
    assert changed.tolist() == [False, False, False]


def test_audit_interface_excludes_validation_and_replication() -> None:
    parameters = tuple(inspect.signature(n.run_mvch_feature_audit).parameters)
    assert "next506_dir" in parameters and "stage_dirs" in parameters
    assert "scigen_discovery_endpoint_dir" in parameters
    assert not any(
        token in name
        for name in parameters
        for token in ("validation", "replication")
    )


def test_audit_fails_closed_on_missing_input(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="NEXT507 input is missing"):
        n.run_mvch_feature_audit(
            scigen_feature_dir=tmp_path / "sf",
            scigen_discovery_endpoint_dir=tmp_path / "se",
            wyformer_feature_dir=tmp_path / "wf",
            wyformer_discovery_endpoint_dir=tmp_path / "we",
            stage_dirs={
                stage: tmp_path / f"n{stage}" for stage in n.REQUIRED_STAGES
            },
            next135_freeze_path=tmp_path / "freeze",
            design_paths={
                stage: tmp_path / f"d{stage}"
                for stage in n.REQUIRED_DESIGN_STAGES
            },
            design_path=tmp_path / "design",
            next412_dir=tmp_path / "n412",
            next506_dir=tmp_path / "n506",
            output_dir=tmp_path / "out",
            require_formal_inputs=False,
        )
