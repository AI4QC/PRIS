from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from src.next190_latent_symmetry_recoverability_audit import (
    HYPOTHESES,
    eligibility_from_metrics,
    normalize_recoverability,
    run_latent_symmetry_recoverability_audit,
    select_recoverability_hypothesis,
)


def test_hypothesis_universe_is_exactly_six_recoverable_high_directions() -> None:
    features = (
        "sym_recovery_onset_rel",
        "sym_recovery_gain_log2",
        "sym_orbit_collapse",
        "sym_recovery_residual_rms_rel",
        "sym_recovery_residual_q95_rel",
        "sym_recovery_residual_max_rel",
    )
    assert HYPOTHESES == {
        f"{feature}__recoverable_high": (feature, 1) for feature in features
    }


@pytest.mark.parametrize(
    ("feature", "values", "expected"),
    [
        ("sym_recovery_onset_rel", [0.0, 0.06, 0.12, 0.24], [0.0, 0.5, 1.0, 1.0]),
        (
            "sym_recovery_gain_log2",
            [0.0, np.log2(48.0) / 2.0, np.log2(48.0)],
            [0.0, 0.5, 1.0],
        ),
        ("sym_orbit_collapse", [-0.1, 0.5, 1.1], [0.0, 0.5, 1.0]),
        ("sym_recovery_residual_q95_rel", [0.0, 0.12, 0.24, 0.48], [0.0, 0.5, 1.0, 1.0]),
    ],
)
def test_physical_normalizers_are_fixed_and_bounded(
    feature: str, values: list[float], expected: list[float]
) -> None:
    actual = normalize_recoverability(feature=feature, values=values)
    assert actual == pytest.approx(expected)


def test_normalizer_preserves_missing_values_and_rejects_unknown_features() -> None:
    actual = normalize_recoverability(
        feature="sym_recovery_onset_rel", values=[np.nan, 0.12]
    )
    assert np.isnan(actual[0]) and actual[1] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="NEXT190 recoverability feature differs"):
        normalize_recoverability(feature="not_frozen", values=[0.0])


def test_eligibility_reuses_every_frozen_cross_source_gate() -> None:
    passing = dict(
        scigen_full_support=0.90,
        wyformer_full_support=0.90,
        scigen_shell_worst_auc=0.55,
        scigen_shell_evaluable_folds=5,
        wyformer_shell_pooled_auc=0.55,
        scigen_full_pooled_auc=0.50,
        wyformer_full_pooled_auc=0.50,
    )
    assert eligibility_from_metrics(**passing)
    for key in passing:
        failing = dict(passing)
        failing[key] = 4 if key.endswith("evaluable_folds") else float(failing[key]) - 1.0e-6
        assert not eligibility_from_metrics(**failing)


def test_selector_ranks_only_eligible_rows_deterministically() -> None:
    records = pd.DataFrame(
        {
            "hypothesis": ["b", "a", "c"],
            "eligible_for_search": [True, True, False],
            "ranking_min_auc": [0.60, 0.60, 0.99],
            "ranking_mean_auc": [0.70, 0.70, 0.99],
        }
    )
    table, selected = select_recoverability_hypothesis(records)
    assert table["hypothesis"].tolist() == ["a", "b", "c"]
    assert selected is not None and selected["hypothesis"] == "a"


def test_formal_interface_has_discovery_but_no_validation_or_replication_paths() -> None:
    parameters = tuple(
        inspect.signature(run_latent_symmetry_recoverability_audit).parameters
    )
    assert "next186_dir" in parameters
    assert "next188_dir" in parameters
    assert "scigen_discovery_endpoint_dir" in parameters
    assert "wyformer_discovery_endpoint_dir" in parameters
    assert not any(
        token in name for name in parameters for token in ("validation", "replication")
    )


def test_formal_audit_fails_closed_on_missing_input(tmp_path) -> None:
    signature = inspect.signature(run_latent_symmetry_recoverability_audit)
    kwargs = {
        name: tmp_path / name
        for name in signature.parameters
        if name != "require_formal_inputs"
    }
    kwargs["require_formal_inputs"] = False
    with pytest.raises(FileNotFoundError, match="NEXT190 input is missing"):
        run_latent_symmetry_recoverability_audit(**kwargs)
