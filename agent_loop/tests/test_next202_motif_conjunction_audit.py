from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from src.next202_motif_conjunction_audit import (
    CONJUNCTIONS,
    FLOOR_LEVELS,
    HYPOTHESES,
    SECONDARY_FEATURES,
    eligibility_from_metrics,
    motif_conjunction_certificate,
    secondary_cleanliness,
    run_motif_conjunction_audit,
    select_motif_conjunction_hypothesis,
    weakest_site_confidence,
)


def test_hypothesis_universe_is_exactly_twenty_four_frozen_high_conjunctions() -> None:
    expected = {
        f"motif_weight_sum_min__{label}__{secondary}__{conjunction}__protected_high": (
            secondary,
            floor,
            conjunction,
            1,
        )
        for secondary in SECONDARY_FEATURES
        for label, floor in FLOOR_LEVELS
        for conjunction in CONJUNCTIONS
    }
    assert HYPOTHESES == expected
    assert len(HYPOTHESES) == 24


def test_weakest_site_confidence_is_the_exact_frozen_ramp_and_missing_safe() -> None:
    actual = weakest_site_confidence(
        [np.nan, -1.0, 0.5, 0.75, 0.875, 1.5], floor_threshold=0.75
    )
    assert np.isnan(actual[0])
    assert actual[1:] == pytest.approx([0.0, 0.0, 0.0, 0.5, 1.0])
    with pytest.raises(ValueError, match="NEXT202 floor threshold differs"):
        weakest_site_confidence([1.0], floor_threshold=1.0)


@pytest.mark.parametrize(
    ("feature", "values", "expected"),
    [
        (
            "motif_global_dispersion_rms",
            [np.nan, -1.0, 0.0, 1.0, 3.0],
            [np.nan, 1.0, 1.0, 0.5, 0.25],
        ),
        (
            "motif_weight_sum_std",
            [np.nan, -1.0, 0.0, 0.25, 0.5, 1.0],
            [np.nan, 1.0, 1.0, 0.5, 0.0, 0.0],
        ),
    ],
)
def test_secondary_cleanliness_uses_only_the_two_exact_physical_maps(
    feature: str, values: list[float], expected: list[float]
) -> None:
    actual = secondary_cleanliness(values, feature=feature)
    assert np.isnan(actual[0])
    assert actual[1:] == pytest.approx(expected[1:])
    with pytest.raises(ValueError, match="NEXT202 secondary feature differs"):
        secondary_cleanliness([0.0], feature="unknown")


@pytest.mark.parametrize(
    ("conjunction", "expected"),
    [
        ("product", [0.0, 0.25, 1.0, np.nan]),
        ("minimum", [0.0, 0.5, 1.0, np.nan]),
    ],
)
def test_conjunction_uses_common_support_and_exact_product_or_minimum(
    conjunction: str, expected: list[float]
) -> None:
    actual = motif_conjunction_certificate(
        weakest_site=[0.0, 0.5, 1.0, np.nan],
        secondary=[1.0, 0.5, 1.0, 1.0],
        conjunction=conjunction,
    )
    assert actual[:3] == pytest.approx(expected[:3])
    assert np.isnan(actual[3])
    with pytest.raises(ValueError, match="NEXT202 conjunction differs"):
        motif_conjunction_certificate(
            weakest_site=[1.0], secondary=[1.0], conjunction="sum"
        )


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
    table, selected = select_motif_conjunction_hypothesis(records)
    assert table["hypothesis"].tolist() == ["a", "b", "c"]
    assert selected is not None and selected["hypothesis"] == "a"


def test_formal_interface_has_discovery_but_no_validation_or_replication_paths() -> None:
    parameters = tuple(inspect.signature(run_motif_conjunction_audit).parameters)
    assert "next201_dir" in parameters
    assert "scigen_discovery_endpoint_dir" in parameters
    assert "wyformer_discovery_endpoint_dir" in parameters
    assert not any(
        token in name for name in parameters for token in ("validation", "replication")
    )


def test_formal_audit_fails_closed_on_missing_input(tmp_path) -> None:
    signature = inspect.signature(run_motif_conjunction_audit)
    kwargs = {
        name: tmp_path / name
        for name in signature.parameters
        if name != "require_formal_inputs"
    }
    kwargs["require_formal_inputs"] = False
    with pytest.raises(FileNotFoundError, match="NEXT202 input is missing"):
        run_motif_conjunction_audit(**kwargs)
