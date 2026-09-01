from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

import src.next183_conditional_nonlocal_closure_audit as n183
from src.next194_signed_local_closure_audit import (
    HYPOTHESES,
    eligibility_from_metrics,
    normalize_signed_local_safety,
    run_signed_local_closure_audit,
    select_signed_local_closure_hypothesis,
    signed_local_closure_certificate,
)


def test_hypothesis_universe_is_exactly_twelve_high_conjunctions() -> None:
    assert HYPOTHESES == {
        f"{feature}__signed_local_safe__{conjunction}__high": (
            feature,
            conjunction,
            1,
        )
        for feature in n183.CLOSURE_FEATURES
        for conjunction in ("product", "minimum")
    }


def test_signed_local_safety_normalization_is_fixed_bounded_and_missing_safe() -> None:
    actual = normalize_signed_local_safety([np.nan, -0.1, 0.0, 0.25, 0.5, 1.0])
    assert np.isnan(actual[0])
    assert actual[1:] == pytest.approx([0.0, 0.0, 0.5, 1.0, 1.0])


@pytest.mark.parametrize(
    ("conjunction", "expected"),
    [
        ("product", [0.0, 0.25, 0.0, np.nan]),
        ("minimum", [0.0, 0.5, 0.0, np.nan]),
    ],
)
def test_certificate_uses_exact_product_or_minimum_and_common_support(
    conjunction: str, expected: list[float]
) -> None:
    actual = signed_local_closure_certificate(
        closure=[0.0, 0.5, 1.0, np.nan],
        signed_local_safety=[0.5, 0.25, 0.0, np.nan],
        conjunction=conjunction,
    )
    assert actual[:3] == pytest.approx(expected[:3])
    assert np.isnan(actual[3])


def test_certificate_rejects_unknown_conjunction_and_out_of_bounds_closure() -> None:
    with pytest.raises(ValueError, match="NEXT194 conjunction differs"):
        signed_local_closure_certificate(
            closure=[0.5], signed_local_safety=[0.5], conjunction="sum"
        )
    with pytest.raises(ValueError, match="NEXT194 closure is outside bounds"):
        signed_local_closure_certificate(
            closure=[1.1], signed_local_safety=[0.5], conjunction="product"
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
    table, selected = select_signed_local_closure_hypothesis(records)
    assert table["hypothesis"].tolist() == ["a", "b", "c"]
    assert selected is not None and selected["hypothesis"] == "a"


def test_formal_interface_has_discovery_but_no_validation_or_replication_paths() -> None:
    parameters = tuple(inspect.signature(run_signed_local_closure_audit).parameters)
    assert "next192_dir" in parameters
    assert "scigen_discovery_endpoint_dir" in parameters
    assert "wyformer_discovery_endpoint_dir" in parameters
    assert not any(
        token in name for name in parameters for token in ("validation", "replication")
    )


def test_formal_audit_fails_closed_on_missing_input(tmp_path) -> None:
    signature = inspect.signature(run_signed_local_closure_audit)
    kwargs = {
        name: tmp_path / name
        for name in signature.parameters
        if name != "require_formal_inputs"
    }
    kwargs["require_formal_inputs"] = False
    with pytest.raises(FileNotFoundError, match="NEXT194 input is missing"):
        run_signed_local_closure_audit(**kwargs)
