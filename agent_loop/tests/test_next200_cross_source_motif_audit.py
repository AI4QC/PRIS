from __future__ import annotations

import inspect

import pandas as pd
import pytest

from src.next200_cross_source_motif_audit import (
    HYPOTHESES,
    eligibility_from_metrics,
    run_cross_source_motif_audit,
    select_motif_hypothesis,
)


HIGH_PROTECTION = {
    "motif_weight_sum_mean",
    "motif_weight_sum_min",
    "motif_cn_dominance_mean",
    "motif_cn_dominance_min",
    "motif_effective_cn_mean",
    "motif_order_strength_mean",
    "motif_order_strength_min",
    "motif_fingerprint_norm_mean",
    "motif_species_centroid_separation_mean",
}

LOW_PROTECTION = {
    "motif_weight_sum_std",
    "motif_cn_dominance_std",
    "motif_cn_entropy_mean",
    "motif_cn_entropy_q95",
    "motif_effective_cn_std",
    "motif_effective_cn_range",
    "motif_order_strength_std",
    "motif_fingerprint_norm_std",
    "motif_same_element_dispersion_rms",
    "motif_same_element_dispersion_q95",
    "motif_same_element_dispersion_max",
    "motif_global_dispersion_rms",
}


def test_hypothesis_universe_is_exactly_one_frozen_physical_direction_per_feature() -> None:
    expected = {
        **{f"{feature}__protected_high": (feature, 1) for feature in HIGH_PROTECTION},
        **{f"{feature}__protected_low": (feature, -1) for feature in LOW_PROTECTION},
    }
    assert HYPOTHESES == expected
    assert len(HYPOTHESES) == 21


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
    table, selected = select_motif_hypothesis(records)
    assert table["hypothesis"].tolist() == ["a", "b", "c"]
    assert selected is not None and selected["hypothesis"] == "a"


def test_formal_interface_has_discovery_but_no_validation_or_replication_paths() -> None:
    parameters = tuple(inspect.signature(run_cross_source_motif_audit).parameters)
    assert "next199_dir" in parameters
    assert "next194_dir" in parameters
    assert "scigen_discovery_endpoint_dir" in parameters
    assert "wyformer_discovery_endpoint_dir" in parameters
    assert not any(
        token in name for name in parameters for token in ("validation", "replication")
    )


def test_formal_audit_fails_closed_on_missing_input(tmp_path) -> None:
    signature = inspect.signature(run_cross_source_motif_audit)
    kwargs = {
        name: tmp_path / name
        for name in signature.parameters
        if name != "require_formal_inputs"
    }
    kwargs["require_formal_inputs"] = False
    with pytest.raises(FileNotFoundError, match="NEXT200 input is missing"):
        run_cross_source_motif_audit(**kwargs)
