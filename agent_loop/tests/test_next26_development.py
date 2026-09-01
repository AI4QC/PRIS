"""Freeze/apply/evaluate contracts for the NEXT26 analytic rule loop."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _tables(n: int = 240) -> tuple[pd.DataFrame, pd.DataFrame]:
    ids = [f"x-{i:04d}" for i in range(n)]
    risk = np.linspace(-2.0, 2.0, n)
    features = pd.DataFrame(
        {
            "material_id": ids,
            "cov_packing": risk,
            "density_proxy": risk + 0.02,
            "nonbond_vdw_q01": -risk,
            "nonbond_clash_frac085": np.maximum(risk, 0),
            "bond_ratio_sd": np.full(n, 0.1),
            "cell_anisotropy": np.full(n, 1.2),
            "volume_pa": 10.0 - risk,
        }
    )
    positive = risk >= 1.2
    endpoints = pd.DataFrame(
        {
            "material_id": ids,
            "force0_max": np.where(positive, 1.2, 0.2),
            "force0_rms": np.where(positive, 0.5, 0.1),
            "energy_drop_pa": np.where(positive, 0.05, 0.01),
            "stress0_norm": np.where(positive, 0.04, 0.005),
        }
    )
    return features, endpoints


def test_severe_endpoint_thresholds_are_inclusive() -> None:
    from src.next26_omc25 import severe_dft_response

    frame = pd.DataFrame(
        {
            "force0_max": [1.0, 0.0, 0.0, 0.0, 0.0],
            "force0_rms": [0.0, 0.4, 0.0, 0.0, 0.0],
            "energy_drop_pa": [0.0, 0.0, 0.04, 0.0, 0.0],
            "stress0_norm": [0.0, 0.0, 0.0, 0.03, 0.0],
        }
    )
    assert severe_dft_response(frame).tolist() == [True, True, True, True, False]


def test_development_search_freezes_small_analytic_rule_and_never_dense_model(
    tmp_path: Path,
) -> None:
    from src.next26_development import FROZEN_RULE_NAME, search_and_freeze

    features, endpoints = _tables()
    out = tmp_path / "freeze"
    manifest = search_and_freeze(features=features, endpoints=endpoints, output_dir=out)

    rule = pd.read_json(out / FROZEN_RULE_NAME, typ="series")
    assert bool(rule["eligible"])
    assert 1 <= len(rule["terms"]) <= 2
    assert rule["formula_family"] in {
        "signed_robust_z",
        "equal_weight_signed_robust_z_sum",
        "conjunctive_signed_robust_z_min",
        "absolute_robust_z",
    }
    assert float(rule["endpoint_auc"]) > 0.95
    assert "coefficients" not in rule
    assert manifest["development_labels_opened"] is True
    assert manifest["prospective_labels_opened"] is False


def test_frozen_rule_application_is_label_free_and_fail_open(tmp_path: Path) -> None:
    from src.next26_apply_rule import apply_frozen_rule
    from src.next26_development import search_and_freeze

    features, endpoints = _tables()
    freeze = tmp_path / "freeze"
    search_and_freeze(features=features, endpoints=endpoints, output_dir=freeze)
    holdout = features.copy()
    holdout.loc[0, [column for column in holdout.columns if column != "material_id"]] = np.nan
    output = tmp_path / "predictions"
    manifest = apply_frozen_rule(
        frozen_rule_path=freeze / "FROZEN_RULE.json",
        rule_manifest_path=freeze / "MANIFEST.json",
        features=holdout,
        output_dir=output,
    )
    predictions = pd.read_parquet(output / "next26_predictions.parquet")

    assert not bool(predictions.loc[0, "reject"])
    assert manifest["labels_opened"] is False
    assert manifest["endpoint_fields_read"] is False
    assert manifest["missing_policy"] == "fail_open_do_not_reject"


def test_decision_metrics_use_wilson_lower_bounds_and_all_four_gates() -> None:
    from src.next26_evaluate import decision_metrics

    positive = np.array([True] * 120 + [False] * 880)
    supported = np.ones(1000, dtype=bool)
    reject = np.array([True] * 120 + [False] * 880)
    metrics = decision_metrics(supported=supported, reject=reject, endpoint_positive=positive)
    assert metrics["coverage_lower"] >= 0.95
    assert metrics["endpoint_negative_protection_lower"] >= 0.95
    assert metrics["endpoint_positive_precision_lower"] >= 0.90
    assert metrics["savings_lower"] >= 0.10
    assert metrics["passes_primary_gates"] is True


def test_search_rejects_feature_columns_that_cross_no_dft_contract(tmp_path: Path) -> None:
    from src.next26_development import search_and_freeze

    features, endpoints = _tables()
    features["dft_energy_proxy"] = 0.0
    with pytest.raises(ValueError, match="no-DFT"):
        search_and_freeze(features=features, endpoints=endpoints, output_dir=tmp_path / "x")


def test_ineligible_search_still_records_best_diagnostic_threshold(tmp_path: Path) -> None:
    from src.next26_development import CANDIDATES_NAME, search_and_freeze

    features, endpoints = _tables(120)
    endpoints.loc[:, ["force0_max", "force0_rms", "energy_drop_pa", "stress0_norm"]] = 0.0
    output = tmp_path / "ineligible"
    manifest = search_and_freeze(features=features, endpoints=endpoints, output_dir=output)
    candidates = pd.read_parquet(output / CANDIDATES_NAME)

    assert manifest["eligible"] is False
    assert candidates["diagnostic_threshold_only"].all()
    assert candidates["threshold"].notna().all()
    assert (candidates["rejected"] > 0).all()
