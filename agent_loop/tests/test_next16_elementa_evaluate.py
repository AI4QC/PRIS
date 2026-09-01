"""Contracts for the frozen NEXT16 ELEMENTA cross-source evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _inputs(tmp_path: Path) -> dict[str, Path]:
    from src.next16_elementa_basin_hull import PROTOCOL as FEATURE_PROTOCOL

    ids = [f"m{i}" for i in range(8)]
    groups = ["Li2O"] * 3 + ["NaCl"] * 2 + ["MgO"] * 3
    scores = [0.02, 0.30, 0.10, 0.03, 0.25, 0.04, 0.08, 0.22]
    basin_decisions = ["KEEP", "REJECT", "KEEP", "KEEP", "REJECT", "KEEP", "KEEP", "REJECT"]
    pauling_decisions = ["KEEP", "REJECT", "ABSTAIN", "KEEP", "ABSTAIN", "KEEP", "KEEP", "REJECT"]
    features = tmp_path / "elementa_basin_hull_features.parquet"
    pd.DataFrame(
        {
            "material_id": ids,
            "rk": groups,
            "supported": [True] * 8,
            "error": [""] * 8,
            "capped_at_max_steps": [False] * 7 + [True],
            "basin_hull_score_ev_per_atom": scores,
            "basin_hull_decision": basin_decisions,
            "pauling_p2_p5_decision": pauling_decisions,
        }
    ).to_parquet(features, index=False)
    feature_manifest = tmp_path / "feature-manifest.json"
    _write_json(
        feature_manifest,
        {
            "protocol": FEATURE_PROTOCOL,
            "elementa_endpoint_bytes_read_by_execution": False,
            "thresholds_refit": False,
            "rule": {
                "formula": "B64 = E_MatterSim_relaxed/N - E_raw_MP_hull(composition)",
                "comparison": ">=",
                "threshold_ev_per_atom": 0.15,
                "failure_policy": "ABSTAIN",
                "selection_origin": "post hoc WBM development sweep; frozen before this ELEMENTA execution",
            },
            "outputs_sha256": {features.name: _sha(features)},
        },
    )

    labels = tmp_path / "elementa_labels.parquet"
    pd.DataFrame(
        {
            "sid": ids,
            "rk": groups,
            "material": ["test"] * 8,
            "formula": groups,
            "e_per_atom": [-2.0, -1.70, -1.90, -3.0, -2.75, -4.0, -3.96, -3.79],
            "nat": [3, 3, 3, 2, 2, 2, 2, 2],
            "final_ionic_step": [10] * 8,
            "final_max_force": [0.01] * 8,
        }
    ).to_parquet(labels, index=False)
    labels_manifest = tmp_path / "labels-manifest.json"
    _write_json(
        labels_manifest,
        {
            "protocol": "2026-08-01-dft-pre-screening-design-v1",
            "input_role": "unrelaxed_x0_only",
            "outputs_sha256": {labels.name: _sha(labels)},
        },
    )
    return locals()


def test_evaluator_keeps_identifiers_external_and_measures_group_regret(
    tmp_path: Path,
) -> None:
    from src import next16_elementa_evaluate as module

    paths = _inputs(tmp_path)
    private = tmp_path / "private"
    aggregate = tmp_path / "aggregate"
    result = module.evaluate_elementa_retrospective(
        features_path=paths["features"],
        features_manifest_path=paths["feature_manifest"],
        labels_path=paths["labels"],
        labels_manifest_path=paths["labels_manifest"],
        private_output_dir=private,
        aggregate_output_dir=aggregate,
        bootstrap_reps=100,
        require_formal_inputs=False,
    )
    method = result["methods"]["next16_basin_hull"]
    assert result["candidate_selected_on"] == "WBM"
    assert result["thresholds_refit"] is False
    assert result["fresh_lockbox"] is False
    assert result["counts"]["complete_composition_groups"] == 3
    assert method["dft_savings"]["estimate"] == pytest.approx(3 / 8)
    assert method["group_minimum_recall"]["estimate"] == 1.0
    assert method["group_best_retention"]["estimate"] == 1.0
    assert method["valuable_recall"]["estimate"] == 1.0
    assert method["high_energy_rejection_recall"]["estimate"] == 1.0
    assert method["reject_precision_above_minimum"]["estimate"] == 1.0
    assert method["all_rejected_groups"] == 0
    assert (private / module.PRIVATE_JOINED_NAME).is_file()
    payload = (aggregate / module.RESULT_NAME).read_bytes()
    assert b'"m0"' not in payload and b'"material_id"' not in payload


def test_evaluator_rejects_a_decision_not_implied_by_the_frozen_score(
    tmp_path: Path,
) -> None:
    from src import next16_elementa_evaluate as module

    paths = _inputs(tmp_path)
    table = pd.read_parquet(paths["features"])
    table.loc[0, "basin_hull_decision"] = "REJECT"
    table.to_parquet(paths["features"], index=False)
    manifest = json.loads(paths["feature_manifest"].read_text(encoding="utf-8"))
    manifest["outputs_sha256"][paths["features"].name] = _sha(paths["features"])
    _write_json(paths["feature_manifest"], manifest)
    with pytest.raises(ValueError, match="decision differs"):
        module.evaluate_elementa_retrospective(
            features_path=paths["features"],
            features_manifest_path=paths["feature_manifest"],
            labels_path=paths["labels"],
            labels_manifest_path=paths["labels_manifest"],
            private_output_dir=tmp_path / "private",
            aggregate_output_dir=tmp_path / "aggregate",
            bootstrap_reps=10,
            require_formal_inputs=False,
        )


def test_evaluator_requires_complete_selected_composition_groups(tmp_path: Path) -> None:
    from src import next16_elementa_evaluate as module

    paths = _inputs(tmp_path)
    labels = pd.read_parquet(paths["labels"])
    labels.loc[len(labels)] = ["extra", "Li2O", "test", "Li2O", -1.8, 3, 10, 0.01]
    labels.to_parquet(paths["labels"], index=False)
    manifest = json.loads(paths["labels_manifest"].read_text(encoding="utf-8"))
    manifest["outputs_sha256"][paths["labels"].name] = _sha(paths["labels"])
    _write_json(paths["labels_manifest"], manifest)
    with pytest.raises(ValueError, match="complete composition groups"):
        module.evaluate_elementa_retrospective(
            features_path=paths["features"],
            features_manifest_path=paths["feature_manifest"],
            labels_path=paths["labels"],
            labels_manifest_path=paths["labels_manifest"],
            private_output_dir=tmp_path / "private",
            aggregate_output_dir=tmp_path / "aggregate",
            bootstrap_reps=10,
            require_formal_inputs=False,
        )


def test_evaluator_cli_cannot_refit_or_select_rows() -> None:
    from src.next16_elementa_evaluate import main

    for forbidden in ("--threshold", "--fit", "--exclude-capped", "--sample", "--groups"):
        with pytest.raises(SystemExit) as exc_info:
            main([forbidden, "x"])
        assert exc_info.value.code == 2
