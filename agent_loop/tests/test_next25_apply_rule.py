"""Contracts for label-blind NEXT25 transport of the frozen NEXT23 law."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> dict[str, object]:
    from src.next25_omatg_holdout import PROTOCOL as COHORT_PROTOCOL

    ids = ["next25-a", "next25-b", "next25-c"]
    metadata = tmp_path / "holdout_metadata.parquet"
    pd.DataFrame(
        {
            "material_id": ids,
            "rk": ["omatg_mp20_csp_linear_ode"] * 3,
            "formula": ["LiO"] * 3,
            "natoms": [2] * 3,
            "input_role": ["unrelaxed_x0_geometry_only"] * 3,
        }
    ).to_parquet(metadata, index=False)
    cohort_manifest = tmp_path / "cohort-MANIFEST.json"
    cohort_manifest.write_text(
        json.dumps(
            {
                "protocol": COHORT_PROTOCOL,
                "input_role": "unrelaxed_x0_geometry_only",
                "all_generator_outputs_retained": True,
                "labels_opened": False,
                "endpoint_artifacts_opened": False,
                "relaxed_structures_opened": False,
                "model_or_proxy_potential_used": False,
                "production_protocol_eligible": True,
                "outputs_sha256": {metadata.name: _sha(metadata)},
            }
        ),
        encoding="utf-8",
    )
    tables = {
        "sivr": pd.DataFrame(
            {
                "material_id": ids,
                "voronoi_q0__sivr_cell_anisotropy": [0.0, 1.0, np.nan],
            }
        ),
        "scbve": pd.DataFrame(
            {"material_id": ids, "scbv_vector_asymmetry_rms": [0.0, 1.0, 1.0]}
        ),
    }
    feature_paths = {}
    feature_manifests = {}
    for source, table in tables.items():
        feature = tmp_path / f"{source}.parquet"
        table.to_parquet(feature, index=False)
        manifest = tmp_path / f"{source}-MANIFEST.json"
        manifest.write_text(
            json.dumps(
                {
                    "input_role": "unrelaxed_x0_geometry_only",
                    "endpoint_fields_read": False,
                    "model_or_proxy_potential_used": False,
                    "coordinates_or_cell_modified": False,
                    "inputs_sha256": {"metadata": _sha(metadata)},
                    "outputs_sha256": {feature.name: _sha(feature)},
                }
            ),
            encoding="utf-8",
        )
        feature_paths[source] = feature
        feature_manifests[source] = manifest
    law = tmp_path / "rule.json"
    law.write_text(
        json.dumps(
            {
                "protocol": "2026-08-02-next23-relaxation-change-rule-freeze-v1",
                "eligible": True,
                "selected_candidate": "B+E",
                "selected_terms": ["B", "E"],
                "threshold": 1.0,
                "base_parameters": {
                    "B": {
                        "source": "sivr",
                        "column": "voronoi_q0__sivr_cell_anisotropy",
                        "direction": 1,
                        "median": 0.0,
                        "scale_iqr": 1.0,
                    },
                    "E": {
                        "source": "scbve",
                        "column": "scbv_vector_asymmetry_rms",
                        "direction": 1,
                        "median": 0.0,
                        "scale_iqr": 1.0,
                    },
                },
                "blind_labels_opened": False,
                "missing_policy": "fail_open_do_not_reject",
                "reject_when": "supported and score >= threshold",
            }
        ),
        encoding="utf-8",
    )
    law_manifest = tmp_path / "law-MANIFEST.json"
    law_manifest.write_text(
        json.dumps(
            {
                "protocol": "2026-08-02-next23-relaxation-change-rule-freeze-v1",
                "blind_labels_opened": False,
                "outputs_sha256": {law.name: _sha(law)},
            }
        ),
        encoding="utf-8",
    )
    return {
        "metadata": metadata,
        "cohort_manifest": cohort_manifest,
        "feature_paths": feature_paths,
        "feature_manifests": feature_manifests,
        "law": law,
        "law_manifest": law_manifest,
    }


def _run(paths: dict[str, object], output: Path):
    from src.next25_apply_rule import apply_transport_rule

    return apply_transport_rule(
        frozen_rule_path=paths["law"],
        rule_manifest_path=paths["law_manifest"],
        cohort_metadata_path=paths["metadata"],
        cohort_manifest_path=paths["cohort_manifest"],
        feature_paths=paths["feature_paths"],
        feature_manifest_paths=paths["feature_manifests"],
        output_dir=output,
    )


def test_next25_scores_exactly_fails_open_and_never_opens_endpoint(tmp_path: Path) -> None:
    from src import next25_apply_rule as module

    paths = _fixture(tmp_path)
    output = tmp_path / "predictions"
    manifest = _run(paths, output)
    predictions = pd.read_parquet(output / module.PREDICTIONS_NAME)
    assert predictions.analytic_supported.tolist() == [True, True, False]
    assert predictions.reject.tolist() == [False, True, False]
    assert predictions.next23_risk_score.iloc[:2].tolist() == pytest.approx([0.0, 2.0])
    assert np.isnan(predictions.next23_risk_score.iloc[2])
    assert manifest["thresholds_refit"] is False
    assert manifest["formula_or_parameters_changed"] is False
    assert manifest["blind_labels_opened"] is False
    assert manifest["endpoint_fields_read"] is False
    assert manifest["counts"] == {"rows": 3, "supported": 2, "rejected": 1}


def test_next25_rejects_boundary_drift_overwrite_and_endpoint_cli(tmp_path: Path) -> None:
    from src.next25_apply_rule import main

    paths = _fixture(tmp_path)
    output = tmp_path / "predictions"
    _run(paths, output)
    with pytest.raises(FileExistsError):
        _run(paths, output)
    manifest = json.loads(paths["cohort_manifest"].read_text())
    manifest["labels_opened"] = True
    paths["cohort_manifest"].write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="boundary"):
        _run(paths, tmp_path / "bad")
    for forbidden in ("--labels", "--endpoint", "--reference", "--relaxed"):
        with pytest.raises(SystemExit) as exc_info:
            main([forbidden, "x"])
        assert exc_info.value.code == 2
