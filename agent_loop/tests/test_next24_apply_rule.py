"""Contracts for label-free NEXT24 transport of the frozen NEXT23 law."""

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
    tmp_path.mkdir(parents=True, exist_ok=True)
    ids = ["generated-a", "generated-b", "generated-c"]
    metadata = tmp_path / "holdout_metadata.parquet"
    pd.DataFrame(
        {
            "material_id": ids,
            "rk": ["ssagen"] * 3,
            "formula": ["LiO"] * 3,
            "natoms": [2] * 3,
            "input_role": ["unrelaxed_x0_geometry_only"] * 3,
        }
    ).to_parquet(metadata, index=False)
    cohort_manifest = tmp_path / "cohort-MANIFEST.json"
    cohort_manifest.write_text(
        json.dumps(
            {
                "protocol": "2026-08-03-next24-ssagen-x0-sanitize-v1",
                "input_role": "unrelaxed_x0_geometry_only",
                "labels_opened": False,
                "endpoint_artifacts_opened": False,
                "relaxed_structures_opened": False,
                "production_protocol_eligible": True,
                "outputs_sha256": {metadata.name: _sha(metadata)},
            }
        ),
        encoding="utf-8",
    )
    frames = {
        "sivr": pd.DataFrame(
            {
                "material_id": ids,
                "voronoi_q0__sivr_cell_anisotropy": [0.0, 1.0, np.nan],
            }
        ),
        "scbve": pd.DataFrame(
            {
                "material_id": ids,
                "scbv_vector_asymmetry_rms": [0.0, 1.0, 1.0],
            }
        ),
    }
    feature_paths: dict[str, Path] = {}
    feature_manifest_paths: dict[str, Path] = {}
    for source, frame in frames.items():
        feature_path = tmp_path / f"{source}-features.parquet"
        frame.to_parquet(feature_path, index=False)
        manifest_path = tmp_path / f"{source}-MANIFEST.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "input_role": "unrelaxed_x0_geometry_only",
                    "endpoint_fields_read": False,
                    "model_or_proxy_potential_used": False,
                    "coordinates_or_cell_modified": False,
                    "inputs_sha256": {"metadata": _sha(metadata)},
                    "outputs_sha256": {feature_path.name: _sha(feature_path)},
                }
            ),
            encoding="utf-8",
        )
        feature_paths[source] = feature_path
        feature_manifest_paths[source] = manifest_path

    law = tmp_path / "NEXT23_FROZEN_RELAXATION_RULE.json"
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
        "feature_manifest_paths": feature_manifest_paths,
        "law": law,
        "law_manifest": law_manifest,
    }


def _apply(paths: dict[str, object], output: Path):
    from src.next24_apply_rule import apply_transport_rule

    return apply_transport_rule(
        frozen_rule_path=paths["law"],
        rule_manifest_path=paths["law_manifest"],
        cohort_metadata_path=paths["metadata"],
        cohort_manifest_path=paths["cohort_manifest"],
        feature_paths=paths["feature_paths"],
        feature_manifest_paths=paths["feature_manifest_paths"],
        output_dir=output,
    )


def test_transport_scores_exactly_uses_selected_sources_and_fails_open(
    tmp_path: Path,
) -> None:
    from src import next24_apply_rule as module

    paths = _fixture(tmp_path)
    output = tmp_path / "predictions"
    manifest = _apply(paths, output)
    predictions = pd.read_parquet(output / module.PREDICTIONS_NAME)
    assert predictions.material_id.tolist() == [
        "generated-a",
        "generated-b",
        "generated-c",
    ]
    assert predictions.analytic_supported.tolist() == [True, True, False]
    assert predictions.next23_risk_score.iloc[:2].tolist() == pytest.approx([0.0, 2.0])
    assert np.isnan(predictions.next23_risk_score.iloc[2])
    assert predictions.reject.tolist() == [False, True, False]
    assert manifest["selected_feature_sources"] == ["scbve", "sivr"]
    assert manifest["thresholds_refit"] is False
    assert manifest["formula_or_parameters_changed"] is False
    assert manifest["blind_labels_opened"] is False
    assert manifest["endpoint_fields_read"] is False
    assert manifest["relaxed_structures_opened"] is False
    assert manifest["counts"] == {"rows": 3, "supported": 2, "rejected": 1}


def test_transport_rejects_missing_extra_or_dft_bearing_feature_sources(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    feature_paths = paths["feature_paths"]
    feature_manifests = paths["feature_manifest_paths"]
    assert isinstance(feature_paths, dict) and isinstance(feature_manifests, dict)

    missing = dict(paths)
    missing["feature_paths"] = {"sivr": feature_paths["sivr"]}
    missing["feature_manifest_paths"] = {"sivr": feature_manifests["sivr"]}
    with pytest.raises(ValueError, match="selected sources"):
        _apply(missing, tmp_path / "missing")

    extra = dict(paths)
    extra["feature_paths"] = {**feature_paths, "madelung": feature_paths["sivr"]}
    extra["feature_manifest_paths"] = {
        **feature_manifests,
        "madelung": feature_manifests["sivr"],
    }
    with pytest.raises(ValueError, match="selected sources"):
        _apply(extra, tmp_path / "extra")

    frame = pd.read_parquet(feature_paths["sivr"])
    frame["dft_energy"] = 0.0
    frame.to_parquet(feature_paths["sivr"], index=False)
    manifest = json.loads(feature_manifests["sivr"].read_text())
    manifest["outputs_sha256"] = {feature_paths["sivr"].name: _sha(feature_paths["sivr"])}
    feature_manifests["sivr"].write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="no-DFT"):
        _apply(paths, tmp_path / "dft-column")


def test_transport_validates_ids_hashes_no_replace_and_endpoint_cli(
    tmp_path: Path,
) -> None:
    from src.next24_apply_rule import main

    paths = _fixture(tmp_path)
    output = tmp_path / "predictions"
    _apply(paths, output)
    with pytest.raises(FileExistsError):
        _apply(paths, output)

    bad = _fixture(tmp_path / "bad-hash")
    bad["law"].write_text(bad["law"].read_text() + "\n")
    with pytest.raises(ValueError, match="hash"):
        _apply(bad, tmp_path / "bad-output")

    for forbidden in ("--labels", "--endpoint", "--relaxed-zip", "--energy"):
        with pytest.raises(SystemExit) as exc_info:
            main([forbidden, str(tmp_path / "x")])
        assert exc_info.value.code == 2

