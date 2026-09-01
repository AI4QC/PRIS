"""Contracts for label-free application of the frozen NEXT23 law."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> dict[str, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    ids = ["wbm-a", "wbm-b", "wbm-c"]
    metadata = tmp_path / "holdout_metadata.parquet"
    pd.DataFrame(
        {
            "material_id": ids,
            "rk": ["LiO"] * 3,
            "formula": ["LiO"] * 3,
            "natoms": [2] * 3,
            "input_role": ["unrelaxed_x0_geometry_only"] * 3,
        }
    ).to_parquet(metadata, index=False)
    cohort_manifest = tmp_path / "cohort-manifest.json"
    cohort_manifest.write_text(
        json.dumps(
            {
                "protocol": "2026-08-02-next23-wbm-relaxation-change-holdout-v1",
                "labels_opened": False,
                "relaxed_structures_opened": False,
                "outputs_sha256": {metadata.name: _sha(metadata)},
            }
        )
    )

    frames = {
        "sivr": pd.DataFrame(
            {
                "material_id": ids,
                "voronoi_q0__sivr_cell_anisotropy": [0.0, 1.0, np.nan],
            }
        ),
        "madelung": pd.DataFrame(
            {"material_id": ids, "nm_point_reduced": [0.0, 0.0, 0.0]}
        ),
        "scbve": pd.DataFrame(
            {"material_id": ids, "scbv_vector_asymmetry_rms": [0.0, 1.0, 1.0]}
        ),
    }
    paths: dict[str, Path] = {
        "metadata": metadata,
        "cohort_manifest": cohort_manifest,
    }
    feature_names = {
        "sivr": "next20_valence_rigidity_features.parquet",
        "madelung": "next21_normalized_madelung_features.parquet",
        "scbve": "next22_bond_valence_equilibrium_features.parquet",
    }
    for source, frame in frames.items():
        feature_path = tmp_path / f"{source}-{feature_names[source]}"
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
            )
        )
        paths[f"{source}_features"] = feature_path
        paths[f"{source}_manifest"] = manifest_path

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
        )
    )
    law_manifest = tmp_path / "law-MANIFEST.json"
    law_manifest.write_text(
        json.dumps(
            {
                "protocol": "2026-08-02-next23-relaxation-change-rule-freeze-v1",
                "blind_labels_opened": False,
                "outputs_sha256": {law.name: _sha(law)},
            }
        )
    )
    paths["law"] = law
    paths["law_manifest"] = law_manifest
    return paths


def _apply(paths: dict[str, Path], output: Path):
    from src.next23_apply_rule import apply_frozen_rule

    return apply_frozen_rule(
        frozen_rule_path=paths["law"],
        rule_manifest_path=paths["law_manifest"],
        cohort_metadata_path=paths["metadata"],
        cohort_manifest_path=paths["cohort_manifest"],
        sivr_features_path=paths["sivr_features"],
        sivr_manifest_path=paths["sivr_manifest"],
        madelung_features_path=paths["madelung_features"],
        madelung_manifest_path=paths["madelung_manifest"],
        scbve_features_path=paths["scbve_features"],
        scbve_manifest_path=paths["scbve_manifest"],
        output_dir=output,
    )


def test_apply_frozen_rule_scores_exactly_and_fails_open(tmp_path: Path) -> None:
    from src import next23_apply_rule as module

    paths = _fixture(tmp_path)
    output = tmp_path / "predictions"
    manifest = _apply(paths, output)
    predictions = pd.read_parquet(output / module.PREDICTIONS_NAME)
    assert predictions.material_id.tolist() == ["wbm-a", "wbm-b", "wbm-c"]
    assert predictions.analytic_supported.tolist() == [True, True, False]
    assert predictions.next23_risk_score.iloc[:2].tolist() == pytest.approx([0.0, 2.0])
    assert np.isnan(predictions.next23_risk_score.iloc[2])
    assert predictions.reject.tolist() == [False, True, False]
    assert manifest["blind_labels_opened"] is False
    assert manifest["counts"] == {"rows": 3, "supported": 2, "rejected": 1}
    assert set(manifest["outputs_sha256"]) == {module.PREDICTIONS_NAME}


def test_apply_validates_hashes_ids_columns_and_no_replace(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    output = tmp_path / "predictions"
    _apply(paths, output)
    with pytest.raises(FileExistsError):
        _apply(paths, output)

    paths = _fixture(tmp_path / "hash-case")
    paths["law"].write_text(paths["law"].read_text() + "\n")
    with pytest.raises(ValueError, match="hash"):
        _apply(paths, tmp_path / "bad-hash")

    paths = _fixture(tmp_path / "column-case")
    frame = pd.read_parquet(paths["sivr_features"])
    frame["dft_energy"] = 0.0
    frame.to_parquet(paths["sivr_features"], index=False)
    feature_manifest = json.loads(paths["sivr_manifest"].read_text())
    feature_manifest["outputs_sha256"] = {
        paths["sivr_features"].name: _sha(paths["sivr_features"])
    }
    paths["sivr_manifest"].write_text(json.dumps(feature_manifest))
    with pytest.raises(ValueError, match="no-DFT"):
        _apply(paths, tmp_path / "bad-column")


def test_apply_cli_has_no_label_or_endpoint_argument(tmp_path: Path) -> None:
    from src.next23_apply_rule import main

    for forbidden in ("--labels", "--endpoint", "--relaxed-zip", "--summary"):
        with pytest.raises(SystemExit) as exc_info:
            main([forbidden, str(tmp_path / "x")])
        assert exc_info.value.code == 2
