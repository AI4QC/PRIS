"""Contracts for one-shot post-freeze NEXT23 blind evaluation."""

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
    n_rows = 400
    ids = [f"wbm-{1 + index % 2}-{index}" for index in range(n_rows)]
    metadata = tmp_path / "holdout_metadata.parquet"
    pd.DataFrame(
        {
            "material_id": ids,
            "rk": ["LiO"] * n_rows,
            "formula": ["LiO"] * n_rows,
            "natoms": [2 + index % 4 for index in range(n_rows)],
            "input_role": ["unrelaxed_x0_geometry_only"] * n_rows,
        }
    ).to_parquet(metadata, index=False)
    cohort_manifest = tmp_path / "cohort-MANIFEST.json"
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

    endpoint = np.r_[
        np.full(100, 0.05), np.full(100, 0.60), np.full(200, 0.30)
    ]
    reject = np.r_[
        np.zeros(100, dtype=bool),
        np.ones(100, dtype=bool),
        np.zeros(200, dtype=bool),
    ]
    predictions = tmp_path / "next23_relaxation_change_predictions.parquet"
    pd.DataFrame(
        {
            "material_id": ids,
            "analytic_supported": [True] * n_rows,
            "next23_risk_score": np.r_[
                np.zeros(100), np.full(100, 5.0), np.ones(200)
            ],
            "reject": reject,
            "input_role": ["unrelaxed_x0_geometry_only"] * n_rows,
        }
    ).to_parquet(predictions, index=False)
    prediction_manifest = tmp_path / "prediction-MANIFEST.json"
    prediction_manifest.write_text(
        json.dumps(
            {
                "protocol": "2026-08-02-next23-label-free-frozen-rule-application-v1",
                "frozen_at_utc": "2026-08-03T00:00:00+00:00",
                "selected_candidate": "B+E",
                "blind_labels_opened": False,
                "endpoint_fields_read": False,
                "outputs_sha256": {predictions.name: _sha(predictions)},
            }
        )
    )

    pauling = tmp_path / "next23_pauling_controls.parquet"
    decisions = np.array(["ABSTAIN"] * n_rows, dtype=object)
    decisions[100:120] = "REJECT"
    decisions[:20] = "KEEP"
    pauling_frame = pd.DataFrame({"material_id": ids})
    for name in ("pauling_p2", "pauling_p3", "pauling_p4", "pauling_p5"):
        pauling_frame[f"{name}_decision"] = decisions
    pauling_frame["pauling_p2_p5_decision"] = decisions
    pauling_frame.to_parquet(pauling, index=False)
    pauling_manifest = tmp_path / "pauling-MANIFEST.json"
    pauling_manifest.write_text(
        json.dumps(
            {
                "protocol": "2026-08-02-next23-wbm-pauling-controls-v1",
                "labels_opened": False,
                "endpoint_artifacts_opened": False,
                "thresholds_refit": False,
                "outputs_sha256": {pauling.name: _sha(pauling)},
            }
        )
    )

    labels = tmp_path / "test_labels.parquet"
    pd.DataFrame(
        {
            "material_id": ids,
            "site_stats_fingerprint_init_final_norm_diff": endpoint,
        }
    ).to_parquet(labels, index=False)
    return {
        "metadata": metadata,
        "cohort_manifest": cohort_manifest,
        "predictions": predictions,
        "prediction_manifest": prediction_manifest,
        "pauling": pauling,
        "pauling_manifest": pauling_manifest,
        "labels": labels,
    }


def _evaluate(paths: dict[str, Path], public: Path, private: Path):
    from src.next23_evaluate import evaluate_frozen_predictions

    return evaluate_frozen_predictions(
        predictions_path=paths["predictions"],
        prediction_manifest_path=paths["prediction_manifest"],
        pauling_controls_path=paths["pauling"],
        pauling_manifest_path=paths["pauling_manifest"],
        cohort_metadata_path=paths["metadata"],
        cohort_manifest_path=paths["cohort_manifest"],
        labels_path=paths["labels"],
        public_output_dir=public,
        private_output_dir=private,
    )


def test_evaluation_opens_labels_after_freeze_and_passes_exact_gates(
    tmp_path: Path,
) -> None:
    from src import next23_evaluate as module

    paths = _fixture(tmp_path)
    public = tmp_path / "public"
    private = tmp_path / "private"
    result = _evaluate(paths, public, private)
    metrics = result["next23"]
    assert metrics["passes_primary_gates"] is True
    assert metrics["coverage_lower"] >= 0.90
    assert metrics["protected_recall_lower"] >= 0.95
    assert metrics["rejection_precision_lower"] >= 0.90
    assert metrics["savings_lower"] >= 0.10
    assert result["beyond_pauling_on_this_endpoint"] is True
    assert result["no_refit_after_blind_opening"] is True
    assert result["claim_scope"] == "relaxation_change_screening_only"
    assert result["continuous_diagnostics"]["spearman_rho"] > 0.0

    public_text = (public / module.RESULT_NAME).read_text()
    assert "wbm-" not in public_text
    joined = pd.read_parquet(private / module.PRIVATE_JOIN_NAME)
    assert joined.material_id.tolist() == sorted(joined.material_id.tolist())
    opening = json.loads((private / module.LABEL_OPENING_NAME).read_text())
    assert opening["predictions_sha256_before_label_opening"] == _sha(
        paths["predictions"]
    )
    assert opening["labels_opened"] is True


def test_evaluation_rejects_tampering_and_never_overwrites(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    public = tmp_path / "public"
    private = tmp_path / "private"
    _evaluate(paths, public, private)
    with pytest.raises(FileExistsError):
        _evaluate(paths, public, private)

    paths = _fixture(tmp_path / "tamper")
    paths["predictions"].write_bytes(paths["predictions"].read_bytes() + b"x")
    with pytest.raises(ValueError, match="hash"):
        _evaluate(paths, tmp_path / "tamper-public", tmp_path / "tamper-private")


def test_evaluation_refuses_incomplete_or_duplicate_blind_labels(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    labels = pd.read_parquet(paths["labels"])
    labels = pd.concat([labels.iloc[:-1], labels.iloc[[0]]], ignore_index=True)
    labels.to_parquet(paths["labels"], index=False)
    with pytest.raises(ValueError, match="one-to-one"):
        _evaluate(paths, tmp_path / "public", tmp_path / "private")

