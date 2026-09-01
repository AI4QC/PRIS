from __future__ import annotations

import numpy as np
import hashlib
import json
from pathlib import Path
import pandas as pd
import pytest

from src.next31_omc25_energy_evaluate import (
    GATES,
    energy_metrics,
    evaluate_frozen_predictions,
    freeze_evaluation_protocol,
)


def test_energy_metrics_use_frozen_energy_thresholds_and_fail_open() -> None:
    energy = np.array([0.005, 0.01, 0.02, 0.04, 0.08])
    supported = np.array([True, False, True, True, True])
    reject = np.array([False, False, False, True, False])
    score = np.array([0.0, np.nan, 0.2, 3.0, 2.0])

    result = energy_metrics(
        energy=energy, supported=supported, reject=reject, score=score
    )

    assert result["counts"]["protected"] == 2
    assert result["counts"]["energy_positive"] == 2
    assert result["coverage"]["numerator"] == 4
    assert result["protected_recall"]["numerator"] == 2
    assert result["reject_precision"]["numerator"] == 1
    assert result["reject_precision"]["denominator"] == 1
    assert result["dft_savings"]["numerator"] == 1
    assert result["auc_energy_positive"] == 1.0


def test_gates_are_frozen_before_prospective_label_opening() -> None:
    assert GATES == {
        "coverage_lower_at_least": 0.95,
        "protected_recall_lower_at_least": 0.95,
        "reject_precision_lower_at_least": 0.70,
        "dft_savings_lower_at_least": 0.02,
        "auc_energy_positive_at_least": 0.85,
    }


def test_freeze_evaluation_protocol_binds_rule_before_labels(tmp_path: Path) -> None:
    rule_path = tmp_path / "rule.json"
    rule_path.write_text(json.dumps({"eligible": True}), encoding="utf-8")
    output_dir = tmp_path / "protocol"

    manifest = freeze_evaluation_protocol(
        frozen_rule_path=rule_path, output_dir=output_dir
    )

    protocol = json.loads(
        (output_dir / "NEXT31_EVALUATION_PROTOCOL.json").read_text()
    )
    assert protocol["gates"] == GATES
    assert protocol["labels_opened"] is False
    assert protocol["energy_positive_min_ev_per_atom"] == 0.04
    assert manifest["labels_opened"] is False


def test_evaluate_frozen_predictions_requires_exact_identity_join(tmp_path: Path) -> None:
    rule_path = tmp_path / "rule.json"
    rule_path.write_text(json.dumps({"eligible": True}), encoding="utf-8")
    protocol_dir = tmp_path / "protocol"
    freeze_evaluation_protocol(
        frozen_rule_path=rule_path, output_dir=protocol_dir
    )
    predictions_path = tmp_path / "predictions.parquet"
    pd.DataFrame(
        {
            "material_id": ["a", "b", "c", "d"],
            "source_shard": ["s1", "s1", "s2", "s2"],
            "analytic_supported": [True, True, True, True],
            "next31_risk_score": [0.0, 0.5, 2.0, 3.0],
            "reject": [False, False, False, True],
        }
    ).to_parquet(predictions_path, index=False)
    prediction_manifest = tmp_path / "prediction_manifest.json"
    prediction_manifest.write_text(
        json.dumps(
            {
                "labels_opened": False,
                "endpoint_artifacts_opened": False,
                "outputs_sha256": {},
                "inputs_sha256": {
                    "frozen_rule": {
                        "sha256": hashlib.sha256(rule_path.read_bytes()).hexdigest()
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    endpoints_path = tmp_path / "endpoints.parquet"
    pd.DataFrame(
        {
            "material_id": ["a", "b", "c", "d"],
            "source_shard": ["s1", "s1", "s2", "s2"],
            "energy_drop_pa": [0.005, 0.02, 0.04, 0.08],
        }
    ).to_parquet(endpoints_path, index=False)

    result = evaluate_frozen_predictions(
        predictions_path=predictions_path,
        prediction_manifest_path=prediction_manifest,
        endpoints_path=endpoints_path,
        evaluation_protocol_path=protocol_dir / "NEXT31_EVALUATION_PROTOCOL.json",
        output_dir=tmp_path / "evaluation",
        require_prediction_output_hash=False,
    )

    assert result["metrics"]["counts"]["rows"] == 4
    assert result["metrics"]["reject_precision"]["estimate"] == 1.0
    assert result["labels_opened_after_predictions_frozen"] is True

    bad_manifest = tmp_path / "bad_prediction_manifest.json"
    bad_manifest.write_text(
        json.dumps(
            {
                "labels_opened": False,
                "endpoint_artifacts_opened": False,
                "outputs_sha256": {},
                "inputs_sha256": {"frozen_rule": {"sha256": "0" * 64}},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="rule binding"):
        evaluate_frozen_predictions(
            predictions_path=predictions_path,
            prediction_manifest_path=bad_manifest,
            endpoints_path=endpoints_path,
            evaluation_protocol_path=protocol_dir
            / "NEXT31_EVALUATION_PROTOCOL.json",
            output_dir=tmp_path / "bad-evaluation",
            require_prediction_output_hash=False,
        )
