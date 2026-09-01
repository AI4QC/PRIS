"""Contracts for one-shot NEXT28 prospective evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_next28_evaluator_uses_frozen_application_manifest(tmp_path: Path) -> None:
    from src.next28_contact_coordination import APPLICATION_PROTOCOL, PREDICTIONS_NAME
    from src.next28_evaluate import evaluate_predictions

    n = 300
    ids = [f"future-{index:04d}" for index in range(n)]
    reject = np.zeros(n, dtype=bool)
    reject[:30] = True
    positive = np.zeros(n, dtype=bool)
    positive[:29] = True
    predictions = tmp_path / PREDICTIONS_NAME
    pd.DataFrame(
        {
            "material_id": ids,
            "source_shard": np.where(np.arange(n) % 2, "a", "b"),
            "analytic_supported": True,
            "next28_risk_score": np.r_[np.full(30, 7.0), np.full(n - 30, 2.0)],
            "reject": reject,
        }
    ).to_parquet(predictions, index=False)
    manifest = tmp_path / "MANIFEST.json"
    manifest.write_text(
        json.dumps(
            {
                "protocol": APPLICATION_PROTOCOL,
                "frozen_at_utc": "2026-08-03T06:00:00+00:00",
                "labels_opened": False,
                "endpoint_fields_read": False,
                "relaxed_structures_opened": False,
                "threshold_refit": False,
                "outputs_sha256": {PREDICTIONS_NAME: _sha(predictions)},
            }
        )
    )
    endpoints = tmp_path / "endpoints.parquet"
    pd.DataFrame(
        {
            "material_id": ids,
            "source_shard": np.where(np.arange(n) % 2, "a", "b"),
            "force0_max": np.where(positive, 1.1, 0.1),
            "force0_rms": 0.1,
            "energy_drop_pa": 0.01,
            "stress0_norm": 0.01,
        }
    ).to_parquet(endpoints, index=False)
    result = evaluate_predictions(
        predictions_path=predictions,
        prediction_manifest_path=manifest,
        endpoints_path=endpoints,
        output_dir=tmp_path / "evaluation",
    )
    assert result["passes_all_prospective_gates"] is True
    assert result["analytic"]["endpoint_positive_precision"] == 29 / 30
    assert result["beyond_pauling_or_dft_claim"] is False
