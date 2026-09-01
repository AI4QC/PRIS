"""Contracts for post-freeze NEXT27 OMC25 endpoint evaluation."""

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
    from src.next27_apply_rule import PREDICTIONS_NAME, PROTOCOL

    tmp_path.mkdir(parents=True, exist_ok=True)
    rows = 200
    ids = [f"omc25-prospective-{index:04d}" for index in range(rows)]
    positive = np.zeros(rows, dtype=bool)
    positive[:80] = True
    reject = np.zeros(rows, dtype=bool)
    reject[:40] = True
    predictions = tmp_path / PREDICTIONS_NAME
    pd.DataFrame(
        {
            "material_id": ids,
            "analytic_supported": True,
            "next27_risk_score": np.linspace(3.0, -2.0, rows),
            "reject": reject,
            "input_role": "unrelaxed_x0_geometry_only",
            "source_shard": np.where(np.arange(rows) % 2, "data0039", "data0030"),
        }
    ).to_parquet(predictions, index=False)
    manifest = tmp_path / "MANIFEST.json"
    manifest.write_text(
        json.dumps(
            {
                "protocol": PROTOCOL,
                "frozen_at_utc": "2026-08-03T05:49:35+00:00",
                "labels_opened": False,
                "endpoint_fields_read": False,
                "relaxed_structures_opened": False,
                "outputs_sha256": {PREDICTIONS_NAME: _sha(predictions)},
            }
        ),
        encoding="utf-8",
    )
    endpoints = tmp_path / "endpoints.parquet"
    pd.DataFrame(
        {
            "material_id": ids,
            "source_shard": np.where(np.arange(rows) % 2, "data0039", "data0030"),
            "force0_max": np.where(positive, 1.1, 0.1),
            "force0_rms": 0.1,
            "energy_drop_pa": 0.01,
            "stress0_norm": 0.01,
            "disp_p90": np.where(positive, 0.4, 0.1),
            "cell_logstrain_max": np.where(positive, 0.2, 0.01),
        }
    ).to_parquet(endpoints, index=False)
    return {"predictions": predictions, "manifest": manifest, "endpoints": endpoints}


def test_next27_evaluation_uses_frozen_gates_and_reports_shards(tmp_path: Path) -> None:
    from src.next27_evaluate import RESULT_NAME, evaluate_predictions

    paths = _fixture(tmp_path)
    output = tmp_path / "evaluation"
    result = evaluate_predictions(
        predictions_path=paths["predictions"],
        prediction_manifest_path=paths["manifest"],
        endpoints_path=paths["endpoints"],
        output_dir=output,
    )

    assert result["passes_all_prospective_gates"] is True
    assert result["prospective_gates"]["endpoint_positive_precision_lower"] == 0.75
    assert result["analytic"]["rejected"] == 40
    assert set(result["by_source_shard"]) == {"data0030", "data0039"}
    assert result["claim_boundary"] == "substantial DFT response, not thermodynamic instability"
    assert result["beyond_pauling_or_dft_claim"] is False
    assert json.loads((output / RESULT_NAME).read_text()) == result


def test_next27_evaluation_rejects_endpoint_shard_mismatch(tmp_path: Path) -> None:
    from src.next27_evaluate import evaluate_predictions

    paths = _fixture(tmp_path)
    endpoints = pd.read_parquet(paths["endpoints"])
    endpoints.loc[0, "source_shard"] = "wrong-shard"
    endpoints.to_parquet(paths["endpoints"], index=False)
    with pytest.raises(ValueError, match="source shards differ"):
        evaluate_predictions(
            predictions_path=paths["predictions"],
            prediction_manifest_path=paths["manifest"],
            endpoints_path=paths["endpoints"],
            output_dir=tmp_path / "shard-mismatch",
        )


def test_next27_evaluation_rejects_identity_mismatch_and_overwrite(tmp_path: Path) -> None:
    from src.next27_evaluate import evaluate_predictions

    paths = _fixture(tmp_path)
    output = tmp_path / "evaluation"
    evaluate_predictions(
        predictions_path=paths["predictions"],
        prediction_manifest_path=paths["manifest"],
        endpoints_path=paths["endpoints"],
        output_dir=output,
    )
    with pytest.raises(FileExistsError):
        evaluate_predictions(
            predictions_path=paths["predictions"],
            prediction_manifest_path=paths["manifest"],
            endpoints_path=paths["endpoints"],
            output_dir=output,
        )

    endpoints = pd.read_parquet(paths["endpoints"]).iloc[:-1]
    endpoints.to_parquet(paths["endpoints"], index=False)
    with pytest.raises(ValueError, match="identities differ"):
        evaluate_predictions(
            predictions_path=paths["predictions"],
            prediction_manifest_path=paths["manifest"],
            endpoints_path=paths["endpoints"],
            output_dir=tmp_path / "mismatch",
        )
