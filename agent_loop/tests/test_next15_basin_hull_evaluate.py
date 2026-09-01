"""Contracts for the sealed NEXT15 WBM retrospective evaluation."""

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
    from src.next14_wbm_evaluate import PROTOCOL as NEXT14_PROTOCOL
    from src.next15_basin_hull import PROTOCOL as FEATURE_PROTOCOL

    ids = [f"m{i}" for i in range(6)]
    features = tmp_path / "basin_hull_features.parquet"
    pd.DataFrame(
        {
            "material_id": ids,
            "rk": ["Li2O", "NaCl", "KBr", "MgO", "Al2O3", "CaS"],
            "supported": [True] * 6,
            "error": [""] * 6,
            "capped_at_max_steps": [False] * 5 + [True],
            "basin_hull_score_ev_per_atom": [0.01, 0.05, 0.30, 0.25, 0.22, 0.10],
            "basin_hull_decision": ["KEEP", "KEEP", "REJECT", "REJECT", "REJECT", "KEEP"],
        }
    ).to_parquet(features, index=False)
    feature_manifest = tmp_path / "feature-manifest.json"
    _write_json(
        feature_manifest,
        {
            "protocol": FEATURE_PROTOCOL,
            "thresholds_refit": False,
            "wbm_endpoint_bytes_read_by_execution": False,
            "rule": {
                "formula": "B64 = E_MatterSim_relaxed/N - E_raw_MP_hull(composition)",
                "comparison": ">=",
                "threshold_ev_per_atom": 0.2,
                "failure_policy": "ABSTAIN",
            },
            "outputs_sha256": {features.name: _sha(features)},
        },
    )

    joined = tmp_path / "joined_predictions_labels.parquet"
    pd.DataFrame(
        {
            "material_id": ids,
            "formula_key": ["Li2O", "NaCl", "KBr", "MgO", "Al2O3", "CaS"],
            "pauling_p2_p5_decision": ["KEEP", "ABSTAIN", "REJECT", "REJECT", "ABSTAIN", "KEEP"],
            "e_above_hull_mp2020_corrected_ppd_mp": [0.0, -0.1, 0.3, 0.25, 0.1, 0.0],
            "site_stats_fingerprint_init_final_norm_diff": [0.0, 0.0, 0.8, 0.7, 0.4, 0.1],
            "stable": [True, True, False, False, False, True],
        }
    ).to_parquet(joined, index=False)
    joined_manifest = tmp_path / "joined-manifest.json"
    _write_json(
        joined_manifest,
        {
            "protocol": NEXT14_PROTOCOL,
            "identifier_bearing": True,
            "storage_role": "external private prediction/label join",
            "outputs_sha256": {joined.name: _sha(joined)},
        },
    )
    return locals()


def test_retrospective_evaluator_keeps_identifiers_external_and_does_not_refit(
    tmp_path: Path,
) -> None:
    from src import next15_basin_hull_evaluate as module

    paths = _inputs(tmp_path)
    private = tmp_path / "private"
    aggregate = tmp_path / "aggregate"
    result = module.evaluate_basin_hull_retrospective(
        features_path=paths["features"],
        features_manifest_path=paths["feature_manifest"],
        next14_joined_path=paths["joined"],
        next14_private_manifest_path=paths["joined_manifest"],
        private_output_dir=private,
        aggregate_output_dir=aggregate,
        bootstrap_reps=100,
        require_formal_inputs=False,
    )
    method = result["methods"]["next15_basin_hull"]
    assert result["labels_previously_opened"] is True
    assert result["fresh_lockbox"] is False
    assert result["thresholds_refit"] is False
    assert method["dft_savings"]["estimate"] == pytest.approx(3 / 6)
    assert method["stable_recall"]["estimate"] == 1.0
    assert method["high_energy_rejection_recall"]["estimate"] == 1.0
    assert method["reject_precision_unstable"]["estimate"] == 1.0
    assert (private / module.PRIVATE_JOINED_NAME).is_file()
    payload = (aggregate / module.RESULT_NAME).read_bytes()
    assert b'"m0"' not in payload and b'"material_id"' not in payload
    with pytest.raises(FileExistsError):
        module.evaluate_basin_hull_retrospective(
            features_path=paths["features"],
            features_manifest_path=paths["feature_manifest"],
            next14_joined_path=paths["joined"],
            next14_private_manifest_path=paths["joined_manifest"],
            private_output_dir=private,
            aggregate_output_dir=tmp_path / "aggregate2",
            bootstrap_reps=10,
            require_formal_inputs=False,
        )


def test_evaluator_rejects_a_decision_not_implied_by_the_sealed_score(tmp_path: Path) -> None:
    from src import next15_basin_hull_evaluate as module

    paths = _inputs(tmp_path)
    table = pd.read_parquet(paths["features"])
    table.loc[0, "basin_hull_decision"] = "REJECT"
    table.to_parquet(paths["features"], index=False)
    manifest = json.loads(paths["feature_manifest"].read_text(encoding="utf-8"))
    manifest["outputs_sha256"][paths["features"].name] = _sha(paths["features"])
    _write_json(paths["feature_manifest"], manifest)
    with pytest.raises(ValueError, match="decision differs"):
        module.evaluate_basin_hull_retrospective(
            features_path=paths["features"],
            features_manifest_path=paths["feature_manifest"],
            next14_joined_path=paths["joined"],
            next14_private_manifest_path=paths["joined_manifest"],
            private_output_dir=tmp_path / "private",
            aggregate_output_dir=tmp_path / "aggregate",
            bootstrap_reps=10,
            require_formal_inputs=False,
        )


def test_evaluator_cli_cannot_refit_or_exclude_rows() -> None:
    from src.next15_basin_hull_evaluate import main

    for forbidden in ("--threshold", "--fit", "--exclude-capped", "--sample"):
        with pytest.raises(SystemExit) as exc_info:
            main([forbidden, "x"])
        assert exc_info.value.code == 2
