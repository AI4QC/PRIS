"""Contracts for opening WBM labels only after all NEXT14 methods are sealed."""

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
    from src.next14_wbm_acsc_features import PROTOCOL as ACSC_PROTOCOL
    from src.next14_wbm_holdout import METADATA_NAME, PROTOCOL as HOLDOUT_PROTOCOL
    from src.next14_wbm_pauling import PROTOCOL as PAULING_PROTOCOL

    ids = [f"m{i}" for i in range(6)]
    formulas = ["Li2O", "NaCl", "KBr", "MgO", "Al2O3", "CaS"]
    keys = formulas
    metadata = tmp_path / METADATA_NAME
    pd.DataFrame(
        {"material_id": ids, "rk": keys, "formula": formulas, "natoms": [3, 2, 2, 2, 5, 2]}
    ).to_parquet(metadata, index=False)
    holdout_manifest = tmp_path / "holdout-manifest.json"
    _write_json(
        holdout_manifest,
        {
            "protocol": HOLDOUT_PROTOCOL,
            "labels_opened": False,
            "endpoint_artifacts_opened": False,
            "outputs_sha256": {METADATA_NAME: _sha(metadata)},
        },
    )

    pauling = tmp_path / "pauling.parquet"
    pd.DataFrame(
        {
            "material_id": ids,
            "pauling_p2_p5_decision": ["KEEP", "ABSTAIN", "REJECT", "REJECT", "ABSTAIN", "KEEP"],
        }
    ).to_parquet(pauling, index=False)
    pauling_manifest = tmp_path / "pauling-manifest.json"
    _write_json(
        pauling_manifest,
        {
            "protocol": PAULING_PROTOCOL,
            "labels_opened": False,
            "endpoint_artifacts_opened": False,
            "thresholds_refit": False,
            "outputs_sha256": {pauling.name: _sha(pauling)},
        },
    )

    acsc = tmp_path / "acsc.parquet"
    pd.DataFrame(
        {
            "material_id": ids,
            "phsc_status": ["resolved_nonnegative"] * 6,
            "phsc_chsc_decision": ["KEEP", "KEEP", "REJECT", "KEEP", "KEEP", "KEEP"],
            "phsc_chsc_acsc_formal_decision": ["KEEP", "KEEP", "REJECT", "REJECT", "REJECT", "KEEP"],
            "phsc_chsc_acsc_nested_decision": ["KEEP", "KEEP", "REJECT", "KEEP", "REJECT", "KEEP"],
            "nested_three_scale_confirmed": [False, False, False, False, True, False],
        }
    ).to_parquet(acsc, index=False)
    acsc_manifest = tmp_path / "acsc-manifest.json"
    _write_json(
        acsc_manifest,
        {
            "protocol": ACSC_PROTOCOL,
            "labels_opened": False,
            "endpoint_artifacts_opened": False,
            "thresholds_refit": False,
            "outputs_sha256": {acsc.name: _sha(acsc)},
        },
    )

    wbm_features = tmp_path / "test_x0_features.parquet"
    pd.DataFrame(
        {
            "material_id": ids,
            "feature_ok": [True] * 6,
            "min_pair_ratio": [1.2, 1.1, 0.7, 0.8, 0.9, 1.3],
        }
    ).to_parquet(wbm_features, index=False)
    labels = tmp_path / "test_labels.parquet"
    pd.DataFrame(
        {
            "material_id": ids,
            "formula": formulas,
            "formula_key": keys,
            "split": ["test"] * 6,
            "stage": ["test"] * 6,
            "e_above_hull_mp2020_corrected_ppd_mp": [0.0, -0.1, 0.3, 0.25, 0.1, 0.0],
            "site_stats_fingerprint_init_final_norm_diff": [0.0, 0.0, 0.8, 0.7, 0.4, 0.1],
            "stable": [True, True, False, False, False, True],
        }
    ).to_parquet(labels, index=False)
    wbm_manifest = tmp_path / "wbm-manifest.json"
    _write_json(
        wbm_manifest,
        {
            "protocol": "2026-08-01-dft-pre-screening-design-v1",
            "input_role": "unrelaxed_x0_only",
            "outputs_sha256": {wbm_features.name: _sha(wbm_features), labels.name: _sha(labels)},
        },
    )
    frozen_rule = tmp_path / "frozen-rule.json"
    _write_json(
        frozen_rule,
        {
            "protocol": "2026-08-01-dft-pre-screening-design-v1",
            "formula": {
                "name": "min_pair_overlap", "family": "min_pair", "mode": "static",
                "pack_low": 0.0, "pack_high": 0.0, "pack_weight": 0.0,
                "scale_penalty": 0.0, "complexity": 1,
            },
            "threshold": {"threshold": 0.05},
        },
    )
    return locals()


def test_isolated_evaluator_logs_opening_and_keeps_identifiers_external(tmp_path: Path) -> None:
    from src import next14_wbm_evaluate as module

    paths = _inputs(tmp_path)
    aggregate = tmp_path / "aggregate"
    private = tmp_path / "private"
    opening = tmp_path / "NEXT14_OPENING.json"
    result = module.evaluate_wbm_holdout(
        metadata_path=paths["metadata"],
        holdout_manifest_path=paths["holdout_manifest"],
        pauling_features_path=paths["pauling"],
        pauling_manifest_path=paths["pauling_manifest"],
        acsc_features_path=paths["acsc"],
        acsc_manifest_path=paths["acsc_manifest"],
        wbm_test_features_path=paths["wbm_features"],
        wbm_manifest_path=paths["wbm_manifest"],
        frozen_wbm_rule_path=paths["frozen_rule"],
        wbm_test_labels_path=paths["labels"],
        opening_log_path=opening,
        private_output_dir=private,
        aggregate_output_dir=aggregate,
        bootstrap_reps=100,
        require_formal_inputs=False,
    )
    assert opening.is_file()
    assert result["labels_opened_after_all_methods_sealed"] is True
    nested = result["methods"]["phsc_chsc_acsc_nested"]
    assert nested["stable_recall"]["estimate"] == 1.0
    assert nested["dft_savings"]["estimate"] == pytest.approx(2 / 6)
    assert nested["high_energy_rejection_recall"]["estimate"] == pytest.approx(1 / 2)
    assert nested["reject_precision_unstable"]["estimate"] == 1.0
    assert result["comparisons_to_pauling"]["phsc_chsc_acsc_nested"]["complete_superiority_over_pauling"] is False
    assert (private / module.PRIVATE_JOINED_NAME).is_file()
    payload = (aggregate / module.RESULT_NAME).read_bytes()
    assert b'"m0"' not in payload and b'"material_id"' not in payload
    with pytest.raises(FileExistsError):
        module.evaluate_wbm_holdout(
            metadata_path=paths["metadata"], holdout_manifest_path=paths["holdout_manifest"],
            pauling_features_path=paths["pauling"], pauling_manifest_path=paths["pauling_manifest"],
            acsc_features_path=paths["acsc"], acsc_manifest_path=paths["acsc_manifest"],
            wbm_test_features_path=paths["wbm_features"], wbm_manifest_path=paths["wbm_manifest"],
            frozen_wbm_rule_path=paths["frozen_rule"], wbm_test_labels_path=paths["labels"],
            opening_log_path=opening, private_output_dir=tmp_path / "p2", aggregate_output_dir=tmp_path / "a2",
            bootstrap_reps=10, require_formal_inputs=False,
        )


def test_evaluator_cli_cannot_refit_or_exclude_rows() -> None:
    from src.next14_wbm_evaluate import main

    for forbidden in ("--threshold", "--fit", "--exclude-abstain", "--sample"):
        with pytest.raises(SystemExit) as exc_info:
            main([forbidden, "x"])
        assert exc_info.value.code == 2

