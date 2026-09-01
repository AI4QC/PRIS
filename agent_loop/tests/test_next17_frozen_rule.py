"""Contracts for the immutable NEXT17 production rule and freeze artifact."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def test_frozen_rule_boundary_and_fail_open_semantics() -> None:
    from src.next17_frozen_rule import (
        FROZEN_THRESHOLD_EV_PER_ATOM,
        next17_frozen_decision,
    )

    assert FROZEN_THRESHOLD_EV_PER_ATOM == 0.06
    assert next17_frozen_decision(0.059999, group_supported=True) == "KEEP"
    assert next17_frozen_decision(0.06, group_supported=True) == "REJECT"
    assert next17_frozen_decision(np.nan, group_supported=False) == "ABSTAIN"
    with pytest.raises(ValueError, match="finite"):
        next17_frozen_decision(np.nan, group_supported=True)


def _inputs(tmp_path: Path, *, promotion: bool = True) -> dict[str, Path]:
    from src.next17_strict_relax_gap import PROTOCOL as FEATURE_PROTOCOL
    from src.next17_strict_relax_gap_evaluate import PROTOCOL as EVALUATION_PROTOCOL

    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"checkpoint")
    features = tmp_path / "features.parquet"
    features.write_bytes(b"sealed-features")
    feature_manifest = tmp_path / "feature-manifest.json"
    _write(
        feature_manifest,
        {
            "protocol": FEATURE_PROTOCOL,
            "formula": (
                "R64s(i) = E_MatterSim_strict_relaxed(i)/N_i "
                "- min_j_in_same_composition E_MatterSim_strict_relaxed(j)/N_j"
            ),
            "elementa_endpoint_bytes_read_by_execution": False,
            "mp_hull_bytes_read_by_execution": False,
            "threshold_selected": False,
            "relaxation": {
                "optimizer": "FIRE",
                "filter": "FRECHETCELLFILTER",
                "fmax_ev_per_a": 0.005,
                "max_prediction_steps": 64,
                "atom_budget": 512,
            },
            "inputs_sha256": {
                "checkpoint": {"path": str(checkpoint), "sha256": _sha(checkpoint)}
            },
            "outputs_sha256": {features.name: _sha(features)},
        },
    )
    result = tmp_path / "development.json"
    _write(
        result,
        {
            "protocol": EVALUATION_PROTOCOL,
            "development_promotion": promotion,
            "fresh_lockbox": False,
            "strict_relax": {
                "selected_threshold_ev_per_atom": 0.06,
                "catalog_scan": {"0.06": {"eligible": True}},
            },
            "selected_comparison_strict_minus_x0": {
                "passes_development_promotion": promotion
            },
            "scientific_improvement_claim": False,
        },
    )
    result_manifest = tmp_path / "development-manifest.json"
    _write(
        result_manifest,
        {
            "protocol": EVALUATION_PROTOCOL,
            "identifier_bearing": False,
            "outputs_sha256": {result.name: _sha(result)},
        },
    )
    return locals()


def test_freeze_binds_rule_model_features_and_development_result(tmp_path: Path) -> None:
    from src import next17_freeze as module

    paths = _inputs(tmp_path)
    output = tmp_path / "freeze"
    frozen = module.freeze_next17(
        strict_features_path=paths["features"],
        strict_manifest_path=paths["feature_manifest"],
        development_result_path=paths["result"],
        development_manifest_path=paths["result_manifest"],
        checkpoint_path=paths["checkpoint"],
        output_dir=output,
        require_formal_inputs=False,
    )
    assert frozen["rule"]["threshold_ev_per_atom"] == 0.06
    assert frozen["rule"]["failure_policy"] == "ABSTAIN entire incomplete group"
    assert frozen["development_promotion_verified"] is True
    assert frozen["inputs_sha256"]["checkpoint"] == _sha(paths["checkpoint"])
    assert (output / module.FROZEN_NAME).is_file()
    with pytest.raises(FileExistsError):
        module.freeze_next17(
            strict_features_path=paths["features"],
            strict_manifest_path=paths["feature_manifest"],
            development_result_path=paths["result"],
            development_manifest_path=paths["result_manifest"],
            checkpoint_path=paths["checkpoint"],
            output_dir=output,
            require_formal_inputs=False,
        )


def test_freeze_rejects_a_failed_development_promotion(tmp_path: Path) -> None:
    from src import next17_freeze as module

    paths = _inputs(tmp_path, promotion=False)
    with pytest.raises(ValueError, match="promotion"):
        module.freeze_next17(
            strict_features_path=paths["features"],
            strict_manifest_path=paths["feature_manifest"],
            development_result_path=paths["result"],
            development_manifest_path=paths["result_manifest"],
            checkpoint_path=paths["checkpoint"],
            output_dir=tmp_path / "freeze",
            require_formal_inputs=False,
        )


def test_freeze_cli_cannot_change_rule_or_relaxation() -> None:
    from src.next17_freeze import main

    for forbidden in ("--threshold", "--fmax", "--max-steps", "--formula"):
        with pytest.raises(SystemExit) as exc_info:
            main([forbidden, "x"])
        assert exc_info.value.code == 2
