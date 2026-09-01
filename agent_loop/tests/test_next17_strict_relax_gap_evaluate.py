"""Contracts for the NEXT17 finite-catalog development evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _inputs(tmp_path: Path) -> dict[str, Path]:
    from src.next17_strict_relax_gap import PROTOCOL as STRICT_PROTOCOL

    ids = ["m0", "m1", "m2", "m3"]
    groups = ["Li2O", "Li2O", "NaCl", "NaCl"]
    strict = tmp_path / "strict.parquet"
    pd.DataFrame(
        {
            "material_id": ids,
            "rk": groups,
            "supported": [True] * 4,
            "strict_group_supported": [True] * 4,
            "strict_relative_gap_ev_per_atom": [0.0, 0.08, 0.0, 0.09],
            "prediction_steps": [8, 9, 10, 11],
            "capped_at_max_steps": [False] * 4,
        }
    ).to_parquet(strict, index=False)
    strict_manifest = tmp_path / "strict-manifest.json"
    _write_json(
        strict_manifest,
        {
            "protocol": STRICT_PROTOCOL,
            "elementa_endpoint_bytes_read_by_execution": False,
            "mp_hull_bytes_read_by_execution": False,
            "threshold_selected": False,
            "outputs_sha256": {strict.name: _sha(strict)},
        },
    )

    x0 = tmp_path / "x0.parquet"
    pd.DataFrame(
        {
            "sid": ids,
            "rk": groups,
            "mattersim_feature_ok": [True] * 4,
            "mattersim_energy_per_atom": [-3.0, -2.92, -4.0, -3.91],
        }
    ).to_parquet(x0, index=False)
    x0_manifest = tmp_path / "x0-manifest.json"
    _write_json(
        x0_manifest,
        {
            "protocol": "2026-08-01-mattersim-x0-baseline-v1",
            "model": {"checkpoint_sha256": "e3df9fa708725e3d453140646c7d1838324b347a3d1214cf1440522146f872b5"},
            "outputs_sha256": {x0.name: _sha(x0)},
        },
    )

    labels = tmp_path / "labels.parquet"
    pd.DataFrame(
        {
            "sid": ids,
            "rk": groups,
            "e_per_atom": [-2.0, -1.8, -3.0, -2.7],
            "final_ionic_step": [10] * 4,
            "final_max_force": [0.01] * 4,
        }
    ).to_parquet(labels, index=False)
    labels_manifest = tmp_path / "labels-manifest.json"
    _write_json(
        labels_manifest,
        {
            "protocol": "2026-08-01-dft-pre-screening-design-v1",
            "input_role": "unrelaxed_x0_only",
            "outputs_sha256": {labels.name: _sha(labels)},
        },
    )
    return locals()


def test_selection_uses_only_the_frozen_catalog_and_maximizes_eligible_savings() -> None:
    from src.next17_strict_relax_gap_evaluate import THRESHOLD_CATALOG, _select_threshold

    scan = {
        str(threshold): {
            "eligible": threshold in {0.06, 0.07},
            "metrics": {"dft_savings": {"estimate": 0.18 if threshold == 0.06 else 0.14}},
        }
        for threshold in THRESHOLD_CATALOG
    }
    assert _select_threshold(scan) == 0.06
    scan["0.07"]["metrics"]["dft_savings"]["estimate"] = 0.18
    assert _select_threshold(scan) == 0.07


def test_evaluator_keeps_identifiers_external_even_when_no_small_sample_candidate(
    tmp_path: Path,
) -> None:
    from src import next17_strict_relax_gap_evaluate as module

    paths = _inputs(tmp_path)
    private, aggregate = tmp_path / "private", tmp_path / "aggregate"
    result = module.evaluate_strict_relax_development(
        strict_features_path=paths["strict"],
        strict_manifest_path=paths["strict_manifest"],
        x0_features_path=paths["x0"],
        x0_manifest_path=paths["x0_manifest"],
        labels_path=paths["labels"],
        labels_manifest_path=paths["labels_manifest"],
        private_output_dir=private,
        aggregate_output_dir=aggregate,
        bootstrap_reps=100,
        require_formal_inputs=False,
    )
    assert result["fresh_lockbox"] is False
    assert result["threshold_catalog"] == list(module.THRESHOLD_CATALOG)
    assert (private / module.PRIVATE_JOINED_NAME).is_file()
    payload = (aggregate / module.RESULT_NAME).read_bytes()
    assert b'"m0"' not in payload and b'"material_id"' not in payload


def test_evaluator_rejects_a_strict_gap_not_bound_by_its_manifest(tmp_path: Path) -> None:
    from src import next17_strict_relax_gap_evaluate as module

    paths = _inputs(tmp_path)
    table = pd.read_parquet(paths["strict"])
    table.loc[0, "strict_relative_gap_ev_per_atom"] = 0.01
    table.to_parquet(paths["strict"], index=False)
    with pytest.raises(ValueError, match="hash differs"):
        module.evaluate_strict_relax_development(
            strict_features_path=paths["strict"],
            strict_manifest_path=paths["strict_manifest"],
            x0_features_path=paths["x0"],
            x0_manifest_path=paths["x0_manifest"],
            labels_path=paths["labels"],
            labels_manifest_path=paths["labels_manifest"],
            private_output_dir=tmp_path / "private",
            aggregate_output_dir=tmp_path / "aggregate",
            bootstrap_reps=10,
            require_formal_inputs=False,
        )


def test_cli_cannot_change_catalog_gates_or_sample() -> None:
    from src.next17_strict_relax_gap_evaluate import main

    for forbidden in ("--threshold", "--catalog", "--safety", "--sample", "--groups"):
        with pytest.raises(SystemExit) as exc_info:
            main([forbidden, "x"])
        assert exc_info.value.code == 2
