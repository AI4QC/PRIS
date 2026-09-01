"""Contracts for isolated evaluation of blinded NEXT13d DFT endpoints."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _queue(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    from src.next13d_acsc_dft_pairs import (
        BLINDED_QUEUE_NAME,
        PRIVATE_PAIRS_NAME,
        PROTOCOL,
    )

    directory = tmp_path / "queue"
    directory.mkdir()
    blind = pd.DataFrame(
        {
            "task_index": [0, 1, 2, 3],
            "task_id": ["task-t0", "task-c0", "task-t1", "task-c1"],
            "formula": ["Li2O", "Li2O", "NaCl", "KCl"],
            "natoms": [3, 3, 2, 2],
            "task_available": [True] * 4,
            "task_prefix": ["tasks/task-t0", "tasks/task-c0", "tasks/task-t1", "tasks/task-c1"],
            "encut_ev": [520] * 4,
        }
    )
    private = pd.DataFrame(
        [
            {"pair_id": "pair-0000", "task_id": "task-t0", "role": "treatment", "sid": "sid-t0", "rk": "Li2|O1", "formula": "Li2O", "natoms": 3, "m5_gap_ev_per_atom": 0.2, "match_tier": "same_rk_same_natoms", "same_rk": True},
            {"pair_id": "pair-0000", "task_id": "task-c0", "role": "control", "sid": "sid-c0", "rk": "Li2|O1", "formula": "Li2O", "natoms": 3, "m5_gap_ev_per_atom": 0.1, "match_tier": "same_rk_same_natoms", "same_rk": True},
            {"pair_id": "pair-0001", "task_id": "task-t1", "role": "treatment", "sid": "sid-t1", "rk": "Na1|Cl1", "formula": "NaCl", "natoms": 2, "m5_gap_ev_per_atom": 0.3, "match_tier": "fallback_same_natoms_chemistry", "same_rk": False},
            {"pair_id": "pair-0001", "task_id": "task-c1", "role": "control", "sid": "sid-c1", "rk": "K1|Cl1", "formula": "KCl", "natoms": 2, "m5_gap_ev_per_atom": 0.2, "match_tier": "fallback_same_natoms_chemistry", "same_rk": False},
        ]
    )
    blind_path = directory / BLINDED_QUEUE_NAME
    private_path = directory / PRIVATE_PAIRS_NAME
    blind.to_parquet(blind_path, index=False)
    private.to_parquet(private_path, index=False)
    manifest = {
        "protocol": PROTOCOL,
        "all_tasks_queued": True,
        "executor_blinded_to_sid_and_role": True,
        "private_mapping_must_be_withheld_from_executor": True,
        "licensed_potcar_contents_included": False,
        "run_protocol": {
            "endpoint_definitions_frozen_before_DFT": {
                "severe_energy_drop_eV_per_atom": 0.10,
                "severe_initial_fmax_eV_per_A": 1.0,
                "severe_max_displacement_angstrom": 0.5,
                "severe_abs_log_volume_ratio": 0.10,
                "nonconvergence_is_severe": True,
                "same_rk_relaxed_energy_primary": True,
                "direction": "treatment_minus_control; positive severity/energy supports ACSC rejection",
            }
        },
        "outputs_sha256": {
            BLINDED_QUEUE_NAME: _sha(blind_path),
            PRIVATE_PAIRS_NAME: _sha(private_path),
        },
    }
    manifest_path = directory / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return blind_path, private_path, manifest_path, directory


def _endpoints(tmp_path: Path, queue_manifest: Path) -> tuple[Path, Path, Path]:
    from src.next13d_acsc_dft_endpoint_evaluate import ENDPOINT_COLUMNS, ENDPOINT_PROTOCOL

    directory = tmp_path / "external-endpoints"
    directory.mkdir()
    rows = [
        # treatments: severe relaxation, and the same-rk treatment is 0.2 eV/atom higher
        ["task-t0", "converged", -20.0, 1.5, 2.0, -18.9, 0.02, 20, 100.0, 120.0, 0.7, 100.0, ""],
        ["task-c0", "converged", -20.0, 0.2, 0.2, -19.5, 0.02, 10, 100.0, 101.0, 0.1, 80.0, ""],
        ["task-t1", "timeout", np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 86400.0, "wall limit"],
        ["task-c1", "converged", -10.0, 0.2, 0.2, -9.9, 0.02, 5, 50.0, 50.5, 0.1, 40.0, ""],
    ]
    table = pd.DataFrame(rows, columns=ENDPOINT_COLUMNS)
    endpoint_path = directory / "endpoints.parquet"
    table.to_parquet(endpoint_path, index=False)
    manifest = {
        "protocol": ENDPOINT_PROTOCOL,
        "queue_manifest_sha256": _sha(queue_manifest),
        "all_tasks_accounted": True,
        "vasp_execution_complete": True,
        "executor_received_private_mapping": False,
        "outputs_sha256": {endpoint_path.name: _sha(endpoint_path)},
    }
    manifest_path = directory / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return endpoint_path, manifest_path, directory


def test_evaluator_accounts_for_failures_and_preserves_pair_direction(tmp_path: Path) -> None:
    from src import next13d_acsc_dft_endpoint_evaluate as module

    blind, private, queue_manifest, _ = _queue(tmp_path)
    endpoints, endpoint_manifest, _ = _endpoints(tmp_path, queue_manifest)
    output = tmp_path / "evaluation"
    result = module.evaluate_paired_dft_endpoints(
        blinded_queue_path=blind,
        private_pairs_path=private,
        queue_manifest_path=queue_manifest,
        endpoints_path=endpoints,
        endpoint_manifest_path=endpoint_manifest,
        output_dir=output,
    )
    assert result["counts"] == {
        "tasks": 4,
        "pairs": 2,
        "converged_tasks": 3,
        "failed_tasks": 0,
        "timeout_tasks": 1,
        "both_converged_pairs": 1,
        "same_rk_both_converged_pairs": 1,
    }
    assert result["primary_severe_relaxation"]["treatment_severe_control_not"] == 2
    assert result["primary_severe_relaxation"]["control_severe_treatment_not"] == 0
    assert result["primary_severe_relaxation"]["one_sided_exact_p"] == pytest.approx(0.25)
    energy = result["same_rk_relaxed_energy_ev_per_atom"]
    assert energy["paired_differences_treatment_minus_control"] == pytest.approx([0.2])
    assert result["scientific_improvement_claim"] is False
    joined = pd.read_parquet(output / module.JOINED_NAME)
    assert joined.loc[joined.task_id.eq("task-t1"), "severe_relaxation"].item()
    assert result["all_endpoints_opened_only_after_full_accounting"] is True


def test_evaluator_rejects_incomplete_or_colocated_endpoints(tmp_path: Path) -> None:
    from src.next13d_acsc_dft_endpoint_evaluate import evaluate_paired_dft_endpoints

    blind, private, queue_manifest, queue_dir = _queue(tmp_path)
    endpoints, endpoint_manifest, _ = _endpoints(tmp_path, queue_manifest)
    table = pd.read_parquet(endpoints).iloc[:-1]
    table.to_parquet(endpoints, index=False)
    manifest = json.loads(endpoint_manifest.read_text())
    manifest["outputs_sha256"][endpoints.name] = _sha(endpoints)
    endpoint_manifest.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="coverage"):
        evaluate_paired_dft_endpoints(
            blinded_queue_path=blind,
            private_pairs_path=private,
            queue_manifest_path=queue_manifest,
            endpoints_path=endpoints,
            endpoint_manifest_path=endpoint_manifest,
            output_dir=tmp_path / "nope",
        )

    colocated = queue_dir / "endpoints.parquet"
    table.to_parquet(colocated, index=False)
    with pytest.raises(ValueError, match="physically separated"):
        evaluate_paired_dft_endpoints(
            blinded_queue_path=blind,
            private_pairs_path=private,
            queue_manifest_path=queue_manifest,
            endpoints_path=colocated,
            endpoint_manifest_path=endpoint_manifest,
            output_dir=tmp_path / "nope2",
        )


def test_evaluator_cli_cannot_refit_frozen_thresholds() -> None:
    from src.next13d_acsc_dft_endpoint_evaluate import main

    for forbidden in ("--threshold", "--energy-drop-cutoff", "--exclude-timeouts"):
        with pytest.raises(SystemExit) as exc_info:
            main([forbidden, "1"])
        assert exc_info.value.code == 2

