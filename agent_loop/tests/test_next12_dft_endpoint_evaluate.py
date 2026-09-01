"""Tests for isolated evaluation of fully accounted NEXT12 DFT endpoints."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _queue(tmp_path: Path) -> tuple[Path, Path, pd.DataFrame]:
    from src.next12_dft_queue import PROTOCOL

    queue_dir = tmp_path / "queue"
    queue_dir.mkdir()
    rows = []
    formulas = ["A", "A", "A", "B", "B", "B"]
    for index, formula in enumerate(formulas):
        high = index in (2, 5)
        rows.append(
            {
                "sid": f"s{index}",
                "formula": formula,
                "natoms": 2,
                "task_available": True,
                "m5_decision": "REJECT" if high else "KEEP",
                "m5_phsc_decision": "REJECT" if high or index == 1 else "KEEP",
                "m5_phsc_chsc_decision": "REJECT" if high or index == 1 else "KEEP",
                "pauling_p2_decision": "REJECT" if index == 2 else "KEEP",
                "pauling_p3_decision": "KEEP",
                "pauling_p4_decision": "KEEP",
                "pauling_p5_decision": "ABSTAIN" if index > 2 else "KEEP",
                "pauling_p2_p5_decision": (
                    "REJECT" if index == 2 else ("ABSTAIN" if index > 2 else "KEEP")
                ),
            }
        )
    table = pd.DataFrame(rows)
    queue_path = queue_dir / "dft_queue.parquet"
    table.to_parquet(queue_path, index=False)
    manifest_path = queue_dir / "MANIFEST.json"
    manifest_path.write_text(
        json.dumps(
            {
                "protocol": PROTOCOL,
                "all_attempts_queued": True,
                "selection_by_gate": False,
                "licensed_potcar_contents_included": False,
                "outputs_sha256": {queue_path.name: _sha(queue_path)},
                "run_protocol": {
                    "endpoint_definitions_frozen_before_DFT": {
                        "near_min_eV_per_atom": 0.001,
                        "valuable_eV_per_atom": 0.05,
                        "high_energy_eV_per_atom": 0.20,
                        "complete_composition_groups_only": True,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return queue_path, manifest_path, table


def _endpoints(
    tmp_path: Path, queue_manifest: Path, *, timeout_sid: str | None = None
) -> tuple[Path, Path]:
    from src.next12_dft_endpoint_evaluate import ENDPOINT_PROTOCOL, ENDPOINT_COLUMNS

    endpoint_dir = tmp_path / "physically-separated-endpoints"
    endpoint_dir.mkdir()
    energy_per_atom = [0.0, 0.02, 0.30, 0.0, 0.10, 0.40]
    rows = []
    for index, value in enumerate(energy_per_atom):
        sid = f"s{index}"
        converged = sid != timeout_sid
        row = {
            "sid": sid,
            "status": "converged" if converged else "timeout",
            "static_energy_ev": -1.0 if converged else np.nan,
            "static_fmax_ev_per_a": 0.2 if converged else np.nan,
            "static_max_stress_gpa": 0.1 if converged else np.nan,
            "relaxed_energy_ev": value * 2 if converged else np.nan,
            "relaxed_fmax_ev_per_a": 0.02 if converged else np.nan,
            "ionic_steps": 10 if converged else 200,
            "initial_volume_angstrom3": 100.0 if converged else np.nan,
            "relaxed_volume_angstrom3": 101.0 if converged else np.nan,
            "max_displacement_angstrom": 0.1 if converged else np.nan,
            "wall_time_seconds": 10.0 + index,
            "error": "" if converged else "scheduler timeout",
        }
        rows.append(row)
    endpoint_path = endpoint_dir / "dft_endpoints.parquet"
    pd.DataFrame(rows, columns=ENDPOINT_COLUMNS).to_parquet(endpoint_path, index=False)
    manifest_path = endpoint_dir / "MANIFEST.json"
    manifest_path.write_text(
        json.dumps(
            {
                "protocol": ENDPOINT_PROTOCOL,
                "queue_manifest_sha256": _sha(queue_manifest),
                "all_attempts_accounted": True,
                "vasp_execution_complete": True,
                "outputs_sha256": {endpoint_path.name: _sha(endpoint_path)},
            }
        ),
        encoding="utf-8",
    )
    return endpoint_path, manifest_path


def test_isolated_evaluator_reports_safety_and_paired_deltas(tmp_path: Path) -> None:
    from src import next12_dft_endpoint_evaluate as module

    queue, queue_manifest, _ = _queue(tmp_path)
    endpoints, endpoint_manifest = _endpoints(tmp_path, queue_manifest)
    output = tmp_path / "evaluation"
    result = module.evaluate_dft_endpoints(
        queue_path=queue,
        queue_manifest_path=queue_manifest,
        endpoints_path=endpoints,
        endpoint_manifest_path=endpoint_manifest,
        output_dir=output,
    )
    assert result["counts"] == {
        "attempts": 6,
        "converged": 6,
        "failed": 0,
        "timeout": 0,
        "complete_groups": 2,
        "evaluable_rows": 6,
        "valuable_rows": 3,
        "high_energy_rows": 2,
    }
    m5 = result["methods"]["m5"]
    combined = result["methods"]["m5_phsc_chsc"]
    assert m5["valuable_recall"]["estimate"] == 1.0
    assert m5["high_energy_rejection_recall"]["estimate"] == 1.0
    assert m5["dft_savings_row_fraction"] == pytest.approx(2 / 6)
    assert combined["valuable_recall"]["estimate"] == pytest.approx(2 / 3)
    assert combined["dft_savings_row_fraction"] == pytest.approx(3 / 6)
    paired = result["paired_vs_m5"]["m5_phsc_chsc"]
    assert paired["net_reject_delta"] == 1
    assert paired["valuable_added_false_rejects"] == 1
    assert paired["high_energy_added_true_rejects"] == 0
    assert result["superiority_gate"]["passes"] is False
    assert result["all_attempts_accounted"] is True
    assert result["endpoint_opening_was_isolated"] is True
    assert (output / module.RESULT_NAME).is_file()


def test_incomplete_composition_group_is_excluded_not_silently_survived(
    tmp_path: Path,
) -> None:
    from src.next12_dft_endpoint_evaluate import evaluate_dft_endpoints

    queue, queue_manifest, _ = _queue(tmp_path)
    endpoints, endpoint_manifest = _endpoints(
        tmp_path, queue_manifest, timeout_sid="s1"
    )
    result = evaluate_dft_endpoints(
        queue_path=queue,
        queue_manifest_path=queue_manifest,
        endpoints_path=endpoints,
        endpoint_manifest_path=endpoint_manifest,
        output_dir=tmp_path / "evaluation",
    )
    assert result["counts"]["timeout"] == 1
    assert result["counts"]["complete_groups"] == 1
    assert result["counts"]["evaluable_rows"] == 3
    assert result["counts"]["valuable_rows"] == 1


def test_endpoint_sid_mismatch_and_overwrite_fail_closed(tmp_path: Path) -> None:
    from src.next12_dft_endpoint_evaluate import evaluate_dft_endpoints

    queue, queue_manifest, _ = _queue(tmp_path)
    endpoints, endpoint_manifest = _endpoints(tmp_path, queue_manifest)
    bad = pd.read_parquet(endpoints).iloc[:-1]
    bad.to_parquet(endpoints, index=False)
    manifest = json.loads(endpoint_manifest.read_text())
    manifest["outputs_sha256"][endpoints.name] = _sha(endpoints)
    endpoint_manifest.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="sid sets differ"):
        evaluate_dft_endpoints(
            queue_path=queue,
            queue_manifest_path=queue_manifest,
            endpoints_path=endpoints,
            endpoint_manifest_path=endpoint_manifest,
            output_dir=tmp_path / "bad",
        )
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        evaluate_dft_endpoints(
            queue_path=queue,
            queue_manifest_path=queue_manifest,
            endpoints_path=endpoints,
            endpoint_manifest_path=endpoint_manifest,
            output_dir=existing,
        )


def test_cli_accepts_no_threshold_override() -> None:
    from src.next12_dft_endpoint_evaluate import main

    for forbidden in ("--valuable-threshold", "--high-energy", "--subset", "--tune"):
        with pytest.raises(SystemExit) as exc_info:
            main([forbidden, "x"])
        assert exc_info.value.code == 2
