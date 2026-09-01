"""Contract tests for the additive CHSC-v0 MatterSim feature runner."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import pytest
from ase import Atoms
from scipy.linalg import logm

from src.next10_lrrc_mattersim_features import BatchPrediction


def _atoms(symbol: str = "H2") -> Atoms:
    return Atoms(
        symbol,
        scaled_positions=[[0.15, 0.25, 0.35], [0.62, 0.71, 0.83]],
        cell=[[8.0, 0.0, 0.0], [0.4, 9.0, 0.0], [0.2, 0.3, 10.0]],
        pbc=True,
    )


class _QuadraticPredictor:
    def __init__(self) -> None:
        self.calls: list[list[Atoms]] = []
        self.reference = _atoms().cell.array.copy()

    def _energy(self, atoms: Atoms) -> float:
        from src.next12_chsc import strain_basis

        relative = np.linalg.solve(self.reference, atoms.cell.array)
        strain = np.real_if_close(logm(relative.T), tol=1000)
        coordinates = np.einsum("aij,ij->a", strain_basis(), strain)
        first = -2.0 if int(atoms.numbers[0]) == 1 else 1.0
        hessian = np.diag([first, 2.0, 3.0, 4.0, 5.0, 6.0])
        return float(0.5 * len(atoms) * coordinates @ hessian @ coordinates)

    def __call__(self, structures: list[Atoms]) -> BatchPrediction:
        self.calls.append([atoms.copy() for atoms in structures])
        return BatchPrediction(
            total_energies_ev=[self._energy(atoms) for atoms in structures],
            forces_ev_per_a=[np.zeros((len(atoms), 3)) for atoms in structures],
            stresses_ev_per_a3=[np.zeros((3, 3)) for _ in structures],
        )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _raw_frame(symbol: str) -> str:
    return f'''2
Lattice="8 0 0 0.4 9 0 0.2 0.3 10" pbc="T T T" Properties=species:S:1:pos:R:3:forces:R:3 endpoint_label=forbidden energy=-999
{symbol} 1.36 2.355 3.5 999 999 999
{symbol} 5.412 6.639 8.3 -999 -999 -999
'''


def _runner_inputs(tmp_path: Path) -> dict[str, Path]:
    from src.next11_geometry_only_frames import build_geometry_only_frames

    tmp_path.mkdir(parents=True, exist_ok=True)
    committee = tmp_path / "mattersim_committee_features.parquet"
    pd.DataFrame(
        {
            "sid": ["sid-b", "sid-a", "sid-ns"],
            "rk": ["rk-b", "rk-a", "rk-ns"],
            "stage": ["threshold_calibration"] * 3,
            "strict_x0_ok": [True, True, False],
            "m5_energy_total_ev": [-2.0, -3.0, -4.0],
        }
    ).to_parquet(committee, index=False)
    roles = tmp_path / "threshold_role_assignments.parquet"
    pd.DataFrame(
        {
            "sid": ["sid-a", "sid-b", "sid-ns"],
            "rk": ["rk-a", "rk-b", "rk-ns"],
            "stage": ["threshold_calibration"] * 3,
            "threshold_role": ["development_gate"] * 3,
        }
    ).to_parquet(roles, index=False)
    raw = tmp_path / "initial_frames.zip"
    with zipfile.ZipFile(raw, "w") as archive:
        archive.writestr("sid-a.extxyz", _raw_frame("H"))
        archive.writestr("nested/sid-b.extxyz", _raw_frame("He"))
    geometry_dir = tmp_path / "geometry"
    build_geometry_only_frames(
        raw_frames_zip_path=raw,
        committee_features_path=committee,
        role_assignments_path=roles,
        output_dir=geometry_dir,
    )

    phsc = tmp_path / "phsc_features.parquet"
    pd.DataFrame(
        {
            "sid": ["sid-ns", "sid-b", "sid-a"],
            "rk": ["rk-ns", "rk-b", "rk-a"],
            "stage": ["threshold_calibration"] * 3,
            "threshold_role": ["development_gate"] * 3,
            "strict_x0_ok": [False, True, True],
            "phsc_status": [
                "abstain_unsupported_geometry",
                "resolved_nonnegative",
                "resolved_negative",
            ],
            "phsc_negative": [None, False, True],
        }
    ).to_parquet(phsc, index=False)
    phsc_manifest = tmp_path / "PHSC-MANIFEST.json"
    phsc_manifest.write_text(
        json.dumps(
            {
                "protocol": "2026-08-02-next11-phsc-mattersim-features-v1",
                "labels_opened": False,
                "input_isolation": {
                    "geometry_only": True,
                    "raw_x0_archive_opened": False,
                    "endpoint_label_artifacts_opened": False,
                },
                "outputs_sha256": {phsc.name: _sha256(phsc)},
            },
            sort_keys=True,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    checkpoint = tmp_path / "MatterSim-v1.0.0-5M.pth"
    checkpoint.write_bytes(b"fake-mattersim-5m")
    return {
        "phsc": phsc,
        "phsc_manifest": phsc_manifest,
        "frames": geometry_dir / "geometry_only_frames.zip",
        "geometry_manifest": geometry_dir / "MANIFEST.json",
        "checkpoint": checkpoint,
    }


def test_batch_uses_complete_85_probe_groups_and_matches_scalar() -> None:
    from src.next12_chsc import CHSCStatus, evaluate_chsc
    from src.next12_chsc_mattersim_features import evaluate_chsc_batch

    supplied = {"sid-z": _atoms("He2"), "sid-a": _atoms("H2")}
    predictor = _QuadraticPredictor()
    observed = evaluate_chsc_batch(
        ["sid-z", "sid-a"],
        [supplied["sid-z"], supplied["sid-a"]],
        predictor,
        structures_per_call=2,
    )

    assert [item.sid for item in observed] == ["sid-a", "sid-z"]
    assert [len(call) for call in predictor.calls] == [170]
    assert observed[0].result.status is CHSCStatus.RESOLVED_NEGATIVE
    assert observed[1].result.status is CHSCStatus.RESOLVED_NONNEGATIVE
    assert all(item.result.energy_call_count == 85 for item in observed)
    assert all(item.hessian_h.shape == (6, 6) for item in observed)
    assert all(item.hessian_h2.shape == (6, 6) for item in observed)

    for item in observed:
        scalar_predictor = _QuadraticPredictor()
        scalar = evaluate_chsc(
            supplied[item.sid],
            lambda atoms: scalar_predictor._energy(atoms),
        )
        assert item.result == scalar


def test_injected_runner_publishes_complete_table_and_label_free_manifest(tmp_path: Path) -> None:
    from src import next12_chsc_mattersim_features as module

    paths = _runner_inputs(tmp_path)
    predictor = _QuadraticPredictor()
    output_dir = tmp_path / "chsc-features"
    manifest = module.run_label_free_features(
        phsc_features_path=paths["phsc"],
        phsc_manifest_path=paths["phsc_manifest"],
        frames_zip_path=paths["frames"],
        geometry_manifest_path=paths["geometry_manifest"],
        checkpoint_path=paths["checkpoint"],
        output_dir=output_dir,
        predictor=predictor,
        device="cpu",
        model_batch_size=32,
        structures_per_call=2,
    )

    table_path = output_dir / module.OUTPUT_NAME
    table = pd.read_parquet(table_path)
    assert list(table.columns) == list(module.OUTPUT_COLUMNS)
    assert table["sid"].tolist() == ["sid-a", "sid-b", "sid-ns"]
    assert table["chsc_status"].tolist() == [
        "resolved_negative",
        "resolved_nonnegative",
        "abstain_unsupported_geometry",
    ]
    assert table["energy_call_count"].tolist() == [85, 85, 0]
    assert all(len(json.loads(value)) == 6 for value in table.loc[:1, "hessian_h_json"])

    loaded = json.loads((output_dir / module.MANIFEST_NAME).read_text("utf-8"))
    assert loaded == manifest
    assert manifest["protocol"] == module.PROTOCOL
    assert manifest["labels_opened"] is False
    assert manifest["input_isolation"] == {
        "geometry_only": True,
        "raw_x0_archive_opened": False,
        "endpoint_label_artifacts_opened": False,
    }
    assert manifest["production_protocol_eligible"] is False
    assert manifest["scientific_improvement_claim"] is False
    assert manifest["counts"] == {
        "selected_rows": 3,
        "strict_rows": 2,
        "nonstrict_rows": 1,
        "resolved_negative_rows": 1,
        "resolved_nonnegative_rows": 1,
        "near_zero_or_inconsistent_rows": 0,
        "abstained_rows": 1,
        "energy_evaluations": 170,
        "batch_predictor_calls": 1,
    }
    assert manifest["criterion"]["name"] == "CHSC-v0"
    assert manifest["criterion"]["energy_evaluations_per_structure"] == 85
    assert manifest["outputs_sha256"] == {module.OUTPUT_NAME: _sha256(table_path)}
    assert set(manifest["executed_source_sha256"]) == set(
        module.EXECUTED_SOURCE_RELATIVE
    )
    assert not list(tmp_path.glob(".chsc-features.staging-*"))


def test_runner_works_after_raw_archive_is_physically_removed(tmp_path: Path) -> None:
    from src.next12_chsc_mattersim_features import run_label_free_features

    paths = _runner_inputs(tmp_path)
    (tmp_path / "initial_frames.zip").unlink()
    output_dir = tmp_path / "without-raw"
    manifest = run_label_free_features(
        phsc_features_path=paths["phsc"],
        phsc_manifest_path=paths["phsc_manifest"],
        frames_zip_path=paths["frames"],
        geometry_manifest_path=paths["geometry_manifest"],
        checkpoint_path=paths["checkpoint"],
        output_dir=output_dir,
        predictor=_QuadraticPredictor(),
        device="cpu",
        structures_per_call=2,
    )
    assert output_dir.is_dir()
    assert manifest["input_isolation"]["raw_x0_archive_opened"] is False


def test_runner_never_overwrites_existing_output(tmp_path: Path) -> None:
    from src.next12_chsc_mattersim_features import run_label_free_features

    paths = _runner_inputs(tmp_path)
    output_dir = tmp_path / "existing"
    output_dir.mkdir()
    (output_dir / "sentinel").write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        run_label_free_features(
            phsc_features_path=paths["phsc"],
            phsc_manifest_path=paths["phsc_manifest"],
            frames_zip_path=paths["frames"],
            geometry_manifest_path=paths["geometry_manifest"],
            checkpoint_path=paths["checkpoint"],
            output_dir=output_dir,
            predictor=_QuadraticPredictor(),
            device="cpu",
            structures_per_call=2,
        )
    assert (output_dir / "sentinel").read_text("utf-8") == "keep"


def test_cli_exposes_no_label_or_endpoint_argument(tmp_path: Path) -> None:
    from src.next12_chsc_mattersim_features import main

    for forbidden in ("--labels", "--endpoint", "--dft-results"):
        with pytest.raises(SystemExit) as exc_info:
            main([forbidden, "x"])
        assert exc_info.value.code == 2
