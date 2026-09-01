"""Contract tests for the additive next11 PHSC MatterSim feature runner."""

from __future__ import annotations

import hashlib
import inspect
import json
import math
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import pytest
from ase import Atoms

from src.next10_lrrc_mattersim_features import BatchPrediction
from src.next11_phsc import PHSCStatus, evaluate_phsc, phsc_probe_group


def _atoms(symbol: str = "H2") -> Atoms:
    return Atoms(
        symbol,
        positions=[[1.1, 1.3, 1.7], [2.4, 2.2, 2.8]],
        cell=[8.0, 9.0, 10.0],
        pbc=True,
    )


def _linear_forces(atoms: Atoms) -> np.ndarray:
    return -np.asarray(atoms.get_positions(), dtype=float)


class _RecordingPredictor:
    def __init__(self) -> None:
        self.calls: list[list[Atoms]] = []

    def __call__(self, structures: list[Atoms]) -> BatchPrediction:
        self.calls.append([atoms.copy() for atoms in structures])
        return BatchPrediction(
            total_energies_ev=[0.0] * len(structures),
            forces_ev_per_a=[_linear_forces(atoms) for atoms in structures],
            stresses_ev_per_a3=[np.zeros((3, 3)) for _ in structures],
        )


class _TelemetryPredictor(_RecordingPredictor):
    def __init__(self, *, device: str, model_batch_size: int) -> None:
        super().__init__()
        self.device = device
        self.model_batch_size = model_batch_size
        self.evaluations = 0
        self.forward_calls = 0

    def __call__(self, structures: list[Atoms]) -> BatchPrediction:
        prediction = super().__call__(structures)
        self.evaluations += len(structures)
        self.forward_calls += math.ceil(len(structures) / self.model_batch_size)
        return prediction

    @property
    def telemetry(self) -> dict[str, object]:
        return {
            "model_parameter_device": self.device,
            "result_tensor_devices": [self.device],
            "forward_calls": self.forward_calls,
            "evaluations": self.evaluations,
            "peak_cuda_memory_bytes": 123456,
        }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _frame(symbol: str, separation: float = 1.4) -> str:
    return f'''2
Lattice="8 0 0 0 9 0 0 0 10" pbc="T T T" Properties=species:S:1:pos:R:3:forces:R:3 endpoint_label=forbidden energy=-999
{symbol} 1.1 1.3 1.7 999 999 999
{symbol} {1.1 + separation} 1.3 1.7 -999 -999 -999
'''


def _runner_inputs(tmp_path: Path) -> dict[str, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    features_path = tmp_path / "mattersim_committee_features.parquet"
    pd.DataFrame(
        {
            "sid": ["sid-b", "sid-fit", "sid-a", "sid-ns"],
            "rk": ["rk-b", "rk-fit", "rk-a", "rk-ns"],
            "stage": ["threshold_calibration"] * 4,
            "strict_x0_ok": [True, True, True, False],
            "m5_energy_total_ev": [-2.0, -3.0, -4.0, -5.0],
        }
    ).to_parquet(features_path, index=False)

    roles_path = tmp_path / "threshold_role_assignments.parquet"
    pd.DataFrame(
        {
            "sid": ["sid-a", "sid-b", "sid-fit", "sid-ns"],
            "rk": ["rk-a", "rk-b", "rk-fit", "rk-ns"],
            "stage": ["threshold_calibration"] * 4,
            "threshold_role": [
                "development_gate",
                "development_gate",
                "threshold_fit",
                "development_gate",
            ],
        }
    ).to_parquet(roles_path, index=False)

    raw_frames_path = tmp_path / "initial_frames.zip"
    with zipfile.ZipFile(raw_frames_path, "w") as archive:
        archive.writestr("nested/sid-b.extxyz", _frame("H"))
        archive.writestr("sid-a.extxyz", _frame("He"))
        archive.writestr("sid-fit.extxyz", _frame("Li", separation=1.2))

    checkpoint_path = tmp_path / "MatterSim-v1.0.0-5M.pth"
    checkpoint_path.write_bytes(b"fake-mattersim-5m")
    manifest_path = tmp_path / "feature-MANIFEST.json"
    manifest = {
        "protocol": "2026-08-01-mattersim-dual-checkpoint-x0-v1",
        "mode": "development",
        "production_protocol_eligible": True,
        "outputs_sha256": {features_path.name: _sha256(features_path)},
        "inputs_sha256": {
            "frames": {
                "path": str(raw_frames_path.resolve()),
                "sha256": _sha256(raw_frames_path),
            }
        },
        "checkpoints": {
            "m5": {
                "path": str(checkpoint_path.resolve()),
                "sha256": _sha256(checkpoint_path),
            }
        },
        "predictor_loaded_checkpoint_sha256": {"m5": _sha256(checkpoint_path)},
    }
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, allow_nan=False), encoding="utf-8"
    )
    from src.next11_geometry_only_frames import (
        MANIFEST_NAME as GEOMETRY_MANIFEST_NAME,
        OUTPUT_ARCHIVE_NAME as GEOMETRY_ARCHIVE_NAME,
        build_geometry_only_frames,
    )

    geometry_dir = tmp_path / "geometry-only"
    build_geometry_only_frames(
        raw_frames_zip_path=raw_frames_path,
        committee_features_path=features_path,
        role_assignments_path=roles_path,
        output_dir=geometry_dir,
    )
    return {
        "features": features_path,
        "roles": roles_path,
        "raw_frames": raw_frames_path,
        "frames": geometry_dir / GEOMETRY_ARCHIVE_NAME,
        "geometry_manifest": geometry_dir / GEOMETRY_MANIFEST_NAME,
        "feature_manifest": manifest_path,
        "checkpoint": checkpoint_path,
    }


def test_batch_streams_complete_four_probe_groups_and_matches_scalar() -> None:
    from src.next11_phsc_mattersim_features import evaluate_phsc_batch

    atoms = _atoms()
    predictor = _RecordingPredictor()

    observed = evaluate_phsc_batch(
        ["sid-a"],
        [atoms],
        predictor,
        groups_per_call=2,
    )

    assert len(observed) == 1
    assert observed[0].sid == "sid-a"
    assert observed[0].result.status is PHSCStatus.RESOLVED_NONNEGATIVE
    assert observed[0].result.force_call_count == 12 * len(atoms)
    scalar = evaluate_phsc(atoms, _linear_forces)
    assert observed[0].result == scalar
    assert [len(call) for call in predictor.calls] == [8, 8, 8]
    first_group = phsc_probe_group(atoms, 0, scalar.h)
    for actual, expected in zip(predictor.calls[0][:4], first_group, strict=True):
        np.testing.assert_allclose(actual.positions, expected.positions, rtol=0.0, atol=0.0)


def test_batch_preserves_cross_sid_mapping_with_nondivisible_group_chunks() -> None:
    from src.next10_lrrc_mattersim_features import BatchPrediction
    from src.next11_phsc_mattersim_features import evaluate_phsc_batch

    calls: list[list[Atoms]] = []

    def species_scaled_forces(atoms: Atoms) -> np.ndarray:
        scale = float(atoms.numbers[0])
        return -scale * np.asarray(atoms.positions, dtype=float)

    def predictor(structures: list[Atoms]) -> BatchPrediction:
        calls.append([atoms.copy() for atoms in structures])
        return BatchPrediction(
            total_energies_ev=[0.0] * len(structures),
            forces_ev_per_a=[species_scaled_forces(atoms) for atoms in structures],
            stresses_ev_per_a3=[np.zeros((3, 3)) for _ in structures],
        )

    supplied = {"sid-z": _atoms("He2"), "sid-a": _atoms("H2")}
    observed = evaluate_phsc_batch(
        ["sid-z", "sid-a"],
        [supplied["sid-z"], supplied["sid-a"]],
        predictor,
        groups_per_call=5,
    )

    assert [item.sid for item in observed] == ["sid-a", "sid-z"]
    assert [len(call) for call in calls] == [20, 20, 8]
    for item in observed:
        expected = evaluate_phsc(supplied[item.sid], species_scaled_forces)
        assert item.result == expected


def test_preinference_probe_failure_is_zero_call_abstain_without_probe_groups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src import next11_phsc_mattersim_features as module
    from src.next11_phsc import PHSCNumericalError

    original = module.phsc_probe_group

    def fail_for_helium(base: Atoms, coordinate: int, h: float):
        if int(base.numbers[0]) == 2 and coordinate == 1:
            raise PHSCNumericalError("h probe is not representable")
        return original(base, coordinate, h)

    monkeypatch.setattr(module, "phsc_probe_group", fail_for_helium)
    predictor = _RecordingPredictor()

    observed = module.evaluate_phsc_batch(
        ["sid-he", "sid-h"],
        [_atoms("He2"), _atoms("H2")],
        predictor,
        groups_per_call=2,
    )

    by_sid = {item.sid: item.result for item in observed}
    assert by_sid["sid-he"].status is PHSCStatus.ABSTAIN_NUMERICAL_FAILURE
    assert by_sid["sid-he"].force_call_count == 0
    assert by_sid["sid-he"].negative is None
    assert "h probe is not representable" in (by_sid["sid-he"].error or "")
    assert by_sid["sid-h"].status is PHSCStatus.RESOLVED_NONNEGATIVE
    assert sum(len(call) for call in predictor.calls) == 12 * len(_atoms("H2"))


def test_injected_runner_seals_exact_feature_table_and_nonproduction_manifest(
    tmp_path: Path,
) -> None:
    from src import next11_phsc_mattersim_features as module

    paths = _runner_inputs(tmp_path)
    predictor = _RecordingPredictor()
    output_dir = tmp_path / "phsc-features"

    manifest = module.run_label_free_features(
        features_path=paths["features"],
        role_assignments_path=paths["roles"],
        frames_zip_path=paths["frames"],
        geometry_manifest_path=paths["geometry_manifest"],
        feature_manifest_path=paths["feature_manifest"],
        checkpoint_path=paths["checkpoint"],
        output_dir=output_dir,
        predictor=predictor,
        device="cpu",
        model_batch_size=8,
        groups_per_call=2,
    )

    assert {path.name for path in output_dir.iterdir()} == {
        module.OUTPUT_NAME,
        module.MANIFEST_NAME,
    }
    table_path = output_dir / module.OUTPUT_NAME
    table = pd.read_parquet(table_path)
    assert list(table.columns) == list(module.OUTPUT_COLUMNS)
    assert table["sid"].tolist() == ["sid-a", "sid-b", "sid-ns"]
    assert table["phsc_status"].tolist() == [
        "resolved_nonnegative",
        "resolved_nonnegative",
        "abstain_unsupported_geometry",
    ]
    assert table["force_call_count"].tolist() == [24, 24, 0]
    assert table["natoms"].tolist() == [2, 2, 0]
    assert table["internal_dim"].tolist() == [3, 3, 0]
    assert [len(call) for call in predictor.calls] == [8] * 6

    loaded = json.loads((output_dir / module.MANIFEST_NAME).read_text("utf-8"))
    assert loaded == manifest
    assert manifest["protocol"] == module.PROTOCOL
    assert manifest["mode"] == "development_gate"
    assert manifest["labels_opened"] is False
    assert manifest["input_isolation"] == {
        "geometry_only": True,
        "geometry_protocol": "2026-08-02-next11-geometry-only-frames-v1",
        "raw_x0_archive_opened": False,
        "endpoint_label_artifacts_opened": False,
    }
    assert manifest["scientific_improvement_claim"] is False
    assert manifest["production_protocol_eligible"] is False
    assert manifest["evidence_role"] == "testing_only_not_scientific_evidence"
    assert manifest["adapter"] == {
        "mode": "injected_test_double",
        "index_alignment": "injected_batch_force_predictor_declared_aligned",
        "index_alignment_verified": False,
        "device": "cpu",
        "model_batch_size": 8,
        "groups_per_call": 2,
        "model_parameter_device": None,
        "result_tensor_devices": [],
        "evaluations": 48,
    }
    assert manifest["counts"] == {
        "feature_rows": 4,
        "role_assignment_rows": 4,
        "selected_rows": 3,
        "strict_rows": 2,
        "nonstrict_rows": 1,
        "probe_eligible_rows": 2,
        "resolved_negative_rows": 0,
        "resolved_nonnegative_rows": 2,
        "near_zero_or_inconsistent_rows": 0,
        "abstained_rows": 1,
        "coordinate_groups": 12,
        "probe_evaluations": 48,
        "batch_predictor_calls": 6,
    }
    assert manifest["execution"] == {
        "batch_predictor_calls": 6,
        "predictor_batch_sizes": [8] * 6,
        "max_predictor_batch_size": 8,
        "forward_calls": None,
        "peak_cuda_memory_bytes": None,
        "wall_time_seconds": manifest["execution"]["wall_time_seconds"],
    }
    assert manifest["execution"]["wall_time_seconds"] > 0.0
    assert manifest["feature_columns"] == list(module.OUTPUT_COLUMNS)
    assert manifest["criterion"] == {
        "name": "PHSC-v0",
        "scope": "fixed_cell_gamma_point_atomic_hessian",
        "step_fraction": 2**-8,
        "probe_order": ["+h", "-h", "+h/2", "-h/2"],
        "force_evaluations_per_atom": 12,
        "primary_decision_proxy": "two_scale_projected_operator_difference",
        "numerical_consistency_proxies_are_confidence_bounds": False,
        "numerical_consistency_proxies_are_rigorous_error_bounds": False,
    }
    assert manifest["formal_expectations"] == {
        "feature_rows": 12990,
        "role_assignment_rows": 4341,
        "selected_rows": 2171,
        "strict_rows": 2164,
        "nonstrict_rows": 7,
        "model_batch_size": 32,
        "groups_per_call": 256,
        "device_contract": "canonical_cuda:N",
        "checkpoint_sha256": module.FROZEN_M5_SHA256,
        "geometry_protocol": "2026-08-02-next11-geometry-only-frames-v1",
        "geometry_only_frames_sha256": module.FROZEN_NEXT11_INPUT_SHA256[
            "geometry_only_frames"
        ],
        "geometry_manifest_sha256": module.FROZEN_NEXT11_INPUT_SHA256[
            "geometry_manifest"
        ],
    }
    assert manifest["outputs_sha256"] == {module.OUTPUT_NAME: _sha256(table_path)}
    assert set(manifest["executed_source_sha256"]) == set(
        module.EXECUTED_SOURCE_RELATIVE
    )
    assert not list(tmp_path.glob(".phsc-features.staging-*"))


def test_runner_uses_sanitized_geometry_without_reopening_raw_x0_archive(
    tmp_path: Path,
) -> None:
    from src import next11_phsc_mattersim_features as module

    paths = _runner_inputs(tmp_path)
    paths["raw_frames"].unlink()
    output_dir = tmp_path / "raw-physically-absent"

    manifest = module.run_label_free_features(
        features_path=paths["features"],
        role_assignments_path=paths["roles"],
        frames_zip_path=paths["frames"],
        geometry_manifest_path=paths["geometry_manifest"],
        feature_manifest_path=paths["feature_manifest"],
        checkpoint_path=paths["checkpoint"],
        output_dir=output_dir,
        predictor=_RecordingPredictor(),
        device="cpu",
        model_batch_size=8,
        groups_per_call=2,
    )

    assert output_dir.is_dir()
    assert manifest["input_isolation"]["raw_x0_archive_opened"] is False
    assert manifest["inputs_sha256"]["source_frames_provenance"]["sha256"]


def test_runner_parses_initial_geometry_snapshot_not_swappable_live_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from src import next11_phsc_mattersim_features as module

    paths = _runner_inputs(tmp_path)
    frozen_archive = paths["frames"].read_bytes()
    frozen_manifest = paths["geometry_manifest"].read_bytes()
    original_loader = module.load_geometry_only_archive
    observed_private_paths: list[tuple[Path, Path]] = []

    def swap_live_inputs_then_load(*, archive_path, manifest_path, expected_sids):
        archive_path = Path(archive_path)
        manifest_path = Path(manifest_path)
        observed_private_paths.append((archive_path, manifest_path))
        assert archive_path != paths["frames"]
        assert manifest_path != paths["geometry_manifest"]
        assert archive_path.read_bytes() == frozen_archive
        assert manifest_path.read_bytes() == frozen_manifest
        paths["frames"].write_bytes(b"attacker replacement")
        paths["geometry_manifest"].write_bytes(b'{"attacker":true}')
        try:
            return original_loader(
                archive_path=archive_path,
                manifest_path=manifest_path,
                expected_sids=expected_sids,
            )
        finally:
            paths["frames"].write_bytes(frozen_archive)
            paths["geometry_manifest"].write_bytes(frozen_manifest)

    monkeypatch.setattr(module, "load_geometry_only_archive", swap_live_inputs_then_load)
    output_dir = tmp_path / "snapshot-bytes"
    module.run_label_free_features(
        features_path=paths["features"],
        role_assignments_path=paths["roles"],
        frames_zip_path=paths["frames"],
        geometry_manifest_path=paths["geometry_manifest"],
        feature_manifest_path=paths["feature_manifest"],
        checkpoint_path=paths["checkpoint"],
        output_dir=output_dir,
        predictor=_RecordingPredictor(),
        device="cpu",
        model_batch_size=8,
        groups_per_call=2,
    )

    assert len(observed_private_paths) == 1
    assert output_dir.is_dir()


def test_cli_has_no_label_surface_and_hardcodes_formal_batch_parameters(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from src import next11_phsc_mattersim_features as module

    assert not any(
        "label" in name.lower()
        for name in inspect.signature(module.run_label_free_features).parameters
    )
    captured: dict[str, object] = {}

    def capture(**kwargs):
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(module, "run_label_free_features", capture)
    paths = [tmp_path / name for name in ("f", "r", "z", "g", "m", "c", "o")]

    assert module.main(
        [
            "--features-path",
            str(paths[0]),
            "--role-assignments-path",
            str(paths[1]),
            "--frames-zip-path",
            str(paths[2]),
            "--geometry-manifest-path",
            str(paths[3]),
            "--feature-manifest-path",
            str(paths[4]),
            "--checkpoint-path",
            str(paths[5]),
            "--output-dir",
            str(paths[6]),
            "--device",
            "cuda:0",
        ]
    ) == 0
    assert captured["predictor"] is None
    assert captured["geometry_manifest_path"] == paths[3]
    assert captured["device"] == "cuda:0"
    assert captured["model_batch_size"] == module.FROZEN_MODEL_BATCH_SIZE == 32
    assert captured["groups_per_call"] == module.FROZEN_GROUPS_PER_CALL == 256
    assert captured["engineering_smoke"] is False


def test_production_stub_requires_frozen_parameters_and_seals_cuda_telemetry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from src import next11_phsc_mattersim_features as module

    paths = _runner_inputs(tmp_path)
    checkpoint_sha256 = _sha256(paths["checkpoint"])
    predictor = _TelemetryPredictor(device="cuda:0", model_batch_size=32)
    monkeypatch.setattr(module, "FROZEN_M5_SHA256", checkpoint_sha256)
    monkeypatch.setattr(
        module,
        "FORMAL_EXPECTED_COUNTS",
        {
            "feature_rows": 4,
            "role_assignment_rows": 4,
            "selected_rows": 3,
            "strict_rows": 2,
            "nonstrict_rows": 1,
        },
    )
    monkeypatch.setattr(module, "_require_frozen_next11_inputs", lambda _snapshots: None)
    monkeypatch.setattr(
        module,
        "_runtime_identity",
        lambda device: {
            "python_version": "3.test",
            "python_implementation": "CPython",
            "platform": "test-platform",
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "ase_version": "test",
            "mattersim_version": "1.2.3",
            "device": device,
            "torch_version": "test",
            "cuda_available": True,
            "cuda_version": "test",
            "gpu_name": "test-gpu",
        },
    )
    monkeypatch.setattr(
        module,
        "_production_predictor",
        lambda checkpoint_path, *, device, batch_size: (
            predictor,
            _sha256(checkpoint_path),
        ),
    )

    manifest = module.run_label_free_features(
        features_path=paths["features"],
        role_assignments_path=paths["roles"],
        frames_zip_path=paths["frames"],
        geometry_manifest_path=paths["geometry_manifest"],
        feature_manifest_path=paths["feature_manifest"],
        checkpoint_path=paths["checkpoint"],
        output_dir=tmp_path / "production",
        predictor=None,
        device="cuda:0",
        model_batch_size=32,
        groups_per_call=256,
    )

    assert manifest["production_protocol_eligible"] is True
    assert manifest["evidence_role"] == "label_free_phsc_feature_generation"
    assert manifest["predictor_loaded_checkpoint_sha256"] == checkpoint_sha256
    assert manifest["adapter"] == {
        "mode": "builtin_indexed_mattersim",
        "index_alignment": "sid_indexed_exact_one_to_one",
        "index_alignment_verified": True,
        "device": "cuda:0",
        "model_batch_size": 32,
        "groups_per_call": 256,
        "model_parameter_device": "cuda:0",
        "result_tensor_devices": ["cuda:0"],
        "evaluations": 48,
    }
    assert manifest["execution"]["batch_predictor_calls"] == 1
    assert manifest["execution"]["predictor_batch_sizes"] == [48]
    assert manifest["execution"]["max_predictor_batch_size"] == 48
    assert manifest["execution"]["forward_calls"] == 2
    assert manifest["execution"]["peak_cuda_memory_bytes"] == 123456

    with pytest.raises(ValueError, match="production PHSC requires"):
        module.run_label_free_features(
            features_path=paths["features"],
            role_assignments_path=paths["roles"],
            frames_zip_path=paths["frames"],
            geometry_manifest_path=paths["geometry_manifest"],
            feature_manifest_path=paths["feature_manifest"],
            checkpoint_path=paths["checkpoint"],
            output_dir=tmp_path / "bad-production",
            predictor=None,
            device="cuda",
            model_batch_size=32,
            groups_per_call=255,
        )


def test_engineering_cuda_smoke_uses_real_adapter_provenance_but_is_nonproduction(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from src import next11_phsc_mattersim_features as module

    paths = _runner_inputs(tmp_path)
    checkpoint_sha256 = _sha256(paths["checkpoint"])
    predictors: list[_TelemetryPredictor] = []
    monkeypatch.setattr(module, "FROZEN_M5_SHA256", checkpoint_sha256)
    monkeypatch.setattr(module, "FROZEN_ENGINEERING_SMOKE_COUNT", 2)
    monkeypatch.setattr(
        module,
        "FORMAL_EXPECTED_COUNTS",
        {
            "feature_rows": 4,
            "role_assignment_rows": 4,
            "selected_rows": 3,
            "strict_rows": 2,
            "nonstrict_rows": 1,
        },
    )
    monkeypatch.setattr(module, "_require_frozen_next11_inputs", lambda _snapshots: None)
    monkeypatch.setattr(
        module,
        "_runtime_identity",
        lambda device: {
            "python_version": "3.test",
            "python_implementation": "CPython",
            "platform": "test-platform",
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "ase_version": "test",
            "mattersim_version": "1.2.3",
            "device": device,
            "torch_version": "test",
            "cuda_available": True,
            "cuda_version": "test",
            "gpu_name": "test-gpu",
        },
    )

    def production_predictor(checkpoint_path: Path, *, device: str, batch_size: int):
        predictor = _TelemetryPredictor(device=device, model_batch_size=batch_size)
        predictors.append(predictor)
        return predictor, _sha256(checkpoint_path)

    monkeypatch.setattr(module, "_production_predictor", production_predictor)
    manifest = module.run_label_free_features(
        features_path=paths["features"],
        role_assignments_path=paths["roles"],
        frames_zip_path=paths["frames"],
        geometry_manifest_path=paths["geometry_manifest"],
        feature_manifest_path=paths["feature_manifest"],
        checkpoint_path=paths["checkpoint"],
        output_dir=tmp_path / "engineering-smoke",
        predictor=None,
        device="cuda:0",
        model_batch_size=32,
        groups_per_call=256,
        engineering_smoke=True,
    )

    assert len(predictors) == 1
    assert manifest["mode"] == "engineering_cuda_smoke"
    assert manifest["production_protocol_eligible"] is False
    assert manifest["evidence_role"] == "engineering_cuda_smoke_only"
    assert manifest["scientific_improvement_claim"] is False
    assert manifest["predictor_loaded_checkpoint_sha256"] == checkpoint_sha256
    assert manifest["adapter"]["mode"] == "builtin_indexed_mattersim"
    assert manifest["adapter"]["index_alignment_verified"] is True
    assert manifest["selection"]["count"] == 2
    assert manifest["selection"]["sids"] == ["sid-a", "sid-b"]
    assert manifest["counts"]["selected_rows"] == 2
    assert manifest["counts"]["strict_rows"] == 2
    assert manifest["execution"]["predictor_batch_sizes"] == [48]
    assert manifest["execution"]["forward_calls"] == 2

    with pytest.raises(ValueError, match="builtin indexed predictor"):
        module.run_label_free_features(
            features_path=paths["features"],
            role_assignments_path=paths["roles"],
            frames_zip_path=paths["frames"],
            geometry_manifest_path=paths["geometry_manifest"],
            feature_manifest_path=paths["feature_manifest"],
            checkpoint_path=paths["checkpoint"],
            output_dir=tmp_path / "invalid-smoke",
            predictor=_RecordingPredictor(),
            device="cpu",
            model_batch_size=8,
            groups_per_call=2,
            engineering_smoke=True,
        )


def test_predictor_failure_is_fatal_without_retry_or_partial_publication(
    tmp_path: Path,
) -> None:
    from src import next11_phsc_mattersim_features as module

    paths = _runner_inputs(tmp_path)
    output_dir = tmp_path / "failed"
    calls = 0

    def exploding_predictor(structures: list[Atoms]) -> BatchPrediction:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("predictor exploded")
        return BatchPrediction(
            total_energies_ev=[0.0] * len(structures),
            forces_ev_per_a=[_linear_forces(atoms) for atoms in structures],
            stresses_ev_per_a3=[np.zeros((3, 3)) for _ in structures],
        )

    with pytest.raises(module.BatchPHSCError, match="predictor exploded"):
        module.run_label_free_features(
            features_path=paths["features"],
            role_assignments_path=paths["roles"],
            frames_zip_path=paths["frames"],
            geometry_manifest_path=paths["geometry_manifest"],
            feature_manifest_path=paths["feature_manifest"],
            checkpoint_path=paths["checkpoint"],
            output_dir=output_dir,
            predictor=exploding_predictor,
            device="cpu",
            model_batch_size=8,
            groups_per_call=2,
        )

    assert calls == 2
    assert not output_dir.exists()
    assert not list(tmp_path.glob(".failed.staging-*"))


def test_missing_coordinate_is_fatal_even_when_predictor_outputs_are_valid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src import next11_phsc_mattersim_features as module

    original = module._probe_groups

    def omit_last(prepared):
        groups = list(original(prepared))
        yield from groups[:-1]

    monkeypatch.setattr(module, "_probe_groups", omit_last)
    predictor = _RecordingPredictor()

    with pytest.raises(module.BatchPHSCError, match="incomplete coordinate set"):
        module.evaluate_phsc_batch(
            ["sid-a"], [_atoms()], predictor, groups_per_call=2
        )

    assert sum(len(call) for call in predictor.calls) == 20


def test_input_drift_aborts_and_atomic_publish_race_preserves_competitor(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from src import next11_phsc_mattersim_features as module

    paths = _runner_inputs(tmp_path)
    drift_output = tmp_path / "drift"
    predictor = _RecordingPredictor()
    original_call = predictor.__call__
    mutated = False

    def mutating_predictor(structures: list[Atoms]) -> BatchPrediction:
        nonlocal mutated
        if not mutated:
            with paths["roles"].open("ab") as stream:
                stream.write(b"drift")
            mutated = True
        return original_call(structures)

    with pytest.raises(RuntimeError, match="input roles changed"):
        module.run_label_free_features(
            features_path=paths["features"],
            role_assignments_path=paths["roles"],
            frames_zip_path=paths["frames"],
            geometry_manifest_path=paths["geometry_manifest"],
            feature_manifest_path=paths["feature_manifest"],
            checkpoint_path=paths["checkpoint"],
            output_dir=drift_output,
            predictor=mutating_predictor,
            device="cpu",
            model_batch_size=8,
            groups_per_call=2,
        )
    assert not drift_output.exists()
    assert not list(tmp_path.glob(".drift.staging-*"))

    paths = _runner_inputs(tmp_path / "race-inputs")
    race_output = tmp_path / "race"
    original_publish = module._atomic_publish_directory_no_replace

    def racing_publish(staging: Path, target: Path) -> None:
        target.mkdir()
        (target / "competitor.txt").write_text("keep", encoding="utf-8")
        original_publish(staging, target)

    monkeypatch.setattr(module, "_atomic_publish_directory_no_replace", racing_publish)
    with pytest.raises(FileExistsError):
        module.run_label_free_features(
            features_path=paths["features"],
            role_assignments_path=paths["roles"],
            frames_zip_path=paths["frames"],
            geometry_manifest_path=paths["geometry_manifest"],
            feature_manifest_path=paths["feature_manifest"],
            checkpoint_path=paths["checkpoint"],
            output_dir=race_output,
            predictor=_RecordingPredictor(),
            device="cpu",
            model_batch_size=8,
            groups_per_call=2,
        )
    assert (race_output / "competitor.txt").read_text("utf-8") == "keep"
    assert not list(tmp_path.glob(".race.staging-*"))


def test_executed_source_drift_aborts_before_publication_without_editing_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from src import next11_phsc_mattersim_features as module

    paths = _runner_inputs(tmp_path)
    output_dir = tmp_path / "source-drift"
    original_sha256 = module._sha256_file
    runner_path = Path(module.__file__).resolve()
    runner_hash_calls = 0

    def drifting_hash(path: Path) -> str:
        nonlocal runner_hash_calls
        if Path(path).resolve() == runner_path:
            runner_hash_calls += 1
            if runner_hash_calls >= 2:
                return "0" * 64
        return original_sha256(path)

    monkeypatch.setattr(module, "_sha256_file", drifting_hash)

    with pytest.raises(RuntimeError, match="executed source.*changed"):
        module.run_label_free_features(
            features_path=paths["features"],
            role_assignments_path=paths["roles"],
            frames_zip_path=paths["frames"],
            geometry_manifest_path=paths["geometry_manifest"],
            feature_manifest_path=paths["feature_manifest"],
            checkpoint_path=paths["checkpoint"],
            output_dir=output_dir,
            predictor=_RecordingPredictor(),
            device="cpu",
            model_batch_size=8,
            groups_per_call=2,
        )

    assert runner_hash_calls == 2
    assert not output_dir.exists()
    assert not list(tmp_path.glob(".source-drift.staging-*"))


def test_staged_parquet_value_drift_aborts_before_publication(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from src import next11_phsc_mattersim_features as module

    paths = _runner_inputs(tmp_path)
    output_dir = tmp_path / "value-drift"
    original_read_parquet = pd.read_parquet

    def corrupt_staged_table(source, *args, **kwargs):
        frame = original_read_parquet(source, *args, **kwargs)
        if isinstance(source, Path) and source.name == module.OUTPUT_NAME:
            frame = frame.copy()
            frame.loc[0, "natoms"] = int(frame.loc[0, "natoms"]) + 1
        return frame

    monkeypatch.setattr(pd, "read_parquet", corrupt_staged_table)
    with pytest.raises(RuntimeError, match="exact value validation"):
        module.run_label_free_features(
            features_path=paths["features"],
            role_assignments_path=paths["roles"],
            frames_zip_path=paths["frames"],
            geometry_manifest_path=paths["geometry_manifest"],
            feature_manifest_path=paths["feature_manifest"],
            checkpoint_path=paths["checkpoint"],
            output_dir=output_dir,
            predictor=_RecordingPredictor(),
            device="cpu",
            model_batch_size=8,
            groups_per_call=2,
        )

    assert not output_dir.exists()
    assert not list(tmp_path.glob(".value-drift.staging-*"))


def test_runner_reuses_next10_indexed_adapter_instead_of_copying_it() -> None:
    from src import next11_phsc_mattersim_features as module

    assert module._production_predictor.__module__ == (
        "src.next10_lrrc_mattersim_features"
    )
    assert "_IndexedMatterSimPredictor" not in module.__dict__
