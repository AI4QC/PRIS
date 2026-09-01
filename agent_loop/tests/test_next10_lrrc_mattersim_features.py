"""Tests for the label-free next10 batched LRRC feature runner."""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from typing import Callable
import zipfile

import numpy as np
import pandas as pd
import pytest
from ase import Atoms

from src.next10_lrrc_mattersim_features import BatchPrediction
from src.next9_lrrc import LRRCStatus, evaluate_lrrc, translation_projected_direction


OUTPUT_COLUMNS = [
    "sid",
    "rk",
    "stage",
    "threshold_role",
    "strict_x0_ok",
    "natoms",
    "lrrc_status",
    "lrrc_negative",
    "d_star_angstrom",
    "h_angstrom",
    "kappa_h_ev_per_a2",
    "kappa_h2_ev_per_a2",
    "kappa_r_ev_per_a2",
    "error_proxy_ev_per_a2",
    "u_num_ev_per_a2",
    "force_call_count",
    "error",
]


def _atoms(
    atomic_number: int,
    *,
    separation: float = 1.4,
) -> Atoms:
    return Atoms(
        numbers=[atomic_number, atomic_number],
        positions=[[0.2, 0.4, 0.6], [0.2 + separation, 0.4, 0.6]],
        cell=[8.0, 8.0, 8.0],
        pbc=True,
    )


def _analytic_forces(atoms: Atoms) -> np.ndarray:
    atomic_number = int(atoms.numbers[0])
    curvature = -2.0 if atomic_number == 2 else 1.0
    vectors = atoms.get_all_distances(mic=True, vector=True)
    return curvature * np.sum(vectors, axis=1)


class _AnalyticBatchPredictor:
    def __init__(self, force_function: Callable[[Atoms], np.ndarray] = _analytic_forces):
        self.force_function = force_function
        self.calls: list[list[Atoms]] = []

    def __call__(self, structures: list[Atoms]) -> BatchPrediction:
        self.calls.append([atoms.copy() for atoms in structures])
        return BatchPrediction(
            total_energies_ev=[0.0] * len(structures),
            forces_ev_per_a=[self.force_function(atoms) for atoms in structures],
            stresses_ev_per_a3=[np.zeros((3, 3)) for _ in structures],
        )


class _TelemetryBatchPredictor:
    def __init__(
        self,
        *,
        device: str,
        peak_cuda_memory_bytes: int,
    ) -> None:
        self.delegate = _AnalyticBatchPredictor()
        self.device = device
        self.peak_cuda_memory_bytes = peak_cuda_memory_bytes
        self.forward_calls = 0
        self.evaluations = 0

    def __call__(self, structures: list[Atoms]) -> BatchPrediction:
        self.forward_calls += 1
        self.evaluations += len(structures)
        return self.delegate(structures)

    @property
    def telemetry(self) -> dict[str, object]:
        return {
            "model_parameter_device": self.device,
            "result_tensor_devices": [self.device],
            "forward_calls": self.forward_calls,
            "evaluations": self.evaluations,
            "peak_cuda_memory_bytes": self.peak_cuda_memory_bytes,
        }


def _runtime_record(*, device: str, cuda_available: bool) -> dict[str, object]:
    return {
        "python_version": "test",
        "python_implementation": "CPython",
        "platform": "test",
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "ase_version": "test",
        "mattersim_version": "1.2.3",
        "torch_version": "test",
        "cuda_available": cuda_available,
        "cuda_version": "test" if cuda_available else "unavailable",
        "gpu_name": "test-gpu" if cuda_available else "unavailable",
        "device": device,
    }


class _FakeTensor:
    def __init__(self, values, *, device: str = "cpu") -> None:
        self.values = np.asarray(values)
        self.device = device

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self) -> np.ndarray:
        return self.values.copy()

    def tolist(self):
        return self.values.tolist()

    def reshape(self, *shape):
        return _FakeTensor(self.values.reshape(*shape), device=self.device)

    def __array__(self, dtype=None):
        return np.asarray(self.values, dtype=dtype)


class _FakeGraph:
    def __init__(self, marker: int, natoms: int) -> None:
        self.marker = marker
        self.num_atoms = natoms


class _FakeGraphBatch:
    def __init__(self, graphs: list[_FakeGraph], *, device: str = "cpu") -> None:
        self.graphs = graphs
        self.next10_input_ordinal = _FakeTensor(
            [graph.next10_input_ordinal for graph in graphs]
        )
        self.num_atoms = _FakeTensor([graph.num_atoms for graph in graphs])
        self.device = device


class _FakeLoader:
    def __init__(
        self,
        dataset: list[_FakeGraph],
        emission: list[int],
        reported_natoms: list[float] | None = None,
    ) -> None:
        self.dataset = dataset
        self.emission = emission
        self.reported_natoms = reported_natoms

    def __iter__(self):
        batch = _FakeGraphBatch([self.dataset[index] for index in self.emission])
        if self.reported_natoms is not None:
            batch.num_atoms = _FakeTensor(self.reported_natoms)
        yield batch


class _FakeModel:
    model_args = {"cutoff": 5.0, "threebody_cutoff": 4.0}

    def __init__(self, device: str) -> None:
        self._parameter = type("Parameter", (), {"device": device})()

    def parameters(self):
        yield self._parameter

    def eval(self) -> None:
        return None


class _FakePotential:
    def __init__(self, *, device: str = "cpu") -> None:
        self.device = device
        self.model = _FakeModel(device)

    def forward(
        self,
        inputs: dict[str, object],
        *,
        include_forces: bool,
        include_stresses: bool,
    ) -> dict[str, _FakeTensor]:
        assert include_forces is True
        assert include_stresses is True
        batch = inputs["graph_batch"]
        markers = [graph.marker for graph in batch.graphs]
        forces = np.concatenate(
            [np.full((graph.num_atoms, 3), graph.marker) for graph in batch.graphs]
        )
        stresses = np.asarray(
            [np.eye(3) * graph.marker for graph in batch.graphs]
        )
        return {
            "total_energy": _FakeTensor(
                np.asarray(markers, dtype=float) * 10.0,
                device=self.device,
            ),
            "forces": _FakeTensor(forces, device=self.device),
            "stresses": _FakeTensor(stresses, device=self.device),
        }


class _FakeCuda:
    def reset_peak_memory_stats(self, _device: str) -> None:
        return None

    def max_memory_allocated(self, _device: str) -> int:
        return 4096


class _FakeTorch:
    def __init__(self) -> None:
        self.cuda = _FakeCuda()


def _fake_indexed_predictor(
    emission: list[int],
    *,
    device: str = "cpu",
    reported_natoms: list[float] | None = None,
):
    from src.next10_lrrc_mattersim_features import _IndexedMatterSimPredictor

    build_calls: list[dict[str, object]] = []

    def build_dataloader(atoms, **kwargs):
        build_calls.append(dict(kwargs))
        dataset = [
            _FakeGraph(int(structure.numbers[0]), len(structure))
            for structure in atoms
        ]
        return _FakeLoader(dataset, emission, reported_natoms)

    predictor = _IndexedMatterSimPredictor(
        potential=_FakePotential(device=device),
        build_dataloader=build_dataloader,
        batch_to_dict=lambda batch, *, device: {"graph_batch": batch},
        torch_module=_FakeTorch(),
        device=device,
        batch_size=8,
    )
    return predictor, build_calls


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _frame(symbol: str, separation: float) -> str:
    return f'''2
Lattice="8 0 0 0 8 0 0 0 8" Properties=species:S:1:pos:R:3:forces:R:3 endpoint_label=forbidden energy=-999
{symbol} 0.2 0.4 0.6 999 999 999
{symbol} {0.2 + separation} 0.4 0.6 -999 -999 -999
'''


def _runner_inputs(tmp_path: Path) -> dict[str, Path]:
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

    frames_path = tmp_path / "initial_frames.zip"
    with zipfile.ZipFile(frames_path, "w") as archive:
        archive.writestr("nested/sid-b.extxyz", _frame("H", 1.4))
        archive.writestr("sid-a.extxyz", _frame("He", 1.4))
        archive.writestr("sid-fit.extxyz", _frame("Li", 1.2))

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
                "path": str(frames_path.resolve()),
                "sha256": _sha256(frames_path),
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
    return {
        "features": features_path,
        "roles": roles_path,
        "frames": frames_path,
        "feature_manifest": manifest_path,
        "checkpoint": checkpoint_path,
    }


def _run_features(tmp_path: Path, *, predictor=None, output_name: str = "output"):
    from src.next10_lrrc_mattersim_features import run_label_free_features

    paths = _runner_inputs(tmp_path)
    if predictor is None:
        predictor = _AnalyticBatchPredictor()
    output_dir = tmp_path / output_name
    manifest = run_label_free_features(
        features_path=paths["features"],
        role_assignments_path=paths["roles"],
        frames_zip_path=paths["frames"],
        feature_manifest_path=paths["feature_manifest"],
        checkpoint_path=paths["checkpoint"],
        output_dir=output_dir,
        predictor=predictor,
        device="cpu",
        batch_size=8,
    )
    return manifest, output_dir, predictor, paths


def _reseal_upstream_manifest(paths: dict[str, Path]) -> None:
    manifest_path = paths["feature_manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["outputs_sha256"] = {
        paths["features"].name: _sha256(paths["features"])
    }
    manifest["inputs_sha256"]["frames"] = {
        "path": str(paths["frames"].resolve()),
        "sha256": _sha256(paths["frames"]),
    }
    manifest["checkpoints"]["m5"] = {
        "path": str(paths["checkpoint"].resolve()),
        "sha256": _sha256(paths["checkpoint"]),
    }
    manifest["predictor_loaded_checkpoint_sha256"]["m5"] = _sha256(
        paths["checkpoint"]
    )
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, allow_nan=False), encoding="utf-8"
    )


def _patch_frozen_fixture_identities(
    monkeypatch: pytest.MonkeyPatch,
    module,
    paths: dict[str, Path],
    *,
    patch_checkpoint: bool = True,
) -> None:
    if patch_checkpoint:
        monkeypatch.setattr(
            module, "FROZEN_M5_SHA256", _sha256(paths["checkpoint"])
        )
    monkeypatch.setattr(
        module,
        "FROZEN_NEXT8_INPUT_SHA256",
        {
            "committee_features": _sha256(paths["features"]),
            "threshold_roles": _sha256(paths["roles"]),
            "frames": _sha256(paths["frames"]),
            "feature_manifest": _sha256(paths["feature_manifest"]),
        },
    )


def test_batch_core_is_sid_sorted_fixed_order_and_scalar_replay_equivalent() -> None:
    from src.next10_lrrc_mattersim_features import evaluate_lrrc_batch

    structures = [_atoms(1), _atoms(2)]
    predictor = _AnalyticBatchPredictor()

    observed = evaluate_lrrc_batch(
        ["sid-z", "sid-a"],
        structures,
        predictor,
    )

    assert [item.sid for item in observed] == ["sid-a", "sid-z"]
    assert [item.force_call_count for item in observed] == [5, 5]
    assert len(predictor.calls) == 5
    assert [len(call) for call in predictor.calls] == [2, 2, 2, 2, 2]
    assert [int(atoms.numbers[0]) for atoms in predictor.calls[0]] == [2, 1]

    source_by_sid = {"sid-z": structures[0], "sid-a": structures[1]}
    for item_index, item in enumerate(observed):
        scalar = evaluate_lrrc(source_by_sid[item.sid], _analytic_forces)
        assert item.result.status is scalar.status is LRRCStatus.OK
        assert item.result.negative is scalar.negative
        assert item.result.d_star == pytest.approx(scalar.d_star, abs=1e-14)
        assert item.result.h == pytest.approx(scalar.h, abs=1e-14)
        assert item.result.kappa_h == pytest.approx(scalar.kappa_h, abs=1e-11)
        assert item.result.kappa_h2 == pytest.approx(scalar.kappa_h2, abs=1e-11)
        assert item.result.kappa_r == pytest.approx(scalar.kappa_r, abs=1e-11)
        assert item.result.error_proxy == pytest.approx(
            scalar.error_proxy, abs=1e-11
        )
        assert item.result.u_num == pytest.approx(scalar.u_num, abs=1e-11)

        base = predictor.calls[0][item_index]
        direction = translation_projected_direction(_analytic_forces(base))
        assert direction is not None
        h = item.result.h
        assert h is not None
        expected_positions = (
            base.positions + h * direction,
            base.positions - h * direction,
            base.positions + 0.5 * h * direction,
            base.positions - 0.5 * h * direction,
        )
        for call, expected in zip(predictor.calls[1:], expected_positions, strict=True):
            np.testing.assert_allclose(call[item_index].positions, expected, rtol=0, atol=0)


def test_stationary_rows_use_only_base_while_moving_rows_use_all_five() -> None:
    from src.next10_lrrc_mattersim_features import evaluate_lrrc_batch

    moving = _atoms(1)
    stationary = _atoms(3, separation=0.0)
    predictor = _AnalyticBatchPredictor()

    observed = evaluate_lrrc_batch(
        ["moving", "stationary"],
        [moving, stationary],
        predictor,
    )

    by_sid = {item.sid: item for item in observed}
    assert by_sid["moving"].result.status is LRRCStatus.OK
    assert by_sid["moving"].force_call_count == 5
    assert by_sid["stationary"].result.status is LRRCStatus.STATIONARY_FALLBACK
    assert by_sid["stationary"].force_call_count == 1
    assert [len(call) for call in predictor.calls] == [2, 1, 1, 1, 1]


def test_preforce_unsupported_geometries_remain_rows_without_prediction() -> None:
    from src.next10_lrrc_mattersim_features import evaluate_lrrc_batch

    one_atom = Atoms(
        numbers=[1],
        positions=[[0.2, 0.4, 0.6]],
        cell=[8.0, 8.0, 8.0],
        pbc=True,
    )
    nonfinite = _atoms(1)
    nonfinite.positions[0, 0] = np.nan
    predictor = _AnalyticBatchPredictor()

    observed = evaluate_lrrc_batch(
        ["one-atom", "nonfinite"],
        [one_atom, nonfinite],
        predictor,
    )

    assert [item.sid for item in observed] == ["nonfinite", "one-atom"]
    assert all(
        item.result.status is LRRCStatus.ABSTAIN_UNSUPPORTED_GEOMETRY
        for item in observed
    )
    assert [item.force_call_count for item in observed] == [0, 0]
    assert predictor.calls == []


def test_postbase_numerical_abstention_remains_a_row_without_probe_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src import next10_lrrc_mattersim_features as module
    from src import next9_lrrc

    underflow_distance = float(np.nextafter(0.0, 1.0))
    monkeypatch.setattr(
        module,
        "median_nearest_neighbor_distance",
        lambda _atoms: underflow_distance,
    )
    monkeypatch.setattr(
        next9_lrrc,
        "median_nearest_neighbor_distance",
        lambda _atoms: underflow_distance,
    )
    predictor = _AnalyticBatchPredictor()

    observed = module.evaluate_lrrc_batch(["numerical"], [_atoms(1)], predictor)

    assert len(observed) == 1
    assert observed[0].result.status is LRRCStatus.ABSTAIN_NUMERICAL_FAILURE
    assert observed[0].force_call_count == 1
    assert [len(call) for call in predictor.calls] == [1]


@pytest.mark.parametrize("failure", ["short", "shape", "nonfinite", "exception"])
def test_batch_predictor_contract_failures_abort_without_partial_results(
    failure: str,
) -> None:
    from src.next10_lrrc_mattersim_features import BatchLRRCError, evaluate_lrrc_batch

    structures = [_atoms(1), _atoms(2)]

    def invalid_predictor(batch: list[Atoms]) -> BatchPrediction:
        if failure == "exception":
            raise RuntimeError("predictor exploded")
        forces = [_analytic_forces(atoms) for atoms in batch]
        if failure == "short":
            forces = forces[:-1]
        elif failure == "shape":
            forces[0] = np.zeros((len(batch[0]), 2))
        elif failure == "nonfinite":
            forces[0] = np.full((len(batch[0]), 3), np.nan)
        return BatchPrediction(
            total_energies_ev=[0.0] * len(batch),
            forces_ev_per_a=forces,
            stresses_ev_per_a3=[np.zeros((3, 3)) for _ in batch],
        )

    with pytest.raises(BatchLRRCError):
        evaluate_lrrc_batch(["a", "b"], structures, invalid_predictor)


def test_indexed_adapter_restores_input_order_for_permuted_equal_natoms_batch() -> None:
    from ase.units import GPa

    predictor, build_calls = _fake_indexed_predictor([1, 0])

    prediction = predictor([_atoms(1), _atoms(2)])

    assert prediction.total_energies_ev == [10.0, 20.0]
    np.testing.assert_allclose(prediction.forces_ev_per_a[0], 1.0)
    np.testing.assert_allclose(prediction.forces_ev_per_a[1], 2.0)
    np.testing.assert_allclose(prediction.stresses_ev_per_a3[0], np.eye(3) * GPa)
    np.testing.assert_allclose(
        prediction.stresses_ev_per_a3[1], np.eye(3) * 2.0 * GPa
    )
    assert build_calls == [
        {
            "cutoff": 5.0,
            "threebody_cutoff": 4.0,
            "batch_size": 8,
            "only_inference": True,
            "shuffle": False,
        }
    ]
    assert predictor.telemetry == {
        "model_parameter_device": "cpu",
        "result_tensor_devices": ["cpu"],
        "forward_calls": 1,
        "evaluations": 2,
        "peak_cuda_memory_bytes": 0,
    }


@pytest.mark.parametrize(
    ("emission", "message"),
    [([1, 1], "duplicate input ordinal"), ([1], "missing input ordinals")],
)
def test_indexed_adapter_rejects_duplicate_or_missing_loader_indices(
    emission: list[int],
    message: str,
) -> None:
    from src.next10_lrrc_mattersim_features import BatchLRRCError

    predictor, _build_calls = _fake_indexed_predictor(emission)

    with pytest.raises(BatchLRRCError, match=message):
        predictor([_atoms(1), _atoms(2)])


@pytest.mark.parametrize(
    "reported_natoms",
    ([2.9, 2.1], [np.nan, 2.0], [np.inf, 2.0]),
)
def test_indexed_adapter_rejects_nonintegral_or_nonfinite_num_atoms(
    reported_natoms: list[float],
) -> None:
    from src.next10_lrrc_mattersim_features import BatchLRRCError

    predictor, _build_calls = _fake_indexed_predictor(
        [0, 1],
        reported_natoms=reported_natoms,
    )

    with pytest.raises(BatchLRRCError, match="num_atoms must be finite exact integers"):
        predictor([_atoms(1), _atoms(2)])


def test_checkpoint_loader_memfd_blocks_swap_load_restore_and_write(
    tmp_path: Path,
) -> None:
    import fcntl
    import os

    from src import next10_lrrc_mattersim_features as module

    checkpoint = tmp_path / "source-checkpoint.pth"
    checkpoint.write_bytes(b"frozen-checkpoint-bytes")
    expected_sha256 = _sha256(checkpoint)
    observed: dict[str, object] = {}
    get_seals = getattr(fcntl, "F_GET_SEALS", 1034)
    seal_write = getattr(fcntl, "F_SEAL_WRITE", 0x0008)
    seal_grow = getattr(fcntl, "F_SEAL_GROW", 0x0004)
    seal_shrink = getattr(fcntl, "F_SEAL_SHRINK", 0x0002)
    seal_seal = getattr(fcntl, "F_SEAL_SEAL", 0x0001)

    class PotentialClass:
        @classmethod
        def from_checkpoint(cls, path, *, device, load_training_state):
            sealed = Path(path)
            observed["path"] = sealed
            probe_fd = os.open(sealed, os.O_RDONLY)
            try:
                try:
                    observed["seals"] = fcntl.fcntl(probe_fd, get_seals)
                except OSError:
                    observed["seals"] = 0
            finally:
                os.close(probe_fd)
            replacement = tmp_path / "malicious-model.pth"
            replacement.write_bytes(b"malicious-model-bytes")
            try:
                os.replace(replacement, sealed)
            except OSError as exc:
                observed["replace_error"] = exc
            else:
                observed["replace_error"] = None
            observed["loaded_bytes"] = sealed.read_bytes()
            if observed["replace_error"] is None:
                sealed.write_bytes(b"frozen-checkpoint-bytes")
            try:
                sealed.write_bytes(b"in-place-malicious-model")
            except OSError as exc:
                observed["write_error"] = exc
            else:
                observed["write_error"] = None
                sealed.write_bytes(b"frozen-checkpoint-bytes")
            assert device == "cpu"
            assert load_training_state is False
            return _FakePotential(device="cpu")

    predictor, loaded_sha256 = module._load_indexed_predictor_from_checkpoint(
        checkpoint_path=checkpoint,
        expected_sha256=expected_sha256,
        potential_class=PotentialClass,
        build_dataloader=lambda *args, **kwargs: None,
        batch_to_dict=lambda *args, **kwargs: None,
        torch_module=_FakeTorch(),
        device="cpu",
        batch_size=8,
    )

    sealed_path = observed["path"]
    assert isinstance(predictor, module._IndexedMatterSimPredictor)
    assert loaded_sha256 == expected_sha256
    assert sealed_path != checkpoint
    assert str(sealed_path).startswith("/proc/self/fd/")
    required_seals = (
        seal_write
        | seal_grow
        | seal_shrink
        | seal_seal
    )
    assert observed["seals"] & required_seals == required_seals
    assert isinstance(observed["replace_error"], OSError)
    assert isinstance(observed["write_error"], OSError)
    assert observed["loaded_bytes"] == b"frozen-checkpoint-bytes"
    assert not sealed_path.exists()
    assert checkpoint.read_bytes() == b"frozen-checkpoint-bytes"


def test_checkpoint_loader_rejects_in_place_write_and_closes_memfd(
    tmp_path: Path,
) -> None:
    from src import next10_lrrc_mattersim_features as module

    checkpoint = tmp_path / "source-checkpoint.pth"
    checkpoint.write_bytes(b"frozen-checkpoint-bytes")
    sealed_paths: list[Path] = []

    class TamperingPotentialClass:
        @classmethod
        def from_checkpoint(cls, path, *, device, load_training_state):
            sealed = Path(path)
            sealed_paths.append(sealed)
            sealed.write_bytes(b"tampered-sealed-copy")
            return _FakePotential(device="cpu")

    with pytest.raises(module.BatchLRRCError, match="checkpoint load failed"):
        module._load_indexed_predictor_from_checkpoint(
            checkpoint_path=checkpoint,
            expected_sha256=_sha256(checkpoint),
            potential_class=TamperingPotentialClass,
            build_dataloader=lambda *args, **kwargs: None,
            batch_to_dict=lambda *args, **kwargs: None,
            torch_module=_FakeTorch(),
            device="cpu",
            batch_size=8,
        )

    assert len(sealed_paths) == 1
    assert not sealed_paths[0].exists()


@pytest.mark.parametrize(
    ("requested", "model_device", "result_devices"),
    [
        ("cuda:1", "cuda:0", ["cuda:0"]),
        ("cuda:0", "cuda:0", ["cuda:1"]),
    ],
)
def test_cuda_indexed_telemetry_requires_one_exact_canonical_device(
    requested: str,
    model_device: str,
    result_devices: list[str],
) -> None:
    from src import next10_lrrc_mattersim_features as module

    predictor = type(
        "TelemetryPredictor",
        (),
        {
            "telemetry": {
                "model_parameter_device": model_device,
                "result_tensor_devices": result_devices,
                "forward_calls": 1,
                "evaluations": 2,
                "peak_cuda_memory_bytes": 1024,
            }
        },
    )()

    with pytest.raises(RuntimeError, match="production CUDA telemetry"):
        module._validated_builtin_telemetry(
            predictor,
            device=requested,
            expected_evaluations=2,
        )


@pytest.mark.parametrize("mismatch", ["positions", "cell", "pbc", "numbers"])
def test_replay_oracle_rejects_probe_geometry_mismatch(mismatch: str) -> None:
    from src.next10_lrrc_mattersim_features import BatchLRRCError, _ReplayForceOracle

    expected = _atoms(1)
    observed = expected.copy()
    if mismatch == "positions":
        observed.positions[0, 0] += 1e-6
    elif mismatch == "cell":
        observed.cell[0, 0] += 1e-6
    elif mismatch == "pbc":
        observed.pbc[0] = False
    else:
        observed.numbers[0] = 2
    replay = _ReplayForceOracle(
        expected_probes=(expected,),
        force_values=(np.zeros((len(expected), 3)),),
    )

    with pytest.raises(BatchLRRCError, match=mismatch):
        replay(observed)


def test_batch_api_is_label_free_and_module_has_no_evaluation_imports() -> None:
    import ast

    from src import next10_lrrc_mattersim_features as module

    public_parameters = inspect.signature(module.run_label_free_features).parameters
    assert not any("label" in name.lower() for name in public_parameters)
    source = Path(module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    } | {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    top_level_imported = {
        alias.name
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    } | {
        node.module
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert not any("gate_diagnostic" in name or "protocol" in name for name in imported)
    assert not any(name.startswith("mattersim") for name in top_level_imported)


def test_label_free_runner_selects_gate_and_seals_strict_artifacts(tmp_path: Path) -> None:
    from src import next10_lrrc_mattersim_features as module

    manifest, output_dir, predictor, paths = _run_features(tmp_path)
    table_path = output_dir / "lrrc_features.parquet"
    manifest_path = output_dir / "MANIFEST.json"

    assert {path.name for path in output_dir.iterdir()} == {
        table_path.name,
        manifest_path.name,
    }
    table = pd.read_parquet(table_path)
    assert list(table.columns) == OUTPUT_COLUMNS
    assert table["sid"].tolist() == ["sid-a", "sid-b", "sid-ns"]
    assert table["threshold_role"].eq("development_gate").all()
    assert table["stage"].eq("threshold_calibration").all()
    assert table["force_call_count"].tolist() == [5, 5, 0]
    assert table.loc[0, "lrrc_status"] == "ok"
    assert table.loc[1, "lrrc_status"] == "ok"
    assert table.loc[2, "lrrc_status"] == "abstain_unsupported_geometry"
    assert table.loc[2, "error"] == "nonstrict_x0"
    assert [len(call) for call in predictor.calls] == [2, 2, 2, 2, 2]
    successful = table["lrrc_status"].eq("ok")
    numeric = [
        "d_star_angstrom",
        "h_angstrom",
        "kappa_h_ev_per_a2",
        "kappa_h2_ev_per_a2",
        "kappa_r_ev_per_a2",
        "error_proxy_ev_per_a2",
        "u_num_ev_per_a2",
    ]
    assert np.isfinite(table.loc[successful, numeric].to_numpy(dtype=float)).all()

    raw_manifest = manifest_path.read_text(encoding="utf-8")
    loaded = json.loads(
        raw_manifest,
        parse_constant=lambda value: (_ for _ in ()).throw(AssertionError(value)),
    )
    assert loaded == manifest
    assert manifest["protocol"] == "2026-08-01-next10-lrrc-mattersim-features-v1"
    assert manifest["mode"] == "development_gate"
    assert manifest["labels_opened"] is False
    assert manifest["selection"] == {
        "stage": "threshold_calibration",
        "threshold_role": "development_gate",
    }
    assert manifest["adapter"]["mode"] == "injected_test_double"
    assert manifest["adapter"]["index_alignment"] == (
        "injected_batch_force_predictor_declared_aligned"
    )
    assert manifest["adapter"]["index_alignment_verified"] is False
    assert manifest["production_protocol_eligible"] is False
    assert manifest["evidence_role"] == "testing_only_not_scientific_evidence"
    assert manifest["scientific_improvement_claim"] is False
    assert manifest["feature_columns"] == OUTPUT_COLUMNS
    assert "output_columns" not in manifest
    assert {
        "torch_version",
        "cuda_available",
        "cuda_version",
        "gpu_name",
    }.issubset(manifest["runtime"])
    assert manifest["counts"] == {
        "feature_rows": 4,
        "role_assignment_rows": 4,
        "selected_rows": 3,
        "strict_rows": 2,
        "nonstrict_rows": 1,
        "ok_rows": 2,
        "stationary_rows": 0,
        "abstained_rows": 1,
        "batch_predictor_calls": 5,
        "force_evaluations": 10,
    }
    expected_input_paths = {
        "committee_features": paths["features"],
        "threshold_roles": paths["roles"],
        "frames": paths["frames"],
        "feature_manifest": paths["feature_manifest"],
        "checkpoint": paths["checkpoint"],
    }
    assert manifest["inputs_sha256"] == {
        role: {"path": str(path.resolve()), "sha256": _sha256(path)}
        for role, path in expected_input_paths.items()
    }
    assert manifest["outputs_sha256"] == {table_path.name: _sha256(table_path)}
    repository_root = Path(module.__file__).resolve().parents[1]
    required_sources = {
        "src/next10_lrrc_mattersim_features.py",
        "src/next9_lrrc.py",
        "src/next8_mattersim_committee_features.py",
        "src/next6_mattersim_baseline.py",
    }
    assert set(manifest["executed_source_sha256"]) == required_sources
    for relative, digest in manifest["executed_source_sha256"].items():
        assert digest == _sha256(repository_root / relative)
    assert manifest["integrity"] == {"prepublish_rehash": "passed"}
    assert not list(tmp_path.glob(".output.staging-*"))


def test_published_parquet_uses_exact_error_strings_and_zero_natoms_for_nonstrict(
    tmp_path: Path,
) -> None:
    def stationary_for_hydrogen(atoms: Atoms) -> np.ndarray:
        if int(atoms.numbers[0]) == 1:
            return np.zeros((len(atoms), 3))
        return _analytic_forces(atoms)

    manifest, output_dir, _predictor, _paths = _run_features(
        tmp_path,
        predictor=_AnalyticBatchPredictor(stationary_for_hydrogen),
    )
    table = pd.read_parquet(output_dir / "lrrc_features.parquet")
    by_sid = table.set_index("sid")

    assert by_sid.loc["sid-a", "lrrc_status"] == "ok"
    assert by_sid.loc["sid-a", "error"] == ""
    assert by_sid.loc["sid-b", "lrrc_status"] == "stationary_fallback"
    assert by_sid.loc["sid-b", "error"] == ""
    assert by_sid.loc["sid-ns", "natoms"] == 0
    assert by_sid.loc["sid-ns", "error"] == "nonstrict_x0"
    assert manifest["counts"]["stationary_rows"] == 1


def test_production_manifest_uses_frozen_evaluator_contract_with_factory_stub(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src import next10_lrrc_mattersim_features as module

    paths = _runner_inputs(tmp_path)
    predictor = _TelemetryBatchPredictor(
        device="cpu",
        peak_cuda_memory_bytes=0,
    )
    _patch_frozen_fixture_identities(monkeypatch, module, paths)
    monkeypatch.setattr(
        module,
        "_runtime_identity",
        lambda device: _runtime_record(device=device, cuda_available=False),
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
        feature_manifest_path=paths["feature_manifest"],
        checkpoint_path=paths["checkpoint"],
        output_dir=tmp_path / "production-contract",
        predictor=None,
        device="cpu",
        batch_size=8,
    )

    assert manifest["protocol"] == module.PROTOCOL
    assert manifest["mode"] == "development_gate"
    assert manifest["evidence_role"] == "label_free_lrrc_feature_generation"
    assert manifest["production_protocol_eligible"] is True
    assert manifest["labels_opened"] is False
    assert manifest["scientific_improvement_claim"] is False
    assert manifest["feature_columns"] == OUTPUT_COLUMNS
    assert set(manifest["outputs_sha256"]) == {"lrrc_features.parquet"}
    assert manifest["predictor_loaded_checkpoint_sha256"] == _sha256(
        paths["checkpoint"]
    )
    assert manifest["adapter"] == {
        "mode": "builtin_indexed_mattersim",
        "index_alignment": "sid_indexed_exact_one_to_one",
        "index_alignment_verified": True,
        "device": "cpu",
        "batch_size": 8,
        "model_parameter_device": "cpu",
        "result_tensor_devices": ["cpu"],
        "evaluations": 10,
    }
    assert manifest["execution"].keys() == {
        "batch_predictor_calls",
        "forward_calls",
        "peak_cuda_memory_bytes",
        "wall_time_seconds",
    }
    assert manifest["execution"]["batch_predictor_calls"] == 5
    assert manifest["execution"]["forward_calls"] == 5
    assert manifest["execution"]["peak_cuda_memory_bytes"] == 0


def test_production_rejects_self_consistent_but_unfrozen_next8_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src import next10_lrrc_mattersim_features as module

    paths = _runner_inputs(tmp_path)
    monkeypatch.setattr(module, "FROZEN_M5_SHA256", _sha256(paths["checkpoint"]))

    def forbidden_factory(*args, **kwargs):
        raise AssertionError("production factory must not see unfrozen inputs")

    monkeypatch.setattr(module, "_production_predictor", forbidden_factory)
    with pytest.raises(ValueError, match="frozen next8 sealed input"):
        module.run_label_free_features(
            features_path=paths["features"],
            role_assignments_path=paths["roles"],
            frames_zip_path=paths["frames"],
            feature_manifest_path=paths["feature_manifest"],
            checkpoint_path=paths["checkpoint"],
            output_dir=tmp_path / "unfrozen-next8-output",
            predictor=None,
            device="cpu",
            batch_size=8,
        )


def test_production_cuda_request_requires_live_cuda_before_model_factory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src import next10_lrrc_mattersim_features as module

    paths = _runner_inputs(tmp_path)
    _patch_frozen_fixture_identities(monkeypatch, module, paths)
    monkeypatch.setattr(
        module,
        "_runtime_identity",
        lambda device: _runtime_record(device=device, cuda_available=False),
    )

    def forbidden_factory(*args, **kwargs):
        raise AssertionError("CUDA-unavailable host must not load MatterSim")

    monkeypatch.setattr(module, "_production_predictor", forbidden_factory)
    with pytest.raises(RuntimeError, match="CUDA.*unavailable"):
        module.run_label_free_features(
            features_path=paths["features"],
            role_assignments_path=paths["roles"],
            frames_zip_path=paths["frames"],
            feature_manifest_path=paths["feature_manifest"],
            checkpoint_path=paths["checkpoint"],
            output_dir=tmp_path / "cuda-unavailable-output",
            predictor=None,
            device="cuda",
            batch_size=8,
        )


def test_production_cuda_rejects_cpu_model_results_and_zero_peak_after_inference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src import next10_lrrc_mattersim_features as module

    paths = _runner_inputs(tmp_path)
    _patch_frozen_fixture_identities(monkeypatch, module, paths)
    monkeypatch.setattr(
        module,
        "_runtime_identity",
        lambda device: _runtime_record(device=device, cuda_available=True),
    )
    predictor = _TelemetryBatchPredictor(
        device="cpu",
        peak_cuda_memory_bytes=0,
    )
    monkeypatch.setattr(
        module,
        "_production_predictor",
        lambda checkpoint_path, *, device, batch_size: (
            predictor,
            _sha256(checkpoint_path),
        ),
    )
    output_dir = tmp_path / "bad-cuda-telemetry"

    with pytest.raises(RuntimeError, match="production CUDA telemetry"):
        module.run_label_free_features(
            features_path=paths["features"],
            role_assignments_path=paths["roles"],
            frames_zip_path=paths["frames"],
            feature_manifest_path=paths["feature_manifest"],
            checkpoint_path=paths["checkpoint"],
            output_dir=output_dir,
            predictor=None,
            device="cuda",
            batch_size=8,
        )

    assert predictor.forward_calls == 5
    assert not output_dir.exists()


@pytest.mark.parametrize(
    ("invalid_case", "message"),
    [
        ("missing_frame", "lacks x0 frame"),
        ("duplicate_frame_stem", "duplicate member stem"),
        ("invalid_strict_frame", "invalid strict x0 frame"),
        ("missing_selected_feature", "sid coverage differs"),
        ("selected_rk_mismatch", "rk values differ"),
    ],
)
def test_invalid_selected_inputs_abort_before_prediction_or_publication(
    tmp_path: Path,
    invalid_case: str,
    message: str,
) -> None:
    from src.next10_lrrc_mattersim_features import run_label_free_features

    paths = _runner_inputs(tmp_path)
    if invalid_case == "missing_frame":
        with zipfile.ZipFile(paths["frames"], "w") as archive:
            archive.writestr("sid-b.extxyz", _frame("H", 1.4))
            archive.writestr("sid-fit.extxyz", _frame("Li", 1.2))
        _reseal_upstream_manifest(paths)
    elif invalid_case == "duplicate_frame_stem":
        with zipfile.ZipFile(paths["frames"], "w") as archive:
            archive.writestr("first/sid-a.extxyz", _frame("He", 1.4))
            archive.writestr("second/sid-a.xyz", _frame("He", 1.4))
            archive.writestr("sid-b.extxyz", _frame("H", 1.4))
        _reseal_upstream_manifest(paths)
    elif invalid_case == "invalid_strict_frame":
        with zipfile.ZipFile(paths["frames"], "w") as archive:
            archive.writestr("sid-a.extxyz", "not an extxyz frame")
            archive.writestr("sid-b.extxyz", _frame("H", 1.4))
        _reseal_upstream_manifest(paths)
    elif invalid_case == "missing_selected_feature":
        features = pd.read_parquet(paths["features"])
        features = features.loc[~features["sid"].eq("sid-a")]
        features.to_parquet(paths["features"], index=False)
        _reseal_upstream_manifest(paths)
    else:
        roles = pd.read_parquet(paths["roles"])
        roles.loc[roles["sid"].eq("sid-a"), "rk"] = "wrong-rk"
        roles.to_parquet(paths["roles"], index=False)
    predictor = _AnalyticBatchPredictor()
    output_dir = tmp_path / "invalid-output"

    with pytest.raises(ValueError, match=message):
        run_label_free_features(
            features_path=paths["features"],
            role_assignments_path=paths["roles"],
            frames_zip_path=paths["frames"],
            feature_manifest_path=paths["feature_manifest"],
            checkpoint_path=paths["checkpoint"],
            output_dir=output_dir,
            predictor=predictor,
            device="cpu",
            batch_size=8,
        )

    assert predictor.calls == []
    assert not output_dir.exists()


@pytest.mark.parametrize(
    ("closure", "message"),
    [
        ("features", "feature hash closure mismatch"),
        ("frames", "frames hash/path mismatch"),
        ("checkpoint", "checkpoint hash/path mismatch"),
        ("loaded_checkpoint", "loaded checkpoint hash mismatch"),
    ],
)
def test_upstream_declared_hash_mismatch_aborts_before_prediction(
    tmp_path: Path,
    closure: str,
    message: str,
) -> None:
    from src.next10_lrrc_mattersim_features import run_label_free_features

    paths = _runner_inputs(tmp_path)
    manifest_path = paths["feature_manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if closure == "features":
        manifest["outputs_sha256"][paths["features"].name] = "0" * 64
    elif closure == "frames":
        manifest["inputs_sha256"]["frames"]["sha256"] = "0" * 64
    elif closure == "checkpoint":
        manifest["checkpoints"]["m5"]["sha256"] = "0" * 64
    else:
        manifest["predictor_loaded_checkpoint_sha256"]["m5"] = "0" * 64
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, allow_nan=False), encoding="utf-8"
    )
    predictor = _AnalyticBatchPredictor()
    output_dir = tmp_path / "bad-closure-output"

    with pytest.raises(ValueError, match=message):
        run_label_free_features(
            features_path=paths["features"],
            role_assignments_path=paths["roles"],
            frames_zip_path=paths["frames"],
            feature_manifest_path=paths["feature_manifest"],
            checkpoint_path=paths["checkpoint"],
            output_dir=output_dir,
            predictor=predictor,
            device="cpu",
            batch_size=8,
        )

    assert predictor.calls == []
    assert not output_dir.exists()


def test_runner_predictor_failure_aborts_without_partial_artifact(
    tmp_path: Path,
) -> None:
    from src.next10_lrrc_mattersim_features import (
        BatchLRRCError,
        run_label_free_features,
    )

    paths = _runner_inputs(tmp_path)
    output_dir = tmp_path / "predictor-failure-output"

    def failing_predictor(_batch: list[Atoms]) -> BatchPrediction:
        raise RuntimeError("injected failure")

    with pytest.raises(BatchLRRCError, match="batch predictor failed at base"):
        run_label_free_features(
            features_path=paths["features"],
            role_assignments_path=paths["roles"],
            frames_zip_path=paths["frames"],
            feature_manifest_path=paths["feature_manifest"],
            checkpoint_path=paths["checkpoint"],
            output_dir=output_dir,
            predictor=failing_predictor,
            device="cpu",
            batch_size=8,
        )

    assert not output_dir.exists()
    assert not list(tmp_path.glob(".predictor-failure-output.staging-*"))


@pytest.mark.parametrize("mutated_role", ["features", "roles", "frames", "feature_manifest", "checkpoint"])
def test_any_input_drift_after_inference_aborts_and_cleans_staging(
    tmp_path: Path,
    mutated_role: str,
) -> None:
    from src.next10_lrrc_mattersim_features import run_label_free_features

    paths = _runner_inputs(tmp_path)
    delegate = _AnalyticBatchPredictor()
    mutated = False

    def mutating_predictor(batch: list[Atoms]) -> BatchPrediction:
        nonlocal mutated
        prediction = delegate(batch)
        if not mutated:
            mutated = True
            with paths[mutated_role].open("ab") as stream:
                stream.write(b"drift")
        return prediction

    output_dir = tmp_path / "drift-output"
    with pytest.raises(RuntimeError, match=f"input {mutated_role} changed"):
        run_label_free_features(
            features_path=paths["features"],
            role_assignments_path=paths["roles"],
            frames_zip_path=paths["frames"],
            feature_manifest_path=paths["feature_manifest"],
            checkpoint_path=paths["checkpoint"],
            output_dir=output_dir,
            predictor=mutating_predictor,
            device="cpu",
            batch_size=8,
        )

    assert not output_dir.exists()
    assert not list(tmp_path.glob(".drift-output.staging-*"))


def test_source_drift_before_publication_aborts_and_cleans_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src import next10_lrrc_mattersim_features as module

    paths = _runner_inputs(tmp_path)
    target = Path(module.__file__).resolve().parents[1] / "src/next9_lrrc.py"
    original_hash = module._sha256_file
    target_calls = 0

    def simulated_hash(path: Path) -> str:
        nonlocal target_calls
        digest = original_hash(path)
        if Path(path).resolve() == target.resolve():
            target_calls += 1
            if target_calls > 1:
                return "0" * 64
        return digest

    monkeypatch.setattr(module, "_sha256_file", simulated_hash)
    output_dir = tmp_path / "source-drift-output"
    with pytest.raises(RuntimeError, match="executed source src/next9_lrrc.py changed"):
        module.run_label_free_features(
            features_path=paths["features"],
            role_assignments_path=paths["roles"],
            frames_zip_path=paths["frames"],
            feature_manifest_path=paths["feature_manifest"],
            checkpoint_path=paths["checkpoint"],
            output_dir=output_dir,
            predictor=_AnalyticBatchPredictor(),
            device="cpu",
            batch_size=8,
        )

    assert not output_dir.exists()
    assert not list(tmp_path.glob(".source-drift-output.staging-*"))


def test_existing_output_is_untouched_without_prediction(tmp_path: Path) -> None:
    from src.next10_lrrc_mattersim_features import run_label_free_features

    paths = _runner_inputs(tmp_path)
    output_dir = tmp_path / "existing-output"
    output_dir.mkdir()
    sentinel = output_dir / "sentinel.txt"
    sentinel.write_text("preserve", encoding="utf-8")
    predictor = _AnalyticBatchPredictor()

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        run_label_free_features(
            features_path=paths["features"],
            role_assignments_path=paths["roles"],
            frames_zip_path=paths["frames"],
            feature_manifest_path=paths["feature_manifest"],
            checkpoint_path=paths["checkpoint"],
            output_dir=output_dir,
            predictor=predictor,
            device="cpu",
            batch_size=8,
        )

    assert sentinel.read_text(encoding="utf-8") == "preserve"
    assert predictor.calls == []


def test_atomic_publish_race_preserves_competing_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src import next10_lrrc_mattersim_features as module

    paths = _runner_inputs(tmp_path)
    output_dir = tmp_path / "race-output"

    def competing_publish(staging: Path, target: Path) -> None:
        assert staging.is_dir()
        target.mkdir()
        (target / "sentinel.txt").write_text("competitor", encoding="utf-8")
        raise FileExistsError(target)

    monkeypatch.setattr(
        module, "_atomic_publish_directory_no_replace", competing_publish
    )
    with pytest.raises(FileExistsError):
        module.run_label_free_features(
            features_path=paths["features"],
            role_assignments_path=paths["roles"],
            frames_zip_path=paths["frames"],
            feature_manifest_path=paths["feature_manifest"],
            checkpoint_path=paths["checkpoint"],
            output_dir=output_dir,
            predictor=_AnalyticBatchPredictor(),
            device="cpu",
            batch_size=8,
        )

    assert (output_dir / "sentinel.txt").read_text(encoding="utf-8") == "competitor"
    assert not list(tmp_path.glob(".race-output.staging-*"))


def test_unfrozen_production_checkpoint_is_rejected_before_factory_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src import next10_lrrc_mattersim_features as module

    paths = _runner_inputs(tmp_path)
    _patch_frozen_fixture_identities(
        monkeypatch,
        module,
        paths,
        patch_checkpoint=False,
    )

    def forbidden_factory(*args, **kwargs):
        raise AssertionError("MatterSim factory must not be reached")

    monkeypatch.setattr(module, "_production_predictor", forbidden_factory)
    with pytest.raises(ValueError, match="does not equal frozen MatterSim 5M"):
        module.run_label_free_features(
            features_path=paths["features"],
            role_assignments_path=paths["roles"],
            frames_zip_path=paths["frames"],
            feature_manifest_path=paths["feature_manifest"],
            checkpoint_path=paths["checkpoint"],
            output_dir=tmp_path / "unfrozen-output",
            predictor=None,
            device="cpu",
            batch_size=8,
        )
