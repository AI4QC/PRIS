import inspect
import hashlib
import json
from pathlib import Path
import sys
import types
import zipfile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src import next7_mattersim_features as features
from src import next7_mattersim_prerelax as prerelax_module
from src.next7_mattersim_prerelax import BatchPrediction, FIRE_PARAMETERS


def _labelled_frame(symbol, displacement, ionic_step):
    return f'''2
Lattice="12 0 0 0 12 0 0 0 12" Properties=species:S:1:pos:R:3:forces:R:3 material_id=raw ionic_step={ionic_step} energy=-999 stress="1 2 3 4 5 6" e_per_atom=-499.5 exact_min=T valuable=T
{symbol} {displacement} 0 0 999 999 999
{symbol} {-displacement} 0 0 -999 -999 -999
'''


def _tiny_inputs(tmp_path):
    elementa = tmp_path / "elementa"
    p9_dir = tmp_path / "p9"
    elementa.mkdir()
    p9_dir.mkdir()
    frames_path = elementa / "elementa_initial_frames.zip"
    with zipfile.ZipFile(frames_path, "w") as archive:
        archive.writestr("nested/sid-z.extxyz", _labelled_frame("He", 1.2, 0))
        archive.writestr("sid-a.extxyz", _labelled_frame("H", 1.0, 0))
        archive.writestr("sid-m.extxyz", _labelled_frame("Li", 1.1, 3))
        archive.writestr("sid-test.extxyz", _labelled_frame("Be", 1.3, 0))

    pd.DataFrame(
        {
            "sid": ["sid-z", "sid-a", "sid-test", "sid-m"],
            "rk": ["He2", "H2", "Be2", "Li2"],
            "geom_min_pair_ratio": [1.0, 1.0, 1.0, 1.0],
            "e_per_atom": [-9.0, -10.0, -8.0, -7.0],
        }
    ).to_parquet(elementa / "elementa_x0_features.parquet", index=False)

    metadata_path = p9_dir / "elementa_x0_p9_features.parquet"
    pd.DataFrame(
        {
            "sid": ["sid-m", "sid-test", "sid-z", "sid-a"],
            "rk": ["Li2", "Be2", "He2", "H2"],
            "material": ["mat-m", "mat-test", "mat-z", "mat-a"],
            "strict_x0_ok": [False, True, True, True],
            "initial_ionic_step": [3, 0, 0, 0],
            "e_per_atom": [-7.0, -8.0, -9.0, -10.0],
        }
    ).to_parquet(metadata_path, index=False)

    stages_path = tmp_path / "stage_assignments.parquet"
    pd.DataFrame(
        {
            "sid": ["sid-a", "sid-m", "sid-z", "sid-test"],
            "rk": ["H2", "Li2", "He2", "Be2"],
            "stage": [
                "formula_selection",
                "search_calibration",
                "search_calibration",
                "test",
            ],
            "delta_e": [0.0, 0.1, 0.2, 0.3],
        }
    ).to_parquet(stages_path, index=False)
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"frozen-fake-model")
    return elementa, p9_dir, stages_path, checkpoint


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _valid_freeze_payload(elementa, p9_dir, stages_path, checkpoint):
    return {
        "protocol": "2026-08-01-mattersim-fewstep-development-freeze-v1",
        "state": "frozen",
        "frozen_at_utc": "2026-08-01T12:00:00+00:00",
        "checkpoint_sha256": _digest(checkpoint),
        "feature_inputs_sha256": {
            "elementa_initial_frames.zip": _digest(
                elementa / "elementa_initial_frames.zip"
            ),
            "elementa_x0_p9_features.parquet": _digest(
                p9_dir / "elementa_x0_p9_features.parquet"
            ),
            "elementa_x0_features.parquet": _digest(
                elementa / "elementa_x0_features.parquet"
            ),
            "stage_assignments.parquet": _digest(stages_path),
        },
        "code_sha256": {
            "next7_mattersim_prerelax.py": _digest(prerelax_module.__file__),
            "next7_mattersim_features.py": _digest(features.__file__),
        },
        "task4_reserved": {"rules": "not validated by feature extraction"},
    }


def _write_valid_freeze(path, elementa, p9_dir, stages_path, checkpoint):
    payload = _valid_freeze_payload(elementa, p9_dir, stages_path, checkpoint)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n")
    return payload


def _replace_zip_member(archive_path, member_name, transform):
    with zipfile.ZipFile(archive_path) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    members[member_name] = transform(members[member_name].decode()).encode()
    with zipfile.ZipFile(archive_path, "w") as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)


class _HarmonicFewstepPredictor:
    def __init__(self):
        self.calls = []
        self.calls_by_atomic_number = {}

    def __call__(self, structures):
        assert len(structures) == 1
        energies = []
        forces = []
        stresses = []
        for atoms in structures:
            assert atoms.info == {}
            assert set(atoms.arrays) == {"numbers", "positions"}
            assert atoms.calc is None
            atomic_number = int(atoms.numbers[0])
            self.calls.append(atomic_number)
            call_number = self.calls_by_atomic_number.get(atomic_number, 0) + 1
            self.calls_by_atomic_number[atomic_number] = call_number
            positions = np.asarray(atoms.positions, dtype=float)
            item_forces = -positions
            if atomic_number == 2 and call_number == 9:
                item_forces = item_forces.copy()
                item_forces[0] = [21.0, 0.0, 0.0]
            energies.append(0.5 * float(np.sum(positions**2)))
            forces.append(item_forces)
            stresses.append(np.zeros(6))
        return BatchPrediction(energies, forces, stresses)


def test_run_fewstep_features_api_has_no_label_input():
    parameters = inspect.signature(features.run_fewstep_features).parameters

    assert tuple(parameters) == (
        "elementa_dir",
        "p9_dir",
        "stage_assignments_path",
        "output_dir",
        "checkpoint",
        "stages",
        "device",
        "atom_budget",
        "structure_cap",
        "inference_batch_size",
        "structure_chunk_size",
        "predictor",
        "frozen_protocol_path",
    )
    assert not any("label" in name for name in parameters)


@pytest.mark.parametrize(
    "stages",
    [(), ("unknown",), "search_calibration", ("search_calibration",) * 2],
)
def test_stages_must_be_a_nonempty_unique_explicit_allowlist(tmp_path, stages):
    with pytest.raises(ValueError, match="stages"):
        features.run_fewstep_features(
            tmp_path,
            tmp_path,
            tmp_path / "stages.parquet",
            tmp_path / "out",
            checkpoint=tmp_path / "model.pth",
            stages=stages,
        )


def test_test_stage_cannot_run_with_development_stages(tmp_path):
    with pytest.raises(ValueError, match="test.*development"):
        features.run_fewstep_features(
            tmp_path,
            tmp_path,
            tmp_path / "stages.parquet",
            tmp_path / "out",
            checkpoint=tmp_path / "model.pth",
            stages=("formula_selection", "test"),
            frozen_protocol_path=tmp_path / "frozen.json",
        )


@pytest.mark.parametrize("frozen", [None, "missing-frozen.json"])
def test_test_stage_requires_an_existing_frozen_protocol(tmp_path, frozen):
    frozen_path = None if frozen is None else tmp_path / frozen
    with pytest.raises(FileNotFoundError, match="frozen protocol"):
        features.run_fewstep_features(
            tmp_path,
            tmp_path,
            tmp_path / "stages.parquet",
            tmp_path / "out",
            checkpoint=tmp_path / "model.pth",
            stages=("test",),
            frozen_protocol_path=frozen_path,
        )


def test_run_fewstep_features_is_label_free_keyed_chunked_and_prefix_aware(
    tmp_path,
):
    elementa, p9_dir, stages_path, checkpoint = _tiny_inputs(tmp_path)
    output = tmp_path / "result"
    predictor = _HarmonicFewstepPredictor()

    manifest = features.run_fewstep_features(
        elementa,
        p9_dir,
        stages_path,
        output,
        checkpoint=checkpoint,
        stages=("search_calibration", "formula_selection"),
        device="cpu",
        atom_budget=4,
        structure_cap=2,
        inference_batch_size=7,
        structure_chunk_size=1,
        predictor=predictor,
    )

    table_path = output / "mattersim_fewstep_features.parquet"
    table = pd.read_parquet(table_path).set_index("sid", drop=False)
    assert table.sid.tolist() == ["sid-a", "sid-m", "sid-z"]
    assert table.loc["sid-a", "material"] == "mat-a"
    assert table.loc["sid-z", "rk"] == "He2"
    assert table.loc["sid-m", "stage"] == "search_calibration"
    assert predictor.calls_by_atomic_number == {1: 9, 2: 9}
    assert 3 not in predictor.calls and 4 not in predictor.calls

    for sid in ("sid-a", "sid-z"):
        assert bool(table.loc[sid, "fewstep_feature_ok"])
        assert table.loc[sid, "fewstep_feature_error"] == ""
        assert table.loc[sid, "force_evaluations"] == 9
        assert table.loc[sid, "optimizer_updates"] == 8
        assert table.loc[sid, "retry_overhead_force_evaluations"] == 0
        assert table.loc[sid, "retry_overhead_optimizer_updates"] == 0
        assert table.loc[sid, "allocated_seconds"] >= 0.0
        for step in (0, 2, 4, 8):
            assert np.isfinite(table.loc[sid, f"k{step}_energy_total_ev"])
            assert np.isfinite(table.loc[sid, f"k{step}_energy_ev_per_atom"])
            assert np.isfinite(
                table.loc[
                    sid,
                    f"k{step}_energy_change_from_previous_snapshot_ev_per_atom",
                ]
            )
            assert np.isfinite(table.loc[sid, f"k{step}_fmax_ev_per_a"])
            assert np.isfinite(
                table.loc[sid, f"k{step}_stress_frobenius_ev_per_a3"]
            )
        assert (
            table.loc[
                sid, "k0_energy_change_from_previous_snapshot_ev_per_atom"
            ]
            == 0.0
        )
        for previous_step, current_step in ((0, 2), (2, 4), (4, 8)):
            assert table.loc[
                sid,
                f"k{current_step}_energy_change_from_previous_snapshot_ev_per_atom",
            ] == pytest.approx(
                table.loc[sid, f"k{current_step}_energy_ev_per_atom"]
                - table.loc[sid, f"k{previous_step}_energy_ev_per_atom"]
            )

    assert [
        bool(table.loc["sid-z", f"k{step}_supported"])
        for step in (0, 2, 4, 8)
    ] == [True, True, True, False]
    assert table.loc["sid-z", "k8_support_reason"] == "force_limit_exceeded"
    assert all(
        bool(table.loc["sid-a", f"k{step}_supported"])
        for step in (0, 2, 4, 8)
    )

    nonstrict = table.loc["sid-m"]
    assert not bool(nonstrict["fewstep_feature_ok"])
    assert nonstrict["fewstep_feature_error"] == "nonzero_initial_ionic_step"
    assert nonstrict["force_evaluations"] == 0
    assert nonstrict["optimizer_updates"] == 0
    assert nonstrict["retry_overhead_force_evaluations"] == 0
    assert nonstrict["retry_overhead_optimizer_updates"] == 0
    assert nonstrict["allocated_seconds"] == 0.0
    for step in (0, 2, 4, 8):
        assert np.isnan(nonstrict[f"k{step}_energy_total_ev"])
        assert np.isnan(
            nonstrict[
                f"k{step}_energy_change_from_previous_snapshot_ev_per_atom"
            ]
        )
        assert not bool(nonstrict[f"k{step}_supported"])
        assert (
            nonstrict[f"k{step}_support_reason"]
            == "nonzero_initial_ionic_step"
        )

    forbidden = (
        "e_per_atom",
        "delta_e",
        "exact_min",
        "near_min",
        "valuable",
        "high_energy",
    )
    assert not any(token in column for token in forbidden for column in table.columns)
    assert set(table["evidence_role"]) == {
        "historically seen discovery; not confirmatory"
    }
    assert manifest["execution"]["predictor_forward_calls"] == 18
    assert manifest["execution"]["total_elapsed_seconds"] == pytest.approx(
        table["allocated_seconds"].sum()
    )


def test_manifest_records_frozen_model_runtime_counts_and_all_hashes(tmp_path):
    elementa, p9_dir, stages_path, checkpoint = _tiny_inputs(tmp_path)
    output = tmp_path / "result"
    predictor = _HarmonicFewstepPredictor()

    manifest = features.run_fewstep_features(
        elementa,
        p9_dir,
        stages_path,
        output,
        checkpoint=checkpoint,
        stages=("search_calibration", "formula_selection"),
        device="cpu",
        atom_budget=4,
        structure_cap=2,
        inference_batch_size=7,
        structure_chunk_size=1,
        predictor=predictor,
    )

    loaded = json.loads((output / "MANIFEST.json").read_text())
    assert loaded == manifest
    assert manifest["protocol"] == "2026-08-01-mattersim-fewstep-prerelax-v1"
    assert manifest["evidence_role"] == (
        "historically seen discovery; not confirmatory"
    )
    assert "label-free" in manifest["input_policy"]
    assert manifest["stages"] == ["search_calibration", "formula_selection"]
    assert manifest["model"] == {
        "package": "mattersim",
        "version": manifest["model"]["version"],
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "device": "cpu",
        "inference_batch_size": 7,
        "atom_budget": 4,
        "structure_cap": 2,
        "structure_chunk_size": 1,
        "snapshot_steps": [0, 2, 4, 8],
        "fire_parameters": dict(FIRE_PARAMETERS),
    }
    assert isinstance(manifest["model"]["version"], str)
    assert set(manifest["runtime"]) == {
        "python_version",
        "torch_version",
        "cuda_available",
        "cuda_version",
        "gpu_name",
    }
    assert manifest["counts"]["input_rows"] == 4
    assert manifest["counts"]["stage_assignment_rows"] == 4
    assert manifest["counts"]["selected_rows"] == 3
    assert manifest["counts"]["strict_x0_rows"] == 2
    assert manifest["counts"]["nonstrict_x0_rows"] == 1
    assert manifest["counts"]["successful_rows"] == 2
    assert manifest["counts"]["failed_rows"] == 1
    assert manifest["counts"]["force_evaluations"] == 18
    assert manifest["counts"]["optimizer_updates"] == 16
    assert manifest["counts"]["retry_overhead_force_evaluations"] == 0
    assert manifest["counts"]["retry_overhead_optimizer_updates"] == 0
    assert manifest["counts"]["supported_at_k8"] == 1
    assert manifest["execution"]["predictor_forward_calls"] == 18
    assert manifest["execution"]["total_elapsed_seconds"] >= 0.0
    assert "peak_cuda_memory_bytes" in manifest["execution"]

    paths = {
        "elementa_initial_frames.zip": elementa / "elementa_initial_frames.zip",
        "elementa_x0_features.parquet": elementa / "elementa_x0_features.parquet",
        "elementa_x0_p9_features.parquet": (
            p9_dir / "elementa_x0_p9_features.parquet"
        ),
        "stage_assignments.parquet": stages_path,
        "model.pth": checkpoint,
    }
    assert manifest["inputs_sha256"] == {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in paths.items()
    }
    table_path = output / "mattersim_fewstep_features.parquet"
    assert manifest["outputs_sha256"] == {
        table_path.name: hashlib.sha256(table_path.read_bytes()).hexdigest()
    }


@pytest.mark.parametrize(
    "existing_name", ["MANIFEST.json", "mattersim_fewstep_features.parquet"]
)
def test_existing_output_artifact_is_never_overwritten(tmp_path, existing_name):
    elementa, p9_dir, stages_path, checkpoint = _tiny_inputs(tmp_path)
    output = tmp_path / "result"
    output.mkdir()
    existing = output / existing_name
    existing.write_bytes(b"preserve-me")
    predictor = _HarmonicFewstepPredictor()

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        features.run_fewstep_features(
            elementa,
            p9_dir,
            stages_path,
            output,
            checkpoint=checkpoint,
            stages=("formula_selection",),
            predictor=predictor,
        )

    assert existing.read_bytes() == b"preserve-me"
    assert predictor.calls == []


def test_test_stage_records_frozen_protocol_hash_and_only_opens_test(tmp_path):
    elementa, p9_dir, stages_path, checkpoint = _tiny_inputs(tmp_path)
    frozen = tmp_path / "frozen_protocol.json"
    _write_valid_freeze(frozen, elementa, p9_dir, stages_path, checkpoint)
    output = tmp_path / "test-result"
    predictor = _HarmonicFewstepPredictor()

    manifest = features.run_fewstep_features(
        elementa,
        p9_dir,
        stages_path,
        output,
        checkpoint=checkpoint,
        stages=("test",),
        device="cpu",
        atom_budget=4,
        inference_batch_size=2,
        structure_chunk_size=4,
        predictor=predictor,
        frozen_protocol_path=frozen,
    )

    table = pd.read_parquet(output / "mattersim_fewstep_features.parquet")
    digest = hashlib.sha256(frozen.read_bytes()).hexdigest()
    assert table.sid.tolist() == ["sid-test"]
    assert predictor.calls_by_atomic_number == {4: 9}
    assert manifest["stages"] == ["test"]
    assert manifest["frozen_protocol"] == {
        "path": str(frozen.resolve()),
        "sha256": digest,
    }
    assert manifest["inputs_sha256"][frozen.name] == digest


@pytest.mark.parametrize(
    "mismatch",
    (
        "malformed",
        "protocol",
        "state",
        "frozen_at_utc",
        "checkpoint_sha256",
        "feature_inputs_sha256",
        "code_sha256",
    ),
)
def test_test_freeze_mismatch_is_rejected_before_zip_is_opened(
    tmp_path, monkeypatch, mismatch
):
    elementa, p9_dir, stages_path, checkpoint = _tiny_inputs(tmp_path)
    frozen = tmp_path / "frozen_protocol.json"
    payload = _valid_freeze_payload(elementa, p9_dir, stages_path, checkpoint)
    if mismatch == "malformed":
        frozen.write_text("not-json\n")
    else:
        if mismatch == "protocol":
            payload["protocol"] = "stale-protocol"
        elif mismatch == "state":
            payload["state"] = "development"
        elif mismatch == "frozen_at_utc":
            payload["frozen_at_utc"] = ""
        elif mismatch == "checkpoint_sha256":
            payload["checkpoint_sha256"] = "0" * 64
        elif mismatch == "feature_inputs_sha256":
            payload["feature_inputs_sha256"] = {
                **payload["feature_inputs_sha256"],
                "elementa_initial_frames.zip": "1" * 64,
            }
        else:
            payload["code_sha256"] = {
                **payload["code_sha256"],
                "next7_mattersim_features.py": "2" * 64,
            }
        frozen.write_text(json.dumps(payload) + "\n")

    def forbidden_zip(*_args, **_kwargs):
        raise AssertionError("test freeze must be validated before frame reads")

    monkeypatch.setattr(features.zipfile, "ZipFile", forbidden_zip)
    with pytest.raises(ValueError, match="frozen protocol"):
        features.run_fewstep_features(
            elementa,
            p9_dir,
            stages_path,
            tmp_path / "result",
            checkpoint=checkpoint,
            stages=("test",),
            predictor=_HarmonicFewstepPredictor(),
            frozen_protocol_path=frozen,
        )


def test_default_mattersim_predictor_is_created_once_for_all_chunks(
    tmp_path, monkeypatch
):
    elementa, p9_dir, stages_path, checkpoint = _tiny_inputs(tmp_path)
    harmonic = _HarmonicFewstepPredictor()
    factory_calls = []

    def fake_factory(checkpoint_arg, *, device, batch_size):
        factory_calls.append((Path(checkpoint_arg), device, batch_size))
        return harmonic

    monkeypatch.setattr(features, "make_mattersim_predictor", fake_factory)
    features.run_fewstep_features(
        elementa,
        p9_dir,
        stages_path,
        tmp_path / "result",
        checkpoint=checkpoint,
        stages=("search_calibration", "formula_selection"),
        device="cpu",
        atom_budget=4,
        inference_batch_size=11,
        structure_chunk_size=1,
    )

    assert factory_calls == [(checkpoint, "cpu", 11)]
    assert len(harmonic.calls) == 18


def test_predictor_failure_writes_one_stable_fail_open_row(tmp_path):
    elementa, p9_dir, stages_path, checkpoint = _tiny_inputs(tmp_path)

    def failing_predictor(_structures):
        raise RuntimeError("synthetic failure containing unstable detail")

    output = tmp_path / "result"
    manifest = features.run_fewstep_features(
        elementa,
        p9_dir,
        stages_path,
        output,
        checkpoint=checkpoint,
        stages=("formula_selection",),
        device="cpu",
        atom_budget=4,
        structure_chunk_size=1,
        predictor=failing_predictor,
    )

    row = pd.read_parquet(output / "mattersim_fewstep_features.parquet").iloc[0]
    assert row["sid"] == "sid-a"
    assert not bool(row["fewstep_feature_ok"])
    assert row["fewstep_feature_error"] == "predictor_failed"
    assert row["force_evaluations"] == 2
    assert row["optimizer_updates"] == 0
    assert row["retry_overhead_force_evaluations"] == 1
    assert row["retry_overhead_optimizer_updates"] == 0
    for step in (0, 2, 4, 8):
        assert np.isnan(row[f"k{step}_energy_total_ev"])
        assert not bool(row[f"k{step}_supported"])
        assert row[f"k{step}_support_reason"] == "predictor_failed"
    assert manifest["counts"]["failed_rows"] == 1


def test_snapshot_summary_failure_is_stable_and_does_not_escape(
    tmp_path, monkeypatch
):
    elementa, p9_dir, stages_path, checkpoint = _tiny_inputs(tmp_path)
    predictor = _HarmonicFewstepPredictor()

    def fail_summary(_snapshots):
        raise RuntimeError("unstable internal summary detail")

    monkeypatch.setattr(features, "summarize_snapshots", fail_summary)
    output = tmp_path / "result"
    features.run_fewstep_features(
        elementa,
        p9_dir,
        stages_path,
        output,
        checkpoint=checkpoint,
        stages=("formula_selection",),
        device="cpu",
        atom_budget=4,
        predictor=predictor,
    )

    row = pd.read_parquet(output / "mattersim_fewstep_features.parquet").iloc[0]
    assert not bool(row["fewstep_feature_ok"])
    assert row["fewstep_feature_error"] == "invalid_snapshots"
    for step in (0, 2, 4, 8):
        assert np.isnan(row[f"k{step}_min_pair_distance_a"])
        assert not bool(row[f"k{step}_supported"])
        assert row[f"k{step}_support_reason"] == "invalid_snapshots"


def test_join_keys_require_nonmissing_rk_values(tmp_path):
    elementa, p9_dir, stages_path, checkpoint = _tiny_inputs(tmp_path)
    metadata_path = p9_dir / "elementa_x0_p9_features.parquet"
    metadata = pd.read_parquet(metadata_path)
    stages = pd.read_parquet(stages_path)
    metadata.loc[metadata["sid"].eq("sid-a"), "rk"] = None
    stages.loc[stages["sid"].eq("sid-a"), "rk"] = None
    metadata.to_parquet(metadata_path, index=False)
    stages.to_parquet(stages_path, index=False)

    with pytest.raises(ValueError, match="rk.*nonmissing"):
        features.run_fewstep_features(
            elementa,
            p9_dir,
            stages_path,
            tmp_path / "result",
            checkpoint=checkpoint,
            stages=("formula_selection",),
            predictor=_HarmonicFewstepPredictor(),
        )


def test_cli_exposes_only_label_free_inputs_and_forwards_protocol_options(
    tmp_path, monkeypatch
):
    captured = {}

    def fake_run(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return {"counts": {"selected_rows": 3}}

    monkeypatch.setattr(features, "run_fewstep_features", fake_run)
    result = features.main(
        [
            "--elementa",
            str(tmp_path / "elementa"),
            "--p9",
            str(tmp_path / "p9"),
            "--stage-assignments",
            str(tmp_path / "stages.parquet"),
            "--output",
            str(tmp_path / "output"),
            "--checkpoint",
            str(tmp_path / "model.pth"),
            "--stages",
            "search_calibration",
            "formula_selection",
            "--device",
            "cpu",
            "--atom-budget",
            "99",
            "--structure-cap",
            "7",
            "--inference-batch-size",
            "5",
            "--structure-chunk-size",
            "13",
        ]
    )

    assert result == 0
    assert captured["args"] == (
        tmp_path / "elementa",
        tmp_path / "p9",
        tmp_path / "stages.parquet",
        tmp_path / "output",
    )
    assert captured["kwargs"] == {
        "checkpoint": tmp_path / "model.pth",
        "stages": ("search_calibration", "formula_selection"),
        "device": "cpu",
        "atom_budget": 99,
        "structure_cap": 7,
        "inference_batch_size": 5,
        "structure_chunk_size": 13,
        "frozen_protocol_path": None,
    }
    assert not any("label" in name for name in captured["kwargs"])


def test_explicit_stage_with_no_rows_writes_empty_schema_without_prediction(tmp_path):
    elementa, p9_dir, stages_path, checkpoint = _tiny_inputs(tmp_path)

    def forbidden_predictor(_structures):
        raise AssertionError("an empty selected stage must not invoke prediction")

    output = tmp_path / "empty-result"
    manifest = features.run_fewstep_features(
        elementa,
        p9_dir,
        stages_path,
        output,
        checkpoint=checkpoint,
        stages=("threshold_calibration",),
        predictor=forbidden_predictor,
    )

    table = pd.read_parquet(output / "mattersim_fewstep_features.parquet")
    assert table.empty
    assert {"sid", "fewstep_feature_ok", "k8_supported"} <= set(table.columns)
    assert manifest["counts"]["selected_rows"] == 0
    assert manifest["execution"]["predictor_forward_calls"] == 0


def test_empty_and_nonempty_outputs_share_an_explicit_arrow_schema(tmp_path):
    elementa, p9_dir, stages_path, checkpoint = _tiny_inputs(tmp_path)
    nonempty_output = tmp_path / "nonempty-result"
    empty_output = tmp_path / "empty-result"

    features.run_fewstep_features(
        elementa,
        p9_dir,
        stages_path,
        nonempty_output,
        checkpoint=checkpoint,
        stages=("formula_selection",),
        device="cpu",
        predictor=_HarmonicFewstepPredictor(),
    )
    features.run_fewstep_features(
        elementa,
        p9_dir,
        stages_path,
        empty_output,
        checkpoint=checkpoint,
        stages=("threshold_calibration",),
        device="cpu",
        predictor=_HarmonicFewstepPredictor(),
    )

    nonempty_schema = pq.read_schema(
        nonempty_output / "mattersim_fewstep_features.parquet"
    )
    empty_schema = pq.read_schema(
        empty_output / "mattersim_fewstep_features.parquet"
    )
    assert empty_schema.equals(nonempty_schema)
    assert all(field.type != pa.null() for field in empty_schema)
    expected_types = {
        "sid": pa.string(),
        "strict_x0_ok": pa.bool_(),
        "initial_ionic_step": pa.int64(),
        "geom_min_pair_ratio": pa.float64(),
        "force_evaluations": pa.int64(),
        "retry_overhead_force_evaluations": pa.int64(),
        "allocated_seconds": pa.float64(),
        "k8_supported": pa.bool_(),
        "k8_support_reason": pa.string(),
        "k8_energy_total_ev": pa.float64(),
    }
    for name, expected_type in expected_types.items():
        assert empty_schema.field(name).type == expected_type


def test_cuda_accounting_resets_and_reads_the_requested_device_index(
    tmp_path, monkeypatch
):
    elementa, p9_dir, stages_path, checkpoint = _tiny_inputs(tmp_path)
    events = []

    class FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def reset_peak_memory_stats(index):
            events.append(("reset", index))

        @staticmethod
        def get_device_name(index):
            events.append(("name", index))
            return f"fake-gpu-{index}"

        @staticmethod
        def max_memory_allocated(index):
            events.append(("max", index))
            return 123456

    fake_torch = types.SimpleNamespace(
        __version__="fake-torch",
        version=types.SimpleNamespace(cuda="fake-cuda"),
        cuda=FakeCuda(),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    harmonic = _HarmonicFewstepPredictor()

    def recording_predictor(structures):
        events.append(("predict", None))
        return harmonic(structures)

    manifest = features.run_fewstep_features(
        elementa,
        p9_dir,
        stages_path,
        tmp_path / "result",
        checkpoint=checkpoint,
        stages=("formula_selection",),
        device="cuda:1",
        predictor=recording_predictor,
    )

    assert events[0] == ("reset", 1)
    assert events[-2:] == [("name", 1), ("max", 1)]
    assert events.count(("reset", 1)) == 1
    assert manifest["runtime"]["gpu_name"] == "fake-gpu-1"
    assert manifest["execution"]["peak_cuda_memory_bytes"] == 123456


@pytest.mark.parametrize(
    ("device", "cuda_available"), (("cpu", True), ("cuda:2", False))
)
def test_cpu_or_unavailable_cuda_has_null_device_metrics(
    monkeypatch, device, cuda_available
):
    calls = []

    class FakeCuda:
        @staticmethod
        def is_available():
            return cuda_available

        @staticmethod
        def reset_peak_memory_stats(index):
            calls.append(("reset", index))

        @staticmethod
        def get_device_name(index):
            calls.append(("name", index))
            return "unexpected"

        @staticmethod
        def max_memory_allocated(index):
            calls.append(("max", index))
            return 1

    monkeypatch.setitem(
        sys.modules,
        "torch",
        types.SimpleNamespace(
            __version__="fake-torch",
            version=types.SimpleNamespace(cuda="fake-cuda"),
            cuda=FakeCuda(),
        ),
    )

    tracking_started = features._reset_peak_cuda_memory(device)
    runtime, peak = features._runtime_metadata(
        device, collect_cuda_metrics=tracking_started
    )

    assert not tracking_started
    assert runtime["gpu_name"] is None
    assert peak is None
    assert calls == []


@pytest.mark.parametrize("pair_ratio", [0.449, np.nan])
def test_unsafe_x0_pair_ratio_keeps_trajectory_but_disables_all_support(
    tmp_path, pair_ratio
):
    elementa, p9_dir, stages_path, checkpoint = _tiny_inputs(tmp_path)
    base_path = elementa / "elementa_x0_features.parquet"
    base = pd.read_parquet(base_path)
    base.loc[base["sid"].eq("sid-a"), "geom_min_pair_ratio"] = pair_ratio
    base.to_parquet(base_path, index=False)
    predictor = _HarmonicFewstepPredictor()
    output = tmp_path / "result"

    features.run_fewstep_features(
        elementa,
        p9_dir,
        stages_path,
        output,
        checkpoint=checkpoint,
        stages=("formula_selection",),
        device="cpu",
        atom_budget=4,
        predictor=predictor,
    )

    row = pd.read_parquet(output / "mattersim_fewstep_features.parquet").iloc[0]
    assert row["geom_min_pair_ratio"] == pytest.approx(pair_ratio, nan_ok=True)
    assert bool(row["fewstep_feature_ok"])
    assert np.isfinite(row["k8_energy_total_ev"])
    assert len(predictor.calls) == 9
    for step in (0, 2, 4, 8):
        assert not bool(row[f"k{step}_supported"])
        assert row[f"k{step}_support_reason"] == "unsafe_x0_pair_ratio"


def test_structure_chunks_are_parsed_and_predicted_in_streaming_order(
    tmp_path, monkeypatch
):
    elementa, p9_dir, stages_path, checkpoint = _tiny_inputs(tmp_path)
    events = []
    real_frame_to_atoms = features.frame_to_atoms
    harmonic = _HarmonicFewstepPredictor()

    def recording_parse(text):
        symbol = text.splitlines()[2].split()[0]
        events.append(f"parse:{symbol}")
        return real_frame_to_atoms(text)

    def recording_predictor(structures):
        events.append(f"predict:{structures[0].get_chemical_symbols()[0]}")
        return harmonic(structures)

    monkeypatch.setattr(features, "frame_to_atoms", recording_parse)
    features.run_fewstep_features(
        elementa,
        p9_dir,
        stages_path,
        tmp_path / "result",
        checkpoint=checkpoint,
        stages=("search_calibration", "formula_selection"),
        device="cpu",
        atom_budget=4,
        structure_chunk_size=1,
        predictor=recording_predictor,
    )

    assert events[:2] == ["parse:He", "predict:He"]
    assert events[10:12] == ["parse:H", "predict:H"]


@pytest.mark.parametrize(
    "invalid_gate",
    ("strict_not_bool", "metadata_nan", "raw_nonzero", "raw_missing"),
)
def test_strict_x0_requires_boolean_metadata_zero_and_raw_zero(
    tmp_path, invalid_gate
):
    elementa, p9_dir, stages_path, checkpoint = _tiny_inputs(tmp_path)
    metadata_path = p9_dir / "elementa_x0_p9_features.parquet"
    metadata = pd.read_parquet(metadata_path)
    frames_path = elementa / "elementa_initial_frames.zip"
    if invalid_gate == "strict_not_bool":
        metadata["strict_x0_ok"] = metadata["strict_x0_ok"].astype(str)
        metadata.to_parquet(metadata_path, index=False)
    elif invalid_gate == "metadata_nan":
        metadata.loc[metadata["sid"].eq("sid-a"), "initial_ionic_step"] = np.nan
        metadata.to_parquet(metadata_path, index=False)
    elif invalid_gate == "raw_nonzero":
        _replace_zip_member(
            frames_path,
            "sid-a.extxyz",
            lambda text: text.replace("ionic_step=0", "ionic_step=3"),
        )
    else:
        _replace_zip_member(
            frames_path,
            "sid-a.extxyz",
            lambda text: text.replace(" ionic_step=0", ""),
        )

    predictor = _HarmonicFewstepPredictor()
    output = tmp_path / "result"
    features.run_fewstep_features(
        elementa,
        p9_dir,
        stages_path,
        output,
        checkpoint=checkpoint,
        stages=("formula_selection",),
        device="cpu",
        atom_budget=4,
        predictor=predictor,
    )

    row = pd.read_parquet(output / "mattersim_fewstep_features.parquet").iloc[0]
    assert predictor.calls == []
    assert not bool(row["strict_x0_ok"])
    assert not bool(row["fewstep_feature_ok"])
    assert row["fewstep_feature_error"] == "nonzero_initial_ionic_step"
    for step in (0, 2, 4, 8):
        assert not bool(row[f"k{step}_supported"])
        assert row[f"k{step}_support_reason"] == "nonzero_initial_ionic_step"
