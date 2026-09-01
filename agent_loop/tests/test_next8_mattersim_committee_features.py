import ast
import fcntl
import hashlib
import inspect
import json
import os
from pathlib import Path
import weakref
import zipfile

import numpy as np
import pandas as pd
import pytest
from ase import Atoms

from src import next6_mattersim_baseline as baseline_module
from src import next6_wbm_features as wbm_features_module
from src import next8_mattersim_committee_features as features


_PRODUCTION_FROZEN_CHECKPOINT_SHA256 = getattr(
    features, "FROZEN_CHECKPOINT_SHA256", None
)
_TEST_FROZEN_CHECKPOINT_SHA256 = {
    "m1": hashlib.sha256(b"fake-1m-checkpoint").hexdigest(),
    "m5": hashlib.sha256(b"fake-5m-checkpoint").hexdigest(),
}


@pytest.fixture(autouse=True)
def _use_fixture_checkpoint_identities(monkeypatch):
    monkeypatch.setattr(
        features,
        "FROZEN_CHECKPOINT_SHA256",
        dict(_TEST_FROZEN_CHECKPOINT_SHA256),
        raising=False,
    )


def _labelled_frame(symbol: str, displacement: float = 1.0) -> str:
    return f'''2
Lattice="12 0 0 0 12 0 0 0 12" Properties=species:S:1:pos:R:3:forces:R:3 material_id=endpoint-id ionic_step=0 final_ionic_step=17 suffix=04 energy=-999 stress="1 2 3 4 5 6" exact_min=T near_min=T valuable=T
{symbol} {displacement} 0 0 999 999 999
{symbol} {-displacement} 0 0 -999 -999 -999
'''


def _tiny_inputs(tmp_path: Path):
    frames_path = tmp_path / "initial_frames.zip"
    with zipfile.ZipFile(frames_path, "w") as archive:
        archive.writestr("nested/sid-B.extxyz", _labelled_frame("He", 1.2))
        archive.writestr("sid-001.extxyz", _labelled_frame("H", 1.0))
        archive.writestr("sid-nonstrict.extxyz", _labelled_frame("Li", 1.1))
        archive.writestr("sid-test.extxyz", _labelled_frame("Be", 1.3))

    metadata_path = tmp_path / "metadata.parquet"
    pd.DataFrame(
        {
            "sid": ["sid-B", "sid-001", "sid-nonstrict", "sid-test"],
            "rk": ["He2|stable", "H2|001", "Li2|raw", "Be2|test"],
            "material": ["mat-B_04", "mat-001_09", "mat-ns_01", "mat-test_08"],
            "strict_x0_ok": [True, True, False, True],
            # Endpoint-only fields are deliberately present in the source.
            "energy": [-99.0, -98.0, -97.0, -96.0],
            "endpoint_forces": [999.0, 999.0, 999.0, 999.0],
            "stress": [999.0, 999.0, 999.0, 999.0],
            "final_ionic_step": [17, 17, 17, 17],
            "suffix": ["04", "09", "01", "08"],
            "near_min": [True, True, False, True],
        }
    ).to_parquet(metadata_path, index=False)

    stages_path = tmp_path / "stages.parquet"
    pd.DataFrame(
        {
            "sid": ["sid-001", "sid-nonstrict", "sid-B", "sid-test"],
            "rk": ["H2|001", "Li2|raw", "He2|stable", "Be2|test"],
            "stage": [
                "formula_selection",
                "threshold_calibration",
                "search_calibration",
                "test",
            ],
            "delta_e": [0.0, 0.1, 0.2, 0.3],
        }
    ).to_parquet(stages_path, index=False)

    checkpoint_1m = tmp_path / "MatterSim-v1.0.0-1M.pth"
    checkpoint_5m = tmp_path / "MatterSim-v1.0.0-5M.pth"
    checkpoint_1m.write_bytes(b"fake-1m-checkpoint")
    checkpoint_5m.write_bytes(b"fake-5m-checkpoint")
    return frames_path, metadata_path, stages_path, {
        "m1": checkpoint_1m,
        "m5": checkpoint_5m,
    }


class _SanitizationCheckingPredictor:
    def __init__(self, checkpoints):
        self.calls = []
        self.loaded_checkpoint_sha256 = {
            model: hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for model, path in checkpoints.items()
        }

    def __call__(self, structures):
        self.calls.append(len(structures))
        output = []
        for atoms in structures:
            assert atoms.info == {}
            assert set(atoms.arrays) == {"numbers", "positions"}
            assert atoms.calc is None
            atomic_number = int(atoms.numbers[0])
            output.append(
                {
                    "m1": features.ModelFeaturePrediction(
                        energy_total_ev=-float(atomic_number),
                        fmax_ev_per_a=0.1 * atomic_number,
                        frms_ev_per_a=0.05 * atomic_number,
                    ),
                    "m5": features.ModelFeaturePrediction(
                        energy_total_ev=-2.0 * atomic_number,
                        fmax_ev_per_a=0.2 * atomic_number,
                        frms_ev_per_a=0.1 * atomic_number,
                    ),
                }
            )
        requested = len(structures)
        telemetry = {
            model: features.ModelEvaluationTelemetry(
                requested_rows=requested,
                attempted_evaluations=requested,
                successful_evaluations=requested,
                forward_calls=1 if requested else 0,
                retry_count=0,
            )
            for model in ("m1", "m5")
        }
        return features.CommitteePredictionBatch(
            predictions=output,
            loaded_checkpoint_sha256=self.loaded_checkpoint_sha256,
            telemetry=telemetry,
        )


class _ClosablePredictor:
    def __init__(self, delegate, *, error=None, device="cpu"):
        self.delegate = delegate
        self.error = error
        self.device = device
        self.close_calls = 0

    def __call__(self, structures):
        if self.error is not None:
            raise self.error
        return self.delegate(structures)

    def close(self):
        self.close_calls += 1


def _run(
    tmp_path: Path,
    *,
    predictor=None,
    mode="development",
    stages=("search_calibration", "formula_selection", "threshold_calibration"),
    output_name="result",
):
    frames_path, metadata_path, stages_path, checkpoints = _tiny_inputs(tmp_path)
    output_dir = tmp_path / output_name
    if predictor is None:
        predictor = _SanitizationCheckingPredictor(checkpoints)
    manifest = features.run_committee_features(
        frames_path,
        metadata_path,
        stages_path,
        output_dir,
        checkpoints=checkpoints,
        stages=stages,
        mode=mode,
        predictor=predictor,
        device="cpu",
        allow_injected_predictor_for_testing=True,
    )
    return manifest, output_dir, predictor, (
        frames_path,
        metadata_path,
        stages_path,
        checkpoints,
    )


def test_api_is_label_free_and_allows_the_real_predictor_by_default():
    parameters = inspect.signature(features.run_committee_features).parameters

    assert tuple(parameters) == (
        "frames_path",
        "metadata_path",
        "stage_assignments_path",
        "output_dir",
        "checkpoints",
        "stages",
        "mode",
        "predictor",
        "device",
        "batch_size",
        "allow_injected_predictor_for_testing",
    )
    assert not any("label" in name for name in parameters)
    assert parameters["predictor"].default is None
    assert parameters["batch_size"].default == 32
    assert parameters["allow_injected_predictor_for_testing"].default is False


def test_injected_predictor_is_rejected_by_default_before_input_reads(
    tmp_path, monkeypatch
):
    frames_path, metadata_path, stages_path, checkpoints = _tiny_inputs(tmp_path)
    predictor = _SanitizationCheckingPredictor(checkpoints)
    parquet_reads = []

    def forbidden_read(*args, **kwargs):
        parquet_reads.append((args, kwargs))
        raise AssertionError("input parquet must not be read")

    monkeypatch.setattr(pd, "read_parquet", forbidden_read)
    output_dir = tmp_path / "result"
    with pytest.raises(ValueError, match="injected predictor.*testing"):
        features.run_committee_features(
            frames_path,
            metadata_path,
            stages_path,
            output_dir,
            checkpoints=checkpoints,
            stages=("formula_selection",),
            mode="development",
            predictor=predictor,
            device="cpu",
        )

    assert parquet_reads == []
    assert predictor.calls == []
    assert not output_dir.exists()


@pytest.mark.parametrize(
    ("mode", "stages"),
    [
        ("development", ()),
        ("development", "search_calibration"),
        ("development", ("unknown",)),
        ("development", ("formula_selection", "formula_selection")),
        ("development", ("test",)),
        ("development", ("formula_selection", "test")),
        ("test", ("formula_selection",)),
        ("test", ("test", "test")),
        ("unknown", ("test",)),
    ],
)
def test_mode_and_stage_allowlists_are_strict_and_test_is_independent(
    tmp_path, mode, stages
):
    frames_path, metadata_path, stages_path, checkpoints = _tiny_inputs(tmp_path)

    with pytest.raises(ValueError, match="mode|stages|test"):
        features.run_committee_features(
            frames_path,
            metadata_path,
            stages_path,
            tmp_path / "result",
            checkpoints=checkpoints,
            stages=stages,
            mode=mode,
            predictor=_SanitizationCheckingPredictor(checkpoints),
            device="cpu",
            allow_injected_predictor_for_testing=True,
        )


def test_development_features_preserve_exact_keys_strip_endpoint_labels_and_abstain(
    tmp_path,
):
    manifest, output_dir, predictor, _inputs = _run(tmp_path)

    table = pd.read_parquet(
        output_dir / "mattersim_committee_features.parquet"
    ).set_index("sid", drop=False)
    assert table["sid"].tolist() == ["sid-001", "sid-B", "sid-nonstrict"]
    assert table.loc["sid-001", "rk"] == "H2|001"
    assert table.loc["sid-B", "rk"] == "He2|stable"
    assert table.loc["sid-001", "material"] == "mat-001_09"
    assert predictor.calls == [2]

    for sid in ("sid-001", "sid-B"):
        assert bool(table.loc[sid, "committee_feature_ok"])
        assert table.loc[sid, "feature_state"] == "READY"
        assert table.loc[sid, "committee_feature_error"] == ""
        assert table.loc[sid, "natoms"] == 2
        assert np.isfinite(table.loc[sid, "m1_energy_total_ev"])
        assert np.isfinite(table.loc[sid, "m1_energy_ev_per_atom"])
        assert np.isfinite(table.loc[sid, "m5_fmax_ev_per_a"])
        assert np.isfinite(table.loc[sid, "m5_frms_ev_per_a"])

    expected_predictions = {
        "sid-001": {
            "m1": (-1.0, -0.5, 0.1, 0.05),
            "m5": (-2.0, -1.0, 0.2, 0.1),
        },
        "sid-B": {
            "m1": (-2.0, -1.0, 0.2, 0.1),
            "m5": (-4.0, -2.0, 0.4, 0.2),
        },
    }
    for sid, models in expected_predictions.items():
        for model, expected in models.items():
            observed = tuple(
                table.loc[sid, f"{model}_{column}"]
                for column in (
                    "energy_total_ev",
                    "energy_ev_per_atom",
                    "fmax_ev_per_a",
                    "frms_ev_per_a",
                )
            )
            assert observed == pytest.approx(expected)

    row = table.loc["sid-nonstrict"]
    assert not bool(row["strict_x0_ok"])
    assert not bool(row["committee_feature_ok"])
    assert row["feature_state"] == "ABSTAIN"
    assert row["committee_feature_error"] == "nonstrict_x0"
    assert row["natoms"] == 0
    assert np.isnan(row["m1_energy_total_ev"])
    assert np.isnan(row["m5_frms_ev_per_a"])

    forbidden = (
        "endpoint",
        "delta_e",
        "exact_min",
        "near_min",
        "valuable",
        "final_ionic_step",
        "suffix",
        "stress",
        "forces",
    )
    assert not any(token in column for token in forbidden for column in table.columns)
    assert manifest["stages"] == [
        "search_calibration",
        "formula_selection",
        "threshold_calibration",
    ]
    assert manifest["mode"] == "development"


def test_test_mode_selects_test_only_without_mixing_development_rows(tmp_path):
    _manifest, output_dir, predictor, _inputs = _run(
        tmp_path, mode="test", stages=("test",)
    )

    table = pd.read_parquet(output_dir / "mattersim_committee_features.parquet")
    assert table[["sid", "rk", "stage"]].to_dict("records") == [
        {"sid": "sid-test", "rk": "Be2|test", "stage": "test"}
    ]
    assert predictor.calls == [1]


@pytest.mark.parametrize("table_name", ["metadata", "stages"])
def test_duplicate_sid_or_composite_key_is_rejected_before_prediction(
    tmp_path, table_name
):
    frames_path, metadata_path, stages_path, checkpoints = _tiny_inputs(tmp_path)
    path = metadata_path if table_name == "metadata" else stages_path
    table = pd.read_parquet(path)
    table = pd.concat([table, table.iloc[[0]]], ignore_index=True)
    table.to_parquet(path, index=False)
    predictor = _SanitizationCheckingPredictor(checkpoints)

    with pytest.raises(ValueError, match="duplicate|unique"):
        features.run_committee_features(
            frames_path,
            metadata_path,
            stages_path,
            tmp_path / "result",
            checkpoints=checkpoints,
            stages=("formula_selection",),
            mode="development",
            predictor=predictor,
            device="cpu",
            allow_injected_predictor_for_testing=True,
        )

    assert predictor.calls == []


def test_rk_mismatch_is_rejected_instead_of_reconstructed(tmp_path):
    frames_path, metadata_path, stages_path, checkpoints = _tiny_inputs(tmp_path)
    stage_table = pd.read_parquet(stages_path)
    stage_table.loc[stage_table["sid"].eq("sid-001"), "rk"] = "mutated-rk"
    stage_table.to_parquet(stages_path, index=False)
    predictor = _SanitizationCheckingPredictor(checkpoints)

    with pytest.raises(ValueError, match="rk"):
        features.run_committee_features(
            frames_path,
            metadata_path,
            stages_path,
            tmp_path / "result",
            checkpoints=checkpoints,
            stages=("formula_selection",),
            mode="development",
            predictor=predictor,
            device="cpu",
            allow_injected_predictor_for_testing=True,
        )

    assert predictor.calls == []


def test_unsupported_frame_is_retained_as_abstain_without_dropping_key(tmp_path):
    frames_path, metadata_path, stages_path, checkpoints = _tiny_inputs(tmp_path)
    with zipfile.ZipFile(frames_path) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    members["sid-001.extxyz"] = b"not-an-extxyz-frame\n"
    with zipfile.ZipFile(frames_path, "w") as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
    predictor = _SanitizationCheckingPredictor(checkpoints)

    features.run_committee_features(
        frames_path,
        metadata_path,
        stages_path,
        tmp_path / "result",
        checkpoints=checkpoints,
        stages=("formula_selection",),
        mode="development",
        predictor=predictor,
        device="cpu",
        allow_injected_predictor_for_testing=True,
    )

    row = pd.read_parquet(
        tmp_path / "result" / "mattersim_committee_features.parquet"
    ).iloc[0]
    assert row["sid"] == "sid-001"
    assert row["rk"] == "H2|001"
    assert row["feature_state"] == "ABSTAIN"
    assert row["committee_feature_error"] == "unsupported_initial_frame"
    assert predictor.calls == []


@pytest.mark.parametrize(
    ("duplicate_name", "emits_duplicate_member_warning"),
    [
        ("sid-001.extxyz", True),
        ("another-directory/sid-001.xyz", False),
    ],
)
def test_archive_duplicate_member_or_stem_is_rejected_before_prediction(
    tmp_path, duplicate_name, emits_duplicate_member_warning
):
    frames_path, metadata_path, stages_path, checkpoints = _tiny_inputs(tmp_path)
    if emits_duplicate_member_warning:
        with pytest.warns(UserWarning, match="Duplicate name"):
            with zipfile.ZipFile(frames_path, "a") as archive:
                archive.writestr(duplicate_name, _labelled_frame("H", 1.4))
    else:
        with zipfile.ZipFile(frames_path, "a") as archive:
            archive.writestr(duplicate_name, _labelled_frame("H", 1.4))
    predictor = _SanitizationCheckingPredictor(checkpoints)
    output_dir = tmp_path / "result"

    with pytest.raises(ValueError, match="duplicate.*member.*stem"):
        features.run_committee_features(
            frames_path,
            metadata_path,
            stages_path,
            output_dir,
            checkpoints=checkpoints,
            stages=("formula_selection",),
            mode="development",
            predictor=predictor,
            device="cpu",
            allow_injected_predictor_for_testing=True,
        )

    assert predictor.calls == []
    assert not output_dir.exists()


def test_predictor_failure_retains_every_strict_row_as_abstain(tmp_path):
    class FailingPredictor:
        def __init__(self):
            self.calls = 0

        def __call__(self, structures):
            self.calls += 1
            assert len(structures) == 2
            raise RuntimeError("unstable model-specific detail")

    predictor = FailingPredictor()
    manifest, output_dir, _predictor, _inputs = _run(
        tmp_path, predictor=predictor
    )

    table = pd.read_parquet(output_dir / "mattersim_committee_features.parquet")
    strict = table.loc[table["strict_x0_ok"]]
    assert predictor.calls == 1
    assert len(strict) == 2
    assert set(strict["feature_state"]) == {"ABSTAIN"}
    assert set(strict["committee_feature_error"]) == {"predictor_failed"}
    assert not strict["committee_feature_ok"].any()
    assert manifest["counts"]["abstained_rows"] == 3


def test_existing_output_directory_is_never_opened_or_overwritten(tmp_path):
    frames_path, metadata_path, stages_path, checkpoints = _tiny_inputs(tmp_path)
    output_dir = tmp_path / "result"
    output_dir.mkdir()
    sentinel = output_dir / "preserve-me"
    sentinel.write_bytes(b"original")
    predictor = _SanitizationCheckingPredictor(checkpoints)

    with pytest.raises(FileExistsError, match="overwrite|existing"):
        features.run_committee_features(
            frames_path,
            metadata_path,
            stages_path,
            output_dir,
            checkpoints=checkpoints,
            stages=("formula_selection",),
            mode="development",
            predictor=predictor,
            device="cpu",
            allow_injected_predictor_for_testing=True,
        )

    assert sentinel.read_bytes() == b"original"
    assert predictor.calls == []


def test_atomic_no_replace_preserves_target_created_during_publication(
    tmp_path, monkeypatch
):
    frames_path, metadata_path, stages_path, checkpoints = _tiny_inputs(tmp_path)
    output_dir = tmp_path / "result"
    real_exists = Path.exists
    target_checks = 0

    def racing_exists(path):
        nonlocal target_checks
        observed = real_exists(path)
        if path == output_dir:
            target_checks += 1
            if target_checks == 1:
                assert not observed
                path.mkdir()
                (path / "racing-writer").write_bytes(b"preserve")
                return False
        return observed

    monkeypatch.setattr(Path, "exists", racing_exists)
    with pytest.raises(FileExistsError, match="overwrite|existing"):
        features.run_committee_features(
            frames_path,
            metadata_path,
            stages_path,
            output_dir,
            checkpoints=checkpoints,
            stages=("formula_selection",),
            mode="development",
            predictor=_SanitizationCheckingPredictor(checkpoints),
            device="cpu",
            allow_injected_predictor_for_testing=True,
        )

    assert (output_dir / "racing-writer").read_bytes() == b"preserve"
    assert not any(
        path.name.startswith(f".{output_dir.name}.staging-")
        for path in tmp_path.iterdir()
    )


def test_manifest_closes_runtime_checkpoints_sources_inputs_counts_and_output(
    tmp_path,
):
    manifest, output_dir, _predictor, inputs = _run(tmp_path)
    frames_path, metadata_path, stages_path, checkpoints = inputs
    table_path = output_dir / "mattersim_committee_features.parquet"

    loaded = json.loads((output_dir / "MANIFEST.json").read_text())
    assert loaded == manifest
    assert manifest["protocol"] == (
        "2026-08-01-mattersim-dual-checkpoint-x0-v1"
    )
    assert set(manifest["runtime"]) == {
        "python_version",
        "python_implementation",
        "platform",
        "torch_version",
        "cuda_available",
        "cuda_version",
        "gpu_name",
        "mattersim_version",
        "device",
    }
    assert manifest["runtime"]["python_version"]
    assert manifest["runtime"]["python_implementation"]
    assert manifest["runtime"]["platform"]
    assert manifest["runtime"]["device"] == "cpu"
    assert isinstance(manifest["runtime"]["mattersim_version"], str)
    predictor_source = Path(inspect.getsourcefile(_SanitizationCheckingPredictor))
    assert manifest["adapter"] == {
        "mode": "injected_test_double",
        "batch_size": 32,
        "device_requested": "cpu",
        "device_resolved": "cpu",
        "implementation": {
            "module": _SanitizationCheckingPredictor.__module__,
            "qualname": _SanitizationCheckingPredictor.__qualname__,
            "source_path": str(predictor_source.resolve()),
            "source_sha256": hashlib.sha256(
                predictor_source.read_bytes()
            ).hexdigest(),
            "source_hash_verified": True,
        },
    }
    assert manifest["production_protocol_eligible"] is False
    assert manifest["evidence_role"] == "testing_only_not_scientific_evidence"

    assert manifest["checkpoints"] == {
        model: {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for model, path in checkpoints.items()
    }
    assert manifest["predictor_loaded_checkpoint_sha256"] == {
        model: hashlib.sha256(path.read_bytes()).hexdigest()
        for model, path in checkpoints.items()
    }
    assert manifest["integrity"] == {"prepublish_rehash": "passed"}
    assert manifest["inputs_sha256"] == {
        "frames": {
            "path": str(frames_path.resolve()),
            "sha256": hashlib.sha256(frames_path.read_bytes()).hexdigest(),
        },
        "metadata": {
            "path": str(metadata_path.resolve()),
            "sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
        },
        "stage_assignments": {
            "path": str(stages_path.resolve()),
            "sha256": hashlib.sha256(stages_path.read_bytes()).hexdigest(),
        },
    }
    repository_root = Path(features.__file__).resolve().parents[1]
    executed_sources = (
        Path(features.__file__).resolve(),
        Path(baseline_module.__file__).resolve(),
        Path(wbm_features_module.__file__).resolve(),
    )
    assert "source_sha256" not in manifest
    assert manifest["executed_source_sha256"] == {
        path.relative_to(repository_root).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in executed_sources
    }

    counts = manifest["counts"]
    assert counts["input_rows"] == 4
    assert counts["stage_assignment_rows"] == 4
    assert counts["selected_rows"] == 3
    assert counts["strict_rows"] == 2
    assert counts["nonstrict_rows"] == 1
    assert counts["successful_rows"] == 2
    assert counts["abstained_rows"] == 1
    assert counts["prediction_rows_requested"] == 2
    assert "prediction_counts" not in counts
    assert "evaluation_counts" not in counts
    assert "prediction_evaluations_attempted" not in counts
    assert "prediction_evaluations_successful" not in counts
    assert counts["model_telemetry"] == {
        "m1": {
            "reported": True,
            "requested_rows": 2,
            "attempted_evaluations": 2,
            "successful_evaluations": 2,
            "forward_calls": 1,
            "retry_count": 0,
        },
        "m5": {
            "reported": True,
            "requested_rows": 2,
            "attempted_evaluations": 2,
            "successful_evaluations": 2,
            "forward_calls": 1,
            "retry_count": 0,
        },
    }
    assert manifest["execution"]["predictor_calls"] == 1
    assert manifest["execution"]["prediction_wall_time_seconds"] >= 0.0
    assert "peak_cuda_memory_bytes" in manifest["execution"]
    assert manifest["outputs_sha256"] == {
        table_path.name: hashlib.sha256(table_path.read_bytes()).hexdigest()
    }


def test_unreported_failure_telemetry_does_not_invent_evaluations(tmp_path):
    def failing_predictor(_structures):
        raise RuntimeError("synthetic failure")

    manifest, _output_dir, _predictor, _inputs = _run(
        tmp_path, predictor=failing_predictor
    )

    assert manifest["counts"]["prediction_rows_requested"] == 2
    assert manifest["counts"]["model_telemetry"] == {
        model: {
            "reported": False,
            "requested_rows": 2,
            "attempted_evaluations": None,
            "successful_evaluations": None,
            "forward_calls": None,
            "retry_count": None,
        }
        for model in ("m1", "m5")
    }
    assert manifest["predictor_loaded_checkpoint_sha256"] == {
        "m1": None,
        "m5": None,
    }


@pytest.mark.parametrize(
    "identity_case", ["missing", "swapped", "duplicate", "mismatch"]
)
def test_loaded_checkpoint_identity_must_match_declared_paths_before_features(
    tmp_path, identity_case
):
    frames_path, metadata_path, stages_path, checkpoints = _tiny_inputs(tmp_path)
    base = _SanitizationCheckingPredictor(checkpoints)

    class WrongIdentityPredictor:
        def __call__(self, structures):
            batch = base(structures)
            loaded = dict(batch.loaded_checkpoint_sha256)
            if identity_case == "missing":
                loaded.pop("m5")
            elif identity_case == "swapped":
                loaded = {"m1": loaded["m5"], "m5": loaded["m1"]}
            elif identity_case == "duplicate":
                loaded["m5"] = loaded["m1"]
            else:
                loaded["m1"] = "0" * 64
            return features.CommitteePredictionBatch(
                predictions=batch.predictions,
                loaded_checkpoint_sha256=loaded,
                telemetry=batch.telemetry,
            )

    output_dir = tmp_path / "result"
    with pytest.raises(ValueError, match="loaded checkpoint"):
        features.run_committee_features(
            frames_path,
            metadata_path,
            stages_path,
            output_dir,
            checkpoints=checkpoints,
            stages=("search_calibration", "formula_selection"),
            mode="development",
            predictor=WrongIdentityPredictor(),
            device="cpu",
            allow_injected_predictor_for_testing=True,
        )

    assert not output_dir.exists()


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("energy_total_ev", np.nan),
        ("energy_total_ev", True),
        ("energy_total_ev", "-1.0"),
        ("fmax_ev_per_a", np.inf),
        ("fmax_ev_per_a", -0.1),
        ("frms_ev_per_a", -0.1),
    ],
)
def test_injected_prediction_scalars_must_be_real_finite_and_nonnegative_forces(
    tmp_path, field, invalid_value
):
    frames_path, metadata_path, stages_path, checkpoints = _tiny_inputs(tmp_path)
    base = _SanitizationCheckingPredictor(checkpoints)

    class InvalidScalarPredictor:
        def __call__(self, structures):
            batch = base(structures)
            predictions = [dict(row) for row in batch.predictions]
            values = {
                "energy_total_ev": -1.0,
                "fmax_ev_per_a": 0.1,
                "frms_ev_per_a": 0.05,
            }
            values[field] = invalid_value
            predictions[0]["m1"] = features.ModelFeaturePrediction(**values)
            return features.CommitteePredictionBatch(
                predictions=predictions,
                loaded_checkpoint_sha256=batch.loaded_checkpoint_sha256,
                telemetry=batch.telemetry,
            )

    output_dir = tmp_path / "result"
    with pytest.raises(ValueError, match="prediction.*scalar|finite|nonnegative"):
        features.run_committee_features(
            frames_path,
            metadata_path,
            stages_path,
            output_dir,
            checkpoints=checkpoints,
            stages=("search_calibration", "formula_selection"),
            mode="development",
            predictor=InvalidScalarPredictor(),
            device="cpu",
            allow_injected_predictor_for_testing=True,
        )

    assert not output_dir.exists()


def test_declared_checkpoint_hashes_must_be_distinct_before_prediction(tmp_path):
    frames_path, metadata_path, stages_path, checkpoints = _tiny_inputs(tmp_path)
    checkpoints["m5"].write_bytes(checkpoints["m1"].read_bytes())
    predictor = _SanitizationCheckingPredictor(checkpoints)
    output_dir = tmp_path / "result"

    with pytest.raises(ValueError, match="checkpoint.*distinct"):
        features.run_committee_features(
            frames_path,
            metadata_path,
            stages_path,
            output_dir,
            checkpoints=checkpoints,
            stages=("formula_selection",),
            mode="development",
            predictor=predictor,
            device="cpu",
            allow_injected_predictor_for_testing=True,
        )

    assert predictor.calls == []
    assert not output_dir.exists()


def test_partial_model_failure_records_only_reported_actual_telemetry(tmp_path):
    frames_path, metadata_path, stages_path, checkpoints = _tiny_inputs(tmp_path)
    base = _SanitizationCheckingPredictor(checkpoints)

    class PartialPredictor:
        def __call__(self, structures):
            batch = base(structures)
            predictions = [dict(row) for row in batch.predictions]
            predictions[1]["m5"] = None
            return features.CommitteePredictionBatch(
                predictions=predictions,
                loaded_checkpoint_sha256=batch.loaded_checkpoint_sha256,
                telemetry={
                    "m1": features.ModelEvaluationTelemetry(
                        requested_rows=2,
                        attempted_evaluations=2,
                        successful_evaluations=2,
                        forward_calls=1,
                        retry_count=0,
                    ),
                    "m5": features.ModelEvaluationTelemetry(
                        requested_rows=2,
                        attempted_evaluations=1,
                        successful_evaluations=1,
                        forward_calls=1,
                        retry_count=0,
                    ),
                },
            )

    output_dir = tmp_path / "result"
    manifest = features.run_committee_features(
        frames_path,
        metadata_path,
        stages_path,
        output_dir,
        checkpoints=checkpoints,
        stages=("search_calibration", "formula_selection"),
        mode="development",
        predictor=PartialPredictor(),
        device="cpu",
        allow_injected_predictor_for_testing=True,
    )

    table = pd.read_parquet(output_dir / "mattersim_committee_features.parquet")
    strict = table.loc[table["strict_x0_ok"]]
    assert int(strict["m1_prediction_ok"].sum()) == 2
    assert int(strict["m5_prediction_ok"].sum()) == 1
    assert int(strict["committee_feature_ok"].sum()) == 1
    assert set(strict.loc[~strict["committee_feature_ok"], "committee_feature_error"]) == {
        "partial_predictor_failure"
    }
    assert manifest["counts"]["model_telemetry"]["m1"] == {
        "reported": True,
        "requested_rows": 2,
        "attempted_evaluations": 2,
        "successful_evaluations": 2,
        "forward_calls": 1,
        "retry_count": 0,
    }
    assert manifest["counts"]["model_telemetry"]["m5"] == {
        "reported": True,
        "requested_rows": 2,
        "attempted_evaluations": 1,
        "successful_evaluations": 1,
        "forward_calls": 1,
        "retry_count": 0,
    }


def test_retry_telemetry_requires_at_least_one_initial_evaluation(tmp_path):
    frames_path, metadata_path, stages_path, checkpoints = _tiny_inputs(tmp_path)
    base = _SanitizationCheckingPredictor(checkpoints)

    class RetryOnlyTelemetryPredictor:
        def __call__(self, structures):
            batch = base(structures)
            predictions = [dict(row) for row in batch.predictions]
            for row in predictions:
                row["m1"] = None
            return features.CommitteePredictionBatch(
                predictions=predictions,
                loaded_checkpoint_sha256=batch.loaded_checkpoint_sha256,
                telemetry={
                    "m1": features.ModelEvaluationTelemetry(
                        requested_rows=2,
                        attempted_evaluations=2,
                        successful_evaluations=0,
                        forward_calls=1,
                        retry_count=2,
                    ),
                    "m5": batch.telemetry["m5"],
                },
            )

    output_dir = tmp_path / "result"
    with pytest.raises(ValueError, match="telemetry"):
        features.run_committee_features(
            frames_path,
            metadata_path,
            stages_path,
            output_dir,
            checkpoints=checkpoints,
            stages=("search_calibration", "formula_selection"),
            mode="development",
            predictor=RetryOnlyTelemetryPredictor(),
            device="cpu",
            allow_injected_predictor_for_testing=True,
        )

    assert not output_dir.exists()


@pytest.mark.parametrize(
    "invalid_case",
    [
        "missing_model",
        "negative",
        "boolean",
        "requested_mismatch",
        "success_exceeds_attempts",
        "success_exceeds_unique_attempts",
        "retry_exceeds_attempts",
        "forward_exceeds_attempts",
        "success_mismatch_predictions",
    ],
)
def test_invalid_or_inconsistent_model_telemetry_fails_closed(
    tmp_path, invalid_case
):
    frames_path, metadata_path, stages_path, checkpoints = _tiny_inputs(tmp_path)
    base = _SanitizationCheckingPredictor(checkpoints)

    class InvalidTelemetryPredictor:
        def __call__(self, structures):
            batch = base(structures)
            telemetry = dict(batch.telemetry)
            values = {
                "requested_rows": 2,
                "attempted_evaluations": 2,
                "successful_evaluations": 2,
                "forward_calls": 1,
                "retry_count": 0,
            }
            if invalid_case == "missing_model":
                telemetry.pop("m5")
            else:
                if invalid_case == "negative":
                    values["attempted_evaluations"] = -1
                elif invalid_case == "boolean":
                    values["forward_calls"] = True
                elif invalid_case == "requested_mismatch":
                    values["requested_rows"] = 1
                elif invalid_case == "success_exceeds_attempts":
                    values["attempted_evaluations"] = 1
                elif invalid_case == "success_exceeds_unique_attempts":
                    values["retry_count"] = 1
                elif invalid_case == "retry_exceeds_attempts":
                    values["retry_count"] = 3
                elif invalid_case == "forward_exceeds_attempts":
                    values["forward_calls"] = 3
                else:
                    values["successful_evaluations"] = 1
                telemetry["m1"] = features.ModelEvaluationTelemetry(**values)
            return features.CommitteePredictionBatch(
                predictions=batch.predictions,
                loaded_checkpoint_sha256=batch.loaded_checkpoint_sha256,
                telemetry=telemetry,
            )

    output_dir = tmp_path / "result"
    with pytest.raises(ValueError, match="telemetry"):
        features.run_committee_features(
            frames_path,
            metadata_path,
            stages_path,
            output_dir,
            checkpoints=checkpoints,
            stages=("search_calibration", "formula_selection"),
            mode="development",
            predictor=InvalidTelemetryPredictor(),
            device="cpu",
            allow_injected_predictor_for_testing=True,
        )

    assert not output_dir.exists()


@pytest.mark.parametrize(
    "mutated_role", ["frames", "metadata", "stage_assignments", "checkpoint_m1"]
)
def test_prepublish_rehash_detects_input_or_checkpoint_mutation_and_cleans_staging(
    tmp_path, mutated_role
):
    frames_path, metadata_path, stages_path, checkpoints = _tiny_inputs(tmp_path)
    paths = {
        "frames": frames_path,
        "metadata": metadata_path,
        "stage_assignments": stages_path,
        "checkpoint_m1": checkpoints["m1"],
    }
    target = paths[mutated_role]
    base = _SanitizationCheckingPredictor(checkpoints)

    class MutatingPredictor:
        def __call__(self, structures):
            batch = base(structures)
            target.write_bytes(target.read_bytes() + b"controlled-mutation")
            return batch

    output_dir = tmp_path / "result"
    with pytest.raises(RuntimeError, match="changed after initial hash"):
        features.run_committee_features(
            frames_path,
            metadata_path,
            stages_path,
            output_dir,
            checkpoints=checkpoints,
            stages=("search_calibration", "formula_selection"),
            mode="development",
            predictor=MutatingPredictor(),
            device="cpu",
            allow_injected_predictor_for_testing=True,
        )

    assert not output_dir.exists()
    assert not any(
        path.name.startswith(f".{output_dir.name}.staging-")
        for path in tmp_path.iterdir()
    )


def test_prepublish_rehash_covers_executed_source_and_cleans_staging(
    tmp_path, monkeypatch
):
    frames_path, metadata_path, stages_path, checkpoints = _tiny_inputs(tmp_path)
    source_path = Path(features.__file__).resolve()
    real_sha256_file = features._sha256_file
    source_hash_calls = 0

    def simulate_source_change(path):
        nonlocal source_hash_calls
        path = Path(path)
        digest = real_sha256_file(path)
        if path.resolve() == source_path:
            source_hash_calls += 1
            if source_hash_calls >= 2:
                return "0" * 64
        return digest

    monkeypatch.setattr(features, "_sha256_file", simulate_source_change)
    output_dir = tmp_path / "result"
    with pytest.raises(RuntimeError, match="changed after initial hash"):
        features.run_committee_features(
            frames_path,
            metadata_path,
            stages_path,
            output_dir,
            checkpoints=checkpoints,
            stages=("search_calibration", "formula_selection"),
            mode="development",
            predictor=_SanitizationCheckingPredictor(checkpoints),
            device="cpu",
            allow_injected_predictor_for_testing=True,
        )

    assert source_hash_calls >= 2
    assert not output_dir.exists()
    assert not any(
        path.name.startswith(f".{output_dir.name}.staging-")
        for path in tmp_path.iterdir()
    )


def test_prepublish_rehash_covers_injected_predictor_source(
    tmp_path, monkeypatch
):
    frames_path, metadata_path, stages_path, checkpoints = _tiny_inputs(tmp_path)
    predictor_source = Path(inspect.getsourcefile(_SanitizationCheckingPredictor))
    real_sha256_file = features._sha256_file
    predictor_source_hash_calls = 0

    def simulate_predictor_source_change(path):
        nonlocal predictor_source_hash_calls
        digest = real_sha256_file(path)
        if Path(path).resolve() == predictor_source.resolve():
            predictor_source_hash_calls += 1
            if predictor_source_hash_calls >= 2:
                return "0" * 64
        return digest

    monkeypatch.setattr(
        features, "_sha256_file", simulate_predictor_source_change
    )
    output_dir = tmp_path / "result"
    with pytest.raises(
        RuntimeError, match="predictor implementation changed"
    ):
        features.run_committee_features(
            frames_path,
            metadata_path,
            stages_path,
            output_dir,
            checkpoints=checkpoints,
            stages=("formula_selection",),
            mode="development",
            predictor=_SanitizationCheckingPredictor(checkpoints),
            device="cpu",
            allow_injected_predictor_for_testing=True,
        )

    assert predictor_source_hash_calls >= 2
    assert not output_dir.exists()


def test_injected_predictor_without_inspectable_source_is_explicitly_unverified(
    tmp_path, monkeypatch
):
    frames_path, metadata_path, stages_path, checkpoints = _tiny_inputs(tmp_path)

    def unavailable_source(_target):
        raise TypeError("source unavailable")

    monkeypatch.setattr(features.inspect, "getsourcefile", unavailable_source)
    monkeypatch.setattr(features.inspect, "getfile", unavailable_source)
    output_dir = tmp_path / "result"
    manifest = features.run_committee_features(
        frames_path,
        metadata_path,
        stages_path,
        output_dir,
        checkpoints=checkpoints,
        stages=("formula_selection",),
        mode="development",
        predictor=_SanitizationCheckingPredictor(checkpoints),
        device="cpu",
        allow_injected_predictor_for_testing=True,
    )

    implementation = manifest["adapter"]["implementation"]
    assert implementation["source_path"] is None
    assert implementation["source_sha256"] is None
    assert implementation["source_hash_verified"] is False
    assert manifest["production_protocol_eligible"] is False


def test_real_mattersim_imports_remain_lazy_and_outside_module_scope():
    source_path = Path(features.__file__).resolve()
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    parent = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parent[child] = node

    mattersim_imports = []
    for node in ast.walk(tree):
        modules = []
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            modules = [node.module or ""]
        if any(
            module == "mattersim" or module.startswith("mattersim.")
            for module in modules
        ):
            mattersim_imports.append(node)

    assert mattersim_imports
    assert all(
        any(
            isinstance(ancestor, (ast.FunctionDef, ast.AsyncFunctionDef))
            for ancestor in _ast_ancestors(node, parent)
        )
        for node in mattersim_imports
    )


def _ast_ancestors(node, parent):
    while node in parent:
        node = parent[node]
        yield node


def _checkpoint_hashes(checkpoints):
    return {
        model: hashlib.sha256(Path(path).read_bytes()).hexdigest()
        for model, path in checkpoints.items()
    }


def _atoms(z=1, *, x=1.0, cell=None, pbc=True):
    if cell is None:
        cell = np.eye(3) * 8.0
    return Atoms(
        numbers=[z],
        positions=[[x, 0.0, 0.0]],
        cell=cell,
        pbc=pbc,
    )


class _FakeHookHandle:
    def __init__(self, hooks, hook):
        self._hooks = hooks
        self._hook = hook

    def remove(self):
        if self._hook in self._hooks:
            self._hooks.remove(self._hook)


class _FakeHookModel:
    def __init__(self, api, model_key, max_z=94):
        self.api = api
        self.model_key = model_key
        self.max_z = max_z
        self.model_args = {"cutoff": 5.0, "threebody_cutoff": 4.0}
        self._forward_pre_hooks = []
        self.direct_forward_calls = 0

    def register_forward_pre_hook(self, hook):
        self._forward_pre_hooks.append(hook)
        return _FakeHookHandle(self._forward_pre_hooks, hook)

    def forward(self, loader):
        # MatterSim 1.2.3 Potential.forward calls model.forward directly,
        # bypassing torch.nn.Module.__call__ and its pre-hook dispatch.
        self.direct_forward_calls += 1
        mode = self.api.behaviors.get(self.model_key, "normal")
        markers = [float(atoms.positions[0, 0]) for atoms in loader["atoms"]]
        if mode == "ordinary_marker7" and 7.0 in markers:
            raise RuntimeError("synthetic ordinary forward failure")
        if mode == "oom_multi" and len(loader["atoms"]) > 1:
            raise RuntimeError("CUDA out of memory while allocating tensor")
        if mode == "fatal_cuda":
            raise RuntimeError("CUDA error: device-side assert triggered")

        offset = 100.0 if self.model_key == "m1" else 200.0
        energies = np.asarray(
            [-offset - float(atoms.numbers[0]) for atoms in loader["atoms"]],
            dtype=float,
        )
        forces = []
        for atoms, marker in zip(loader["atoms"], markers, strict=True):
            force = np.zeros((len(atoms), 3), dtype=float)
            force[:, 0] = float(atoms.numbers[0])
            if mode == "nan_marker8" and marker == 8.0:
                energies[len(forces)] = np.nan
            if mode == "bad_force_marker9" and marker == 9.0:
                force = np.zeros((len(atoms), 2), dtype=float)
            if mode == "inf_force_marker10" and marker == 10.0:
                force[0, 0] = np.inf
            forces.append(force)
        if mode == "short_batch" and len(loader["atoms"]) > 1:
            energies = energies[:-1]
        return energies, forces, None


class _FakePotentialInstance:
    def __init__(self, api, model_key):
        self.api = api
        self.model_key = model_key
        self.model_name = f"fake-{model_key}"
        self.model = _FakeHookModel(api, model_key)
        self.original_model_forward = self.model.forward

    def predict_properties(
        self, loader, *, include_forces, include_stresses
    ):
        assert include_forces is True
        assert include_stresses is False
        self.api.predict_calls.append(
            (self.model_key, tuple(loader["atoms"]))
        )
        return self.model.forward(loader)


class _FakeMatterSimAPI:
    def __init__(self, *, behaviors=None, fail_load_model=None):
        self.behaviors = dict(behaviors or {})
        self.fail_load_model = fail_load_model
        self.load_calls = []
        self.load_paths = []
        self.loader_calls = []
        self.predict_calls = []
        self.potentials = []
        api = self

        class Potential:
            @classmethod
            def from_checkpoint(
                cls, *, load_path, device, load_training_state
            ):
                return api.from_checkpoint(
                    load_path=load_path,
                    device=device,
                    load_training_state=load_training_state,
                )

        self.Potential = Potential

    def from_checkpoint(self, *, load_path, device, load_training_state):
        assert load_training_state is False
        assert load_path.startswith("/proc/self/fd/")
        fd = int(Path(load_path).name)
        expected_seals = (
            features._F_SEAL_SEAL
            | features._F_SEAL_SHRINK
            | features._F_SEAL_GROW
            | features._F_SEAL_WRITE
        )
        assert fcntl.fcntl(fd, features._F_GET_SEALS) & expected_seals == expected_seals
        payload = Path(load_path).read_bytes()
        if b"fake-1m-checkpoint" in payload:
            model_key = "m1"
        elif b"fake-5m-checkpoint" in payload:
            model_key = "m5"
        else:
            raise RuntimeError("corrupt or unknown checkpoint payload")
        self.load_calls.append((model_key, payload, device, load_training_state))
        self.load_paths.append(load_path)
        if model_key == self.fail_load_model:
            raise RuntimeError(f"synthetic corrupt {model_key} checkpoint")
        potential = _FakePotentialInstance(self, model_key)
        self.potentials.append(potential)
        return potential

    def build_dataloader(self, atoms_list, **kwargs):
        atoms = tuple(atoms_list)
        self.loader_calls.append((atoms, dict(kwargs)))
        if any(float(item.positions[0, 0]) == 13.0 for item in atoms):
            raise ValueError("synthetic graph construction failure")
        return {"atoms": atoms}


def _real_adapter(tmp_path, monkeypatch, *, behaviors=None, batch_size=16):
    _frames, _metadata, _stages, checkpoints = _tiny_inputs(tmp_path)
    api = _FakeMatterSimAPI(behaviors=behaviors)
    api_loads = []

    def load_api():
        api_loads.append(True)
        return api.Potential, api.build_dataloader

    monkeypatch.setattr(features, "_load_mattersim_api", load_api)
    predictor = features.MatterSimCommitteePredictor(
        checkpoints=checkpoints,
        expected_checkpoint_sha256=_checkpoint_hashes(checkpoints),
        device="cpu",
        batch_size=batch_size,
    )
    return predictor, api, api_loads, checkpoints


def test_production_checkpoint_identity_mapping_is_frozen_to_reviewed_models():
    assert dict(_PRODUCTION_FROZEN_CHECKPOINT_SHA256) == {
        "m1": "28b0b0b0f13efefee06b47ea4c9105a26bd3e2c8396da193430da96b3b49a8be",
        "m5": "e3df9fa708725e3d453140646c7d1838324b347a3d1214cf1440522146f872b5",
    }


def test_forward_wrapper_counts_direct_model_forward_and_is_recoverable():
    class Model:
        def forward(self, loader):
            return loader

    model = Model()
    tracker = features._ForwardTelemetryTracker()
    original_forward = model.forward

    handle = features._install_forward_wrapper(model, tracker)
    with pytest.raises(
        features.FatalCommitteePredictionError, match="already|duplicate"
    ):
        features._install_forward_wrapper(model, tracker)

    tracker.start(1)
    tracker.set_pending((0,))
    assert model.forward("one-native-batch") == "one-native-batch"
    tracker.clear_pending()
    assert tracker.telemetry(
        requested_rows=1, successful_evaluations=1
    ).forward_calls == 1
    handle.remove()
    handle.remove()
    assert model.forward == original_forward


def test_adapter_observes_direct_model_forward_without_manual_hook_dispatch(
    tmp_path, monkeypatch
):
    predictor, api, _api_loads, _checkpoints = _real_adapter(
        tmp_path, monkeypatch
    )

    batch = predictor([_atoms(1)])

    assert batch.telemetry["m1"].forward_calls == 1
    assert batch.telemetry["m5"].forward_calls == 1
    assert [potential.model.direct_forward_calls for potential in api.potentials] == [
        1,
        1,
    ]
    assert all(
        potential.model._forward_pre_hooks == [] for potential in api.potentials
    )


def test_real_adapter_lazy_loads_m1_then_m5_once_from_sealed_bytes(
    tmp_path, monkeypatch
):
    predictor, api, api_loads, checkpoints = _real_adapter(tmp_path, monkeypatch)
    expected_hashes = _checkpoint_hashes(checkpoints)

    first = predictor([_atoms(1)])
    second = predictor([_atoms(2)])

    assert api_loads == [True]
    assert [call[0] for call in api.load_calls] == ["m1", "m5"]
    assert [call[2:] for call in api.load_calls] == [
        ("cpu", False),
        ("cpu", False),
    ]
    assert first.loaded_checkpoint_sha256 == expected_hashes
    assert second.loaded_checkpoint_sha256 == expected_hashes
    assert all(not Path(path).exists() for path in api.load_paths)
    assert len(api.load_calls) == 2


def test_close_is_reverse_order_idempotent_clears_state_and_disables_calls(
    tmp_path, monkeypatch
):
    _frames, _metadata, _stages, checkpoints = _tiny_inputs(tmp_path)
    monkeypatch.setattr(features, "_cuda_is_available", lambda: False)
    predictor = features.MatterSimCommitteePredictor(
        checkpoints=checkpoints,
        expected_checkpoint_sha256=_checkpoint_hashes(checkpoints),
        device="cpu",
        batch_size=8,
    )
    removed = []

    class Handle:
        def __init__(self, name):
            self.name = name

        def remove(self):
            removed.append(self.name)

    predictor._forward_handles = [Handle("m1"), Handle("m5")]
    predictor._potentials = {"m1": object(), "m5": object()}
    predictor._model_args = {"m1": ("model", 5.0, 4.0)}
    predictor._max_z = {"m1": 94}
    predictor._trackers = {"m1": object()}
    predictor._loaded_hashes = dict(_checkpoint_hashes(checkpoints))
    predictor._dataloader_builder = lambda _atoms: None

    predictor.close()
    predictor.close()

    assert removed == ["m5", "m1"]
    assert predictor._forward_handles == []
    assert predictor._potentials == {}
    assert predictor._model_args == {}
    assert predictor._max_z == {}
    assert predictor._trackers == {}
    assert predictor._loaded_hashes == {}
    assert predictor._dataloader_builder is None
    with pytest.raises(features.FatalCommitteePredictionError, match="closed"):
        predictor([_atoms(1)])


def test_close_restores_forward_and_releases_models_without_gc(
    tmp_path, monkeypatch
):
    predictor, api, _api_loads, _checkpoints = _real_adapter(
        tmp_path, monkeypatch
    )
    predictor([_atoms(1)])
    potentials = list(api.potentials)
    potential_refs = [weakref.ref(item) for item in potentials]
    model_refs = [weakref.ref(item.model) for item in potentials]
    originals = [item.original_model_forward for item in potentials]

    predictor.close()
    predictor.close()

    assert [item.model.forward for item in potentials] == originals
    api.potentials.clear()
    del originals
    del potentials
    assert [reference() for reference in potential_refs] == [None, None]
    assert [reference() for reference in model_refs] == [None, None]


def test_context_manager_closes_after_prediction_exception(
    tmp_path, monkeypatch
):
    predictor, api, _api_loads, _checkpoints = _real_adapter(
        tmp_path,
        monkeypatch,
        behaviors={"m1": "fatal_cuda"},
    )

    with pytest.raises(features.FatalCommitteePredictionError, match="CUDA|device"):
        with predictor as entered:
            assert entered is predictor
            predictor([_atoms(1)])

    assert len(api.potentials) == 2
    assert all(
        item.model.forward == item.original_model_forward
        for item in api.potentials
    )
    with pytest.raises(features.FatalCommitteePredictionError, match="closed"):
        predictor([_atoms(1)])


@pytest.mark.parametrize("failure", ["hash_mismatch", "duplicate", "corrupt"])
def test_checkpoint_snapshot_or_load_failure_is_fatal(
    tmp_path, monkeypatch, failure
):
    _frames, _metadata, _stages, checkpoints = _tiny_inputs(tmp_path)
    hashes = _checkpoint_hashes(checkpoints)
    api = _FakeMatterSimAPI(fail_load_model="m5" if failure == "corrupt" else None)
    monkeypatch.setattr(
        features,
        "_load_mattersim_api",
        lambda: (api.Potential, api.build_dataloader),
    )
    if failure == "hash_mismatch":
        hashes["m1"] = "0" * 64
    elif failure == "duplicate":
        hashes["m5"] = hashes["m1"]

    with pytest.raises(features.FatalCommitteePredictionError, match="checkpoint"):
        predictor = features.MatterSimCommitteePredictor(
            checkpoints=checkpoints,
            expected_checkpoint_sha256=hashes,
            device="cpu",
            batch_size=8,
        )
        predictor([_atoms(1)])

    assert all(not Path(path).exists() for path in api.load_paths)
    if failure == "corrupt":
        assert len(api.potentials) == 1
        potential = api.potentials[0]
        assert potential.model.forward == potential.original_model_forward


@pytest.mark.parametrize("identity_case", ["swapped", "arbitrary"])
def test_runner_rejects_nonfrozen_checkpoint_identity_before_prediction_or_publish(
    tmp_path, identity_case
):
    frames_path, metadata_path, stages_path, checkpoints = _tiny_inputs(tmp_path)
    if identity_case == "swapped":
        checkpoints = {"m1": checkpoints["m5"], "m5": checkpoints["m1"]}
    else:
        checkpoints["m5"].write_bytes(b"arbitrary-different-checkpoint")
    predictor = _SanitizationCheckingPredictor(checkpoints)
    output_dir = tmp_path / "result"

    with pytest.raises(
        features.FatalCommitteePredictionError, match="frozen checkpoint"
    ):
        features.run_committee_features(
            frames_path,
            metadata_path,
            stages_path,
            output_dir,
            checkpoints=checkpoints,
            stages=("formula_selection",),
            mode="development",
            predictor=predictor,
            device="cpu",
            allow_injected_predictor_for_testing=True,
        )

    assert predictor.calls == []
    assert not output_dir.exists()


def test_real_adapter_uses_native_single_batch_api_and_computes_force_scalars(
    tmp_path, monkeypatch
):
    predictor, api, _api_loads, _checkpoints = _real_adapter(
        tmp_path, monkeypatch, batch_size=8
    )

    batch = predictor([_atoms(1), _atoms(3)])

    assert len(api.loader_calls) == 2
    assert [call[1] for call in api.loader_calls] == [
        {
            "model_type": "fake-m1",
            "cutoff": 5.0,
            "threebody_cutoff": 4.0,
            "batch_size": 2,
            "only_inference": True,
            "shuffle": False,
        },
        {
            "model_type": "fake-m5",
            "cutoff": 5.0,
            "threebody_cutoff": 4.0,
            "batch_size": 2,
            "only_inference": True,
            "shuffle": False,
        },
    ]
    assert [call[0] for call in api.predict_calls] == ["m1", "m5"]
    assert batch.predictions[0]["m1"] == features.ModelFeaturePrediction(
        energy_total_ev=-101.0,
        fmax_ev_per_a=1.0,
        frms_ev_per_a=1.0,
    )
    assert batch.predictions[1]["m5"].energy_total_ev == -203.0
    assert batch.predictions[1]["m5"].fmax_ev_per_a == 3.0
    assert batch.telemetry == {
        model: features.ModelEvaluationTelemetry(
            requested_rows=2,
            attempted_evaluations=2,
            successful_evaluations=2,
            forward_calls=1,
            retry_count=0,
        )
        for model in ("m1", "m5")
    }


def test_invalid_structures_are_rejected_before_forward_but_high_force_is_valid(
    tmp_path, monkeypatch
):
    predictor, api, _api_loads, _checkpoints = _real_adapter(tmp_path, monkeypatch)
    empty = Atoms(cell=np.eye(3), pbc=True)
    too_heavy = _atoms(95)
    nonfinite = _atoms(1)
    nonfinite.positions[0, 0] = np.nan
    degenerate = _atoms(1, cell=np.zeros((3, 3)), pbc=True)
    high_force = _atoms(94)

    batch = predictor([empty, too_heavy, nonfinite, degenerate, high_force])

    for row in batch.predictions[:4]:
        assert row == {"m1": None, "m5": None}
    assert batch.predictions[4]["m1"].fmax_ev_per_a == 94.0
    assert batch.predictions[4]["m5"].fmax_ev_per_a == 94.0
    assert [len(call[1]) for call in api.predict_calls] == [1, 1]
    assert batch.telemetry == {
        model: features.ModelEvaluationTelemetry(
            requested_rows=5,
            attempted_evaluations=1,
            successful_evaluations=1,
            forward_calls=1,
            retry_count=0,
        )
        for model in ("m1", "m5")
    }


def test_graph_build_failure_isolated_without_counting_attempt_or_retry(
    tmp_path, monkeypatch
):
    predictor, _api, _api_loads, _checkpoints = _real_adapter(
        tmp_path, monkeypatch, batch_size=8
    )

    batch = predictor([_atoms(1, x=1.0), _atoms(2, x=13.0), _atoms(3, x=3.0)])

    assert [row["m1"] is not None for row in batch.predictions] == [True, False, True]
    assert [row["m5"] is not None for row in batch.predictions] == [True, False, True]
    assert batch.telemetry == {
        model: features.ModelEvaluationTelemetry(
            requested_rows=3,
            attempted_evaluations=2,
            successful_evaluations=2,
            forward_calls=2,
            retry_count=0,
        )
        for model in ("m1", "m5")
    }


def test_ordinary_single_row_failure_in_m1_still_runs_m5_with_exact_retries(
    tmp_path, monkeypatch
):
    predictor, api, _api_loads, _checkpoints = _real_adapter(
        tmp_path,
        monkeypatch,
        behaviors={"m1": "ordinary_marker7"},
        batch_size=8,
    )

    batch = predictor([_atoms(1, x=1.0), _atoms(2, x=7.0), _atoms(3, x=3.0)])

    assert [row["m1"] is not None for row in batch.predictions] == [True, False, True]
    assert all(row["m5"] is not None for row in batch.predictions)
    assert [call[0] for call in api.predict_calls] == [
        "m1",
        "m1",
        "m1",
        "m1",
        "m1",
        "m5",
    ]
    assert batch.telemetry["m1"] == features.ModelEvaluationTelemetry(
        requested_rows=3,
        attempted_evaluations=8,
        successful_evaluations=2,
        forward_calls=5,
        retry_count=5,
    )
    assert batch.telemetry["m5"] == features.ModelEvaluationTelemetry(
        requested_rows=3,
        attempted_evaluations=3,
        successful_evaluations=3,
        forward_calls=1,
        retry_count=0,
    )


def test_oom_clears_cache_and_bisects_while_counting_real_forwards(
    tmp_path, monkeypatch
):
    predictor, _api, _api_loads, _checkpoints = _real_adapter(
        tmp_path,
        monkeypatch,
        behaviors={"m1": "oom_multi"},
        batch_size=8,
    )
    cache_clears = []
    monkeypatch.setattr(features, "_empty_cuda_cache", lambda: cache_clears.append(True))

    batch = predictor([_atoms(1), _atoms(2), _atoms(3)])

    assert all(row["m1"] is not None for row in batch.predictions)
    assert batch.telemetry["m1"] == features.ModelEvaluationTelemetry(
        requested_rows=3,
        attempted_evaluations=8,
        successful_evaluations=3,
        forward_calls=5,
        retry_count=5,
    )
    assert cache_clears == [True, True]


def test_non_oom_cuda_failure_is_fatal_without_retrying_or_running_m5(
    tmp_path, monkeypatch
):
    predictor, api, _api_loads, _checkpoints = _real_adapter(
        tmp_path,
        monkeypatch,
        behaviors={"m1": "fatal_cuda"},
        batch_size=8,
    )

    with pytest.raises(features.FatalCommitteePredictionError, match="CUDA|device"):
        predictor([_atoms(1), _atoms(2)])

    assert [call[0] for call in api.predict_calls] == ["m1"]


def test_short_batch_is_bisected_and_original_prediction_order_is_preserved(
    tmp_path, monkeypatch
):
    predictor, _api, _api_loads, _checkpoints = _real_adapter(
        tmp_path,
        monkeypatch,
        behaviors={"m1": "short_batch"},
        batch_size=8,
    )

    batch = predictor([_atoms(3), _atoms(1), _atoms(2)])

    assert [row["m1"].energy_total_ev for row in batch.predictions] == [
        -103.0,
        -101.0,
        -102.0,
    ]
    assert batch.telemetry["m1"] == features.ModelEvaluationTelemetry(
        requested_rows=3,
        attempted_evaluations=8,
        successful_evaluations=3,
        forward_calls=5,
        retry_count=5,
    )


@pytest.mark.parametrize(
    ("mode", "marker"),
    [("nan_marker8", 8.0), ("bad_force_marker9", 9.0), ("inf_force_marker10", 10.0)],
)
def test_nonfinite_or_bad_shape_output_is_localized_to_one_model_row(
    tmp_path, monkeypatch, mode, marker
):
    predictor, _api, _api_loads, _checkpoints = _real_adapter(
        tmp_path,
        monkeypatch,
        behaviors={"m1": mode},
        batch_size=8,
    )

    batch = predictor([_atoms(1, x=1.0), _atoms(2, x=marker), _atoms(3, x=3.0)])

    assert [row["m1"] is not None for row in batch.predictions] == [True, False, True]
    assert all(row["m5"] is not None for row in batch.predictions)
    assert batch.telemetry["m1"] == features.ModelEvaluationTelemetry(
        requested_rows=3,
        attempted_evaluations=3,
        successful_evaluations=2,
        forward_calls=1,
        retry_count=0,
    )


def test_runner_reraises_fatal_predictor_error_and_never_publishes(tmp_path):
    frames_path, metadata_path, stages_path, checkpoints = _tiny_inputs(tmp_path)
    output_dir = tmp_path / "result"

    def fatal_predictor(_structures):
        raise features.FatalCommitteePredictionError("fatal CUDA context")

    with pytest.raises(features.FatalCommitteePredictionError, match="fatal CUDA"):
        features.run_committee_features(
            frames_path,
            metadata_path,
            stages_path,
            output_dir,
            checkpoints=checkpoints,
            stages=("formula_selection",),
            mode="development",
            predictor=fatal_predictor,
            device="cpu",
            allow_injected_predictor_for_testing=True,
        )

    assert not output_dir.exists()


def test_runner_builds_default_adapter_only_after_checkpoint_hashing(
    tmp_path, monkeypatch
):
    frames_path, metadata_path, stages_path, checkpoints = _tiny_inputs(tmp_path)
    expected = _checkpoint_hashes(checkpoints)
    builtin_class = features.MatterSimCommitteePredictor
    events = []
    real_hash = features._sha256_file
    api = _FakeMatterSimAPI()
    monkeypatch.setattr(
        features,
        "_load_mattersim_api",
        lambda: (api.Potential, api.build_dataloader),
    )

    def recording_hash(path):
        digest = real_hash(path)
        if Path(path) in set(checkpoints.values()):
            events.append(("hash", Path(path).name))
        return digest

    def factory(*, checkpoints, expected_checkpoint_sha256, device, batch_size):
        events.append(("factory", tuple(checkpoints), device, batch_size))
        assert expected_checkpoint_sha256 == expected
        return builtin_class(
            checkpoints=checkpoints,
            expected_checkpoint_sha256=expected_checkpoint_sha256,
            device=device,
            batch_size=batch_size,
        )

    monkeypatch.setattr(features, "_sha256_file", recording_hash)
    monkeypatch.setattr(features, "MatterSimCommitteePredictor", factory)

    manifest = features.run_committee_features(
        frames_path,
        metadata_path,
        stages_path,
        tmp_path / "result",
        checkpoints=checkpoints,
        stages=("formula_selection",),
        mode="development",
        device="cpu",
        batch_size=11,
    )

    factory_index = next(i for i, item in enumerate(events) if item[0] == "factory")
    assert {item[1] for item in events[:factory_index] if item[0] == "hash"} == {
        path.name for path in checkpoints.values()
    }
    assert events[factory_index] == ("factory", ("m1", "m5"), "cpu", 11)
    assert manifest["counts"]["successful_rows"] == 1
    assert len(api.potentials) == 2
    assert all(
        item.model.forward == item.original_model_forward
        for item in api.potentials
    )
    assert manifest["adapter"] == {
        "mode": "builtin_mattersim",
        "batch_size": 11,
        "device_requested": "cpu",
        "device_resolved": "cpu",
        "implementation": {
            "module": builtin_class.__module__,
            "qualname": builtin_class.__qualname__,
            "source_path": str(Path(features.__file__).resolve()),
            "source_sha256": hashlib.sha256(
                Path(features.__file__).read_bytes()
            ).hexdigest(),
            "source_hash_verified": True,
        },
    }
    assert manifest["production_protocol_eligible"] is True
    assert manifest["evidence_role"] == "protocol_feature_generation"


def test_builtin_constructor_returning_nonexact_predictor_is_rejected_before_prediction(
    tmp_path, monkeypatch
):
    frames_path, metadata_path, stages_path, checkpoints = _tiny_inputs(tmp_path)
    delegate = _SanitizationCheckingPredictor(checkpoints)
    substituted = _ClosablePredictor(delegate)
    monkeypatch.setattr(
        features,
        "MatterSimCommitteePredictor",
        lambda **_kwargs: substituted,
    )
    output_dir = tmp_path / "result"

    with pytest.raises(
        features.FatalCommitteePredictionError,
        match="built-in.*identity|exact built-in",
    ):
        features.run_committee_features(
            frames_path,
            metadata_path,
            stages_path,
            output_dir,
            checkpoints=checkpoints,
            stages=("formula_selection",),
            mode="development",
            device="cpu",
        )

    assert delegate.calls == []
    assert substituted.close_calls == 1
    assert not output_dir.exists()
    assert not any(
        path.name.startswith(f".{output_dir.name}.staging-")
        for path in tmp_path.iterdir()
    )


def test_owned_nonexact_predictor_is_guarded_before_device_access_and_preserves_identity_error(
    tmp_path, monkeypatch
):
    frames_path, metadata_path, stages_path, checkpoints = _tiny_inputs(tmp_path)

    class DangerousSubstitute:
        def __init__(self):
            self.device_reads = 0
            self.close_calls = 0

        @property
        def device(self):
            self.device_reads += 1
            raise RuntimeError("device getter must not run")

        def close(self):
            self.close_calls += 1
            raise RuntimeError("synthetic cleanup failure")

    substituted = DangerousSubstitute()
    monkeypatch.setattr(
        features,
        "MatterSimCommitteePredictor",
        lambda **_kwargs: substituted,
    )
    parquet_reads = []

    def forbidden_read(*args, **kwargs):
        parquet_reads.append((args, kwargs))
        raise AssertionError("input parquet must not be read")

    monkeypatch.setattr(pd, "read_parquet", forbidden_read)
    output_dir = tmp_path / "result"

    with pytest.raises(
        features.FatalCommitteePredictionError,
        match="built-in.*identity|exact built-in",
    ) as captured:
        features.run_committee_features(
            frames_path,
            metadata_path,
            stages_path,
            output_dir,
            checkpoints=checkpoints,
            stages=("formula_selection",),
            mode="development",
            device="cpu",
        )

    assert substituted.device_reads == 0
    assert substituted.close_calls == 1
    assert captured.value.__cause__ is not None
    assert "close failed" in str(captured.value.__cause__)
    assert parquet_reads == []
    assert not output_dir.exists()
    assert not any(
        path.name.startswith(f".{output_dir.name}.staging-")
        for path in tmp_path.iterdir()
    )


def test_owned_exact_predictor_is_closed_when_device_stringification_fails(
    tmp_path, monkeypatch
):
    frames_path, metadata_path, stages_path, checkpoints = _tiny_inputs(tmp_path)
    builtin_class = features.MatterSimCommitteePredictor
    created = []

    class DangerousDevice:
        def __str__(self):
            raise RuntimeError("synthetic device stringification failure")

    def factory(**kwargs):
        predictor = builtin_class(**kwargs)
        original_close = predictor.close
        predictor.close_calls = 0

        def recording_close():
            predictor.close_calls += 1
            original_close()

        predictor.close = recording_close
        predictor.device = DangerousDevice()
        created.append(predictor)
        return predictor

    monkeypatch.setattr(features, "MatterSimCommitteePredictor", factory)
    parquet_reads = []

    def forbidden_read(*args, **kwargs):
        parquet_reads.append((args, kwargs))
        raise AssertionError("input parquet must not be read")

    monkeypatch.setattr(pd, "read_parquet", forbidden_read)
    output_dir = tmp_path / "result"

    with pytest.raises(RuntimeError, match="device stringification"):
        features.run_committee_features(
            frames_path,
            metadata_path,
            stages_path,
            output_dir,
            checkpoints=checkpoints,
            stages=("formula_selection",),
            mode="development",
            device="cpu",
        )

    assert len(created) == 1
    assert created[0].close_calls == 1
    assert parquet_reads == []
    assert not output_dir.exists()
    assert not any(
        path.name.startswith(f".{output_dir.name}.staging-")
        for path in tmp_path.iterdir()
    )


def test_owned_close_failure_occurs_before_publication_and_leaves_no_artifacts(
    tmp_path, monkeypatch
):
    frames_path, metadata_path, stages_path, checkpoints = _tiny_inputs(tmp_path)
    api = _FakeMatterSimAPI()
    monkeypatch.setattr(
        features,
        "_load_mattersim_api",
        lambda: (api.Potential, api.build_dataloader),
    )
    original_remove = features._ForwardWrapperHandle.remove

    def failing_remove(handle):
        original_remove(handle)
        raise RuntimeError("synthetic close failure")

    monkeypatch.setattr(
        features._ForwardWrapperHandle,
        "remove",
        failing_remove,
    )
    output_dir = tmp_path / "result"

    with pytest.raises(
        features.FatalCommitteePredictionError,
        match="cleanup|close",
    ):
        features.run_committee_features(
            frames_path,
            metadata_path,
            stages_path,
            output_dir,
            checkpoints=checkpoints,
            stages=("formula_selection",),
            mode="development",
            device="cpu",
        )

    assert not output_dir.exists()
    assert not any(
        path.name.startswith(f".{output_dir.name}.staging-")
        for path in tmp_path.iterdir()
    )
    assert len(api.potentials) == 2
    assert all(
        item.model.forward == item.original_model_forward
        for item in api.potentials
    )


@pytest.mark.parametrize("failure_path", ["predictor_call", "publication"])
def test_runner_always_closes_owned_builtin_on_failure(
    tmp_path, monkeypatch, failure_path
):
    frames_path, metadata_path, stages_path, checkpoints = _tiny_inputs(tmp_path)
    api = _FakeMatterSimAPI(
        behaviors={"m1": "fatal_cuda"}
        if failure_path == "predictor_call"
        else None
    )
    monkeypatch.setattr(
        features,
        "_load_mattersim_api",
        lambda: (api.Potential, api.build_dataloader),
    )
    if failure_path == "publication":
        monkeypatch.setattr(
            features,
            "_atomic_publish_directory_no_replace",
            lambda _source, _target: (_ for _ in ()).throw(
                OSError("synthetic publication failure")
            ),
        )

    output_dir = tmp_path / "result"
    with pytest.raises(
        (features.FatalCommitteePredictionError, OSError),
        match="CUDA|device|publication failure",
    ):
        features.run_committee_features(
            frames_path,
            metadata_path,
            stages_path,
            output_dir,
            checkpoints=checkpoints,
            stages=("formula_selection",),
            mode="development",
            device="cpu",
            batch_size=4,
        )

    assert not output_dir.exists()
    assert len(api.potentials) == 2
    assert all(
        item.model.forward == item.original_model_forward
        for item in api.potentials
    )


def test_runner_never_closes_caller_injected_predictor(tmp_path):
    frames_path, metadata_path, stages_path, checkpoints = _tiny_inputs(tmp_path)
    injected = _ClosablePredictor(
        _SanitizationCheckingPredictor(checkpoints)
    )

    features.run_committee_features(
        frames_path,
        metadata_path,
        stages_path,
        tmp_path / "result",
        checkpoints=checkpoints,
        stages=("formula_selection",),
        mode="development",
        predictor=injected,
        device="cpu",
        allow_injected_predictor_for_testing=True,
    )

    assert injected.close_calls == 0


def test_explicit_unavailable_cuda_fails_fast_and_auto_resolves_once(
    tmp_path, monkeypatch
):
    _frames, _metadata, _stages, checkpoints = _tiny_inputs(tmp_path)
    monkeypatch.setattr(features, "_cuda_is_available", lambda: False)

    with pytest.raises(features.FatalCommitteePredictionError, match="CUDA"):
        features.MatterSimCommitteePredictor(
            checkpoints=checkpoints,
            expected_checkpoint_sha256=_checkpoint_hashes(checkpoints),
            device="cuda",
            batch_size=8,
        )

    predictor = features.MatterSimCommitteePredictor(
        checkpoints=checkpoints,
        expected_checkpoint_sha256=_checkpoint_hashes(checkpoints),
        device="auto",
        batch_size=8,
    )
    assert predictor.device == "cpu"


def test_cli_is_label_free_and_routes_all_reproducibility_arguments(
    tmp_path, monkeypatch
):
    captured = {}

    def fake_run(frames, metadata, stages_path, output, **kwargs):
        captured.update(
            frames=frames,
            metadata=metadata,
            stages_path=stages_path,
            output=output,
            **kwargs,
        )
        return {"protocol": "test"}

    monkeypatch.setattr(features, "run_committee_features", fake_run)
    paths = [
        tmp_path / name
        for name in (
            "frames.zip",
            "metadata.parquet",
            "stages.parquet",
            "out",
            "1M.pth",
            "5M.pth",
        )
    ]
    exit_code = features.main(
        [
            "--frames", str(paths[0]),
            "--metadata", str(paths[1]),
            "--stages", str(paths[2]),
            "--output", str(paths[3]),
            "--m1", str(paths[4]),
            "--m5", str(paths[5]),
            "--device", "cpu",
            "--batch-size", "17",
            "--mode", "test",
        ]
    )

    assert exit_code == 0
    assert captured == {
        "frames": paths[0],
        "metadata": paths[1],
        "stages_path": paths[2],
        "output": paths[3],
        "checkpoints": {"m1": paths[4], "m5": paths[5]},
        "stages": ("test",),
        "mode": "test",
        "predictor": None,
        "device": "cpu",
        "batch_size": 17,
    }
