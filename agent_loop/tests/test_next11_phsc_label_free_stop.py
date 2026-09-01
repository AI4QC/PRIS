"""Adversarial tests for the PHSC-v0 label-free necessary-condition stop."""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
from pathlib import Path
from types import MappingProxyType
import zipfile

import numpy as np
import pandas as pd
import pytest

from src.next8_mattersim_committee_protocol import (
    DEVELOPMENT_FREEZE_PROTOCOL,
    THRESHOLD_SPLIT_SALT,
    derive_disagreement_cutoffs,
    serialize_formula_catalog,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def _feature_row(sid: str, rk: str, stage: str, *, energy: float = 0.0) -> dict[str, object]:
    return {
        "sid": sid,
        "rk": rk,
        "stage": stage,
        "committee_feature_ok": True,
        "m1_prediction_ok": True,
        "m5_prediction_ok": True,
        "m1_energy_ev_per_atom": float(energy),
        "m5_energy_ev_per_atom": float(energy),
        "m1_fmax_ev_per_a": 0.2,
        "m1_frms_ev_per_a": 0.1,
        "m5_fmax_ev_per_a": 0.2,
        "m5_frms_ev_per_a": 0.1,
        "strict_x0_ok": True,
    }


def _phsc_row(
    module: object,
    sid: str,
    rk: str,
    *,
    status: str,
) -> dict[str, object]:
    success = not status.startswith("abstain_")
    negative: object
    if status == "resolved_negative":
        negative = True
    elif success:
        negative = False
    else:
        negative = pd.NA
    diagnostics = {
        "d_star_angstrom": 1.5,
        "h_angstrom": 1.5 / 256.0,
        "lambda_h_ev_per_a2": -1.0 if negative is True else 1.0,
        "lambda_h2_ev_per_a2": -1.0 if negative is True else 1.0,
        "lambda_r_ev_per_a2": -1.0 if negative is True else 1.0,
        "e_num_ev_per_a2": 0.0,
        "u_num_ev_per_a2": -1.0 if negative is True else 1.0,
        "l_num_ev_per_a2": -1.0 if negative is True else 1.0,
        "tau_alg_ev_per_a2": 1e-12,
        "antisymmetric_norm_h_ev_per_a2": 0.0,
        "antisymmetric_norm_h2_ev_per_a2": 0.0,
        "acoustic_residual_h_ev_per_a2": 0.0,
        "acoustic_residual_h2_ev_per_a2": 0.0,
    }
    if not success:
        diagnostics = {name: np.nan for name in diagnostics}
    row = {
        "sid": sid,
        "rk": rk,
        "stage": "threshold_calibration",
        "threshold_role": "development_gate",
        "strict_x0_ok": True,
        "natoms": 2,
        "internal_dim": 3,
        "phsc_status": status,
        "phsc_negative": negative,
        **diagnostics,
        "force_call_count": 24,
        "error": "" if success else "synthetic attributed failure",
    }
    assert tuple(row) == module.PHSC_FEATURE_COLUMNS
    return row


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, sort_keys=True, allow_nan=False, separators=(",", ":")),
        encoding="utf-8",
    )


def _rewrite_phsc_table(paths: dict[str, Path], table: pd.DataFrame) -> None:
    table.to_parquet(paths["phsc_features"], index=False)
    manifest = json.loads(paths["phsc_manifest"].read_text(encoding="utf-8"))
    manifest["outputs_sha256"] = {
        paths["phsc_features"].name: _sha256(paths["phsc_features"])
    }
    _write_json(paths["phsc_manifest"], manifest)


def _write_geometry_artifact(
    tmp_path: Path,
    *,
    sids: list[str],
    raw_frames_path: Path,
    committee_path: Path,
    roles_path: Path,
    feature_rows: int,
    role_rows: int,
) -> tuple[Path, Path]:
    from src import next11_geometry_only_frames as geometry

    output = tmp_path / "geometry-only"
    output.mkdir()
    archive_path = output / geometry.OUTPUT_ARCHIVE_NAME
    frame = '''2
Lattice="8 0 0 0 9 0 0 0 10" Properties=species:S:1:pos:R:3 pbc="T T T"
H 1.1 1.3 1.7
H 2.5 1.3 1.7
'''
    with zipfile.ZipFile(
        archive_path, "w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for sid in sorted(sids):
            info = zipfile.ZipInfo(f"{sid}.extxyz", (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, frame, compresslevel=9)
    source_path = Path(geometry.__file__).resolve()
    manifest = {
        "protocol": geometry.PROTOCOL,
        "mode": "development_gate",
        "endpoint_label_artifacts_opened": False,
        "raw_x0_archive_bytes_read": True,
        "raw_x0_nongeometry_values_converted_or_exported": False,
        "input_role": "unrelaxed_x0_geometry_only",
        "selection": {
            "stage": "threshold_calibration",
            "threshold_role": "development_gate",
            "strict_x0_ok": True,
        },
        "geometry_schema": dict(geometry.GEOMETRY_SCHEMA),
        "dropped_field_names": {
            "comment": ["endpoint_label"],
            "atom_properties": ["forces"],
        },
        "inputs_sha256": {
            "raw_frames": {
                "path": str(raw_frames_path.resolve()),
                "sha256": _sha256(raw_frames_path),
            },
            "committee_features": {
                "path": str(committee_path.resolve()),
                "sha256": _sha256(committee_path),
            },
            "threshold_roles": {
                "path": str(roles_path.resolve()),
                "sha256": _sha256(roles_path),
            },
        },
        "executed_source_sha256": {
            geometry.EXECUTED_SOURCE_RELATIVE: _sha256(source_path)
        },
        "integrity": {"prepublish_rehash": "passed"},
        "counts": {
            "feature_rows": feature_rows,
            "role_assignment_rows": role_rows,
            "development_gate_rows": len(sids),
            "strict_rows": len(sids),
            "output_frames": len(sids),
            "total_atoms": 2 * len(sids),
            "raw_archive_file_members": feature_rows,
        },
        "sid_order_sha256": hashlib.sha256(
            ("\n".join(sorted(sids)) + "\n").encode("utf-8")
        ).hexdigest(),
        "outputs_sha256": {
            geometry.OUTPUT_ARCHIVE_NAME: _sha256(archive_path)
        },
        "scientific_improvement_claim": False,
    }
    manifest_path = output / geometry.MANIFEST_NAME
    _write_json(manifest_path, manifest)
    return archive_path, manifest_path


def _fake_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    gate_rows: int,
    negative_rows: int,
    reject_abstain_sid: str | None = None,
) -> dict[str, Path]:
    from src import next11_phsc_label_free_stop as module
    from src.next11_phsc_mattersim_features import (
        EXECUTED_SOURCE_RELATIVE as PHSC_SOURCE_RELATIVE,
        OUTPUT_COLUMNS,
    )

    tmp_path.mkdir(parents=True, exist_ok=True)
    assert tuple(OUTPUT_COLUMNS) == module.PHSC_FEATURE_COLUMNS
    if negative_rows > gate_rows:
        raise ValueError("negative_rows exceeds gate_rows")

    committee_rows = [
        _feature_row("search-a", "search-rk-a", "search_calibration"),
        _feature_row("search-b", "search-rk-b", "search_calibration"),
        _feature_row("selection-a", "selection-rk", "formula_selection"),
        _feature_row("fit-a", "fit-rk", "threshold_calibration"),
    ]
    gate_sids = [f"gate-{index:03d}" for index in range(gate_rows)]
    for sid in gate_sids:
        committee_rows.append(
            _feature_row(sid, f"rk-{sid}", "threshold_calibration")
        )
    if reject_abstain_sid is not None:
        # Give this row a positive within-group gap so the frozen 0.5 rule rejects it.
        target = next(row for row in committee_rows if row["sid"] == reject_abstain_sid)
        target["rk"] = "rk-reject"
        target["m1_energy_ev_per_atom"] = 2.0
        target["m5_energy_ev_per_atom"] = 2.0
        companion_sid = next(sid for sid in gate_sids if sid != reject_abstain_sid)
        companion = next(row for row in committee_rows if row["sid"] == companion_sid)
        companion["rk"] = "rk-reject"

    committee = pd.DataFrame(committee_rows)
    committee_path = tmp_path / "mattersim_committee_features.parquet"
    committee.to_parquet(committee_path, index=False)

    threshold = committee.loc[committee["stage"].eq("threshold_calibration")]
    roles = threshold[["sid", "rk", "stage"]].copy()
    roles["threshold_role"] = np.where(
        roles["sid"].eq("fit-a"), "threshold_fit", "development_gate"
    )
    roles["split_salt"] = THRESHOLD_SPLIT_SALT
    roles_path = tmp_path / "threshold_role_assignments.parquet"
    roles.to_parquet(roles_path, index=False)

    frames_path = tmp_path / "initial_frames.zip"
    frames_path.write_bytes(b"fake label-free frames")
    checkpoint_path = tmp_path / "mattersim-5m.pth"
    checkpoint_path.write_bytes(b"fake sealed checkpoint")
    geometry_archive_path, geometry_manifest_path = _write_geometry_artifact(
        tmp_path,
        sids=gate_sids,
        raw_frames_path=frames_path,
        committee_path=committee_path,
        roles_path=roles_path,
        feature_rows=len(committee),
        role_rows=len(roles),
    )
    committee_manifest_path = tmp_path / "committee-MANIFEST.json"
    committee_manifest = {
        "protocol": "2026-08-01-mattersim-dual-checkpoint-x0-v1",
        "mode": "development",
        "production_protocol_eligible": True,
        "outputs_sha256": {committee_path.name: _sha256(committee_path)},
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
    _write_json(committee_manifest_path, committee_manifest)

    search = committee.loc[committee["stage"].eq("search_calibration")].copy()
    cutoffs = derive_disagreement_cutoffs(search)
    serialized_catalog = serialize_formula_catalog(cutoffs)
    rules: list[dict[str, object]] = []
    for track, alpha, within in (
        ("primary", 0.01, "max"),
        ("comparator", 0.035, "min"),
    ):
        for role, formula in (("selected", "AGREE995"), ("m5_baseline", "M5")):
            rules.append(
                {
                    "role": role,
                    "formula": formula,
                    "track": track,
                    "threshold": 0.5,
                    "threshold_state": "finite",
                    "threshold_source_role": "threshold_fit",
                    "alpha": alpha,
                    "within_group": within,
                    "operator": "score > threshold",
                    "unsupported_decision": "ABSTAIN",
                }
            )
    protocol_path = tmp_path / "FROZEN_PROTOCOL.json"
    protocol = {
        "protocol": DEVELOPMENT_FREEZE_PROTOCOL,
        "state": "frozen",
        "catalog": {
            "serialized": serialized_catalog,
            "sha256": hashlib.sha256(serialized_catalog.encode()).hexdigest(),
        },
        "cutoff_provenance": {
            "catalog_serialization_sha256": hashlib.sha256(
                serialized_catalog.encode()
            ).hexdigest(),
            "feature_manifest_sha256": _sha256(committee_manifest_path),
            "feature_sha256": _sha256(committee_path),
            "protocol_code_sha256": _sha256(
                Path("src/next8_mattersim_committee_protocol.py")
            ),
        },
        "development_artifacts_sha256": {
            "threshold_role_assignments.parquet": _sha256(roles_path)
        },
        "selection": {"state": "selected", "name": "AGREE995"},
        "final_rules": rules,
        "split": {
            "development_gate_groups": int(
                roles.loc[roles["threshold_role"].eq("development_gate"), "rk"].nunique()
            ),
            "threshold_fit_groups": 1,
            "ordering": "sha256(salt+'\\0'+rk),rk",
            "salt": THRESHOLD_SPLIT_SALT,
        },
    }
    _write_json(protocol_path, protocol)

    phsc_rows: list[dict[str, object]] = []
    for index, sid in enumerate(gate_sids):
        rk = str(roles.loc[roles["sid"].eq(sid), "rk"].iloc[0])
        status = "resolved_negative" if index < negative_rows else "resolved_nonnegative"
        if sid == reject_abstain_sid:
            status = "abstain_numerical_failure"
        phsc_rows.append(_phsc_row(module, sid, rk, status=status))
    phsc = pd.DataFrame(phsc_rows, columns=module.PHSC_FEATURE_COLUMNS)
    phsc_path = tmp_path / "phsc_features.parquet"
    phsc.to_parquet(phsc_path, index=False)

    statuses = phsc["phsc_status"].astype(str)
    calls = phsc["force_call_count"].astype(int)
    coordinate_groups = int(calls.sum() // 4)
    remaining_groups = coordinate_groups
    predictor_batch_sizes: list[int] = []
    while remaining_groups:
        batch_groups = min(remaining_groups, 256)
        predictor_batch_sizes.append(4 * batch_groups)
        remaining_groups -= batch_groups
    predictor_calls = len(predictor_batch_sizes)
    phsc_manifest_path = tmp_path / "phsc-MANIFEST.json"
    repo_root = Path.cwd().resolve()
    executed_sources = {
        relative: _sha256(repo_root / relative) for relative in PHSC_SOURCE_RELATIVE
    }
    phsc_manifest = {
        "protocol": "2026-08-02-next11-phsc-mattersim-features-v1",
        "mode": "development_gate",
        "labels_opened": False,
        "selection": {
            "stage": "threshold_calibration",
            "threshold_role": "development_gate",
        },
        "input_isolation": {
            "geometry_only": True,
            "geometry_protocol": "2026-08-02-next11-geometry-only-frames-v1",
            "raw_x0_archive_opened": False,
            "endpoint_label_artifacts_opened": False,
        },
        "adapter": {
            "mode": "builtin_indexed_mattersim",
            "index_alignment": "sid_indexed_exact_one_to_one",
            "index_alignment_verified": True,
            "device": "cuda:0",
            "model_batch_size": 32,
            "groups_per_call": 256,
            "model_parameter_device": "cuda:0",
            "result_tensor_devices": ["cuda:0"],
            "evaluations": int(calls.sum()),
        },
        "predictor_loaded_checkpoint_sha256": _sha256(checkpoint_path),
        "production_protocol_eligible": True,
        "evidence_role": "label_free_phsc_feature_generation",
        "runtime": {
            "python_version": "3.11-test",
            "python_implementation": "CPython",
            "platform": "Linux-test",
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "ase_version": "test",
            "mattersim_version": "1.2.3",
            "device": "cuda:0",
            "torch_version": "test+cu",
            "cuda_available": True,
            "cuda_version": "test",
            "gpu_name": "test gpu",
        },
        "inputs_sha256": {
            "committee_features": {
                "path": str(committee_path.resolve()),
                "sha256": _sha256(committee_path),
            },
            "threshold_roles": {
                "path": str(roles_path.resolve()),
                "sha256": _sha256(roles_path),
            },
            "geometry_only_frames": {
                "path": str(geometry_archive_path.resolve()),
                "sha256": _sha256(geometry_archive_path),
            },
            "geometry_manifest": {
                "path": str(geometry_manifest_path.resolve()),
                "sha256": _sha256(geometry_manifest_path),
            },
            "source_frames_provenance": {
                "path": str(frames_path.resolve()),
                "sha256": _sha256(frames_path),
            },
            "feature_manifest": {
                "path": str(committee_manifest_path.resolve()),
                "sha256": _sha256(committee_manifest_path),
            },
            "checkpoint": {
                "path": str(checkpoint_path.resolve()),
                "sha256": _sha256(checkpoint_path),
            },
        },
        "executed_source_sha256": executed_sources,
        "integrity": {"prepublish_rehash": "passed"},
        "feature_columns": list(module.PHSC_FEATURE_COLUMNS),
        "criterion": {
            "name": "PHSC-v0",
            "scope": "fixed_cell_gamma_point_atomic_hessian",
            "step_fraction": 0.00390625,
            "probe_order": ["+h", "-h", "+h/2", "-h/2"],
            "force_evaluations_per_atom": 12,
            "primary_decision_proxy": "two_scale_projected_operator_difference",
            "numerical_consistency_proxies_are_confidence_bounds": False,
            "numerical_consistency_proxies_are_rigorous_error_bounds": False,
        },
        "formal_expectations": {
            "feature_rows": len(committee),
            "role_assignment_rows": len(roles),
            "selected_rows": len(phsc),
            "strict_rows": len(phsc),
            "nonstrict_rows": 0,
            "model_batch_size": 32,
            "groups_per_call": 256,
            "device_contract": "canonical_cuda:N",
            "checkpoint_sha256": _sha256(checkpoint_path),
            "geometry_protocol": "2026-08-02-next11-geometry-only-frames-v1",
            "geometry_only_frames_sha256": _sha256(geometry_archive_path),
            "geometry_manifest_sha256": _sha256(geometry_manifest_path),
        },
        "counts": {
            "feature_rows": len(committee),
            "role_assignment_rows": len(roles),
            "selected_rows": len(phsc),
            "strict_rows": len(phsc),
            "nonstrict_rows": 0,
            "probe_eligible_rows": int((calls > 0).sum()),
            "resolved_negative_rows": int(statuses.eq("resolved_negative").sum()),
            "resolved_nonnegative_rows": int(statuses.eq("resolved_nonnegative").sum()),
            "near_zero_or_inconsistent_rows": int(
                statuses.eq("near_zero_or_inconsistent").sum()
            ),
            "abstained_rows": int(statuses.str.startswith("abstain_").sum()),
            "coordinate_groups": coordinate_groups,
            "probe_evaluations": int(calls.sum()),
            "batch_predictor_calls": predictor_calls,
        },
        "execution": {
            "batch_predictor_calls": predictor_calls,
            "predictor_batch_sizes": predictor_batch_sizes,
            "max_predictor_batch_size": max(predictor_batch_sizes),
            "forward_calls": sum(
                (batch_size + 31) // 32 for batch_size in predictor_batch_sizes
            ),
            "peak_cuda_memory_bytes": 1024,
            "wall_time_seconds": 1.0,
        },
        "scientific_improvement_claim": False,
        "outputs_sha256": {phsc_path.name: _sha256(phsc_path)},
    }
    _write_json(phsc_manifest_path, phsc_manifest)

    monkeypatch.setattr(
        module,
        "FROZEN_INPUT_SHA256",
        MappingProxyType(
            {
                "committee_features": _sha256(committee_path),
                "committee_manifest": _sha256(committee_manifest_path),
                "threshold_roles": _sha256(roles_path),
                "frozen_protocol": _sha256(protocol_path),
            }
        ),
    )
    monkeypatch.setattr(module, "FROZEN_GATE_ROWS", gate_rows)
    monkeypatch.setattr(module, "FROZEN_STRICT_ROWS", gate_rows)
    monkeypatch.setattr(module, "FROZEN_NONSTRICT_ROWS", 0)
    monkeypatch.setattr(module, "FROZEN_FEATURE_ROWS", len(committee), raising=False)
    monkeypatch.setattr(module, "FROZEN_ROLE_ROWS", len(roles), raising=False)
    monkeypatch.setattr(
        module,
        "FROZEN_GEOMETRY_SHA256",
        MappingProxyType(
            {
                "geometry_only_frames": _sha256(geometry_archive_path),
                "geometry_manifest": _sha256(geometry_manifest_path),
            }
        ),
        raising=False,
    )
    monkeypatch.setattr(
        module, "FROZEN_RAW_ARCHIVE_FILE_MEMBERS", len(committee), raising=False
    )
    monkeypatch.setattr(
        module, "FROZEN_GEOMETRY_TOTAL_ATOMS", 2 * gate_rows, raising=False
    )
    monkeypatch.setattr(
        module,
        "PHSC_FROZEN_SOURCE_SHA256",
        MappingProxyType(executed_sources),
        raising=False,
    )
    return {
        "committee_features": committee_path,
        "committee_manifest": committee_manifest_path,
        "roles": roles_path,
        "phsc_features": phsc_path,
        "phsc_manifest": phsc_manifest_path,
        "protocol": protocol_path,
        "raw_frames": frames_path,
        "geometry_only_frames": geometry_archive_path,
        "geometry_manifest": geometry_manifest_path,
        "checkpoint": checkpoint_path,
    }


def _run(module: object, paths: dict[str, Path], output: Path) -> Path:
    return module.run_label_free_stop(
        committee_features_path=paths["committee_features"],
        committee_manifest_path=paths["committee_manifest"],
        role_assignments_path=paths["roles"],
        phsc_features_path=paths["phsc_features"],
        phsc_manifest_path=paths["phsc_manifest"],
        frozen_protocol_path=paths["protocol"],
        output_dir=output,
    )


def test_public_api_has_no_label_next8_manifest_or_baseline_metric_input() -> None:
    from src import next11_phsc_label_free_stop as module

    assert tuple(inspect.signature(module.run_label_free_stop).parameters) == (
        "committee_features_path",
        "committee_manifest_path",
        "role_assignments_path",
        "phsc_features_path",
        "phsc_manifest_path",
        "frozen_protocol_path",
        "output_dir",
    )
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    }
    assert "src.next10_lrrc_gate_diagnostic" not in imports
    source = Path(module.__file__).read_text(encoding="utf-8")
    assert "baseline_metrics_path" not in source
    assert "next8_manifest_path" not in source


@pytest.mark.parametrize(("negative_rows", "expected_pass"), [(65, False), (66, True)])
def test_frozen_65_66_boundary_is_label_free_and_exact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    negative_rows: int,
    expected_pass: bool,
) -> None:
    from src import next11_phsc_label_free_stop as module

    paths = _fake_artifacts(
        tmp_path, monkeypatch, gate_rows=70, negative_rows=negative_rows
    )
    output = tmp_path / f"stop-{negative_rows}"

    assert _run(module, paths, output) == output

    result = json.loads((output / module.RESULT_NAME).read_text(encoding="utf-8"))
    primary = result["policies"]["primary"]["M5"]
    assert primary["net_reject_delta"] == negative_rows
    assert primary["nonreject_to_reject"] == negative_rows
    assert primary["reject_to_nonreject"] == 0
    assert result["necessary_condition"] == {
        "cohort_rows": 70,
        "required_net_reject_delta": 66,
        "observed_primary_m5_net_reject_delta": negative_rows,
        "passes": expected_pass,
        "labels_opened": False,
    }
    assert sum(
        primary["transition_matrix"][source][target]
        for source in module.DECISIONS
        for target in module.DECISIONS
    ) == 70


def test_reject_to_abstain_is_deducted_and_all_nine_transitions_are_published(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src import next11_phsc_label_free_stop as module

    paths = _fake_artifacts(
        tmp_path,
        monkeypatch,
        gate_rows=70,
        negative_rows=66,
        reject_abstain_sid="gate-069",
    )
    output = tmp_path / "deducted"

    _run(module, paths, output)

    result = json.loads((output / module.RESULT_NAME).read_text(encoding="utf-8"))
    primary = result["policies"]["primary"]["M5"]
    assert primary["transition_matrix"]["REJECT"]["ABSTAIN"] == 1
    assert primary["reject_to_nonreject"] == 1
    assert primary["net_reject_delta"] == (
        primary["nonreject_to_reject"] - primary["reject_to_nonreject"]
    )
    assert set(primary["transition_matrix"]) == set(module.DECISIONS)
    assert all(
        set(primary["transition_matrix"][source]) == set(module.DECISIONS)
        for source in module.DECISIONS
    )


def test_only_allowlisted_parquets_are_opened_and_unexpected_label_argument_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src import next11_phsc_label_free_stop as module

    paths = _fake_artifacts(tmp_path, monkeypatch, gate_rows=4, negative_rows=0)
    allowed = {
        paths["committee_features"].read_bytes(),
        paths["roles"].read_bytes(),
        paths["phsc_features"].read_bytes(),
    }
    original = pd.read_parquet
    observed: list[bytes] = []

    def sentinel(source: object, *args: object, **kwargs: object) -> pd.DataFrame:
        if not hasattr(source, "read"):
            raise AssertionError("parquet parser received a path instead of snapshotted bytes")
        payload = source.read()
        assert payload in allowed, "attempted to parse a non-allowlisted parquet payload"
        observed.append(payload)
        from io import BytesIO

        return original(BytesIO(payload), *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", sentinel)
    _run(module, paths, tmp_path / "sentinel")
    assert set(observed) == allowed
    with pytest.raises(TypeError):
        module.run_label_free_stop(  # type: ignore[call-arg]
            committee_features_path=paths["committee_features"],
            committee_manifest_path=paths["committee_manifest"],
            role_assignments_path=paths["roles"],
            phsc_features_path=paths["phsc_features"],
            phsc_manifest_path=paths["phsc_manifest"],
            frozen_protocol_path=paths["protocol"],
            output_dir=tmp_path / "forbidden",
            labels_path=tmp_path / "forbidden-labels.parquet",
        )


def test_nonproduction_phsc_artifact_and_duplicate_json_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src import next11_phsc_label_free_stop as module

    paths = _fake_artifacts(tmp_path, monkeypatch, gate_rows=4, negative_rows=0)
    manifest = json.loads(paths["phsc_manifest"].read_text(encoding="utf-8"))
    manifest["production_protocol_eligible"] = False
    _write_json(paths["phsc_manifest"], manifest)
    monkeypatch.setattr(
        module,
        "FROZEN_INPUT_SHA256",
        MappingProxyType(
            {
                **dict(module.FROZEN_INPUT_SHA256),
                "committee_manifest": _sha256(paths["committee_manifest"]),
            }
        ),
    )
    with pytest.raises(ValueError, match="production"):
        _run(module, paths, tmp_path / "nonproduction")
    assert not (tmp_path / "nonproduction").exists()

    paths = _fake_artifacts(tmp_path / "duplicate", monkeypatch, gate_rows=4, negative_rows=0)
    duplicate = b'{"protocol":"x","protocol":"y"}'
    paths["phsc_manifest"].write_bytes(duplicate)
    with pytest.raises(ValueError, match="duplicate JSON key"):
        _run(module, paths, tmp_path / "duplicate-output")


@pytest.mark.parametrize(
    ("block", "field", "replacement"),
    [
        ("criterion", "primary_decision_proxy", "single_scale_eigenvalue"),
        ("formal_expectations", "groups_per_call", 255),
    ],
)
def test_phsc_manifest_requires_exact_frozen_scientific_contract_blocks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    block: str,
    field: str,
    replacement: object,
) -> None:
    from src import next11_phsc_label_free_stop as module

    paths = _fake_artifacts(tmp_path, monkeypatch, gate_rows=4, negative_rows=0)
    manifest = json.loads(paths["phsc_manifest"].read_text(encoding="utf-8"))
    manifest[block][field] = replacement
    _write_json(paths["phsc_manifest"], manifest)

    with pytest.raises(ValueError, match=block):
        _run(module, paths, tmp_path / "invalid-contract")
    assert not (tmp_path / "invalid-contract").exists()


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("d_star_angstrom", -1.5),
        ("h_angstrom", 0.01),
        ("e_num_ev_per_a2", -0.01),
        ("tau_alg_ev_per_a2", -1e-12),
        ("antisymmetric_norm_h_ev_per_a2", -0.01),
        ("antisymmetric_norm_h2_ev_per_a2", -0.01),
        ("acoustic_residual_h_ev_per_a2", -0.01),
        ("acoustic_residual_h2_ev_per_a2", -0.01),
        ("u_num_ev_per_a2", 1.25),
        ("l_num_ev_per_a2", 0.75),
    ],
)
def test_successful_phsc_rows_require_frozen_numerical_invariants(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    replacement: float,
) -> None:
    from src import next11_phsc_label_free_stop as module

    paths = _fake_artifacts(tmp_path, monkeypatch, gate_rows=4, negative_rows=0)
    table = pd.read_parquet(paths["phsc_features"])
    table.loc[0, field] = replacement
    _rewrite_phsc_table(paths, table)

    with pytest.raises(ValueError, match="PHSC.*numerical"):
        _run(module, paths, tmp_path / "invalid-numerics")
    assert not (tmp_path / "invalid-numerics").exists()


def test_successful_phsc_status_must_equal_frozen_classifier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src import next11_phsc_label_free_stop as module

    paths = _fake_artifacts(tmp_path, monkeypatch, gate_rows=4, negative_rows=0)
    table = pd.read_parquet(paths["phsc_features"])
    table.loc[0, "phsc_status"] = "resolved_negative"
    table.loc[0, "phsc_negative"] = True
    _rewrite_phsc_table(paths, table)
    manifest = json.loads(paths["phsc_manifest"].read_text(encoding="utf-8"))
    manifest["counts"]["resolved_negative_rows"] = 1
    manifest["counts"]["resolved_nonnegative_rows"] = 3
    _write_json(paths["phsc_manifest"], manifest)

    with pytest.raises(ValueError, match="frozen classifier"):
        _run(module, paths, tmp_path / "contradictory-status")


def test_phsc_manifest_requires_canonical_cuda_device_and_exact_forward_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src import next11_phsc_label_free_stop as module

    paths = _fake_artifacts(tmp_path, monkeypatch, gate_rows=4, negative_rows=0)
    manifest = json.loads(paths["phsc_manifest"].read_text(encoding="utf-8"))
    manifest["adapter"]["device"] = "cuda:00"
    manifest["adapter"]["model_parameter_device"] = "cuda:00"
    manifest["adapter"]["result_tensor_devices"] = ["cuda:00"]
    manifest["runtime"]["device"] = "cuda:00"
    _write_json(paths["phsc_manifest"], manifest)
    with pytest.raises(ValueError, match="canonical cuda:N"):
        _run(module, paths, tmp_path / "noncanonical-device")

    paths = _fake_artifacts(
        tmp_path / "forward", monkeypatch, gate_rows=4, negative_rows=0
    )
    manifest = json.loads(paths["phsc_manifest"].read_text(encoding="utf-8"))
    manifest["execution"]["forward_calls"] += 1
    _write_json(paths["phsc_manifest"], manifest)
    with pytest.raises(ValueError, match="forward-call"):
        _run(module, paths, tmp_path / "wrong-forward-calls")


@pytest.mark.parametrize(
    "mutation",
    [
        "nonpositive",
        "split_probe_group",
        "too_large",
        "wrong_length",
        "wrong_sum",
        "short_nonfinal",
        "wrong_max",
    ],
)
def test_phsc_execution_requires_exact_frozen_predictor_batch_sequence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    from src import next11_phsc_label_free_stop as module

    paths = _fake_artifacts(tmp_path, monkeypatch, gate_rows=70, negative_rows=0)
    manifest = json.loads(paths["phsc_manifest"].read_text(encoding="utf-8"))
    execution = manifest["execution"]
    sizes = execution["predictor_batch_sizes"]
    if mutation == "nonpositive":
        sizes[0] = 0
    elif mutation == "split_probe_group":
        sizes[0] = 1022
    elif mutation == "too_large":
        sizes[0] = 1028
    elif mutation == "wrong_length":
        sizes.append(4)
    elif mutation == "wrong_sum":
        sizes[-1] -= 4
    elif mutation == "short_nonfinal":
        sizes[0] -= 4
        sizes[-1] += 4
    elif mutation == "wrong_max":
        execution["max_predictor_batch_size"] -= 4
    else:  # pragma: no cover - parameterization is frozen above.
        raise AssertionError(mutation)
    _write_json(paths["phsc_manifest"], manifest)

    with pytest.raises(ValueError, match="predictor batch"):
        _run(module, paths, tmp_path / f"invalid-batches-{mutation}")


@pytest.mark.parametrize("mutation", ["missing", "mismatch"])
def test_phsc_sources_require_exact_prefrozen_sha_allowlist(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    from src import next11_phsc_label_free_stop as module

    paths = _fake_artifacts(tmp_path, monkeypatch, gate_rows=4, negative_rows=0)
    frozen = dict(module.PHSC_FROZEN_SOURCE_SHA256)
    selected = next(iter(frozen))
    if mutation == "missing":
        del frozen[selected]
    else:
        frozen[selected] = "0" * 64
    monkeypatch.setattr(
        module, "PHSC_FROZEN_SOURCE_SHA256", MappingProxyType(frozen)
    )

    with pytest.raises(ValueError, match="frozen PHSC source"):
        _run(module, paths, tmp_path / f"source-{mutation}")


def test_phsc_frozen_source_allowlist_is_the_reviewed_exact_mapping() -> None:
    from src import next11_phsc_label_free_stop as module

    assert dict(module.PHSC_FROZEN_SOURCE_SHA256) == {
        "src/next11_phsc_mattersim_features.py": (
            "35f3e4435034d5915c048a53bf7ff47aa5c404fec23f69f2656eeab4a55c3d19"
        ),
        "src/next11_phsc.py": (
            "0016d021e6109c224f33b938f20e28bc2fe5063170dc1af3362695dddc6c3fda"
        ),
        "src/next11_geometry_only_frames.py": (
            "2d8a9129140ff30258fdf6d40ccc1f9ecd5a1d4db50eb23278080ba58083d2cf"
        ),
        "src/next10_lrrc_mattersim_features.py": (
            "9de42b45e6b526dfe2807921dbd680229a887c1a0c1f0cee1b1ed9ff47da44f1"
        ),
        "src/next9_lrrc.py": (
            "16f70dbdcfbe17e45157be79db33077d81ffb2ea841c7a3fe13a308c347a1c90"
        ),
        "src/next8_mattersim_committee_features.py": (
            "32153365d4e22a253ddb1869d9cd9b0a2b658dff3475639a27e3fbe576317909"
        ),
        "src/next6_mattersim_baseline.py": (
            "fd874b08f17e489d438e57db984c711265b8a14236eeed58098c4914e94bfecb"
        ),
        "src/next6_wbm_build.py": (
            "3edb1e24bb515e9a4057658974836e71f19851840cfef8f6cd053d7016a16d9a"
        ),
        "src/next6_wbm_features.py": (
            "c6a71370a5108a562452c7670d72d364e521fa4017842725b9a661dcde65f55f"
        ),
        "src/next6_wbm_protocol.py": (
            "73a538df7bfa046d3aed791dd54b6a79923f9dc9c33f19196185e7bc4004e299"
        ),
    }
    assert set(module.PHSC_FROZEN_SOURCE_SHA256) == set(
        module.PHSC_EXECUTED_SOURCE_RELATIVE
    )


def test_stop_uses_sanitized_geometry_after_raw_archive_is_deleted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src import next11_phsc_label_free_stop as module

    paths = _fake_artifacts(tmp_path, monkeypatch, gate_rows=4, negative_rows=0)
    raw_record = {
        "path": str(paths["raw_frames"].resolve()),
        "sha256": _sha256(paths["raw_frames"]),
    }
    paths["raw_frames"].unlink()

    output = tmp_path / "raw-deleted"
    _run(module, paths, output)

    manifest = json.loads((output / module.MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["inputs_sha256"]["source_frames_provenance"] == raw_record
    assert "frames" not in manifest["inputs_sha256"]


@pytest.mark.parametrize("artifact", ["geometry_only_frames", "geometry_manifest"])
def test_tampered_sanitized_geometry_artifacts_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact: str,
) -> None:
    from src import next11_phsc_label_free_stop as module

    paths = _fake_artifacts(tmp_path, monkeypatch, gate_rows=4, negative_rows=0)
    paths[artifact].write_bytes(paths[artifact].read_bytes() + b"tamper")

    with pytest.raises(ValueError, match="geometry"):
        _run(module, paths, tmp_path / f"tampered-{artifact}")


def test_phsc_input_isolation_and_raw_provenance_are_exact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src import next11_phsc_label_free_stop as module

    paths = _fake_artifacts(tmp_path, monkeypatch, gate_rows=4, negative_rows=0)
    manifest = json.loads(paths["phsc_manifest"].read_text(encoding="utf-8"))
    manifest["input_isolation"]["raw_x0_archive_opened"] = True
    _write_json(paths["phsc_manifest"], manifest)
    with pytest.raises(ValueError, match="input isolation"):
        _run(module, paths, tmp_path / "raw-opened")

    paths = _fake_artifacts(
        tmp_path / "provenance", monkeypatch, gate_rows=4, negative_rows=0
    )
    manifest = json.loads(paths["phsc_manifest"].read_text(encoding="utf-8"))
    manifest["inputs_sha256"]["source_frames_provenance"]["sha256"] = "0" * 64
    _write_json(paths["phsc_manifest"], manifest)
    with pytest.raises(ValueError, match="source frames provenance"):
        _run(module, paths, tmp_path / "wrong-raw-provenance")


def test_geometry_manifest_raw_provenance_must_match_frozen_committee_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src import next11_phsc_label_free_stop as module

    paths = _fake_artifacts(tmp_path, monkeypatch, gate_rows=4, negative_rows=0)
    geometry_manifest = json.loads(
        paths["geometry_manifest"].read_text(encoding="utf-8")
    )
    geometry_manifest["inputs_sha256"]["raw_frames"]["sha256"] = "0" * 64
    _write_json(paths["geometry_manifest"], geometry_manifest)
    changed_sha = _sha256(paths["geometry_manifest"])
    phsc_manifest = json.loads(
        paths["phsc_manifest"].read_text(encoding="utf-8")
    )
    phsc_manifest["inputs_sha256"]["geometry_manifest"]["sha256"] = changed_sha
    phsc_manifest["formal_expectations"]["geometry_manifest_sha256"] = changed_sha
    _write_json(paths["phsc_manifest"], phsc_manifest)
    monkeypatch.setattr(
        module,
        "FROZEN_GEOMETRY_SHA256",
        MappingProxyType(
            {
                "geometry_only_frames": _sha256(paths["geometry_only_frames"]),
                "geometry_manifest": changed_sha,
            }
        ),
    )

    with pytest.raises(ValueError, match="geometry.*raw frames provenance"):
        _run(module, paths, tmp_path / "geometry-wrong-raw-provenance")


def test_manifest_hashes_outputs_and_second_publication_preserves_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src import next11_phsc_label_free_stop as module

    paths = _fake_artifacts(tmp_path, monkeypatch, gate_rows=4, negative_rows=0)
    output = tmp_path / "published"
    _run(module, paths, output)
    original = {path.name: path.read_bytes() for path in output.iterdir()}

    manifest = json.loads((output / module.MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["protocol"] == module.PROTOCOL
    assert manifest["labels_opened"] is False
    assert manifest["scientific_improvement_claim"] is False
    assert manifest["production_protocol_eligible"] is True
    assert manifest["outputs_sha256"] == {
        module.RESULT_NAME: _sha256(output / module.RESULT_NAME)
    }
    assert set(manifest["executed_source_sha256"]) == set(
        module.EXECUTED_SOURCE_RELATIVE
    )
    assert set(manifest["inputs_sha256"]) == {
        "committee_features",
        "committee_manifest",
        "threshold_roles",
        "phsc_features",
        "phsc_manifest",
        "frozen_protocol",
        "geometry_only_frames",
        "geometry_manifest",
        "source_frames_provenance",
        "checkpoint",
    }

    with pytest.raises(FileExistsError):
        _run(module, paths, output)
    assert {path.name: path.read_bytes() for path in output.iterdir()} == original
    assert not list(tmp_path.glob(".published.staging-*"))
