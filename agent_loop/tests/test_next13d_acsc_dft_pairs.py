"""Contracts for the label-free, blinded NEXT13d paired DFT queue."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

from ase import Atoms
import numpy as np
import pandas as pd
import pytest


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    nested = pd.DataFrame(
        [
            {"sid": "t-a", "rk": "Li2|O1", "m5_gap_ev_per_atom": 0.10, "m5_phsc_chsc_decision": "KEEP", "nested_three_scale_confirmed": True},
            {"sid": "t-b", "rk": "Li2|O1", "m5_gap_ev_per_atom": 0.20, "m5_phsc_chsc_decision": "KEEP", "nested_three_scale_confirmed": True},
            {"sid": "t-c", "rk": "Na1|Cl1", "m5_gap_ev_per_atom": 0.30, "m5_phsc_chsc_decision": "KEEP", "nested_three_scale_confirmed": True},
            {"sid": "old-reject", "rk": "Li2|O1", "m5_gap_ev_per_atom": 0.0, "m5_phsc_chsc_decision": "REJECT", "nested_three_scale_confirmed": True},
            {"sid": "c-a", "rk": "Li2|O1", "m5_gap_ev_per_atom": 0.11, "m5_phsc_chsc_decision": "KEEP", "nested_three_scale_confirmed": False},
            {"sid": "c-b", "rk": "Li2|O1", "m5_gap_ev_per_atom": 0.19, "m5_phsc_chsc_decision": "KEEP", "nested_three_scale_confirmed": False},
            {"sid": "c-c", "rk": "K1|Cl1", "m5_gap_ev_per_atom": 0.28, "m5_phsc_chsc_decision": "KEEP", "nested_three_scale_confirmed": False},
            {"sid": "near-zero", "rk": "Na1|Cl1", "m5_gap_ev_per_atom": 0.31, "m5_phsc_chsc_decision": "KEEP", "nested_three_scale_confirmed": False},
        ]
    )
    acsc = pd.DataFrame(
        [
            {"sid": "t-a", "natoms": 3, "acsc_status": "resolved_negative"},
            {"sid": "t-b", "natoms": 6, "acsc_status": "resolved_negative"},
            {"sid": "t-c", "natoms": 4, "acsc_status": "resolved_negative"},
            {"sid": "old-reject", "natoms": 3, "acsc_status": "resolved_negative"},
            {"sid": "c-a", "natoms": 3, "acsc_status": "resolved_nonnegative"},
            {"sid": "c-b", "natoms": 9, "acsc_status": "resolved_nonnegative"},
            {"sid": "c-c", "natoms": 4, "acsc_status": "resolved_nonnegative"},
            {"sid": "near-zero", "natoms": 4, "acsc_status": "near_zero_or_inconsistent"},
        ]
    )
    return nested, acsc


def test_matching_is_label_free_unique_and_hierarchical() -> None:
    from src.next13d_acsc_dft_pairs import select_blinded_pairs

    nested, acsc = _tables()
    pairs = select_blinded_pairs(nested, acsc)

    assert pairs["treatment_sid"].tolist() == ["t-a", "t-b", "t-c"]
    assert pairs["control_sid"].is_unique
    assert pairs.set_index("treatment_sid")["match_tier"].to_dict() == {
        "t-a": "same_rk_same_natoms",
        "t-b": "same_rk_diff_natoms",
        "t-c": "fallback_same_natoms_chemistry",
    }
    assert "old-reject" not in set(pairs["treatment_sid"])
    assert "near-zero" not in set(pairs["control_sid"])


def test_opaque_ids_are_deterministic_and_hide_role_and_sid() -> None:
    from src.next13d_acsc_dft_pairs import opaque_task_id

    first = opaque_task_id(pair_index=7, role="treatment", sid="elem-123")
    assert first == opaque_task_id(pair_index=7, role="treatment", sid="elem-123")
    assert first != opaque_task_id(pair_index=7, role="control", sid="elem-456")
    assert first.startswith("task-") and len(first) == 21
    assert "elem-123" not in first and "treatment" not in first


def _write_inputs(tmp_path: Path) -> dict[str, Path]:
    from src.next13_acsc_old_cohort import PROTOCOL as ACSC_PROTOCOL
    from src.next13c_acsc_nested_overlap import PROTOCOL as NESTED_PROTOCOL

    nested, acsc = _tables()
    nested_path = tmp_path / "nested.parquet"
    acsc_path = tmp_path / "acsc.parquet"
    nested.to_parquet(nested_path, index=False)
    acsc.to_parquet(acsc_path, index=False)
    nested_manifest = tmp_path / "nested-manifest.json"
    nested_manifest.write_text(
        json.dumps(
            {
                "protocol": NESTED_PROTOCOL,
                "labels_opened": False,
                "endpoint_artifacts_opened": False,
                "outputs_sha256": {nested_path.name: _sha(nested_path)},
            }
        ),
        encoding="utf-8",
    )
    acsc_manifest = tmp_path / "acsc-manifest.json"
    acsc_manifest.write_text(
        json.dumps(
            {
                "protocol": ACSC_PROTOCOL,
                "endpoint_artifacts_opened": False,
                "outputs_sha256": {acsc_path.name: _sha(acsc_path)},
            }
        ),
        encoding="utf-8",
    )
    frames = tmp_path / "frames.zip"
    frames.write_bytes(b"loader fixture")
    geometry_manifest = tmp_path / "geometry-manifest.json"
    geometry_manifest.write_text("{}", encoding="utf-8")
    return {
        "nested": nested_path,
        "nested_manifest": nested_manifest,
        "acsc": acsc_path,
        "acsc_manifest": acsc_manifest,
        "frames": frames,
        "geometry_manifest": geometry_manifest,
    }


def _potcars(tmp_path: Path) -> Path:
    root = tmp_path / "potcars"
    for potential, enmax in (("Cl", 350), ("K_sv", 300), ("Li_sv", 310), ("Na_pv", 320), ("O", 400)):
        directory = root / potential
        directory.mkdir(parents=True)
        (directory / "POTCAR").write_text(
            f"TITEL = PAW_PBE {potential} TEST\nENMAX = {enmax}; ENMIN = 1\nLICENSED-{potential}\n",
            encoding="utf-8",
        )
    return root


def test_local_standard_potcar_suffix_is_supported_without_copying_contents(tmp_path: Path) -> None:
    from src.next13d_acsc_dft_pairs import _paired_potcar_record

    root = tmp_path / "potcars"
    directory = root / "B"
    directory.mkdir(parents=True)
    source = directory / "POTCAR.B"
    source.write_text(
        "TITEL = PAW_GGA B TEST\nENMAX = 318.644; ENMIN = 238.983 eV\nLICENSED-B\n",
        encoding="utf-8",
    )
    record = _paired_potcar_record(root, "B", "B")
    assert record["source_relative"] == "B/POTCAR.B"
    assert record["enmax_ev"] == pytest.approx(318.644)
    assert record["source_sha256"] == record["content_sha256"] == _sha(source)


def _geometry_loader(*, archive_path: Path, manifest_path: Path, expected_sids: list[str]):
    del archive_path, manifest_path
    formulas = {
        "t-a": "Li2O", "t-b": "Li4O2", "t-c": "Na2Cl2",
        "c-a": "Li2O", "c-b": "Li6O3", "c-c": "K2Cl2",
    }
    structures = []
    for sid in sorted(expected_sids):
        symbols = formulas[sid]
        n = len(Atoms(symbols))
        structures.append(
            Atoms(symbols, scaled_positions=np.linspace(0.1, 0.8, n * 3).reshape(n, 3), cell=np.eye(3) * 6, pbc=True)
        )
    return sorted(expected_sids), structures


def test_queue_is_blinded_license_safe_and_fully_accounted(tmp_path: Path) -> None:
    from src import next13d_acsc_dft_pairs as module

    inputs = _write_inputs(tmp_path)
    output = tmp_path / "queue"
    manifest = module.build_paired_dft_queue(
        nested_features_path=inputs["nested"],
        nested_manifest_path=inputs["nested_manifest"],
        acsc_features_path=inputs["acsc"],
        acsc_manifest_path=inputs["acsc_manifest"],
        frames_zip_path=inputs["frames"],
        geometry_manifest_path=inputs["geometry_manifest"],
        potcar_root=_potcars(tmp_path),
        output_dir=output,
        potential_map={"Cl": "Cl", "K": "K_sv", "Li": "Li_sv", "Na": "Na_pv", "O": "O"},
        geometry_loader=_geometry_loader,
        expected_pair_count=3,
        expected_tier_counts=None,
    )
    blind = pd.read_parquet(output / module.BLINDED_QUEUE_NAME)
    private = pd.read_parquet(output / module.PRIVATE_PAIRS_NAME)
    assert len(blind) == 6 and blind.task_id.is_unique
    assert set(blind.columns).isdisjoint({"sid", "role", "acsc_status", "nested_three_scale_confirmed"})
    assert len(private) == 6 and set(private.role) == {"treatment", "control"}
    assert private.groupby("pair_id").size().eq(2).all()
    assert manifest["counts"]["pairs"] == 3
    assert manifest["counts"]["tasks"] == 6
    assert manifest["labels_opened"] is False
    assert manifest["dft_endpoints_opened"] is False
    assert manifest["licensed_potcar_contents_included"] is False

    archive_bytes = (output / module.TASKS_NAME).read_bytes()
    for forbidden in (b"t-a", b"t-b", b"t-c", b"c-a", b"c-b", b"c-c", b"treatment", b"control", b"LICENSED-"):
        assert forbidden not in archive_bytes
    with zipfile.ZipFile(output / module.TASKS_NAME) as archive:
        names = archive.namelist()
        assert len([name for name in names if name.endswith("/TASK.json")]) == 6
        assert not any(name.endswith("/POTCAR") for name in names)
        task = json.loads(archive.read(next(name for name in names if name.endswith("/TASK.json"))))
        assert set(task).isdisjoint({"sid", "role", "pair_id"})
        assert task["task_id"].startswith("task-")
    assert b"t-a" not in (output / module.BLINDED_QUEUE_NAME).read_bytes()


def test_queue_refuses_overwrite_and_cli_exposes_no_endpoint_input(tmp_path: Path) -> None:
    from src.next13d_acsc_dft_pairs import build_paired_dft_queue, main

    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        build_paired_dft_queue(
            nested_features_path=tmp_path / "x",
            nested_manifest_path=tmp_path / "x",
            acsc_features_path=tmp_path / "x",
            acsc_manifest_path=tmp_path / "x",
            frames_zip_path=tmp_path / "x",
            geometry_manifest_path=tmp_path / "x",
            potcar_root=tmp_path / "x",
            output_dir=existing,
        )
    for forbidden in ("--labels", "--dft-results", "--endpoint"):
        with pytest.raises(SystemExit) as exc_info:
            main([forbidden, "x"])
        assert exc_info.value.code == 2
