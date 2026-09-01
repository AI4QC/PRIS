"""Contracts for the additive next11 geometry-only x0 archive."""

from __future__ import annotations

import ast
import hashlib
import inspect
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import pytest


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _raw_frame(
    symbol: str,
    *,
    material: str,
    offset: float = 0.0,
    pbc: str = "T T T",
) -> str:
    return f'''2
material={material} endpoint_label=DO_NOT_COPY_{material} energy=-9876.543 Lattice="8 0 0 0 9 0 0 0 10" pbc="{pbc}" Properties=species:S:1:pos:R:3:forces:R:3:charge:R:1
{symbol} {1.1 + offset} 1.3 1.7 111 112 113 991
{symbol} {2.5 + offset} 1.3 1.7 211 212 213 992
'''


def _inputs(tmp_path: Path) -> dict[str, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    features = tmp_path / "mattersim_committee_features.parquet"
    pd.DataFrame(
        {
            "sid": ["sid-b", "sid-fit", "sid-a", "sid-ns"],
            "rk": ["rk-b", "rk-fit", "rk-a", "rk-ns"],
            "stage": ["threshold_calibration"] * 4,
            "strict_x0_ok": [True, True, True, False],
            "endpoint_label_forbidden": [
                "FEATURE_SECRET_B",
                "FEATURE_SECRET_FIT",
                "FEATURE_SECRET_A",
                "FEATURE_SECRET_NS",
            ],
        }
    ).to_parquet(features, index=False)

    roles = tmp_path / "threshold_role_assignments.parquet"
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
    ).to_parquet(roles, index=False)

    frames = tmp_path / "elementa_initial_frames.zip"
    with zipfile.ZipFile(frames, "w") as archive:
        archive.writestr(
            "nested/sid-b.extxyz",
            _raw_frame("H", material="RAW_SECRET_B", offset=0.25),
        )
        archive.writestr(
            "sid-a.extxyz",
            _raw_frame("He", material="RAW_SECRET_A", offset=-0.125),
        )
        archive.writestr(
            "sid-fit.extxyz",
            _raw_frame("Li", material="RAW_SECRET_FIT"),
        )
    return {"features": features, "roles": roles, "frames": frames}


def _strict_frame(
    *,
    comment_suffix: str = "",
    properties: str = "species:S:1:pos:R:3",
    atom_suffix: str = "",
    pbc: str = "T T T",
) -> str:
    return f'''2
Lattice="8 0 0 0 9 0 0 0 10" Properties={properties} pbc="{pbc}"{comment_suffix}
H 1.1 1.3 1.7{atom_suffix}
H 2.5 1.3 1.7{atom_suffix}
'''


def _write_manual_artifact(
    tmp_path: Path,
    module,
    *,
    frames: dict[str, str],
) -> tuple[Path, Path]:
    archive_path = tmp_path / module.OUTPUT_ARCHIVE_NAME
    with zipfile.ZipFile(
        archive_path, "w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for sid, text in sorted(frames.items()):
            info = zipfile.ZipInfo(f"{sid}.extxyz", (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, text, compresslevel=9)
    source_path = Path(module.__file__).resolve()
    sid_order = sorted(frames)
    manifest = {
        "protocol": module.PROTOCOL,
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
        "geometry_schema": dict(module.GEOMETRY_SCHEMA),
        "dropped_field_names": {"comment": [], "atom_properties": []},
        "inputs_sha256": {
            "raw_frames": {"path": "/sealed/raw.zip", "sha256": "a" * 64},
            "committee_features": {
                "path": "/sealed/features.parquet",
                "sha256": "b" * 64,
            },
            "threshold_roles": {
                "path": "/sealed/roles.parquet",
                "sha256": "c" * 64,
            },
        },
        "executed_source_sha256": {
            module.EXECUTED_SOURCE_RELATIVE: _sha256(source_path)
        },
        "integrity": {"prepublish_rehash": "passed"},
        "counts": {
            "feature_rows": len(frames),
            "role_assignment_rows": len(frames),
            "development_gate_rows": len(frames),
            "strict_rows": len(frames),
            "output_frames": len(frames),
            "total_atoms": 2 * len(frames),
            "raw_archive_file_members": len(frames),
        },
        "sid_order_sha256": hashlib.sha256(
            ("\n".join(sid_order) + "\n").encode("utf-8")
        ).hexdigest(),
        "outputs_sha256": {
            module.OUTPUT_ARCHIVE_NAME: _sha256(archive_path)
        },
        "scientific_improvement_claim": False,
    }
    manifest_path = tmp_path / module.MANIFEST_NAME
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return archive_path, manifest_path


def test_build_is_deterministic_geometry_only_and_records_names_not_values(
    tmp_path: Path,
) -> None:
    from src import next11_geometry_only_frames as module

    paths = _inputs(tmp_path)
    output_a = tmp_path / "geometry-a"
    output_b = tmp_path / "geometry-b"

    manifest_a = module.build_geometry_only_frames(
        raw_frames_zip_path=paths["frames"],
        committee_features_path=paths["features"],
        role_assignments_path=paths["roles"],
        output_dir=output_a,
    )
    manifest_b = module.build_geometry_only_frames(
        raw_frames_zip_path=paths["frames"],
        committee_features_path=paths["features"],
        role_assignments_path=paths["roles"],
        output_dir=output_b,
    )

    archive_a = output_a / module.OUTPUT_ARCHIVE_NAME
    archive_b = output_b / module.OUTPUT_ARCHIVE_NAME
    manifest_path_a = output_a / module.MANIFEST_NAME
    manifest_path_b = output_b / module.MANIFEST_NAME
    assert archive_a.read_bytes() == archive_b.read_bytes()
    assert manifest_path_a.read_bytes() == manifest_path_b.read_bytes()
    assert manifest_a == manifest_b

    with zipfile.ZipFile(archive_a) as archive:
        assert archive.namelist() == ["sid-a.extxyz", "sid-b.extxyz"]
        payload = b"".join(archive.read(name) for name in archive.namelist())
        assert archive.comment == b""
        for info in archive.infolist():
            assert info.extra == b""
            assert info.comment == b""
            assert info.date_time == (1980, 1, 1, 0, 0, 0)
        for name in archive.namelist():
            lines = archive.read(name).decode("utf-8").splitlines()
            assert lines[1].split(' Properties="', 1)[1].split('"', 1)[0] == (
                "species:S:1:pos:R:3"
            )
            assert lines[1].count("=") == 3
            assert all(len(line.split()) == 4 for line in lines[2:])
    for forbidden in (
        b"DO_NOT_COPY",
        b"RAW_SECRET",
        b"FEATURE_SECRET",
        b"-9876.543",
        b" 111 ",
        b" 991",
    ):
        assert forbidden not in archive_a.read_bytes()
        assert forbidden not in payload
        assert forbidden not in manifest_path_a.read_bytes()

    assert manifest_a["protocol"] == module.PROTOCOL
    assert manifest_a["endpoint_label_artifacts_opened"] is False
    assert manifest_a["raw_x0_archive_bytes_read"] is True
    assert manifest_a["raw_x0_nongeometry_values_converted_or_exported"] is False
    assert manifest_a["scientific_improvement_claim"] is False
    assert manifest_a["dropped_field_names"] == {
        "comment": ["endpoint_label", "energy", "material"],
        "atom_properties": ["charge", "forces"],
    }
    assert manifest_a["geometry_schema"] == module.GEOMETRY_SCHEMA
    assert manifest_a["counts"] == {
        "feature_rows": 4,
        "role_assignment_rows": 4,
        "development_gate_rows": 3,
        "strict_rows": 2,
        "output_frames": 2,
        "total_atoms": 4,
        "raw_archive_file_members": 3,
    }
    assert set(manifest_a["inputs_sha256"]) == {
        "raw_frames",
        "committee_features",
        "threshold_roles",
    }
    assert manifest_a["outputs_sha256"] == {
        module.OUTPUT_ARCHIVE_NAME: _sha256(archive_a)
    }
    assert manifest_a["executed_source_sha256"] == {
        module.EXECUTED_SOURCE_RELATIVE: _sha256(Path(module.__file__).resolve())
    }


def test_loader_enforces_manifest_closure_exact_coverage_and_clean_atoms(
    tmp_path: Path,
) -> None:
    from src import next11_geometry_only_frames as module

    paths = _inputs(tmp_path)
    output_dir = tmp_path / "geometry"
    module.build_geometry_only_frames(
        raw_frames_zip_path=paths["frames"],
        committee_features_path=paths["features"],
        role_assignments_path=paths["roles"],
        output_dir=output_dir,
    )
    archive_path = output_dir / module.OUTPUT_ARCHIVE_NAME
    manifest_path = output_dir / module.MANIFEST_NAME

    assert module.validate_geometry_only_archive(
        archive_path=archive_path,
        manifest_path=manifest_path,
        expected_sids=["sid-b", "sid-a"],
    ) == ("sid-a", "sid-b")
    sids, structures = module.load_geometry_only_archive(
        archive_path=archive_path,
        manifest_path=manifest_path,
        expected_sids=["sid-b", "sid-a"],
    )
    assert sids == ["sid-a", "sid-b"]
    assert [atoms.get_chemical_symbols() for atoms in structures] == [
        ["He", "He"],
        ["H", "H"],
    ]
    assert [sorted(atoms.arrays) for atoms in structures] == [
        ["numbers", "positions"],
        ["numbers", "positions"],
    ]
    assert all(atoms.calc is None and atoms.info == {} for atoms in structures)
    assert all(np.array_equal(atoms.pbc, [True, True, True]) for atoms in structures)

    with pytest.raises(ValueError, match="exact sid coverage"):
        module.load_geometry_only_archive(
            archive_path=archive_path,
            manifest_path=manifest_path,
            expected_sids=["sid-a"],
        )


@pytest.mark.parametrize(
    ("frame", "match"),
    [
        (_strict_frame(comment_suffix=" energy=-1"), "comment keys"),
        (
            _strict_frame(
                properties="species:S:1:pos:R:3:forces:R:3",
                atom_suffix=" 0 0 0",
            ),
            "Properties",
        ),
        (_strict_frame() + _strict_frame(), "single frame"),
        (_strict_frame(pbc="T T F"), "fully periodic"),
    ],
)
def test_validator_rejects_noncanonical_or_multiframe_members(
    tmp_path: Path,
    frame: str,
    match: str,
) -> None:
    from src import next11_geometry_only_frames as module

    archive_path, manifest_path = _write_manual_artifact(
        tmp_path, module, frames={"sid-a": frame}
    )
    with pytest.raises(ValueError, match=match):
        module.load_geometry_only_archive(
            archive_path=archive_path,
            manifest_path=manifest_path,
            expected_sids=["sid-a"],
        )


def test_validator_rejects_nested_member_and_manifest_or_archive_tampering(
    tmp_path: Path,
) -> None:
    from src import next11_geometry_only_frames as module

    archive_path, manifest_path = _write_manual_artifact(
        tmp_path, module, frames={"sid-a": _strict_frame()}
    )
    original_manifest = manifest_path.read_text("utf-8")

    manifest_path.write_text(
        original_manifest.replace(
            '"scientific_improvement_claim": false',
            '"scientific_improvement_claim": false, "protocol": "duplicate"',
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate key"):
        module.validate_geometry_only_archive(
            archive_path=archive_path,
            manifest_path=manifest_path,
            expected_sids=["sid-a"],
        )

    manifest_path.write_text(original_manifest, encoding="utf-8")
    archive_path.write_bytes(archive_path.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="archive hash"):
        module.validate_geometry_only_archive(
            archive_path=archive_path,
            manifest_path=manifest_path,
            expected_sids=["sid-a"],
        )

    nested_dir = tmp_path / "nested"
    nested_dir.mkdir()
    nested_archive, nested_manifest = _write_manual_artifact(
        nested_dir, module, frames={"nested/sid-a": _strict_frame()}
    )
    with pytest.raises(ValueError, match="root-level"):
        module.validate_geometry_only_archive(
            archive_path=nested_archive,
            manifest_path=nested_manifest,
            expected_sids=["sid-a"],
        )


def test_validator_rejects_noncanonical_unix_member_metadata(tmp_path: Path) -> None:
    from src import next11_geometry_only_frames as module

    archive_path, manifest_path = _write_manual_artifact(
        tmp_path, module, frames={"sid-a": _strict_frame()}
    )
    with zipfile.ZipFile(archive_path) as archive:
        payload = archive.read("sid-a.extxyz")
    with zipfile.ZipFile(archive_path, "w") as archive:
        info = zipfile.ZipInfo("sid-a.extxyz", (1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        info.create_system = 0
        info.external_attr = 0
        archive.writestr(info, payload, compresslevel=9)
    manifest = json.loads(manifest_path.read_text("utf-8"))
    manifest["outputs_sha256"][module.OUTPUT_ARCHIVE_NAME] = _sha256(archive_path)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="metadata"):
        module.validate_geometry_only_archive(
            archive_path=archive_path,
            manifest_path=manifest_path,
            expected_sids=["sid-a"],
        )


@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        (
            lambda paths: pd.DataFrame(
                {
                    "sid": ["sid-a", "sid-a"],
                    "rk": ["rk-a", "rk-a"],
                    "stage": ["threshold_calibration"] * 2,
                    "threshold_role": ["development_gate"] * 2,
                }
            ).to_parquet(paths["roles"], index=False),
            "unique",
        ),
        (
            lambda paths: pd.DataFrame(
                {
                    "sid": ["sid-a"],
                    "rk": ["wrong"],
                    "stage": ["threshold_calibration"],
                    "threshold_role": ["development_gate"],
                }
            ).to_parquet(paths["roles"], index=False),
            "rk values differ",
        ),
    ],
)
def test_selection_is_fail_closed_and_publishes_nothing(
    tmp_path: Path,
    mutator,
    match: str,
) -> None:
    from src import next11_geometry_only_frames as module

    paths = _inputs(tmp_path)
    mutator(paths)
    output_dir = tmp_path / "geometry"
    with pytest.raises(ValueError, match=match):
        module.build_geometry_only_frames(
            raw_frames_zip_path=paths["frames"],
            committee_features_path=paths["features"],
            role_assignments_path=paths["roles"],
            output_dir=output_dir,
        )
    assert not output_dir.exists()
    assert not list(tmp_path.glob(".geometry.staging-*"))


def test_build_rejects_multiframe_target_duplicate_sid_and_preexisting_output(
    tmp_path: Path,
) -> None:
    from src import next11_geometry_only_frames as module

    paths = _inputs(tmp_path)
    with zipfile.ZipFile(paths["frames"], "a") as archive:
        archive.writestr("other/sid-a.extxyz", _raw_frame("He", material="duplicate"))
    output_dir = tmp_path / "geometry"
    with pytest.raises(ValueError, match="duplicate member stem sid"):
        module.build_geometry_only_frames(
            raw_frames_zip_path=paths["frames"],
            committee_features_path=paths["features"],
            role_assignments_path=paths["roles"],
            output_dir=output_dir,
        )
    assert not output_dir.exists()

    clean = _inputs(tmp_path / "clean")
    output_dir.mkdir()
    marker = output_dir / "keep.txt"
    marker.write_text("user data", encoding="utf-8")
    with pytest.raises(FileExistsError):
        module.build_geometry_only_frames(
            raw_frames_zip_path=clean["frames"],
            committee_features_path=clean["features"],
            role_assignments_path=clean["roles"],
            output_dir=output_dir,
        )
    assert marker.read_text("utf-8") == "user data"


def test_dangling_symlink_output_is_rejected_before_inputs_are_opened(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from src import next11_geometry_only_frames as module

    target = tmp_path / "geometry"
    target.symlink_to(tmp_path / "missing-target", target_is_directory=True)

    def forbidden_snapshot(_path: Path):
        raise AssertionError("inputs must not be opened for a preexisting target")

    monkeypatch.setattr(module, "_snapshot", forbidden_snapshot)
    with pytest.raises(FileExistsError):
        module.build_geometry_only_frames(
            raw_frames_zip_path=tmp_path / "raw.zip",
            committee_features_path=tmp_path / "features.parquet",
            role_assignments_path=tmp_path / "roles.parquet",
            output_dir=target,
        )


def test_raw_parser_obeys_property_schema_and_rejects_invalid_geometry(
    tmp_path: Path,
) -> None:
    from src import next11_geometry_only_frames as module

    paths = _inputs(tmp_path)
    reordered = '''2
energy=-7 pbc="T T T" Properties=forces:R:3:pos:R:3:species:S:1 Lattice="8 0 0 0 9 0 0 0 10"
8 9 10 1.1 1.3 1.7 H
18 19 20 2.5 1.3 1.7 H
'''
    with zipfile.ZipFile(paths["frames"], "w") as archive:
        archive.writestr("sid-a.extxyz", reordered)
        archive.writestr("sid-b.extxyz", _raw_frame("H", material="b"))
    output_dir = tmp_path / "geometry"
    module.build_geometry_only_frames(
        raw_frames_zip_path=paths["frames"],
        committee_features_path=paths["features"],
        role_assignments_path=paths["roles"],
        output_dir=output_dir,
    )
    sids, structures = module.load_geometry_only_archive(
        archive_path=output_dir / module.OUTPUT_ARCHIVE_NAME,
        manifest_path=output_dir / module.MANIFEST_NAME,
        expected_sids=["sid-a", "sid-b"],
    )
    assert sids == ["sid-a", "sid-b"]
    assert np.array_equal(structures[0].positions[0], [1.1, 1.3, 1.7])

    bad = _inputs(tmp_path / "bad")
    with zipfile.ZipFile(bad["frames"], "w") as archive:
        archive.writestr(
            "sid-a.extxyz",
            _raw_frame("He", material="a").replace("1.1 1.3", "nan 1.3", 1),
        )
        archive.writestr("sid-b.extxyz", _raw_frame("H", material="b"))
    with pytest.raises(ValueError, match="finite"):
        module.build_geometry_only_frames(
            raw_frames_zip_path=bad["frames"],
            committee_features_path=bad["features"],
            role_assignments_path=bad["roles"],
            output_dir=tmp_path / "bad-output",
        )


def test_public_surface_has_no_endpoint_label_input_and_cli_uses_only_x0_inputs() -> None:
    from src import next11_geometry_only_frames as module

    parameters = inspect.signature(module.build_geometry_only_frames).parameters
    assert tuple(parameters) == (
        "raw_frames_zip_path",
        "committee_features_path",
        "role_assignments_path",
        "output_dir",
    )
    assert not any("label" in name.lower() or "endpoint" in name.lower() for name in parameters)
    source = Path(module.__file__).read_text("utf-8")
    tree = ast.parse(source)
    string_literals = {
        node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert "elementa_labels.parquet" not in string_literals
    assert not any(
        isinstance(node, (ast.Import, ast.ImportFrom))
        and "next6_elementa_initial" in ast.unparse(node)
        for node in ast.walk(tree)
    )


def test_manifest_top_level_schema_is_exact(tmp_path: Path) -> None:
    from src import next11_geometry_only_frames as module

    paths = _inputs(tmp_path)
    output_dir = tmp_path / "geometry"
    manifest = module.build_geometry_only_frames(
        raw_frames_zip_path=paths["frames"],
        committee_features_path=paths["features"],
        role_assignments_path=paths["roles"],
        output_dir=output_dir,
    )
    assert set(manifest) == {
        "protocol",
        "mode",
        "endpoint_label_artifacts_opened",
        "raw_x0_archive_bytes_read",
        "raw_x0_nongeometry_values_converted_or_exported",
        "input_role",
        "selection",
        "geometry_schema",
        "dropped_field_names",
        "inputs_sha256",
        "executed_source_sha256",
        "integrity",
        "counts",
        "sid_order_sha256",
        "outputs_sha256",
        "scientific_improvement_claim",
    }
