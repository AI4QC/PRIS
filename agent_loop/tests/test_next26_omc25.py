"""Contracts for the additive NEXT26 OMC25 source and endpoint reader."""

from __future__ import annotations

import json
from pathlib import Path
import zlib

from ase import Atoms
from ase.io.jsonio import encode
import lmdb
import numpy as np
import pandas as pd
import pytest


def _record(identity: tuple[str, int, str, str], frame: int, shift: float) -> dict:
    refcode, z_value, step, xtal_id = identity
    numbers = np.array([6, 1, 1], dtype=int)
    positions = np.array(
        [[0.0 + shift, 0.0, 0.0], [1.0 + shift, 0.0, 0.0], [0.0, 1.0, 0.0]]
    )
    return {
        "numbers": numbers,
        "positions": positions,
        "pbc": np.array([True, True, True]),
        "cell": np.eye(3) * (5.0 + shift),
        "energy": -10.0 - frame * 0.15,
        "forces": np.full((3, 3), 1.2 / (frame + 1)),
        "stress": np.full(6, 0.02 / (frame + 1)),
        "data": {
            "sid": f"{refcode}-{z_value}-{step}-{xtal_id}-{frame}",
            "csd_refcode": refcode,
            "z_value": z_value,
            "genarris_step": step,
            "xtal.id": xtal_id,
            "source": f"root/{refcode}/{z_value}/{step}/{xtal_id}",
        },
    }


def _source(tmp_path: Path) -> tuple[Path, Path]:
    identities = [
        ("AAAAAA", 1, "gener", "abc"),
        ("BBBBBB", 2, "press", "def"),
        ("CCCCCC", 1, "gener", "ghi"),
    ]
    db = tmp_path / "fixture.aselmdb"
    env = lmdb.open(str(db), subdir=False, map_size=1 << 24)
    key = 1
    with env.begin(write=True) as txn:
        for identity, count in zip(identities, [3, 3, 2], strict=True):
            for frame in range(count):
                payload = encode(_record(identity, frame, frame * 0.1)).encode()
                txn.put(str(key).encode(), zlib.compress(payload))
                key += 1
        txn.put(b"nextid", zlib.compress(json.dumps(key).encode()))
    env.close()
    catalogue = tmp_path / "catalogue.csv"
    pd.DataFrame(
        {
            "csd_refcode": [x[0] for x in identities],
            "z_value": [x[1] for x in identities],
            "genarris_step": [x[2] for x in identities],
            "xtal.id": [x[3] for x in identities],
            "split": ["val"] * 3,
            "nframes": [3, 3, 3],
            "mol.composition": ["C H2"] * 3,
            "xtal.composition": ["C H2"] * 3,
            "mol.natoms": [3] * 3,
            "xtal.natoms": [3] * 3,
        }
    ).to_csv(catalogue)
    return db, catalogue


def test_metadata_scanner_selects_only_complete_chronological_trajectories(
    tmp_path: Path,
) -> None:
    from src.next26_omc25 import scan_complete_trajectories

    db, catalogue = _source(tmp_path)
    result = scan_complete_trajectories(db_path=db, catalogue_path=catalogue)

    assert [row.material_id for row in result] == [
        "AAAAAA-1-gener-abc",
        "BBBBBB-2-press-def",
    ]
    assert [row.frame_indices for row in result] == [(0, 1, 2), (0, 1, 2)]
    assert [row.lmdb_keys for row in result] == [(1, 2, 3), (4, 5, 6)]


def test_x0_sanitizer_projects_geometry_and_never_exports_dft_fields(
    tmp_path: Path,
) -> None:
    from src.next11_geometry_only_frames import _load_archive_only
    from src.next26_omc25 import GEOMETRY_NAME, MANIFEST_NAME, METADATA_NAME, sanitize_x0

    db, catalogue = _source(tmp_path)
    output = tmp_path / "x0"
    manifest = sanitize_x0(
        db_path=db,
        catalogue_path=catalogue,
        output_dir=output,
        exclude_refcodes={"BBBBBB"},
    )

    metadata = pd.read_parquet(output / METADATA_NAME)
    assert metadata.material_id.tolist() == ["AAAAAA-1-gener-abc"]
    assert metadata.columns.tolist() == [
        "material_id",
        "csd_refcode",
        "z_value",
        "genarris_step",
        "xtal_id",
        "natoms",
        "input_role",
    ]
    _, frames = _load_archive_only(output / GEOMETRY_NAME, tuple(metadata.material_id))
    assert len(frames) == 1
    np.testing.assert_array_equal(frames[0].positions, _record(("AAAAAA", 1, "gener", "abc"), 0, 0)["positions"])
    assert frames[0].calc is None
    assert not frames[0].info
    payload = (output / GEOMETRY_NAME).read_bytes().lower()
    for token in (b"energy", b"force", b"stress", b"dft", b"relax"):
        assert token not in payload
    assert manifest["endpoint_numeric_fields_parsed"] is False
    assert manifest["labels_opened"] is False
    assert (output / MANIFEST_NAME).is_file()
    with pytest.raises(FileExistsError):
        sanitize_x0(db_path=db, catalogue_path=catalogue, output_dir=output)


def test_endpoint_builder_uses_first_and_last_frames_and_flags_severe_response(
    tmp_path: Path,
) -> None:
    from src.next26_omc25 import build_endpoint_table, severe_dft_response

    db, catalogue = _source(tmp_path)
    result = build_endpoint_table(db_path=db, catalogue_path=catalogue)

    assert result.material_id.tolist() == [
        "AAAAAA-1-gener-abc",
        "BBBBBB-2-press-def",
    ]
    row = result.iloc[0]
    assert row.energy_drop_pa == pytest.approx(0.1)
    assert row.force0_max == pytest.approx(np.sqrt(3) * 1.2)
    assert row.force1_max == pytest.approx(np.sqrt(3) * 0.4)
    assert row.frame_first == 0 and row.frame_last == 2
    assert bool(severe_dft_response(result).iloc[0]) is True


def test_geometry_projection_rejects_nonperiodic_or_malformed_records() -> None:
    from src.next26_omc25 import project_x0_payload

    record = _record(("AAAAAA", 1, "gener", "abc"), 0, 0)
    record["pbc"] = np.array([True, False, True])
    with pytest.raises(ValueError, match="periodic"):
        project_x0_payload(encode(record).encode())
