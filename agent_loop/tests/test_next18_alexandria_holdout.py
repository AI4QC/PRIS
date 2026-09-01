"""Contracts for the geometry-only Alexandria two-shard holdout."""

from __future__ import annotations

import bz2
import hashlib
import json
from pathlib import Path
import zipfile

from pymatgen.core import Lattice, Structure
import pytest


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _calculation(structure: Structure, endpoint_energy: float) -> list[dict]:
    moved = structure.copy()
    moved.translate_sites([0], [0.01, 0.0, 0.0], frac_coords=False)
    return [
        {
            "PREC": "Accurate",
            "ENMAX": 520,
            "kpoints": {"grid": [2, 2, 2]},
            "steps": [
                {
                    "structure": structure.as_dict(),
                    "energy": 123456.789,
                    "forces": [[9.0, 9.0, 9.0]] * len(structure),
                    "stress": [8.0] * 6,
                },
                {
                    "structure": moved.as_dict(),
                    "energy": endpoint_energy,
                    "forces": [[0.0, 0.0, 0.0]] * len(structure),
                    "stress": [0.0] * 6,
                },
            ],
        }
    ]


def _write_shard(path: Path, rows: dict[str, list[dict]]) -> None:
    with bz2.open(path, "wt", encoding="utf-8") as stream:
        json.dump(rows, stream, sort_keys=True)


def _inputs(tmp_path: Path) -> tuple[Path, Path]:
    lattice = Lattice.cubic(5.0)
    li2o = Structure(lattice, ["Li", "Li", "O"], [[0, 0, 0], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25]])
    nacl = Structure(lattice, ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    mgo = Structure(lattice, ["Mg", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    shard0 = tmp_path / "pbe_0000.json.bz2"
    shard1 = tmp_path / "pbe_0001.json.bz2"
    _write_shard(
        shard0,
        {
            "a": _calculation(li2o, -10.0),
            "b": _calculation(nacl, -8.0),
            "singleton": _calculation(mgo, -7.0),
        },
    )
    _write_shard(
        shard1,
        {
            "c": _calculation(li2o, -9.0),
            "d": _calculation(nacl, -7.0),
        },
    )
    return shard0, shard1


def test_streaming_reader_yields_top_level_records(tmp_path: Path) -> None:
    from src.next18_alexandria_holdout import iter_bz2_object

    shard0, _ = _inputs(tmp_path)
    rows = list(iter_bz2_object(shard0))
    assert [key for key, _ in rows] == ["a", "b", "singleton"]
    assert all(isinstance(value, list) for _, value in rows)


def test_holdout_selects_all_repeated_compositions_and_strips_endpoints(tmp_path: Path) -> None:
    from src import next18_alexandria_holdout as module

    shard0, shard1 = _inputs(tmp_path)
    target = tmp_path / "holdout"
    manifest = module.build_alexandria_holdout(
        shard_0000_path=shard0,
        shard_0001_path=shard1,
        output_dir=target,
        require_formal_inputs=False,
        expected_source_rows=5,
        expected_source_groups=3,
        expected_selected_groups=2,
        expected_selected_rows=4,
    )
    assert manifest["counts"]["selected_groups"] == 2
    assert manifest["counts"]["selected_rows"] == 4
    assert manifest["raw_container_endpoint_bytes_present"] is True
    assert manifest["endpoint_fields_accessed_by_sanitizer"] is False
    assert manifest["fresh_never_read_lockbox"] is False
    with zipfile.ZipFile(target / module.GEOMETRY_NAME) as archive:
        assert archive.namelist() == ["a.extxyz", "b.extxyz", "c.extxyz", "d.extxyz"]
        payload = b"".join(archive.read(name) for name in archive.namelist())
    for forbidden in (b"123456.789", b"energy", b"forces", b"stress"):
        assert forbidden not in payload
    with pytest.raises(FileExistsError):
        module.build_alexandria_holdout(
            shard_0000_path=shard0,
            shard_0001_path=shard1,
            output_dir=target,
            require_formal_inputs=False,
            expected_source_rows=5,
            expected_source_groups=3,
            expected_selected_groups=2,
            expected_selected_rows=4,
        )


def test_holdout_cli_cannot_select_by_endpoint_or_change_shards() -> None:
    from src.next18_alexandria_holdout import main

    for forbidden in ("--labels", "--energy", "--sample", "--group-min", "--extra-shard"):
        with pytest.raises(SystemExit) as exc_info:
            main([forbidden, "x"])
        assert exc_info.value.code == 2
