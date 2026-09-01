"""Contracts for the NEXT25 geometry-blind OMatG composition cohort."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import pickle

import lmdb
import numpy as np
import pandas as pd
import pytest
import torch


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record(numbers: list[int], *, marker: float = 1.0) -> dict[str, object]:
    return {
        "atomic_numbers": torch.tensor(numbers, dtype=torch.int32),
        "pos": torch.full((len(numbers), 3), marker, dtype=torch.float64),
        "cell": torch.eye(3, dtype=torch.float64) * (marker + 2.0),
        "band_gap": marker,
        "ids": f"SECRET-{marker}",
        "pbc": torch.ones(3, dtype=torch.bool),
    }


def _write_lmdb(path: Path, rows: list[list[int]]) -> None:
    with (
        lmdb.Environment(str(path), subdir=False, map_size=1 << 24, lock=False) as env,
        env.begin(write=True) as txn,
    ):
        for index, numbers in enumerate(rows):
            txn.put(str(index).encode(), pickle.dumps(_record(numbers, marker=index + 1.0)))


def _source(tmp_path: Path) -> tuple[Path, Path, Path]:
    # Train/validation overlap must be excluded by reduced composition.
    train = tmp_path / "train.lmdb"
    val = tmp_path / "val.lmdb"
    test = tmp_path / "test.lmdb"
    _write_lmdb(train, [[3, 8], [11, 17]])  # LiO, NaCl
    _write_lmdb(val, [[19, 35]])  # KBr
    _write_lmdb(
        test,
        [
            [3, 8],  # LiO: train overlap
            [12, 8],  # MgO: duplicate reduced formula
            [12, 12, 8, 8],  # MgO duplicate with a larger cell
            [20, 9, 9],  # CaF2: eligible
            [14, 8, 8],  # SiO2: eligible
            [2],  # He: one atom
            [19, 35],  # KBr: validation overlap
            [13, 13, 8, 8, 8],  # Al2O3: eligible
            [26, 26, 8, 8, 8],  # Fe2O3: eligible
        ],
    )
    return train, val, test


class _AtomicNumbersOnly(Mapping[str, object]):
    """Mapping that proves the extractor never requests hidden fields."""

    def __getitem__(self, key: str) -> object:
        if key != "atomic_numbers":
            raise AssertionError(f"forbidden field accessed: {key}")
        return torch.tensor([13, 13, 8, 8, 8], dtype=torch.int64)

    def __iter__(self):
        raise AssertionError("record iteration would expose field names")

    def __len__(self) -> int:
        raise AssertionError("record length is not required")


def test_record_extractor_accesses_atomic_numbers_only() -> None:
    from src.next25_omatg_compositions import extract_composition_record

    row = extract_composition_record(_AtomicNumbersOnly(), source_index=17)
    assert row == {
        "source_index": 17,
        "atomic_numbers": [8, 8, 8, 13, 13],
        "natoms": 5,
        "formula": "Al2O3",
        "reduced_formula": "Al2O3",
    }


def test_lmdb_reader_uses_numeric_key_order_beyond_single_digits(tmp_path: Path) -> None:
    from src.next25_omatg_compositions import _read_compositions

    source = tmp_path / "twelve.lmdb"
    _write_lmdb(source, [[8, 8]] * 12)
    rows = _read_compositions(source, role="test")
    assert [row["source_index"] for row in rows] == list(range(12))


def test_selector_freezes_unique_unseen_compositions_and_dummy_lmdb(tmp_path: Path) -> None:
    from src import next25_omatg_compositions as module

    train, val, test = _source(tmp_path)
    output = tmp_path / "cohort"
    result = module.freeze_composition_cohort(
        train_lmdb_path=train,
        val_lmdb_path=val,
        test_lmdb_path=test,
        output_dir=output,
        sample_size=3,
        min_atoms=2,
        max_atoms=20,
        require_formal_inputs=False,
    )

    assert {path.name for path in output.iterdir()} == {
        module.COHORT_NAME,
        module.COMPOSITIONS_LMDB_NAME,
        module.MANIFEST_NAME,
    }
    cohort = pd.read_parquet(output / module.COHORT_NAME)
    assert cohort.columns.tolist() == [
        "material_id",
        "source_split",
        "source_index",
        "formula",
        "reduced_formula",
        "atomic_numbers_json",
        "natoms",
        "selection_key",
        "selection_rank",
        "input_role",
    ]
    eligible = [
        {"source_index": 3, "formula": "CaF2", "reduced_formula": "CaF2"},
        {"source_index": 4, "formula": "SiO2", "reduced_formula": "SiO2"},
        {"source_index": 7, "formula": "Al2O3", "reduced_formula": "Al2O3"},
        {"source_index": 8, "formula": "Fe2O3", "reduced_formula": "Fe2O3"},
    ]
    expected = sorted(
        eligible,
        key=lambda row: module.selection_key(
            source_index=row["source_index"],
            formula=row["formula"],
            reduced_formula=row["reduced_formula"],
        ),
    )[:3]
    assert cohort.source_index.tolist() == [row["source_index"] for row in expected]
    assert cohort.selection_rank.tolist() == [0, 1, 2]
    assert cohort.source_split.eq("test").all()
    assert cohort.input_role.eq("composition_only").all()
    assert not set(cohort.reduced_formula) & {"LiO", "MgO", "KBr", "He"}
    assert not set(cohort.columns) & {
        "pos",
        "cell",
        "band_gap",
        "ids",
        "energy",
        "label",
        "endpoint",
    }

    with lmdb.open(
        str(output / module.COMPOSITIONS_LMDB_NAME),
        subdir=False,
        readonly=True,
        lock=False,
    ) as env, env.begin() as txn:
        assert txn.stat()["entries"] == 3
        for row_index, row in cohort.iterrows():
            payload = txn.get(str(row_index).encode())
            assert payload is not None
            dummy = pickle.loads(payload)
            expected_numbers = json.loads(row.atomic_numbers_json)
            assert dummy["atomic_numbers"].tolist() == expected_numbers
            assert tuple(dummy["pos"].shape) == (row.natoms, 3)
            assert tuple(dummy["cell"].shape) == (3, 3)
            np.testing.assert_array_equal(dummy["cell"].numpy(), np.eye(3) * 3.0)
            assert dummy["ids"] == row.material_id
            assert set(dummy) == {"pos", "cell", "atomic_numbers", "ids"}

    assert result["protocol"] == module.PROTOCOL
    assert result["input_role"] == "composition_only"
    assert result["reference_geometry_fields_accessed"] is False
    assert result["property_label_fields_accessed"] is False
    assert result["labels_opened"] is False
    assert result["counts"] == {
        "train_rows": 2,
        "validation_rows": 1,
        "test_rows": 9,
        "test_size_eligible_rows": 8,
        "test_unique_reduced_formula_rows": 6,
        "test_unique_unseen_rows": 4,
        "selected_rows": 3,
        "selected_atoms": int(cohort.natoms.sum()),
    }
    assert result["inputs_sha256"] == {
        "train_lmdb": {"path": str(train.resolve()), "sha256": _sha(train)},
        "val_lmdb": {"path": str(val.resolve()), "sha256": _sha(val)},
        "test_lmdb": {"path": str(test.resolve()), "sha256": _sha(test)},
    }
    assert result["outputs_sha256"] == {
        module.COHORT_NAME: _sha(output / module.COHORT_NAME),
        module.COMPOSITIONS_LMDB_NAME: _sha(output / module.COMPOSITIONS_LMDB_NAME),
    }


def test_selector_refuses_invalid_sources_shortfall_overwrite_and_label_cli(
    tmp_path: Path,
) -> None:
    from src.next25_omatg_compositions import freeze_composition_cohort, main

    train, val, test = _source(tmp_path)
    with pytest.raises(ValueError, match="eligible"):
        freeze_composition_cohort(
            train_lmdb_path=train,
            val_lmdb_path=val,
            test_lmdb_path=test,
            output_dir=tmp_path / "shortfall",
            sample_size=5,
            require_formal_inputs=False,
        )

    existing = tmp_path / "existing"
    existing.mkdir()
    marker = existing / "keep"
    marker.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        freeze_composition_cohort(
            train_lmdb_path=train,
            val_lmdb_path=val,
            test_lmdb_path=test,
            output_dir=existing,
            sample_size=2,
            require_formal_inputs=False,
        )
    assert marker.read_text(encoding="utf-8") == "keep"

    for forbidden in ("--labels", "--endpoint", "--reference-zip", "--band-gap"):
        with pytest.raises(SystemExit) as exc_info:
            main([forbidden, "x"])
        assert exc_info.value.code == 2


@pytest.mark.parametrize(
    "numbers",
    [
        torch.tensor([], dtype=torch.int64),
        torch.tensor([0, 8], dtype=torch.int64),
        torch.tensor([8.0, 8.0], dtype=torch.float64),
        np.array([[8, 8]], dtype=np.int64),
    ],
)
def test_record_extractor_rejects_invalid_atomic_numbers(numbers: object) -> None:
    from src.next25_omatg_compositions import extract_composition_record

    with pytest.raises(ValueError, match="atomic_numbers"):
        extract_composition_record({"atomic_numbers": numbers}, source_index=0)
