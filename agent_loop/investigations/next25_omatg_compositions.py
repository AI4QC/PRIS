#!/usr/bin/env python3
"""Freeze a geometry-blind MP-20 composition cohort for OMatG CSP."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import pickle
import re
import shutil
import tempfile
from typing import Any

import lmdb
import numpy as np
import pandas as pd
from pymatgen.core import Composition, Element
import torch

from src.next19_feature_build import _publish_directory_no_replace


PROTOCOL = "2026-08-03-next25-omatg-composition-only-v1"
COHORT_NAME = "composition_cohort.parquet"
COMPOSITIONS_LMDB_NAME = "compositions_only.lmdb"
MANIFEST_NAME = "MANIFEST.json"
SELECTION_SALT = "next25-omatg-mp20-unique-unseen-composition-v1"
FORMAL_SAMPLE_SIZE = 512
FORMAL_MIN_ATOMS = 2
FORMAL_MAX_ATOMS = 20
FORMAL_INPUT_SHA256: Mapping[str, str] = {
    "train_lmdb": "ca21166f6d0ecaac7278629ef813eac734797bcbe98f34d580993a2ae14c545b",
    "val_lmdb": "14f73bbea446909f92af23bc06902bb3d2f08bede34c1ca08837cff62671cd43",
    "test_lmdb": "74223481f135274e54375c52e38ace5ab7a2403f367a4dd8352cfd1874b2986d",
}
OMATG_GIT_COMMIT = "fcb9ba2c2cfd70505b0f142a5b3c44944d78e7f0"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _hash_record(path: Path, digest: str) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": digest}


def extract_composition_record(
    record: Mapping[str, object], *, source_index: int
) -> dict[str, object]:
    """Project one trusted LMDB record through ``atomic_numbers`` only."""

    if type(source_index) is not int or source_index < 0:
        raise ValueError("source_index must be a non-negative exact integer")
    try:
        raw = record["atomic_numbers"]
    except (KeyError, TypeError) as exc:
        raise ValueError("record lacks atomic_numbers") from exc
    if isinstance(raw, torch.Tensor):
        if raw.ndim != 1 or raw.dtype not in (torch.int32, torch.int64):
            raise ValueError("atomic_numbers must be a one-dimensional integer tensor")
        values = raw.detach().cpu().tolist()
    elif isinstance(raw, np.ndarray):
        if raw.ndim != 1 or raw.dtype.kind not in "iu":
            raise ValueError("atomic_numbers must be a one-dimensional integer array")
        values = raw.tolist()
    elif isinstance(raw, (list, tuple)):
        if any(type(value) is not int for value in raw):
            raise ValueError("atomic_numbers sequence must contain exact integers")
        values = list(raw)
    else:
        raise ValueError("atomic_numbers has an unsupported representation")
    if not values or any(type(value) is not int or not 1 <= value <= 118 for value in values):
        raise ValueError("atomic_numbers must contain atomic numbers from 1 through 118")

    # A site ordering can encode information about the source structure.  Sort
    # the multiset so the released input carries composition and atom count only.
    atomic_numbers = sorted(values)
    counts = Counter(atomic_numbers)
    composition = Composition(
        {Element.from_Z(atomic_number): count for atomic_number, count in counts.items()}
    )
    formula_parts: list[str] = []
    for token in composition.formula.split():
        match = re.fullmatch(r"([A-Z][a-z]*)([0-9]+(?:\.[0-9]+)?)", token)
        if match is None or not float(match.group(2)).is_integer():
            raise ValueError("atomic_numbers did not yield an integer formula")
        count = int(float(match.group(2)))
        formula_parts.append(match.group(1) + (str(count) if count != 1 else ""))
    formula = "".join(formula_parts)
    reduced_formula = composition.reduced_formula.replace(" ", "")
    return {
        "source_index": source_index,
        "atomic_numbers": atomic_numbers,
        "natoms": len(atomic_numbers),
        "formula": formula,
        "reduced_formula": reduced_formula,
    }


def _read_compositions(path: Path, *, role: str) -> list[dict[str, object]]:
    try:
        environment = lmdb.open(
            str(path),
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            max_readers=1,
        )
    except (lmdb.Error, OSError) as exc:
        raise ValueError(f"invalid {role} LMDB") from exc
    rows: list[dict[str, object]] = []
    try:
        with environment.begin(write=False) as transaction:
            number_records = int(transaction.stat()["entries"])
            for expected_index in range(number_records):
                payload = transaction.get(str(expected_index).encode("ascii"))
                if payload is None:
                    raise ValueError(f"{role} LMDB keys must be consecutive decimal indices")
                try:
                    record = pickle.loads(payload)
                except Exception as exc:
                    raise ValueError(f"invalid pickled record in {role} LMDB") from exc
                if not isinstance(record, Mapping):
                    raise ValueError(f"{role} LMDB record must be a mapping")
                rows.append(
                    extract_composition_record(record, source_index=expected_index)
                )
    finally:
        environment.close()
    if not rows:
        raise ValueError(f"{role} LMDB is empty")
    return rows


def selection_key(*, source_index: int, formula: str, reduced_formula: str) -> str:
    payload = (
        f"{SELECTION_SALT}\n{source_index}\n{formula}\n{reduced_formula}\n"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_composition_lmdb(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if os.path.lexists(path):
        raise FileExistsError(str(path))
    try:
        environment = lmdb.Environment(
            str(path), subdir=False, map_size=int(1e10), lock=False
        )
    except (lmdb.Error, OSError) as exc:
        raise ValueError("could not create composition-only LMDB") from exc
    try:
        with environment.begin(write=True) as transaction:
            for index, row in enumerate(rows):
                numbers = json.loads(str(row["atomic_numbers_json"]))
                natoms = int(row["natoms"])
                if len(numbers) != natoms:
                    raise ValueError("composition-only atom count differs")
                # Reproduce the official OMatG create_compositions dummy input:
                # cubic 3-A cell and evenly spaced fractional coordinates.
                fractional = np.array(
                    [(atom_index / natoms, 0.0, 0.0) for atom_index in range(natoms)],
                    dtype=np.float64,
                )
                cell = np.eye(3, dtype=np.float64) * 3.0
                dummy = {
                    "pos": torch.from_numpy(fractional @ cell),
                    "cell": torch.from_numpy(cell),
                    "atomic_numbers": torch.tensor(numbers, dtype=torch.int32),
                    "ids": str(row["material_id"]),
                }
                transaction.put(str(index).encode("ascii"), pickle.dumps(dummy))
        environment.sync()
    finally:
        environment.close()


def _verify_unchanged(paths: Mapping[str, Path], hashes: Mapping[str, str]) -> None:
    for role, path in paths.items():
        if _sha256(path) != hashes[role]:
            raise ValueError(f"{role} changed before publication")


def freeze_composition_cohort(
    *,
    train_lmdb_path: Path,
    val_lmdb_path: Path,
    test_lmdb_path: Path,
    output_dir: Path,
    sample_size: int = FORMAL_SAMPLE_SIZE,
    min_atoms: int = FORMAL_MIN_ATOMS,
    max_atoms: int = FORMAL_MAX_ATOMS,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    """Select and publish a deterministic composition-only OMatG cohort."""

    target = Path(output_dir).resolve()
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if type(sample_size) is not int or sample_size <= 0:
        raise ValueError("sample_size must be a positive exact integer")
    if (
        type(min_atoms) is not int
        or type(max_atoms) is not int
        or not 1 <= min_atoms <= max_atoms
    ):
        raise ValueError("atom bounds must be ordered positive exact integers")
    paths = {
        "train_lmdb": Path(train_lmdb_path).resolve(),
        "val_lmdb": Path(val_lmdb_path).resolve(),
        "test_lmdb": Path(test_lmdb_path).resolve(),
    }
    for role, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{role} is not a file: {path}")
    input_hashes = {role: _sha256(path) for role, path in paths.items()}
    formal_identity = (
        input_hashes == dict(FORMAL_INPUT_SHA256)
        and sample_size == FORMAL_SAMPLE_SIZE
        and min_atoms == FORMAL_MIN_ATOMS
        and max_atoms == FORMAL_MAX_ATOMS
    )
    if require_formal_inputs and not formal_identity:
        raise ValueError("formal NEXT25 inputs or sampling constants differ")

    train = _read_compositions(paths["train_lmdb"], role="train")
    validation = _read_compositions(paths["val_lmdb"], role="validation")
    test = _read_compositions(paths["test_lmdb"], role="test")
    seen_formulas = {
        str(row["reduced_formula"]) for row in train + validation
    }
    test_reduced_counts = Counter(str(row["reduced_formula"]) for row in test)
    size_eligible = [
        row for row in test if min_atoms <= int(row["natoms"]) <= max_atoms
    ]
    unique_rows = [
        row
        for row in size_eligible
        if test_reduced_counts[str(row["reduced_formula"])] == 1
    ]
    eligible = [
        row for row in unique_rows if str(row["reduced_formula"]) not in seen_formulas
    ]
    for row in eligible:
        row["selection_key"] = selection_key(
            source_index=int(row["source_index"]),
            formula=str(row["formula"]),
            reduced_formula=str(row["reduced_formula"]),
        )
    eligible.sort(key=lambda row: str(row["selection_key"]))
    if len(eligible) < sample_size:
        raise ValueError(
            f"only {len(eligible)} eligible unique unseen compositions for "
            f"sample_size={sample_size}"
        )
    selected = eligible[:sample_size]

    cohort_rows: list[dict[str, Any]] = []
    for rank, row in enumerate(selected):
        cohort_rows.append(
            {
                "material_id": f"next25-omatg-mp20-{rank:04d}",
                "source_split": "test",
                "source_index": int(row["source_index"]),
                "formula": str(row["formula"]),
                "reduced_formula": str(row["reduced_formula"]),
                "atomic_numbers_json": json.dumps(
                    row["atomic_numbers"], separators=(",", ":")
                ),
                "natoms": int(row["natoms"]),
                "selection_key": str(row["selection_key"]),
                "selection_rank": rank,
                "input_role": "composition_only",
            }
        )
    cohort = pd.DataFrame(cohort_rows)

    repository_root = Path(__file__).resolve().parents[1]
    source_paths = {
        "src/next25_omatg_compositions.py": Path(__file__).resolve(),
        "src/next19_feature_build.py": repository_root
        / "src/next19_feature_build.py",
    }
    manifest: dict[str, object] = {
        "protocol": PROTOCOL,
        "mode": "composition_only_generator_input_freeze",
        "source_generator": "omatg_mp20_csp_linear_ode",
        "source_revision": {
            "repository": "https://github.com/FERMat-ML/OMatG",
            "git_commit": OMATG_GIT_COMMIT,
        },
        "input_role": "composition_only",
        "allowed_record_fields_accessed": ["atomic_numbers"],
        "reference_geometry_fields_accessed": False,
        "property_label_fields_accessed": False,
        "reference_material_identifiers_accessed": False,
        "labels_opened": False,
        "endpoint_artifacts_opened": False,
        "same_composition_candidates_used": False,
        "site_order_canonicalized": True,
        "selection": {
            "salt": SELECTION_SALT,
            "ranking": "ascending SHA-256(salt, source_index, formula, reduced_formula)",
            "test_reduced_formula_multiplicity": 1,
            "reduced_formula_absent_from_train_and_validation": True,
            "sample_size": sample_size,
            "min_atoms": min_atoms,
            "max_atoms": max_atoms,
        },
        "counts": {
            "train_rows": len(train),
            "validation_rows": len(validation),
            "test_rows": len(test),
            "test_size_eligible_rows": len(size_eligible),
            "test_unique_reduced_formula_rows": len(unique_rows),
            "test_unique_unseen_rows": len(eligible),
            "selected_rows": len(cohort),
            "selected_atoms": int(cohort["natoms"].sum()),
        },
        "inputs_sha256": {
            role: _hash_record(paths[role], input_hashes[role]) for role in paths
        },
        "executed_source_sha256": {
            relative: _sha256(path) for relative, path in source_paths.items()
        },
        "production_protocol_eligible": bool(formal_identity),
        "scientific_improvement_claim": False,
    }

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent)
    )
    try:
        cohort_path = staging / COHORT_NAME
        lmdb_path = staging / COMPOSITIONS_LMDB_NAME
        cohort.to_parquet(cohort_path, index=False)
        _write_composition_lmdb(lmdb_path, cohort.to_dict("records"))
        manifest["outputs_sha256"] = {
            COHORT_NAME: _sha256(cohort_path),
            COMPOSITIONS_LMDB_NAME: _sha256(lmdb_path),
        }
        (staging / MANIFEST_NAME).write_bytes(_json_bytes(manifest))
        _verify_unchanged(paths, input_hashes)
        for relative, path in source_paths.items():
            if _sha256(path) != manifest["executed_source_sha256"][relative]:
                raise ValueError(f"executed source changed before publication: {relative}")
        _publish_directory_no_replace(staging, target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-lmdb", type=Path, required=True)
    parser.add_argument("--val-lmdb", type=Path, required=True)
    parser.add_argument("--test-lmdb", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=FORMAL_SAMPLE_SIZE)
    parser.add_argument("--min-atoms", type=int, default=FORMAL_MIN_ATOMS)
    parser.add_argument("--max-atoms", type=int, default=FORMAL_MAX_ATOMS)
    parser.add_argument("--allow-nonformal-inputs", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = freeze_composition_cohort(
        train_lmdb_path=args.train_lmdb,
        val_lmdb_path=args.val_lmdb,
        test_lmdb_path=args.test_lmdb,
        output_dir=args.output_dir,
        sample_size=args.sample_size,
        min_atoms=args.min_atoms,
        max_atoms=args.max_atoms,
        require_formal_inputs=not args.allow_nonformal_inputs,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
