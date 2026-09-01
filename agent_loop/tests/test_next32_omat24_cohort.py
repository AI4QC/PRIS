from __future__ import annotations

import json
from pathlib import Path
import zlib

import lmdb
import numpy as np
import pandas as pd
import pytest

from src.next32_omat24_cohort import (
    decode_dft_endpoint,
    project_identity_geometry,
    publish_dft_endpoints,
    sanitize_omat24_cohort,
    select_parent_unique,
)


def _payload(
    *,
    sid: str = "parent-a_rattled-300_seed_0",
    parent_id: str = "parent-a",
    endpoint_sentinel: bool = True,
) -> bytes:
    endpoint: object = {"sentinel": "must-not-be-decoded"} if endpoint_sentinel else -4.0
    return json.dumps(
        {
            "numbers": [11, 17],
            "positions": [[0.0, 0.0, 0.0], [1.5, 1.5, 1.5]],
            "cell": [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]],
            "pbc": [True, True, True],
            "energy": endpoint,
            "forces": endpoint,
            "stress": endpoint,
            "data": {
                "sid": sid,
                "parent_id": parent_id,
                "task_type": "Structure Optimization",
                "energy_corrected_mp2020": endpoint,
            },
        }
    ).encode("utf-8")


def test_geometry_projection_skips_dft_endpoint_values() -> None:
    metadata, atoms = project_identity_geometry(_payload())

    assert metadata == {
        "sid": "parent-a_rattled-300_seed_0",
        "parent_id": "parent-a",
        "task_type": "Structure Optimization",
    }
    assert atoms.numbers.tolist() == [11, 17]
    assert atoms.pbc.tolist() == [True, True, True]
    assert atoms.calc is None
    assert atoms.info == {}


def test_parent_unique_selection_is_deterministic_and_excludes_parents() -> None:
    rows = [
        {"sid": "a-2", "parent_id": "a", "record_key": 2},
        {"sid": "a-1", "parent_id": "a", "record_key": 1},
        {"sid": "b-1", "parent_id": "b", "record_key": 3},
        {"sid": "c-1", "parent_id": "c", "record_key": 4},
    ]

    first = select_parent_unique(rows, salt="fixed", limit=2, exclude_parent_ids={"c"})
    second = select_parent_unique(list(reversed(rows)), salt="fixed", limit=2, exclude_parent_ids={"c"})

    assert first == second
    assert len(first) == 2
    assert len({row["parent_id"] for row in first}) == 2
    assert all(row["parent_id"] != "c" for row in first)


def test_dft_endpoint_decoding_is_separate_and_validates_shapes() -> None:
    value = json.loads(_payload(endpoint_sentinel=False))
    value["forces"] = [[3.0, 4.0, 0.0], [0.0, 0.0, 0.0]]
    value["stress"] = [0.01, 0.02, 0.02, 0.0, 0.0, 0.0]

    endpoint = decode_dft_endpoint(json.dumps(value).encode("utf-8"))

    assert endpoint["force_max"] == pytest.approx(5.0)
    assert endpoint["force_rms"] == pytest.approx(np.sqrt(12.5))
    assert endpoint["stress_norm"] == pytest.approx(0.03)
    assert endpoint["energy_per_atom"] == pytest.approx(-2.0)

    value["forces"] = [[1.0, 2.0]]
    with pytest.raises(ValueError, match="forces"):
        decode_dft_endpoint(json.dumps(value).encode("utf-8"))


def _write_db(path: Path, records: list[bytes]) -> None:
    env = lmdb.open(str(path), subdir=False, map_size=16 * 1024 * 1024)
    try:
        with env.begin(write=True) as transaction:
            for index, payload in enumerate(records, start=1):
                transaction.put(str(index).encode(), zlib.compress(payload))
            transaction.put(b"nextid", zlib.compress(str(len(records) + 1).encode()))
    finally:
        env.close()


def test_sanitize_then_open_identity_locked_endpoints(tmp_path: Path) -> None:
    raw = tmp_path / "tiny.aselmdb"
    records: list[bytes] = []
    for sid, parent in (("a-1", "a"), ("a-2", "a"), ("b-1", "b"), ("c-1", "c")):
        value = json.loads(_payload(sid=sid, parent_id=parent, endpoint_sentinel=False))
        value["forces"] = [[3.0, 4.0, 0.0], [0.0, 0.0, 0.0]]
        value["stress"] = [0.01, 0.02, 0.02, 0.0, 0.0, 0.0]
        records.append(json.dumps(value).encode())
    _write_db(raw, records)
    cohort_dir = tmp_path / "cohort"

    cohort_manifest = sanitize_omat24_cohort(
        db_path=raw,
        source_name="rattled-test",
        salt="fixed",
        limit=2,
        exclude_parent_ids={"c"},
        output_dir=cohort_dir,
    )

    metadata_path = cohort_dir / "next32_cohort.parquet"
    metadata = pd.read_parquet(metadata_path)
    assert len(metadata) == 2
    assert metadata.parent_id.nunique() == 2
    assert metadata.input_role.eq("unrelaxed_x0_geometry_only").all()
    assert cohort_manifest["labels_opened"] is False
    assert cohort_manifest["endpoint_numeric_fields_parsed"] is False

    endpoint_dir = tmp_path / "endpoints"
    endpoint_manifest = publish_dft_endpoints(
        db_path=raw,
        metadata_path=metadata_path,
        cohort_manifest_path=cohort_dir / "MANIFEST.json",
        identity_lock_path=metadata_path,
        identity_lock_manifest_path=cohort_dir / "MANIFEST.json",
        output_dir=endpoint_dir,
    )

    endpoints = pd.read_parquet(endpoint_dir / "next32_dft_endpoints.parquet")
    assert set(endpoints.material_id) == set(metadata.material_id)
    assert endpoints.force_max.eq(5.0).all()
    assert endpoint_manifest["labels_opened"] is True
    with pytest.raises(FileExistsError):
        sanitize_omat24_cohort(
            db_path=raw,
            source_name="rattled-test",
            salt="fixed",
            limit=2,
            output_dir=cohort_dir,
        )
