from __future__ import annotations

import json
from pathlib import Path
import zlib

import lmdb
import pandas as pd
import pytest

from src.next39_omat24_trajectory_cohort import (
    freeze_trajectory_cohort,
    parse_trajectory_sid,
    select_eligible_trajectories,
)


def _payload(
    *,
    sid: str,
    parent_id: str,
    position: float | object = 1.5,
    task_type: str = "Structure Optimization",
) -> bytes:
    sentinel = {"sentinel": "DFT-value-must-not-be-decoded"}
    return json.dumps(
        {
            "numbers": [11, 17],
            "positions": [[0.0, 0.0, 0.0], [position, position, position]],
            "cell": [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]],
            "pbc": [True, True, True],
            "energy": sentinel,
            "forces": sentinel,
            "stress": sentinel,
            "data": {
                "sid": sid,
                "parent_id": parent_id,
                "task_type": task_type,
                "energy_corrected_mp2020": sentinel,
            },
        }
    ).encode()


def _write_db(path: Path, records: list[bytes]) -> None:
    env = lmdb.open(str(path), subdir=False, map_size=16 * 1024 * 1024)
    try:
        with env.begin(write=True) as transaction:
            for index, payload in enumerate(records, start=1):
                transaction.put(str(index).encode(), zlib.compress(payload))
            transaction.put(b"nextid", zlib.compress(str(len(records) + 1).encode()))
    finally:
        env.close()


def test_parse_trajectory_sid_uses_final_integer_suffix() -> None:
    assert parse_trajectory_sid("parent_a_rattled-300_seed_20") == (
        "parent_a_rattled-300_seed",
        20,
    )
    with pytest.raises(ValueError, match="trajectory step"):
        parse_trajectory_sid("parent_a_rattled-300_seed")


def test_selection_is_step_and_parent_unique_and_order_independent() -> None:
    rows = [
        {"sid": "a_t0_0", "parent_id": "p", "task_type": "Structure Optimization", "record_key": 1},
        {"sid": "a_t0_20", "parent_id": "p", "task_type": "Structure Optimization", "record_key": 2},
        {"sid": "a_t1_0", "parent_id": "p", "task_type": "Structure Optimization", "record_key": 3},
        {"sid": "a_t1_30", "parent_id": "p", "task_type": "Structure Optimization", "record_key": 4},
        {"sid": "b_t0_0", "parent_id": "b", "task_type": "Structure Optimization", "record_key": 5},
        {"sid": "b_t0_19", "parent_id": "b", "task_type": "Structure Optimization", "record_key": 6},
        {"sid": "c_t0_0", "parent_id": "c", "task_type": "Single Point", "record_key": 7},
        {"sid": "c_t0_40", "parent_id": "c", "task_type": "Single Point", "record_key": 8},
        {"sid": "d_t0_0", "parent_id": "d", "task_type": "Structure Optimization", "record_key": 9},
        {"sid": "d_t0_20", "parent_id": "d", "task_type": "Structure Optimization", "record_key": 10},
        {"sid": "d_t0_40", "parent_id": "d", "task_type": "Structure Optimization", "record_key": 11},
    ]

    first = select_eligible_trajectories(rows, minimum_latest_step=20, salt="fixed")
    second = select_eligible_trajectories(
        list(reversed(rows)), minimum_latest_step=20, salt="fixed"
    )

    assert first == second
    assert {row["parent_id"] for row in first} == {"p", "d"}
    assert len(first) == 2
    d_row = next(row for row in first if row["parent_id"] == "d")
    assert d_row["initial_step"] == 0
    assert d_row["latest_step"] == 40
    assert d_row["latest_record_key"] == 11


def test_freezer_decodes_only_step_zero_geometry(tmp_path: Path) -> None:
    database = tmp_path / "tiny.aselmdb"
    _write_db(
        database,
        [
            _payload(sid="a_traj_0", parent_id="a"),
            _payload(
                sid="a_traj_20",
                parent_id="a",
                position={"sentinel": "later-geometry-must-remain-closed"},
            ),
        ],
    )
    output = tmp_path / "cohort"

    manifest = freeze_trajectory_cohort(
        db_path=database,
        source_name="rattled-relax",
        salt="next39-test",
        minimum_latest_step=20,
        output_dir=output,
    )

    table = pd.read_parquet(output / "next39_trajectory_cohort.parquet")
    assert table[["initial_step", "latest_step"]].to_dict("records") == [
        {"initial_step": 0, "latest_step": 20}
    ]
    assert manifest["later_geometry_opened"] is False
    assert manifest["dft_numeric_fields_parsed"] is False
    assert manifest["counts"]["selected_trajectories"] == 1
    with pytest.raises(FileExistsError):
        freeze_trajectory_cohort(
            db_path=database,
            source_name="rattled-relax",
            salt="next39-test",
            minimum_latest_step=20,
            output_dir=output,
        )
