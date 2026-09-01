from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.next39_omat24_trajectory_cohort import freeze_trajectory_cohort
from src.next40_omat24_short_source import (
    FILTERED_DB_NAME,
    build_short_horizon_source,
)
from tests.test_next39_omat24_trajectory_cohort import _payload, _write_db


def test_short_source_excludes_used_and_long_parents_without_opening_geometry(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.aselmdb"
    _write_db(
        source,
        [
            _payload(sid="short_traj_0", parent_id="short"),
            _payload(
                sid="short_traj_5",
                parent_id="short",
                position={"sentinel": "later-geometry-stays-opaque"},
            ),
            _payload(sid="used_traj_0", parent_id="used"),
            _payload(sid="used_traj_3", parent_id="used", position=1.6),
            _payload(sid="long_traj_0", parent_id="long"),
            _payload(sid="long_traj_20", parent_id="long", position=1.7),
        ],
    )
    excluded = tmp_path / "excluded.txt"
    excluded.write_text("used\n", encoding="utf-8")
    output = tmp_path / "filtered"

    manifest = build_short_horizon_source(
        db_path=source,
        excluded_parent_ids_path=excluded,
        salt="next40-test",
        minimum_latest_step=1,
        maximum_latest_step=19,
        output_dir=output,
    )

    assert manifest["counts"]["selected_parents"] == 1
    assert manifest["counts"]["copied_records"] == 2
    assert manifest["later_geometry_opened"] is False
    assert manifest["dft_values_read"] is False
    cohort = tmp_path / "cohort"
    freeze_trajectory_cohort(
        db_path=output / FILTERED_DB_NAME,
        source_name="short",
        salt="fixed",
        minimum_latest_step=1,
        output_dir=cohort,
    )
    table = pd.read_parquet(cohort / "next39_trajectory_cohort.parquet")
    assert table.parent_id.tolist() == ["short"]
    assert table.latest_step.tolist() == [5]
    with pytest.raises(FileExistsError):
        build_short_horizon_source(
            db_path=source,
            excluded_parent_ids_path=excluded,
            salt="next40-test",
            minimum_latest_step=1,
            maximum_latest_step=19,
            output_dir=output,
        )
