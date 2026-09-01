from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.next39_omat24_trajectory_cohort import freeze_trajectory_cohort
from src.next41_contact_guard_features import (
    FEATURE_NAME,
    build_contact_guard_features,
)
from tests.test_next39_omat24_trajectory_cohort import _payload, _write_db


def test_contact_features_publish_from_frozen_step0_only(tmp_path: Path) -> None:
    database = tmp_path / "tiny.aselmdb"
    _write_db(
        database,
        [
            _payload(sid="a_traj_0", parent_id="a", position=1.5),
            _payload(
                sid="a_traj_2",
                parent_id="a",
                position={"sentinel": "later-remains-closed"},
            ),
        ],
    )
    cohort = tmp_path / "cohort"
    freeze_trajectory_cohort(
        db_path=database,
        source_name="test",
        salt="fixed",
        minimum_latest_step=1,
        output_dir=cohort,
    )
    output = tmp_path / "features"

    manifest = build_contact_guard_features(
        metadata_path=cohort / "next39_trajectory_cohort.parquet",
        frames_zip_path=cohort / "geometry_only_frames.zip",
        upstream_manifest_path=cohort / "MANIFEST.json",
        output_dir=output,
    )

    table = pd.read_parquet(output / FEATURE_NAME)
    assert len(table) == 1
    assert table.contact_supported.tolist() == [True]
    assert table.cov_q01.notna().all()
    assert manifest["later_geometry_opened"] is False
    assert manifest["dft_values_read"] is False
    with pytest.raises(FileExistsError):
        build_contact_guard_features(
            metadata_path=cohort / "next39_trajectory_cohort.parquet",
            frames_zip_path=cohort / "geometry_only_frames.zip",
            upstream_manifest_path=cohort / "MANIFEST.json",
            output_dir=output,
        )
