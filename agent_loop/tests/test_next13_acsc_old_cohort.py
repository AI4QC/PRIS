"""Contracts for the frozen old-cohort ACSC incremental audit."""

from __future__ import annotations

import pandas as pd
import pytest
from pathlib import Path


def _phsc_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sid": ["c", "a", "b", "ns"],
            "rk": ["rk-c", "rk-a", "rk-b", "rk-ns"],
            "stage": ["threshold_calibration"] * 4,
            "threshold_role": ["development_gate"] * 4,
            "strict_x0_ok": [True, True, True, False],
            "natoms": [2, 2, 2, 0],
            "phsc_status": [
                "resolved_nonnegative",
                "resolved_nonnegative",
                "resolved_negative",
                "abstain_unsupported_geometry",
            ],
        }
    )


def _chsc_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sid": ["a", "b", "c", "ns"],
            "rk": ["rk-a", "rk-b", "rk-c", "rk-ns"],
            "stage": ["threshold_calibration"] * 4,
            "threshold_role": ["development_gate"] * 4,
            "strict_x0_ok": [True, True, True, False],
            "natoms": [2, 2, 2, 0],
            "chsc_status": [
                "resolved_nonnegative",
                "resolved_nonnegative",
                "resolved_negative",
                "abstain_unsupported_geometry",
            ],
        }
    )


def test_selection_is_sorted_intersection_of_both_resolved_nonnegative_states() -> None:
    from src.next13_acsc_old_cohort import eligible_upstream_table

    eligible, counts = eligible_upstream_table(_phsc_table(), _chsc_table())

    assert eligible["sid"].tolist() == ["a"]
    assert eligible.iloc[0].to_dict() == {
        "sid": "a",
        "rk": "rk-a",
        "stage": "threshold_calibration",
        "threshold_role": "development_gate",
        "natoms": 2,
        "upstream_phsc_status": "resolved_nonnegative",
        "upstream_chsc_status": "resolved_nonnegative",
    }
    assert counts == {
        "upstream_rows": 4,
        "upstream_strict_rows": 3,
        "eligible_both_resolved_nonnegative_rows": 1,
    }


def test_selection_fails_closed_on_upstream_alignment_drift() -> None:
    from src.next13_acsc_old_cohort import eligible_upstream_table

    chsc = _chsc_table()
    chsc.loc[chsc["sid"] == "a", "rk"] = "different"
    with pytest.raises(ValueError, match="metadata alignment"):
        eligible_upstream_table(_phsc_table(), chsc)


def test_input_snapshots_explicitly_retain_only_tables_and_feature_manifests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src import next13_acsc_old_cohort as module

    observed: dict[str, bool] = {}

    def fake_snapshot(path: Path, *, include_data: bool) -> tuple[str, bool]:
        observed[path.name] = include_data
        return path.name, include_data

    monkeypatch.setattr(module, "_snapshot", fake_snapshot)
    paths = {
        "phsc_features": Path("phsc.parquet"),
        "phsc_manifest": Path("phsc.json"),
        "chsc_features": Path("chsc.parquet"),
        "chsc_manifest": Path("chsc.json"),
        "geometry_only_frames": Path("frames.zip"),
        "geometry_manifest": Path("geometry.json"),
        "checkpoint": Path("model.pth"),
    }

    module._snapshot_inputs(paths)

    assert observed == {
        "phsc.parquet": True,
        "phsc.json": True,
        "chsc.parquet": True,
        "chsc.json": True,
        "frames.zip": False,
        "geometry.json": False,
        "model.pth": False,
    }


def test_cli_exposes_no_label_or_endpoint_argument() -> None:
    from src.next13_acsc_old_cohort import main

    for forbidden in ("--labels", "--endpoint", "--dft-results"):
        with pytest.raises(SystemExit) as exc_info:
            main([forbidden, "x"])
        assert exc_info.value.code == 2
