"""Contracts for true held-kind next4 stability evaluation."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from next4_law_stability import iter_true_loko, main, signed_delta_summary  # noqa: E402


def test_iter_true_loko_never_exposes_held_kind_to_training():
    discovery = pd.DataFrame(
        {
            "sid": ["a", "b", "c", "d", "e"],
            "kind": ["S1", "S1", "S2", "S2", "S3"],
        }
    )
    calibration = pd.DataFrame(
        {"sid": ["f", "g", "h"], "kind": ["S1", "S2", "S4"]}
    )
    folds = list(iter_true_loko(discovery, calibration))
    assert [fold[0] for fold in folds] == ["S1", "S2", "S3"]
    for held, training, held_discovery, held_calibration in folds:
        assert held not in set(training["kind"])
        assert set(held_discovery["kind"]) == {held}
        assert set(held_calibration["kind"]).issubset({held})
        assert len(training) + len(held_discovery) == len(discovery)


def test_signed_delta_summary_discloses_direction_and_absolute_instability():
    summary = signed_delta_summary([0.10, -0.05, 0.0])
    assert summary["signed_mean"] == (0.10 - 0.05) / 3
    assert summary["mean_absolute"] == (0.10 + 0.05) / 3
    assert summary["positive"] == 1
    assert summary["negative"] == 1
    assert summary["zero"] == 1


def test_stability_cli_refuses_to_overwrite(tmp_path):
    output = tmp_path / "existing.json"
    output.write_text("keep", encoding="utf-8")
    try:
        main(
            [
                "--isolated-dir",
                str(tmp_path / "missing"),
                "--real-descriptors",
                str(tmp_path / "missing-real.parquet"),
                "--bad-descriptors",
                str(tmp_path / "missing-bad.parquet"),
                "--real-sixfam",
                str(tmp_path / "missing-six-real.parquet"),
                "--bad-sixfam",
                str(tmp_path / "missing-six-bad.parquet"),
                "--real-corrected",
                str(tmp_path / "missing-corrected-real.parquet"),
                "--bad-corrected",
                str(tmp_path / "missing-corrected-bad.parquet"),
                "--real-guards",
                str(tmp_path / "missing-guard-real.parquet"),
                "--bad-guards",
                str(tmp_path / "missing-guard-bad.parquet"),
                "--reference-report",
                str(tmp_path / "missing-reference.json"),
                "--out",
                str(output),
            ]
        )
    except SystemExit as exc:
        assert "refusing to overwrite" in str(exc)
    else:
        raise AssertionError("main must refuse output overwrite")
