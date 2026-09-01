"""Contracts for all-295 unknown-fails-closed falsification."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from next4_falsification import evaluate_rule_set, main  # noqa: E402


def test_evaluate_rule_set_keeps_all_rows_and_fails_unknown_closed():
    frame = pd.DataFrame(
        {
            "x": [0.5, 2.0, np.nan, 0.5],
            "g": [0.0, 0.0, 0.0, np.nan],
        }
    )
    rules = [
        {
            "description": "if g <= 0 then x <= 1",
            "feature": "x",
            "family": "fixed-guarded-one-sided",
            "origin": "test",
            "side": "hi",
            "thresholds": [1.0],
            "real_coverage": 1.0,
            "bad_coverage": 1.0,
            "guard_feature": "g",
            "guard_side": "lo",
            "guard_threshold": 0.0,
        }
    ]
    result = evaluate_rule_set(frame, rules)
    assert result["n"] == 4
    assert result["pass_rate"] == 0.25
    assert result["known_rate"] == 0.5
    assert result["unknown_count"] == 2
    assert result["known_only_pass_rate"] == 0.5
    assert result["joint_required_feature_coverage"] == 0.5
    assert result["required_feature_coverage"] == {"g": 0.75, "x": 0.75}


def test_falsification_cli_refuses_to_overwrite(tmp_path):
    output = tmp_path / "existing.json"
    output.write_text("keep", encoding="utf-8")
    try:
        main(
            [
                "--features-dir",
                str(tmp_path / "missing"),
                "--law-report",
                str(tmp_path / "missing-law.json"),
                "--fp-p235",
                str(tmp_path / "missing-p235.parquet"),
                "--fp-guards",
                str(tmp_path / "missing-guards.parquet"),
                "--fp-sixfam",
                str(tmp_path / "missing-sixfam.parquet"),
                "--fp-corrected",
                str(tmp_path / "missing-corrected.parquet"),
                "--out",
                str(output),
            ]
        )
    except SystemExit as exc:
        assert "refusing to overwrite" in str(exc)
    else:
        raise AssertionError("main must refuse output overwrite")
    assert output.read_text(encoding="utf-8") == "keep"
