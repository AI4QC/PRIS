"""Tests for post-hoc LOKO rule recurrence diagnostics."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from next5_consensus_diagnostic import main as consensus_main  # noqa: E402
from next5_consensus_diagnostic import recurrent_semantic_rules  # noqa: E402
from next5_consensus_diagnostic import semantic_rule_key  # noqa: E402


def _rule(threshold: float, *, coverage: float = 1.0) -> dict[str, object]:
    return {
        "description": f"x <= {threshold}",
        "feature": "x",
        "family": "one-sided",
        "origin": "new",
        "side": "hi",
        "thresholds": [threshold],
        "guard_feature": None,
        "guard_side": None,
        "guard_threshold": None,
        "real_coverage": coverage,
        "bad_coverage": coverage,
    }


def test_semantic_key_ignores_coverage_but_not_threshold():
    assert semantic_rule_key(_rule(2.0, coverage=0.8)) == semantic_rule_key(
        _rule(2.0, coverage=1.0)
    )
    assert semantic_rule_key(_rule(2.0)) != semantic_rule_key(_rule(2.1))


def test_recurrence_counts_each_held_kind_at_most_once_and_keeps_maximum():
    folds = {
        "S1": {"rules": [_rule(2.0), _rule(2.0)]},
        "S2": {"rules": [_rule(2.0), _rule(3.0)]},
        "S3": {"rules": [_rule(2.0)]},
        "S4": {"rules": [_rule(3.0)]},
    }
    result = recurrent_semantic_rules(folds, min_count=2)
    assert [entry["count"] for entry in result] == [3, 2]
    assert result[0]["held_kinds"] == ["S1", "S2", "S3"]
    assert result[0]["is_maximum_recurrence"] is True
    assert result[1]["is_maximum_recurrence"] is False


def test_consensus_cli_refuses_to_overwrite_before_loading_inputs(tmp_path):
    output = tmp_path / "existing.json"
    output.write_text("keep", encoding="utf-8")
    with np.testing.assert_raises_regex(SystemExit, "refusing to overwrite"):
        consensus_main(
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
                "--input-domain-audit",
                str(tmp_path / "missing-audit.json"),
                "--reference-report",
                str(tmp_path / "missing-law.json"),
                "--loko-report",
                str(tmp_path / "missing-loko.json"),
                "--out",
                str(output),
            ]
        )
    assert output.read_text(encoding="utf-8") == "keep"
