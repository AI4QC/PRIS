"""Tests for the soft-margin all-295 fail-closed audit."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from next5_falsification import development_gate  # noqa: E402
from next5_falsification import main as falsification_main  # noqa: E402


def test_development_gate_uses_full_295_pass_delta_and_joint_coverage():
    result = development_gate(
        {"pass_rate": 0.90},
        {"pass_rate": 0.87, "joint_required_feature_coverage": 0.90},
        prior_gate=True,
    )
    assert np.isclose(result["all_295_pass_delta_vs_existing_loop"], -0.03)
    assert result["passes_joint_coverage_gate"] is True
    assert result["passes_all_295_false_positive_gate"] is True
    assert result["combined_development_gate_before_loko"] is True


def test_development_gate_cannot_rescue_a_failed_prior_gate():
    result = development_gate(
        {"pass_rate": 0.90},
        {"pass_rate": 0.95, "joint_required_feature_coverage": 1.0},
        prior_gate=False,
    )
    assert result["combined_development_gate_before_loko"] is False


def test_falsification_cli_refuses_to_overwrite_before_loading_inputs(tmp_path):
    output = tmp_path / "existing.json"
    output.write_text("keep", encoding="utf-8")
    with np.testing.assert_raises_regex(SystemExit, "refusing to overwrite"):
        falsification_main(
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
    assert output.read_text(encoding="utf-8") == "keep"
