"""Contracts for the sequential soft-margin anion search."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from next5_law_search import main as search_main  # noqa: E402
from next5_law_search import joint_feature_coverage_by_stratum  # noqa: E402
from next5_law_search import paired_soft_strata  # noqa: E402
from next5_law_search import soften_paired_floors  # noqa: E402


def test_soften_paired_floors_uses_distinct_full_and_cell_margins():
    strict = {
        "anion:O": 0.996,
        "anion:O:fold:0": 0.98,
        "anion:N": 0.001,
        "anion:N:fold:2": 0.005,
    }
    metadata = {
        "anion:O": {"fold": "all"},
        "anion:O:fold:0": {"fold": 0},
        "anion:N": {"fold": "all"},
        "anion:N:fold:2": {"fold": 2},
    }

    softened = soften_paired_floors(
        strict,
        metadata,
        full_anion_margin=0.0025,
        cell_margin=0.01,
    )

    assert softened["anion:O"] == 0.9935
    assert softened["anion:O:fold:0"] == 0.97
    assert softened["anion:N"] == 0.0
    assert softened["anion:N:fold:2"] == 0.0


def test_soften_paired_floors_rejects_negative_or_mismatched_inputs():
    strict = {"anion:O": 0.99}
    metadata = {"anion:N": {"fold": "all"}}
    with np.testing.assert_raises_regex(ValueError, "matching keys"):
        soften_paired_floors(strict, metadata)
    with np.testing.assert_raises_regex(ValueError, "non-negative"):
        soften_paired_floors(
            strict,
            {"anion:O": {"fold": "all"}},
            full_anion_margin=-0.1,
        )


def test_paired_soft_strata_pool_rare_anions_before_applying_full_margin():
    rows = [
        {"source_id": f"O-{index}", "anion": "O"}
        for index in range(220)
    ]
    rows += [
        {"source_id": f"Br-{index}", "anion": "Br"}
        for index in range(70)
    ]
    rows += [
        {"source_id": f"I-{index}", "anion": "I"}
        for index in range(80)
    ]
    frame = pd.DataFrame(rows)
    baseline = np.asarray([(index % 17) != 0 for index in range(len(frame))])

    strata, floors, metadata = paired_soft_strata(
        frame,
        baseline,
        n_folds=4,
        min_anion_rows=200,
        min_cell_rows=50,
        full_anion_margin=0.0025,
        cell_margin=0.01,
    )

    other = strata["anion:other-anions"]
    assert int(other.sum()) == 150
    assert metadata["anion:other-anions"]["members"] == ["Br", "I"]
    assert metadata["anion:other-anions"]["fold"] == "all"
    assert floors["anion:other-anions"] == max(
        0.0, float(baseline[other].mean()) - 0.0025
    )


def test_joint_feature_coverage_includes_target_and_guard_columns():
    frame = pd.DataFrame(
        {
            "target": [1.0, np.nan, 2.0, 3.0],
            "guard": [1.0, 1.0, np.nan, 1.0],
        }
    )
    records = [{"feature": "target", "guard_feature": "guard"}]
    strata = {
        "left": np.asarray([True, True, False, False]),
        "right": np.asarray([False, False, True, True]),
    }

    coverage = joint_feature_coverage_by_stratum(frame, records, strata)

    assert coverage["features"] == ["guard", "target"]
    assert coverage["overall"] == 0.5
    assert coverage["by_stratum"] == {"left": 0.5, "right": 0.5}
    assert coverage["minimum_stratum"] == 0.5


def test_search_cli_refuses_to_overwrite_before_loading_inputs(tmp_path):
    output = tmp_path / "existing.json"
    output.write_text("keep", encoding="utf-8")
    with np.testing.assert_raises_regex(SystemExit, "refusing to overwrite"):
        search_main(
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
                "--out",
                str(output),
            ]
        )
    assert output.read_text(encoding="utf-8") == "keep"
