"""Contract tests for the robust anion-aware additive law search."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from better_law_search import LawCandidate  # noqa: E402
from next4_law_search import (  # noqa: E402
    build_fixed_guard_candidates,
    build_next4_candidate_sets,
    deterministic_real_folds,
    paired_robust_strata,
    robust_pareto_beam,
    main as search_main,
)


def _candidate(
    name: str,
    real_mask: list[bool],
    bad_mask: list[bool],
) -> LawCandidate:
    return LawCandidate(
        description=name,
        feature=name,
        family="one-sided",
        origin="test",
        side="hi",
        thresholds=(1.0,),
        real_mask=np.asarray(real_mask, dtype=bool),
        bad_mask=np.asarray(bad_mask, dtype=bool),
        real_coverage=1.0,
        bad_coverage=1.0,
    )


def test_deterministic_real_folds_are_stable_under_row_reordering():
    frame = pd.DataFrame({"source_id": ["r7", "r2", "r9", "r1", "r5"]})
    original = dict(
        zip(frame["source_id"], deterministic_real_folds(frame, n_folds=4))
    )
    reordered = frame.sample(frac=1.0, random_state=17).reset_index(drop=True)
    shuffled = dict(
        zip(reordered["source_id"], deterministic_real_folds(reordered, n_folds=4))
    )
    assert original == shuffled
    assert set(original.values()).issubset({0, 1, 2, 3})


def test_paired_robust_strata_use_exact_baseline_rate_and_drop_small_cells():
    rows: list[dict[str, object]] = []
    for anion, total in (("O", 280), ("N", 260), ("Br", 120)):
        for index in range(total):
            rows.append({"source_id": f"{anion}-{index}", "anion": anion})
    frame = pd.DataFrame(rows)
    baseline = np.asarray(
        [(index % 13) != 0 for index in range(len(frame))], dtype=bool
    )

    strata, floors, metadata = paired_robust_strata(
        frame,
        baseline,
        n_folds=4,
        min_anion_rows=200,
        min_cell_rows=50,
    )

    assert "anion:O" in strata
    assert "anion:N" in strata
    assert "anion:Br" not in strata
    assert all(not name.startswith("anion:Br:") for name in strata)
    assert any(name.startswith("anion:O:fold:") for name in strata)
    assert any(name.startswith("anion:N:fold:") for name in strata)
    assert set(strata) == set(floors) == set(metadata)
    for name, mask in strata.items():
        assert int(mask.sum()) >= 50
        assert floors[name] == float(baseline[mask].mean())
        assert metadata[name]["n"] == int(mask.sum())
        assert metadata[name]["baseline_satisfaction"] == floors[name]


def test_robust_beam_rejects_pooled_candidate_that_breaks_one_real_stratum():
    candidates = [
        _candidate(
            "pooled-but-tail-unsafe",
            [True, True, False, True],
            [False, False, False, True],
        ),
        _candidate(
            "tail-safe",
            [False, True, True, True],
            [False, True, False, True],
        ),
    ]
    result = robust_pareto_beam(
        candidates,
        real_size=4,
        bad_size=4,
        bad_kinds=np.asarray(["S1", "S1", "S2", "S2"], dtype=object),
        satisfaction_floor=0.75,
        max_rules=1,
        width=4,
        min_gain=0.0,
        real_strata={"tail": np.asarray([False, False, True, True])},
        stratum_floors={"tail": 1.0},
    )
    assert result.indices == (1,)


def test_robust_beam_prioritizes_minimum_kind_before_pooled_rejection():
    candidates = [
        _candidate(
            "high-pooled-zero-worst-kind",
            [True, True],
            [False, False, False, True],
        ),
        _candidate(
            "balanced",
            [True, True],
            [False, True, True, False],
        ),
    ]
    result = robust_pareto_beam(
        candidates,
        real_size=2,
        bad_size=4,
        bad_kinds=np.asarray(["S1", "S1", "S1", "S2"], dtype=object),
        satisfaction_floor=1.0,
        max_rules=1,
        width=4,
        min_gain=0.0,
    )
    assert result.indices == (1,)


def test_fixed_guards_only_extend_named_mechanism_targets():
    real = pd.DataFrame(
        {
            "p7c_an_short_contact_frac": [0.0, 0.5, np.nan],
            "bvloc_parameter_exact_fraction": [0.95, 0.80, np.nan],
        }
    )
    bad = pd.DataFrame(
        {
            "p7c_an_short_contact_frac": [0.0, 0.5, np.nan],
            "bvloc_parameter_exact_fraction": [0.95, 0.80, np.nan],
        }
    )
    targets = [
        _candidate("p2c_an_sa_like_fraction_max", [False] * 3, [False] * 3),
        _candidate("p2c_an_sa_effective_cn_max", [False] * 3, [False] * 3),
        _candidate("bvloc_an_absolute_mismatch_max", [False] * 3, [False] * 3),
        _candidate("p9c_bond_mismatch_max", [False] * 3, [False] * 3),
    ]
    guarded = build_fixed_guard_candidates(real, bad, targets)
    assert len(guarded) == 2
    by_feature = {candidate.feature: candidate for candidate in guarded}
    p2 = by_feature["p2c_an_sa_like_fraction_max"]
    assert p2.guard_feature == "p7c_an_short_contact_frac"
    assert p2.guard_side == "lo"
    assert p2.guard_threshold == 0.0
    np.testing.assert_array_equal(p2.real_mask, [False, True, True])
    bvloc = by_feature["bvloc_an_absolute_mismatch_max"]
    assert bvloc.guard_feature == "bvloc_parameter_exact_fraction"
    assert bvloc.guard_side == "hi"
    assert bvloc.guard_threshold == 0.9
    np.testing.assert_array_equal(bvloc.real_mask, [False, True, True])


def test_next4_pool_substitutes_corrected_features_and_drops_tainted_columns():
    rng = np.random.default_rng(5)
    n_real, n_bad = 300, 250
    real = pd.DataFrame(
        {
            "source_id": [f"r{i}" for i in range(n_real)],
            "anion": "O",
            "old_feat": rng.normal(size=n_real),
            "p2vor_an_sa_like_fraction_max": rng.uniform(0, 1, n_real),
            "p2c_an_sa_like_fraction_max": rng.uniform(0, 1, n_real),
            "p6c_an_gap_ratio_max": rng.uniform(1, 2, n_real),
            "p7c_an_short_contact_frac": rng.choice([0.0, 0.5], n_real),
            "p9c_bond_mismatch_max": rng.uniform(0, 2, n_real),
            "bvloc_parameter_exact_fraction": rng.uniform(0, 1, n_real),
        }
    )
    bad = pd.DataFrame(
        {
            "sid": [f"b{i}" for i in range(n_bad)],
            "parent": [f"p{i % 50}" for i in range(n_bad)],
            "kind": ["S1", "S2", "S3", "S4", "S5"] * 50,
            "anion": "O",
            "old_feat": rng.normal(size=n_bad),
            "p2vor_an_sa_like_fraction_max": rng.uniform(0, 1, n_bad),
            "p2c_an_sa_like_fraction_max": rng.uniform(0.5, 1.5, n_bad),
            "p6c_an_gap_ratio_max": rng.uniform(0.5, 1.5, n_bad),
            "p7c_an_short_contact_frac": rng.choice([0.0, 0.5], n_bad),
            "p9c_bond_mismatch_max": rng.uniform(1, 3, n_bad),
            "bvloc_parameter_exact_fraction": rng.uniform(0, 1, n_bad),
        }
    )
    pools, _counts = build_next4_candidate_sets(
        real,
        bad,
        min_coverage=0.5,
        max_guard_targets=20,
    )
    existing_features = {candidate.feature for candidate in pools["existing_loop"]}
    additive_features = {
        candidate.feature for candidate in pools["additive_corrected_loop"]
    }
    assert "old_feat" in existing_features
    assert not any(feature.startswith("p2c_") for feature in existing_features)
    assert "p2c_an_sa_like_fraction_max" in additive_features
    assert "p6c_an_gap_ratio_max" in additive_features
    assert "p9c_bond_mismatch_max" in additive_features
    assert "p2vor_an_sa_like_fraction_max" not in additive_features
    assert "p7c_an_short_contact_frac" not in additive_features


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
