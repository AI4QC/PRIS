"""Contract tests for the np-next-20260801 search drivers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from next_law_search import (  # noqa: E402
    FROZEN_P235_SEARCH_FEATURES,
    build_next_candidate_sets,
    is_frozen_next_search_feature,
    load_isolated_search_frames,
)
from next_formula_search import next_eligible_features  # noqa: E402


def test_frozen_vocabulary_shape():
    assert len(FROZEN_P235_SEARCH_FEATURES) == 43
    # 18 P2 + 7 P3 + 18 P5; P1's frozen 18 live in better_search.
    assert "p3haw_nnls_relres" in FROZEN_P235_SEARCH_FEATURES
    assert "p2vor_an_sa_like_fraction_q95" in FROZEN_P235_SEARCH_FEATURES
    assert "p5hop_cat_mefir_rel_min" in FROZEN_P235_SEARCH_FEATURES
    # Diagnostics are never searchable.
    for diagnostic in (
        "p2vor_site_coverage",
        "p3haw_n_bonds",
        "p5hop_mefir_iterations",
        "p5hop_mefir_converged_fraction",
        "bvlocx_site_coverage",
        "bvloc_cat_relative_mismatch_min",  # expanded P1 stays out
        "bvloc_site_coverage",
    ):
        assert not is_frozen_next_search_feature(diagnostic)
    for member in (
        "bvloc_cat_absolute_mismatch_mean",
        "bvloc_an_effective_cn_q95",
        "p2vor_cat_sa_effective_cn_mean",
        "p3haw_pauling_gap",
        "p5hop_an_econ_delta_max",
    ):
        assert is_frozen_next_search_feature(member)


def _toy_frames():
    rng = np.random.default_rng(0)
    n_real, n_bad = 400, 300
    real = pd.DataFrame(
        {
            "source_id": [f"r{i}" for i in range(n_real)],
            "split": "discovery",
            "anion": "O",
            "old_feat": rng.normal(0.0, 1.0, n_real),
            "p2vor_cat_sa_effective_cn_mean": rng.normal(6.0, 1.0, n_real),
            "p3haw_nnls_relres": np.abs(rng.normal(0.01, 0.005, n_real)),
            "p5hop_cat_econ_strict_mean": rng.normal(6.0, 1.0, n_real),
            "bvloc_cat_absolute_mismatch_mean": np.abs(
                rng.normal(0.1, 0.05, n_real)
            ),
            "p2vor_site_coverage": np.ones(n_real),
            "fi": rng.uniform(0.2, 0.9, n_real),
        }
    )
    bad = pd.DataFrame(
        {
            "sid": [f"b{i}" for i in range(n_bad)],
            "psplit": "discovery",
            "parent": [f"p{i % 50}" for i in range(n_bad)],
            "kind": ["S1", "S2", "S3", "S4", "S5"] * (n_bad // 5),
            "anion": "O",
            "old_feat": rng.normal(0.0, 1.0, n_bad),
            "p2vor_cat_sa_effective_cn_mean": rng.normal(6.0, 1.0, n_bad),
            "p3haw_nnls_relres": np.abs(rng.normal(0.3, 0.1, n_bad)),
            "p5hop_cat_econ_strict_mean": rng.normal(6.0, 1.0, n_bad),
            "bvloc_cat_absolute_mismatch_mean": np.abs(
                rng.normal(0.3, 0.1, n_bad)
            ),
            "p2vor_site_coverage": np.ones(n_bad),
            "fi": rng.uniform(0.2, 0.9, n_bad),
        }
    )
    return real, bad


def test_candidate_pools_keep_new_families_out_of_existing():
    real, bad = _toy_frames()
    pools, counts = build_next_candidate_sets(
        real, bad, min_coverage=0.5, max_guard_targets=20
    )
    existing_features = {candidate.feature for candidate in pools["existing_loop"]}
    assert "old_feat" in existing_features
    assert not any(
        feature.startswith(("bvloc", "p2vor_", "p3haw_", "p5hop_"))
        for feature in existing_features
    )
    additive_features = {candidate.feature for candidate in pools["additive_bvloc_loop"]}
    assert "p3haw_nnls_relres" in additive_features  # separates the toy classes
    assert "p2vor_site_coverage" not in additive_features
    assert counts["new_features_eligible"] == 4
    assert counts["combined_candidates"] > counts["old_candidates"]


def test_formula_eligibility_respects_family_boundary():
    real, _ = _toy_frames()
    frame = real.rename(columns={"source_id": "source_id"})
    frame["rk"] = "A"
    frame["e_hull"] = 0.0
    existing = next_eligible_features(frame, include_new=False, min_coverage=0.5)
    additive = next_eligible_features(frame, include_new=True, min_coverage=0.5)
    assert existing == ["fi", "old_feat"]
    assert "p3haw_nnls_relres" in additive
    assert "bvloc_cat_absolute_mismatch_mean" in additive
    assert "p2vor_site_coverage" not in additive


def test_isolated_loader_rejects_lockbox(tmp_path):
    real = pd.DataFrame(
        {
            "source_id": ["a", "b"],
            "split": ["discovery", "lockbox"],
        }
    )
    bad = pd.DataFrame(
        {
            "sid": ["a_S1", "b_S1"],
            "psplit": ["discovery", "discovery"],
            "kind": ["S1", "S1"],
            "parent": ["a", "b"],
        }
    )
    real.to_parquet(tmp_path / "law_real.parquet", index=False)
    bad.to_parquet(tmp_path / "law_bad.parquet", index=False)
    desc_r = pd.DataFrame({"source_id": ["a", "b"], "x": [1.0, 2.0]})
    desc_b = pd.DataFrame({"sid": ["a_S1", "b_S1"], "x": [1.0, 2.0]})
    desc_r.to_parquet(tmp_path / "dr.parquet", index=False)
    desc_b.to_parquet(tmp_path / "db.parquet", index=False)
    with pytest.raises(ValueError, match="lockbox"):
        load_isolated_search_frames(tmp_path, tmp_path / "dr.parquet", tmp_path / "db.parquet")
