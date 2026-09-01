"""Contract tests for the np-next-20260801c search vocabulary."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from next3_law_search import (  # noqa: E402
    FROZEN_SIXFAM_SEARCH_FEATURES,
    NEXT3_GUARD_COLUMNS,
    build_next3_candidate_sets,
    is_frozen_sixfam_search_feature,
)


def test_sixfam_vocabulary_shape():
    assert len(FROZEN_SIXFAM_SEARCH_FEATURES) == 33
    assert "p4csm_cat_q95" in FROZEN_SIXFAM_SEARCH_FEATURES
    assert "p6gap_an_gap_ratio_max" in FROZEN_SIXFAM_SEARCH_FEATURES
    assert "p7poly_an_contact_min" in FROZEN_SIXFAM_SEARCH_FEATURES
    assert "p8nnj_cat_jaccard_mean" in FROZEN_SIXFAM_SEARCH_FEATURES
    assert "p9lew_bond_mismatch_max" in FROZEN_SIXFAM_SEARCH_FEATURES
    assert "p10vor_an_freevol_mean" in FROZEN_SIXFAM_SEARCH_FEATURES
    # previous 61 stay searchable; diagnostics never are
    assert is_frozen_sixfam_search_feature("bvloc_cat_absolute_mismatch_mean")
    assert is_frozen_sixfam_search_feature("p3haw_nnls_relres")
    assert is_frozen_sixfam_search_feature("p2vor_an_sa_like_fraction_max")
    for diagnostic in (
        "p4csm_site_coverage",
        "p6gap_site_coverage",
        "p7poly_an_coverage",
        "p8nnj_site_coverage",
        "p10vor_site_coverage",
    ):
        assert not is_frozen_sixfam_search_feature(diagnostic)
    assert "p7poly_an_contact_min" in NEXT3_GUARD_COLUMNS
    assert len(NEXT3_GUARD_COLUMNS) == 11


def test_next3_pools_separate_families():
    rng = np.random.default_rng(0)
    n_real, n_bad = 300, 250
    real = pd.DataFrame(
        {
            "source_id": [f"r{i}" for i in range(n_real)],
            "split": "discovery",
            "anion": "O",
            "old_feat": rng.normal(0.0, 1.0, n_real),
            "p4csm_cat_max": rng.uniform(0.0, 2.0, n_real),
            "p6gap_cat_gap_ratio_mean": rng.normal(1.5, 0.2, n_real),
            "p7poly_an_contact_min": rng.uniform(1.5, 3.0, n_real),
            "z_an_abs": np.full(n_real, 2.0),
            "fi": rng.uniform(0.2, 0.9, n_real),
        }
    )
    bad = pd.DataFrame(
        {
            "sid": [f"b{i}" for i in range(n_bad)],
            "psplit": "discovery",
            "parent": [f"p{i % 40}" for i in range(n_bad)],
            "kind": ["S1", "S2", "S3", "S4", "S5"] * (n_bad // 5),
            "anion": "O",
            "old_feat": rng.normal(0.0, 1.0, n_bad),
            "p4csm_cat_max": rng.uniform(2.0, 6.0, n_bad),
            "p6gap_cat_gap_ratio_mean": rng.normal(1.2, 0.2, n_bad),
            "p7poly_an_contact_min": rng.uniform(1.5, 3.0, n_bad),
            "z_an_abs": np.full(n_bad, 2.0),
            "fi": rng.uniform(0.2, 0.9, n_bad),
        }
    )
    pools, counts = build_next3_candidate_sets(
        real, bad, min_coverage=0.5, max_guard_targets=20
    )
    existing_features = {c.feature for c in pools["existing_loop"]}
    assert "old_feat" in existing_features
    assert not any(
        f.startswith(("p4csm_", "p6gap_", "p7poly_", "p8nnj_", "p9lew_", "p10vor_", "bvloc", "p2vor_", "p3haw_", "p5hop_"))
        for f in existing_features
    )
    additive_features = {c.feature for c in pools["additive_bvloc_loop"]}
    assert "p4csm_cat_max" in additive_features
    assert "z_an_abs" not in additive_features  # guard-only column
    assert counts["new_features_eligible"] == 3
