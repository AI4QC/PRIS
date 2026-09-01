import math
import json

import numpy as np
import pandas as pd

from src.next6_wbm_calibrate import (
    CandidateSpec,
    candidate_catalog,
    choose_formula,
    evaluate_catalog,
    run_calibration,
    score_candidate,
)


def _feature_row(ok: bool = True) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "material_id": ["x"],
            "feature_ok": [ok],
            "min_pair_ratio": [0.5],
            "repulsion_p2_l080": [3.0],
            "packing_l080": [2.0],
            "repulsion_p2_l090": [2.0],
            "packing_l090": [1.5],
            "repulsion_p2_l100": [1.0],
            "packing_l100": [1.0],
            "repulsion_p2_l110": [0.8],
            "packing_l110": [0.75],
            "repulsion_p2_l120": [0.7],
            "packing_l120": [0.5],
        }
    )


def test_static_and_scale_envelope_scores_follow_frozen_positive_formula():
    # Break caught: maximizing instead of minimizing over scale reverses the
    # intended volume-envelope relaxation.
    static = CandidateSpec(
        name="static",
        family="born_pack",
        mode="static",
        pack_low=0.0,
        pack_high=1.0,
        pack_weight=1.0,
        scale_penalty=0.0,
        complexity=2,
    )
    envelope = CandidateSpec(
        name="envelope",
        family="born_pack",
        mode="envelope",
        pack_low=0.0,
        pack_high=1.0,
        pack_weight=1.0,
        scale_penalty=0.0,
        complexity=3,
    )

    assert score_candidate(_feature_row(), static).tolist() == [1.0]
    assert score_candidate(_feature_row(), envelope).tolist() == [0.7]


def test_unsupported_feature_row_remains_nan_for_abstention():
    # Break caught: filling an unsupported score with zero turns abstention into KEEP.
    spec = CandidateSpec(
        name="rep",
        family="repulsion",
        mode="static",
        pack_low=0.0,
        pack_high=0.0,
        pack_weight=0.0,
        scale_penalty=0.0,
        complexity=1,
    )
    assert math.isnan(score_candidate(_feature_row(ok=False), spec)[0])


def test_candidate_catalog_has_only_nonnegative_sparse_physics_terms():
    # Break caught: an unconstrained negative coefficient could reward overlap and
    # turn the interpretable law search into an arbitrary regression.
    catalog = candidate_catalog()
    assert len(catalog) == len({spec.name for spec in catalog})
    assert all(spec.pack_weight >= 0 for spec in catalog)
    assert all(spec.scale_penalty >= 0 for spec in catalog)
    assert all(spec.complexity <= 3 for spec in catalog)


def test_formula_choice_prefers_savings_then_lower_complexity():
    # Break caught: choosing by row order makes the frozen rule non-deterministic.
    frontier = pd.DataFrame(
        {
            "name": ["complex", "simple", "unsafe"],
            "certified": [True, True, False],
            "dft_savings": [0.30, 0.30, 0.90],
            "complexity": [3, 1, 1],
        }
    )
    assert choose_formula(frontier)["name"] == "simple"


def test_catalog_evaluation_uses_only_aligned_ids_and_returns_full_frontier():
    # Break caught: positional joins can attach a label to the wrong material ID.
    features = pd.concat([_feature_row(), _feature_row()], ignore_index=True)
    features["material_id"] = ["a", "b"]
    labels = pd.DataFrame({"material_id": ["b", "a"], "stable": [False, True]})
    specs = [
        CandidateSpec("rep", "repulsion", "static", 0, 0, 0, 0, 1),
        CandidateSpec("overlap", "min_pair", "static", 0, 0, 0, 0, 1),
    ]

    got = evaluate_catalog(
        features,
        labels,
        specs=specs,
        max_false_negative_ucb=1.0,
        confidence=0.95,
    )

    assert got.name.tolist() == ["overlap", "rep"]
    assert got.n.tolist() == [2, 2]
    assert got.n_stable.tolist() == [1, 1]


def test_run_calibration_writes_json_serializable_frozen_rule(tmp_path):
    # Break caught: pandas/numpy scalar parameters otherwise make the final frozen
    # rule fail only after the expensive catalog evaluation has completed.
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    for stage in ("formula_selection", "threshold_calibration"):
        features = pd.concat([_feature_row(), _feature_row()], ignore_index=True)
        features["material_id"] = [f"{stage}-a", f"{stage}-b"]
        labels = pd.DataFrame(
            {
                "material_id": [f"{stage}-a", f"{stage}-b"],
                "stable": [True, False],
            }
        )
        features.to_parquet(artifacts / f"{stage}_x0_features.parquet", index=False)
        labels.to_parquet(artifacts / f"{stage}_labels.parquet", index=False)

    output = tmp_path / "output"
    run_calibration(
        artifacts,
        output,
        max_false_negative_ucb=1.0,
        confidence=0.95,
    )

    loaded = json.loads((output / "frozen_rule.json").read_text())
    assert loaded["formula"]["complexity"] in {1, 2, 3}
    assert isinstance(loaded["threshold"]["n_reject"], int)
