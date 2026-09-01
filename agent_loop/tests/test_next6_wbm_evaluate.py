import json

import numpy as np
import pandas as pd
import pytest

from src.next6_wbm_evaluate import (
    apply_frozen_rule,
    classification_metrics,
    run_test_evaluation,
)


def _features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "material_id": ["a", "b", "c"],
            "feature_ok": [True, True, False],
            "min_pair_ratio": [0.5, 1.0, np.nan],
            "repulsion_p2_l080": [2.0, 0.0, np.nan],
            "packing_l080": [1.0, 1.0, np.nan],
            "repulsion_p2_l090": [1.5, 0.0, np.nan],
            "packing_l090": [1.0, 1.0, np.nan],
            "repulsion_p2_l100": [1.0, 0.0, np.nan],
            "packing_l100": [1.0, 1.0, np.nan],
            "repulsion_p2_l110": [0.8, 0.0, np.nan],
            "packing_l110": [1.0, 1.0, np.nan],
            "repulsion_p2_l120": [0.7, 0.0, np.nan],
            "packing_l120": [1.0, 1.0, np.nan],
        }
    )


def _frozen() -> dict:
    return {
        "formula": {
            "name": "repulsion_static",
            "family": "repulsion",
            "mode": "static",
            "pack_low": 0.0,
            "pack_high": 0.0,
            "pack_weight": 0.0,
            "scale_penalty": 0.0,
            "complexity": 1,
        },
        "threshold": {"threshold": 0.5},
    }


def test_frozen_rule_rejects_high_scores_and_abstains_on_missing_features():
    # Break caught: using <= at deployment would reverse the calibrated score direction.
    got = apply_frozen_rule(_features(), _frozen())
    assert got.decision.tolist() == ["REJECT", "KEEP", "ABSTAIN"]
    assert got.score.tolist()[:2] == [1.0, 0.0]
    assert np.isnan(got.score.iloc[2])


def test_classification_metrics_count_abstain_as_predicted_stable_dft_work():
    # Break caught: dropping abstentions changes both precision and reported budget.
    stable = np.array([True, True, False, False])
    decisions = np.array(["KEEP", "REJECT", "REJECT", "ABSTAIN"])
    scores = np.array([0.1, 1.0, 0.9, np.nan])
    got = classification_metrics(stable, decisions, scores, top_ks=(2,))

    assert got["stable_precision"] == 0.5
    assert got["stable_recall"] == 0.5
    assert got["stable_f1"] == 0.5
    assert got["dft_savings"] == 0.5
    assert got["abstention_rate"] == 0.25
    assert got["daf"] == 1.0
    assert got["top_2_precision"] == 0.5


@pytest.mark.filterwarnings("error")
def test_test_evaluation_opens_label_partition_once_by_default(tmp_path):
    # Break caught: silently rerunning the test partition enables iterative test tuning.
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    features = _features()
    labels = pd.DataFrame(
        {
            "material_id": ["a", "b", "c"],
            "formula_key": ["H2O", "LiO", "ClNa"],
            "stable": [False, True, False],
        }
    )
    features.to_parquet(artifacts / "test_x0_features.parquet", index=False)
    labels.to_parquet(artifacts / "test_labels.parquet", index=False)
    frozen_path = tmp_path / "frozen_rule.json"
    frozen_path.write_text(json.dumps(_frozen()))
    output = tmp_path / "result"

    metrics = run_test_evaluation(artifacts, frozen_path, output, bootstrap_reps=20)
    assert metrics["n"] == 3
    assert (output / "test_metrics.json").is_file()
    assert len((artifacts / "TEST_OPENINGS.jsonl").read_text().splitlines()) == 1

    with pytest.raises(RuntimeError, match="already opened"):
        run_test_evaluation(artifacts, frozen_path, output, bootstrap_reps=20)
