from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.next60_odac23_robust_scaffold_endpoint import ENDPOINT_COLUMN
from src.next82_prlr_discovery_evaluation import (
    evaluate_frozen_prlr_discovery,
    replication_ready,
)


def _frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n_rows = 240
    material_id = [f"m{index:03d}" for index in range(n_rows)]
    endpoint = np.tile(np.asarray([0.0] * 20 + [0.1] * 20 + [0.3] * 20), 4)
    score = endpoint + np.linspace(0.0, 1.0e-4, n_rows)
    features = pd.DataFrame(
        {
            "material_id": material_id,
            "partition_role": "discovery",
            "defective": np.repeat([False, False, True, True], 60),
            "open_metal_site": np.repeat([False, True, False, True], 60),
        }
    )
    predictions = pd.DataFrame(
        {
            "material_id": material_id,
            "partition_role": "discovery",
            "prlr_risk": score,
            "supported": True,
            "reject": endpoint >= 0.3,
        }
    )
    labels = pd.DataFrame(
        {
            "material_id": material_id,
            "partition_role": "discovery",
            ENDPOINT_COLUMN: endpoint,
        }
    )
    return features, predictions, labels


def test_frozen_prediction_evaluation_can_pass_all_gates_and_safety_margin() -> None:
    result = evaluate_frozen_prlr_discovery(*_frames())

    assert result["passes_original_gates"]
    assert result["passes_replication_readiness_margin"]
    assert result["metrics"]["reject_precision_lower"] >= 0.80
    assert result["rows"] == 240


def test_replication_ready_requires_point_eight_precision_lower_bound() -> None:
    metrics = {
        "coverage_lower": 0.99,
        "protected_recall_lower": 0.98,
        "reject_precision_lower": 0.799,
        "savings_lower": 0.03,
        "pooled_extreme_auc": 0.80,
        "macro_stratum_auc": 0.75,
        "worst_stratum_auc": 0.70,
    }

    assert not replication_ready(metrics)
    assert replication_ready({**metrics, "reject_precision_lower": 0.80})


def test_evaluation_rejects_non_discovery_labels() -> None:
    features, predictions, labels = _frames()
    labels["partition_role"] = "internal_replication"

    with pytest.raises(ValueError, match="discovery"):
        evaluate_frozen_prlr_discovery(features, predictions, labels)
