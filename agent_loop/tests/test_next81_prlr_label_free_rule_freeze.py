from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.next60_odac23_robust_scaffold_endpoint import ENDPOINT_COLUMN
from src.next81_prlr_label_free_rule_freeze import freeze_label_free_prlr_rule


def _features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "material_id": [f"m{index:02d}" for index in range(22)],
            "partition_role": ["discovery"] * 20
            + ["internal_validation", "internal_replication"],
            "repulsive_load_supported": [True] * 22,
            "prlr_risk": np.arange(22, dtype=float),
        }
    )


def test_freeze_uses_only_unlabeled_discovery_q95_and_applies_to_all_roles() -> None:
    result = freeze_label_free_prlr_rule(_features())
    expected = float(np.quantile(np.arange(20, dtype=float), 0.95, method="inverted_cdf"))

    assert result["formula"]["threshold"] == expected
    assert result["formula"]["feature"] == "prlr_risk"
    assert result["formula"]["missing_policy"] == "KEEP"
    predictions = result["predictions"]
    assert len(predictions) == 22
    assert predictions.loc[predictions.material_id.eq("m20"), "reject"].item()
    assert result["summary"]["threshold_fit_rows"] == 20


def test_freeze_rejects_any_endpoint_column() -> None:
    features = _features().assign(**{ENDPOINT_COLUMN: 0.0})

    with pytest.raises(ValueError, match="endpoint"):
        freeze_label_free_prlr_rule(features)


def test_freeze_rejects_duplicate_identity_or_missing_discovery_score() -> None:
    duplicate = pd.concat([_features(), _features().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="identity"):
        freeze_label_free_prlr_rule(duplicate)

    unsupported = _features()
    unsupported.loc[0, "repulsive_load_supported"] = False
    with pytest.raises(ValueError, match="discovery"):
        freeze_label_free_prlr_rule(unsupported)
