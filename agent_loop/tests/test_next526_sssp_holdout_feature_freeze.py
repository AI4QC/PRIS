import pandas as pd
import pytest

import src.next526_sssp_holdout_feature_freeze as n


def _metadata():
    return pd.DataFrame(
        {
            "material_id": ["v1", "v2", "r1", "r2"],
            "reduced_formula": ["AB", "AC", "AD", "AE"],
            "chemical_system": ["A-B", "A-C", "A-D", "A-E"],
            "natoms": [2, 2, 2, 2],
            "partition_role": [
                "internal_validation",
                "internal_validation",
                "internal_replication",
                "internal_replication",
            ],
            "input_role": ["raw_x0"] * 4,
        }
    )


def test_select_holdout_metadata_is_role_exact_and_sorted():
    selected = n.select_holdout_metadata(
        _metadata().sample(frac=1.0, random_state=4),
        expected_rows={"internal_validation": 2, "internal_replication": 2},
    )
    assert list(selected) == ["internal_validation", "internal_replication"]
    assert selected["internal_validation"]["material_id"].tolist() == ["v1", "v2"]
    assert selected["internal_replication"]["material_id"].tolist() == ["r1", "r2"]


def test_select_holdout_metadata_rejects_endpoint_fields_and_duplicate_ids():
    bad = _metadata()
    bad["distortion_ratio"] = 0.0
    with pytest.raises(ValueError, match="endpoint field"):
        n.select_holdout_metadata(
            bad,
            expected_rows={"internal_validation": 2, "internal_replication": 2},
        )
    duplicate = _metadata()
    duplicate.loc[3, "material_id"] = "v1"
    with pytest.raises(ValueError, match="metadata differs"):
        n.select_holdout_metadata(
            duplicate,
            expected_rows={"internal_validation": 2, "internal_replication": 2},
        )


def test_boundary_flags_keep_all_endpoint_and_model_inputs_closed():
    assert set(n.BOUNDARY_FLAGS.values()) == {False}
    assert n.BOUNDARY_FLAGS["validation_endpoint_opened"] is False
    assert n.BOUNDARY_FLAGS["replication_endpoint_opened"] is False
    assert n.BOUNDARY_FLAGS["dft_values_used_by_features"] is False
