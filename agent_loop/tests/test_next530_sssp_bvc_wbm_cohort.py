from pathlib import Path

import pandas as pd
import pytest

import src.next530_sssp_bvc_wbm_cohort as n


def test_selection_key_is_salted_stable_and_id_sensitive():
    assert n.selection_key("wbm-1") == n.selection_key("wbm-1")
    assert n.selection_key("wbm-1") != n.selection_key("wbm-2")
    with pytest.raises(ValueError, match="material_id"):
        n.selection_key("")


def test_exclusion_union_is_exact_and_rejects_label_fields(tmp_path: Path):
    left = tmp_path / "left.parquet"
    right = tmp_path / "right.parquet"
    pd.DataFrame({"material_id": ["a", "b"], "natoms": [2, 3]}).to_parquet(left)
    pd.DataFrame({"material_id": ["b", "c"], "formula": ["AB", "AC"]}).to_parquet(right)
    result = n.validated_exclusion_union(
        [left, right], source_ids={"a", "b", "c", "d"}
    )
    assert result == ("a", "b", "c")

    bad = tmp_path / "bad.parquet"
    pd.DataFrame({"material_id": ["a"], "endpoint": [0.0]}).to_parquet(bad)
    with pytest.raises(ValueError, match="label-free contract"):
        n.validated_exclusion_union([bad], source_ids={"a"})


def test_formal_cohort_constants_match_frozen_design():
    assert n.SAMPLE_SIZE == 8192
    assert n.MIN_ATOMS == 2
    assert n.MAX_ATOMS == 12
    assert n.SELECTION_SALT == "next530-sssp-bvc-wbm-relaxation-v1"
