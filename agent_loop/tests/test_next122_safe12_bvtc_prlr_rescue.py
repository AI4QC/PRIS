from __future__ import annotations

import inspect

import pandas as pd
import pytest

from src.next122_safe12_bvtc_prlr_rescue import (
    EXPECTED_CANDIDATE_COUNT,
    EXPECTED_FREEZE_SHA256,
    EXPECTED_INPUT_SHA256,
    PROTOCOL,
    RESCUE_TERM_IDS,
    RESCUE_WEIGHT,
    build_rescue_candidate_specs,
    run_safe12_bvtc_prlr_rescue,
    select_safe12_bases,
)


def _next121_row(*, key: str, safe: bool) -> dict[str, object]:
    return {
        "candidate_key": key,
        "base_term_ids_json": '["old_a","old_b","old_c","cmvo"]',
        "base_weights_json": "[1.0,2.0,0.5,1.0]",
        "optional_term_ids_json": '["bvtbd_a","bvtbd_b"]',
        "optional_weights_json": "[0.1,0.1]",
        "passes_safe_all_cells": safe,
        "safe_passing_cells": 12 if safe else 11,
    }


def test_safe12_selection_flattens_complete_physical_formula() -> None:
    records = pd.DataFrame(
        [_next121_row(key="safe", safe=True), _next121_row(key="unsafe", safe=False)]
    )
    selected = select_safe12_bases(records)
    assert selected["prior_candidate_key"].tolist() == ["safe"]
    assert selected["term_ids_json"].tolist() == [
        '["old_a","old_b","old_c","cmvo","bvtbd_a","bvtbd_b"]'
    ]
    assert selected["weights_json"].tolist() == [
        "[1.0,2.0,0.5,1.0,0.1,0.1]"
    ]


def test_each_base_has_exactly_four_frozen_rescue_variants() -> None:
    assert PROTOCOL == "2026-08-08-next122-safe12-bvtc-prlr-rescue-v1"
    assert EXPECTED_FREEZE_SHA256 == (
        "65573952e7d4e6c70e63c7d9e39cf0ebae34d43ffd07cccfbf86fc88bb75522e"
    )
    assert RESCUE_TERM_IDS == (
        "bvtc_correction_rms__high",
        "prlr_bar_stress_amplification__high",
    )
    assert RESCUE_WEIGHT == 0.1
    bases = select_safe12_bases(pd.DataFrame([_next121_row(key="safe", safe=True)]))
    old = {"old_a", "old_b", "old_c", "cmvo", "bvtbd_a", "bvtbd_b", *RESCUE_TERM_IDS}
    specs = build_rescue_candidate_specs(base_records=bases, old_term_ids=old)
    assert len(specs) == 4
    assert len({spec["candidate_key"] for spec in specs}) == 4
    added = [
        tuple(term for term in spec["base_term_ids"] if term in RESCUE_TERM_IDS)
        for spec in specs
    ]
    assert set(added) == {
        (),
        (RESCUE_TERM_IDS[0],),
        (RESCUE_TERM_IDS[1],),
        RESCUE_TERM_IDS,
    }
    assert EXPECTED_CANDIDATE_COUNT == 14_292


def test_rescue_rejects_base_that_already_contains_target_term() -> None:
    row = _next121_row(key="duplicate", safe=True)
    row["optional_term_ids_json"] = '["bvtc_correction_rms__high"]'
    row["optional_weights_json"] = "[0.1]"
    bases = select_safe12_bases(pd.DataFrame([row]))
    with pytest.raises(ValueError, match="already contains rescue term"):
        build_rescue_candidate_specs(
            base_records=bases,
            old_term_ids={"old_a", "old_b", "old_c", "cmvo", *RESCUE_TERM_IDS},
        )


def test_formal_next121_input_identity_is_exact() -> None:
    assert EXPECTED_INPUT_SHA256["next121_search_records"] == (
        "6cdbecf01ab8d7ca45395a6eef3c2dc6181e72cff0c5234b254c4213e26009db"
    )


def test_runner_hashes_candidate_universe_before_endpoint_reads() -> None:
    parameters = inspect.signature(run_safe12_bvtc_prlr_rescue).parameters
    assert not any(
        "validation" in name or "replication" in name for name in parameters
    )
    source = inspect.getsource(run_safe12_bvtc_prlr_rescue)
    selection = source.index("select_safe12_bases(")
    candidates = source.index("build_rescue_candidate_specs(")
    catalogue_hash = source.index("label_free_catalogue_sha256")
    scigen_endpoint = source.index('pd.read_parquet(paths["scigen_endpoint"])')
    wyformer_endpoint = source.index('pd.read_parquet(paths["wyformer_endpoint"])')
    assert selection < candidates < catalogue_hash < scigen_endpoint
    assert catalogue_hash < wyformer_endpoint
