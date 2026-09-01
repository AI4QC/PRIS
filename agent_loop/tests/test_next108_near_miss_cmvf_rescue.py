from __future__ import annotations

import inspect

import pandas as pd

from src.next108_near_miss_cmvf_rescue import (
    PROTOCOL,
    build_rescue_candidate_specs,
    build_rescue_configurations,
    run_near_miss_cmvf_rescue,
    select_near_miss_bases,
)


def _eligible_terms() -> list[dict[str, object]]:
    terms = []
    for metric in ("reallocation", "overload", "log_scale_mismatch"):
        feature = f"cmvf_expanded_{metric}"
        terms.append(
            {
                "term_id": f"{feature}__high",
                "feature": feature,
                "direction": 1,
                "transform": "log1p_nonnegative",
                "group": "cmvf_expanded",
                "support_column": "cmvf_expanded_supported",
                "center": 0.0,
                "scale": 1.0,
            }
        )
    return terms


def _base_row(value: float, key: str) -> dict[str, object]:
    return {
        "candidate_key": key,
        "term_ids_json": '["old_a"]',
        "weights_json": "[1.0]",
        "passes_source_auc_gates": value >= 0.75,
        "scigen_pooled_auc": value,
        "scigen_macro_auc": value - 0.15,
        "scigen_worst_auc": value - 0.20,
        "wyformer_pooled_auc": value,
        "wyformer_macro_auc": value - 0.15,
        "wyformer_worst_auc": value - 0.20,
    }


def test_near_miss_pool_uses_all_six_finite_auc_margins() -> None:
    records = pd.DataFrame(
        [
            _base_row(0.75, "pass"),
            _base_row(0.74, "boundary"),
            _base_row(0.7399, "outside"),
            _base_row(float("nan"), "nonfinite"),
        ]
    )

    selected = select_near_miss_bases(records, tolerance=0.01)

    assert selected["candidate_key"].tolist() == ["pass", "boundary"]


def test_rescue_grammar_has_only_25_expanded_flow_pairs_and_26_per_base() -> None:
    configurations = build_rescue_configurations(_eligible_terms())

    assert PROTOCOL == "2026-08-04-next108-near-miss-cmvf-rescue-v1"
    assert len(configurations) == 25
    for configuration in configurations:
        assert [component["term_id"] for component in configuration["components"]] == [
            "cmvf_expanded_overload__high",
            "cmvf_expanded_reallocation__high",
        ]
    specs = build_rescue_candidate_specs(
        base_records=pd.DataFrame([_base_row(0.74, "near-miss")]),
        old_term_ids={"old_a"},
        configurations=configurations,
    )
    assert len(specs) == 26


def test_runner_has_no_validation_or_replication_interface_and_freezes_before_labels() -> None:
    parameters = inspect.signature(run_near_miss_cmvf_rescue).parameters
    assert not any(
        "validation" in name or "replication" in name for name in parameters
    )
    source = inspect.getsource(run_near_miss_cmvf_rescue)
    calibration = source.index("calibrate_optional_terms(")
    configurations = source.index("build_rescue_configurations(")
    scigen_endpoint = source.index('pd.read_parquet(paths["scigen_endpoint"])')
    wyformer_endpoint = source.index('pd.read_parquet(paths["wyformer_endpoint"])')
    assert calibration < configurations < scigen_endpoint
    assert configurations < wyformer_endpoint
