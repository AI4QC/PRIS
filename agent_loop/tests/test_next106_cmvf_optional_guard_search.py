from __future__ import annotations

import inspect
import json

import numpy as np
import pandas as pd

from src.next106_cmvf_optional_guard_search import (
    OPTIONAL_TERM_TEMPLATES,
    OPTIONAL_WEIGHT_GRID,
    PROTOCOL,
    build_optional_guard_candidate_specs,
    calibrate_optional_terms,
    compose_optional_guard_score,
    run_cmvf_optional_guard_search,
)


def test_protocol_templates_and_runner_boundary_are_frozen() -> None:
    assert PROTOCOL == "2026-08-04-next106-cmvf-optional-guard-search-v1"
    assert OPTIONAL_WEIGHT_GRID == (0.25, 0.5, 1.0, 2.0, 4.0)
    assert len(OPTIONAL_TERM_TEMPLATES) == 6
    assert {term["feature"] for term in OPTIONAL_TERM_TEMPLATES} == {
        f"cmvf_{mode}_{metric}"
        for mode in ("core", "expanded")
        for metric in ("reallocation", "overload", "log_scale_mismatch")
    }
    assert all(term["direction"] == 1 for term in OPTIONAL_TERM_TEMPLATES)
    parameters = inspect.signature(run_cmvf_optional_guard_search).parameters
    assert {"scigen_discovery_endpoint_dir", "wyformer_discovery_endpoint_dir"} <= set(
        parameters
    )
    assert not any("validation" in name or "replication" in name for name in parameters)


def test_optional_guard_keeps_base_support_when_cmvf_is_missing() -> None:
    score, supported = compose_optional_guard_score(
        base_score=np.asarray([1.0, 2.0, 3.0]),
        base_supported=np.asarray([True, True, False]),
        guard_risk=np.asarray([4.0, 0.0, 7.0]),
        guard_active=np.asarray([True, False, True]),
        guard_weight=0.5,
    )

    assert np.array_equal(supported, [True, True, False])
    assert score[0] == 3.0
    assert score[1] == 2.0
    assert np.isnan(score[2])


def test_optional_term_calibration_uses_only_active_label_free_rows() -> None:
    values = [0.1, 0.2, 0.4, 0.8, 0.15, 0.3, 0.6, 1.2, 2.4]
    features = pd.DataFrame(
        {
            "source_dataset": ["scigen"] * 10 + ["wyformer"] * 10,
            "cmvf_core_supported": [True] * 4 + [False] * 6 + [True] * 5 + [False] * 5,
            "cmvf_core_reallocation": [*values[:4], *([np.nan] * 6), *values[4:], *([np.nan] * 5)],
        }
    )
    template = ({
        "term_id": "cmvf_core_reallocation__high",
        "feature": "cmvf_core_reallocation",
        "direction": 1,
        "transform": "log1p_nonnegative",
        "group": "cmvf_core",
        "support_column": "cmvf_core_supported",
    },)

    eligible, excluded = calibrate_optional_terms(
        features,
        templates=template,
        min_source_coverage=0.15,
        min_unique_values=8,
    )

    assert excluded == []
    assert len(eligible) == 1
    assert eligible[0]["finite_rows"] == 9
    assert eligible[0]["source_coverage"] == {"scigen": 0.4, "wyformer": 0.5}


def test_candidate_catalogue_adds_zero_or_one_cmvf_guard() -> None:
    base_records = pd.DataFrame(
        {
            "passes_source_auc_gates": [True],
            "term_ids_json": [json.dumps(["old_a"])],
            "weights_json": [json.dumps([2.0])],
        }
    )
    optional_terms = [{"term_id": term["term_id"]} for term in OPTIONAL_TERM_TEMPLATES]

    specs = build_optional_guard_candidate_specs(
        base_records=base_records,
        old_term_ids={"old_a"},
        optional_terms=optional_terms,
    )

    assert len(specs) == 1 + len(optional_terms) * len(OPTIONAL_WEIGHT_GRID)
    assert sum(spec["optional_term_id"] is None for spec in specs) == 1


def test_runner_source_calibrates_terms_before_opening_endpoints() -> None:
    source = inspect.getsource(run_cmvf_optional_guard_search)
    thread_guard = source.index(
        'next105_manifest.get("solver_thread_environment")'
    )
    calibration = source.index("calibrate_optional_terms(")
    scigen_endpoint_open = source.index('pd.read_parquet(paths["scigen_endpoint"])')
    wyformer_endpoint_open = source.index('pd.read_parquet(paths["wyformer_endpoint"])')

    assert thread_guard < calibration
    assert calibration < scigen_endpoint_open
    assert calibration < wyformer_endpoint_open
