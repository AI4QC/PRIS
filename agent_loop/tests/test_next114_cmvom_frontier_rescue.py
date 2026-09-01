from __future__ import annotations

import inspect
import math

import numpy as np
import pandas as pd
import pytest

from src.next103_dobvr_optional_guard_search import _optional_term_risk
from src.next114_cmvom_frontier_rescue import (
    BASE_REPRODUCTION_AUC_TOLERANCE,
    EXPECTED_CANDIDATE_COUNT,
    EXPECTED_FREEZE_SHA256,
    FROZEN_TERM_SPECS,
    PROTOCOL,
    build_cmvom_guard_configurations,
    build_frontier_candidate_specs,
    materialize_cmvom_tail_terms,
    run_cmvom_frontier_rescue,
    select_frontier_bases,
    _verify_base_reproduction,
)


def _next111_row(
    *,
    key: str,
    auc: bool,
    safe_cells: int,
    safe_all: bool,
) -> dict[str, object]:
    return {
        "candidate_key": key,
        "base_term_ids_json": '["old_a","old_b","old_c"]',
        "base_weights_json": "[1.0,2.0,0.5]",
        "optional_term_ids_json": '["cmvo_core_component_balance_gap__high"]',
        "optional_weights_json": "[1.0]",
        "passes_source_auc_gates": auc,
        "safe_passing_cells": safe_cells,
        "passes_safe_all_cells": safe_all,
    }


def test_frozen_frontier_selection_keeps_only_the_two_conflict_routes() -> None:
    records = pd.DataFrame(
        [
            _next111_row(key="auc-frontier", auc=True, safe_cells=10, safe_all=False),
            _next111_row(key="auc-behind", auc=True, safe_cells=9, safe_all=False),
            _next111_row(key="safe-frontier", auc=False, safe_cells=12, safe_all=True),
            _next111_row(key="neither", auc=False, safe_cells=11, safe_all=False),
        ]
    )

    selected = select_frontier_bases(records)

    assert selected["prior_candidate_key"].tolist() == [
        "auc-frontier",
        "safe-frontier",
    ]
    assert selected["term_ids_json"].tolist() == [
        '["old_a","old_b","old_c","cmvo_core_component_balance_gap__high"]',
        '["old_a","old_b","old_c","cmvo_core_component_balance_gap__high"]',
    ]
    assert selected["weights_json"].tolist() == [
        "[1.0,2.0,0.5,1.0]",
        "[1.0,2.0,0.5,1.0]",
    ]


def test_frozen_grammar_has_63_configurations_and_64_specs_per_base() -> None:
    configurations = build_cmvom_guard_configurations(FROZEN_TERM_SPECS)

    assert PROTOCOL == "2026-08-08-next114-cmvom-frontier-rescue-v1"
    assert EXPECTED_FREEZE_SHA256 == (
        "83414bc8c38cff32406f5f5f8b761993aa8e43d08b8c528fe04bbe726b76827d"
    )
    assert len(FROZEN_TERM_SPECS) == 3
    assert len(configurations) == 63
    assert sum(len(item["components"]) == 1 for item in configurations) == 15
    assert sum(len(item["components"]) == 2 for item in configurations) == 48
    base = pd.DataFrame(
        {
            "prior_candidate_key": ["frontier"],
            "term_ids_json": [
                '["old_a","old_b","old_c","cmvo_core_component_balance_gap__high"]'
            ],
            "weights_json": ["[1.0,2.0,0.5,1.0]"],
        }
    )
    specs = build_frontier_candidate_specs(
        base_records=base,
        old_term_ids={
            "old_a",
            "old_b",
            "old_c",
            "cmvo_core_component_balance_gap__high",
        },
        configurations=configurations,
    )
    assert len(specs) == 64
    assert len({spec["candidate_key"] for spec in specs}) == 64
    assert EXPECTED_CANDIDATE_COUNT == 2688


def test_tail_materialization_is_exact_and_missing_is_inactive() -> None:
    columns: dict[str, object] = {
        "cmvom_core_supported": [True, True, False]
    }
    for spec in FROZEN_TERM_SPECS:
        columns[str(spec["raw_feature"])] = [
                spec["center"],
                spec["center"] + spec["scale"],
                np.nan,
        ]
    rows = pd.DataFrame(columns)

    extended, terms = materialize_cmvom_tail_terms(rows)

    assert len(terms) == 3
    for term in terms:
        risk, active = _optional_term_risk(extended, term)
        assert active.tolist() == [True, True, False]
        assert risk[0] == 0.0
        assert math.isclose(risk[1], 1.0, rel_tol=0.0, abs_tol=1.0e-12)
        assert risk[2] == 0.0


def test_runner_has_no_validation_or_replication_interface_and_freezes_first() -> None:
    parameters = inspect.signature(run_cmvom_frontier_rescue).parameters
    assert not any(
        "validation" in name or "replication" in name for name in parameters
    )
    source = inspect.getsource(run_cmvom_frontier_rescue)
    materialization = source.index("materialize_cmvom_tail_terms(")
    configurations = source.index("build_cmvom_guard_configurations(")
    catalogue_hash = source.index("label_free_catalogue_sha256")
    scigen_endpoint = source.index('pd.read_parquet(paths["scigen_endpoint"])')
    wyformer_endpoint = source.index('pd.read_parquet(paths["wyformer_endpoint"])')
    assert materialization < configurations < catalogue_hash < scigen_endpoint
    assert catalogue_hash < wyformer_endpoint


def test_base_reproduction_allows_only_tie_scale_auc_noise() -> None:
    metric_names = (
        "scigen_pooled_auc",
        "scigen_macro_auc",
        "scigen_worst_auc",
        "wyformer_pooled_auc",
        "wyformer_macro_auc",
        "wyformer_worst_auc",
    )
    prior_record = {
        **{name: 0.75 for name in metric_names},
        "safe_passing_cells": 10,
        "passes_source_auc_gates": True,
        "passes_safe_all_cells": False,
    }
    prior = pd.DataFrame(
        {
            "term_ids_json": ['["a","b","c","d"]'],
            "weights_json": ["[1.0,2.0,0.5,1.0]"],
            "_prior_record": [prior_record],
        }
    )
    observed = {
        **{name: 0.75 + 1.9e-5 for name in metric_names},
        "base_term_ids_json": '["a","b","c","d"]',
        "base_weights_json": "[1.0,2.0,0.5,1.0]",
        "optional_configuration_id": None,
        "safe_passing_cells": 10,
        "passes_source_auc_gates": True,
        "passes_safe_all_cells": False,
    }

    assert BASE_REPRODUCTION_AUC_TOLERANCE == 2.0e-5
    _verify_base_reproduction(result_records=[observed], prior=prior)

    observed["scigen_pooled_auc"] = 0.75 + 2.1e-5
    with pytest.raises(RuntimeError, match="do not reproduce"):
        _verify_base_reproduction(result_records=[observed], prior=prior)
