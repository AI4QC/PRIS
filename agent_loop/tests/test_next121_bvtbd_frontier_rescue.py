from __future__ import annotations

import inspect
import json

import numpy as np
import pandas as pd
import pytest

from src.next103_dobvr_optional_guard_search import _optional_term_risk
from src.next121_bvtbd_frontier_rescue import (
    EXPECTED_CANDIDATE_COUNT,
    EXPECTED_FREEZE_SHA256,
    EXPECTED_INPUT_SHA256,
    FROZEN_TERM_SPECS,
    PROTOCOL,
    build_bvtbd_guard_configurations,
    build_frontier_candidate_specs,
    materialize_bvtbd_composite_guard_terms,
    materialize_bvtbd_tail_terms,
    run_bvtbd_frontier_rescue,
    select_frontier_bases,
)


def _next117_row(
    *, key: str, auc: bool, safe_cells: int, safe_all: bool
) -> dict[str, object]:
    return {
        "candidate_key": key,
        "base_term_ids_json": '["old_a","old_b","old_c","cmvo"]',
        "base_weights_json": "[1.0,2.0,0.5,1.0]",
        "optional_term_ids_json": '["hcid_a","hcid_b"]',
        "optional_weights_json": "[0.25,0.5]",
        "passes_source_auc_gates": auc,
        "safe_passing_cells": safe_cells,
        "passes_safe_all_cells": safe_all,
    }


def test_frontier_selection_keeps_auc_best_safe_and_all_safe_routes() -> None:
    records = pd.DataFrame(
        [
            _next117_row(key="auc-best", auc=True, safe_cells=11, safe_all=False),
            _next117_row(key="auc-behind", auc=True, safe_cells=10, safe_all=False),
            _next117_row(key="safe", auc=False, safe_cells=12, safe_all=True),
            _next117_row(key="neither", auc=False, safe_cells=11, safe_all=False),
        ]
    )

    selected = select_frontier_bases(records)

    assert selected["prior_candidate_key"].tolist() == ["auc-best", "safe"]
    assert selected["term_ids_json"].tolist() == [
        '["old_a","old_b","old_c","cmvo","hcid_a","hcid_b"]',
        '["old_a","old_b","old_c","cmvo","hcid_a","hcid_b"]',
    ]
    assert selected["weights_json"].tolist() == [
        "[1.0,2.0,0.5,1.0,0.25,0.5]",
        "[1.0,2.0,0.5,1.0,0.25,0.5]",
    ]


def test_frozen_grammar_has_116_configurations_and_117_specs_per_base() -> None:
    configurations = build_bvtbd_guard_configurations(FROZEN_TERM_SPECS)

    assert PROTOCOL == "2026-08-08-next121-bvtbd-frontier-rescue-v1"
    assert EXPECTED_FREEZE_SHA256 == (
        "b2d85550b09a2785f59890ef3ab957fd4974c06fa7be76c12d01242a404dcec8"
    )
    assert len(FROZEN_TERM_SPECS) == 4
    assert len(configurations) == 116
    assert sum(len(item["components"]) == 1 for item in configurations) == 20
    assert sum(len(item["components"]) == 2 for item in configurations) == 96
    base = pd.DataFrame(
        {
            "prior_candidate_key": ["frontier"],
            "term_ids_json": ['["old_a","old_b","old_c","hcid"]'],
            "weights_json": ["[1.0,2.0,0.5,1.0]"],
        }
    )
    specs = build_frontier_candidate_specs(
        base_records=base,
        old_term_ids={"old_a", "old_b", "old_c", "hcid"},
        configurations=configurations,
    )
    assert len(specs) == 117
    assert len({spec["candidate_key"] for spec in specs}) == 117
    assert EXPECTED_CANDIDATE_COUNT == 59_319


def test_bvtbd_composite_accepts_frozen_tenth_weight() -> None:
    rows = pd.DataFrame(
        {
            "natoms": [2],
            "bvtbd_supported": [True],
            "bvtbd_required_linf_budget": [1.0],
            "bvtbd_minimum_motion_rms": [0.1],
            "bvtbd_cell_strain_frobenius": [1.0],
            "bvtbd_deformation_debt_tau10": [0.75],
        }
    )
    extended, terms = materialize_bvtbd_tail_terms(rows)
    configuration = next(
        item
        for item in build_bvtbd_guard_configurations(FROZEN_TERM_SPECS)
        if len(item["components"]) == 1
        and item["components"][0]["weight"] == 0.1
    )
    final, composite, mapping = materialize_bvtbd_composite_guard_terms(
        features=extended,
        eligible_terms=terms,
        configurations=[configuration],
    )
    assert len(composite) == 1
    assert composite[0]["components"][0]["weight"] == 0.1
    assert composite[0]["term_id"] in mapping
    assert final[composite[0]["support_column"]].tolist() == [True]


def test_bvtbd_materialization_uses_physical_budgets_and_missing_is_inactive() -> None:
    rows = pd.DataFrame(
        {
            # Unsupported single-site rows must deactivate the optional guard;
            # their metadata must not make the whole catalogue fail.
            "natoms": [2, 2, 1],
            "bvtbd_supported": [True, True, False],
            "bvtbd_required_linf_budget": [0.05, 1.0, np.nan],
            "bvtbd_minimum_motion_rms": [0.05, 0.1, np.nan],
            "bvtbd_cell_strain_frobenius": [0.01, 10.0, np.nan],
            "bvtbd_deformation_debt_tau10": [0.25, 0.75, np.nan],
        }
    )

    extended, terms = materialize_bvtbd_tail_terms(rows)

    assert len(terms) == 4
    by_id = {term["term_id"]: term for term in terms}
    expected_second = {
        "bvtbd_required_linf_budget_decades__high": 1.0,
        "bvtbd_deformation_debt_tau10__high": 0.5,
        "bvtbd_coordinate_localization__high": pytest.approx(
            max(0.0, (1.0 / (0.1 * np.sqrt(12.0)) - 0.5) / 0.5)
        ),
        "bvtbd_cell_strain_budget_decades__high": 2.0,
    }
    for term_id, expected in expected_second.items():
        term = by_id[term_id]
        raw = extended[term["raw_feature"]].to_numpy(float)
        assert raw[0] == 0.0
        assert raw[1] == expected
        risk, active = _optional_term_risk(extended, term)
        assert active.tolist() == [True, True, False]
        assert risk[2] == 0.0
        assert np.isfinite(risk[:2]).all()


def test_formal_next120_input_identity_is_exact() -> None:
    assert EXPECTED_INPUT_SHA256["next120_wyformer_features"] == (
        "dafa685d47d02a1ed6c20c1ec3ecc4e54aaa571119b104365032705dd4974892"
    )


def test_runner_hashes_label_free_catalogue_before_endpoint_reads() -> None:
    parameters = inspect.signature(run_bvtbd_frontier_rescue).parameters
    assert not any(
        "validation" in name or "replication" in name for name in parameters
    )
    source = inspect.getsource(run_bvtbd_frontier_rescue)
    materialization = source.index("materialize_bvtbd_tail_terms(")
    configurations = source.index("build_bvtbd_guard_configurations(")
    catalogue_hash = source.index("label_free_catalogue_sha256")
    scigen_endpoint = source.index('pd.read_parquet(paths["scigen_endpoint"])')
    wyformer_endpoint = source.index('pd.read_parquet(paths["wyformer_endpoint"])')
    assert materialization < configurations < catalogue_hash < scigen_endpoint
    assert catalogue_hash < wyformer_endpoint


def test_frontier_rejects_duplicate_flattened_terms() -> None:
    row = _next117_row(key="duplicate", auc=True, safe_cells=11, safe_all=False)
    row["optional_term_ids_json"] = '["cmvo"]'
    row["optional_weights_json"] = "[0.5]"
    with pytest.raises(ValueError, match="flattened frontier formula"):
        select_frontier_bases(pd.DataFrame([row]))
