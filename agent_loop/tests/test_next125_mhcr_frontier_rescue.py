from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.next125_mhcr_frontier_rescue import (
    EXPECTED_CANDIDATE_COUNT,
    EXPECTED_FREEZE_SHA256,
    FROZEN_TERM_SPECS,
    PROTOCOL,
    build_frontier_candidate_specs,
    build_mhcr_guard_configurations,
    materialize_mhcr_composite_guard_terms,
    materialize_mhcr_tail_terms,
    search_optional_guard_laws_parallel,
    select_frontier_bases,
    verify_base_reproduction,
)
from src.next87_scigen_sparse_law_search import assign_group_folds


def test_freeze_identity_and_term_universe_are_exact() -> None:
    assert PROTOCOL == "2026-08-08-next125-mhcr-frontier-rescue-v1"
    assert EXPECTED_FREEZE_SHA256 == (
        "e89ebcb4604ed4ae9f13fd205b6a5acf07b521948f62723d9d1bd68a85d10dde"
    )
    assert EXPECTED_CANDIDATE_COUNT == 57_178
    assert [spec["raw_feature"] for spec in FROZEN_TERM_SPECS] == [
        "mhcr_core_positive_deficit_gain_tau50",
        "mhcr_core_negative_deficit_gain_tau50",
        "mhcr_expanded_positive_deficit_gain_tau50",
        "mhcr_expanded_negative_deficit_gain_tau50",
    ]


def _feature_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "mhcr_core_supported": [True, True, False],
            "mhcr_expanded_supported": [True, False, True],
            "mhcr_core_positive_deficit_gain_tau50": [0.25, 0.5, np.nan],
            "mhcr_core_negative_deficit_gain_tau50": [0.0, 0.75, np.nan],
            "mhcr_expanded_positive_deficit_gain_tau50": [0.1, np.nan, 0.4],
            "mhcr_expanded_negative_deficit_gain_tau50": [0.2, np.nan, 0.6],
        }
    )


def test_tail_materialization_is_reversible_bounded_and_fail_open() -> None:
    table, terms = materialize_mhcr_tail_terms(_feature_frame())
    assert len(terms) == 4
    for term in terms:
        raw = term["raw_feature"]
        encoded = term["feature"]
        active = term["support_column"]
        source_raw = term["source_raw_feature"]
        mask = table[active].to_numpy(bool)
        assert np.allclose(np.arcsinh(table.loc[mask, encoded]), table.loc[mask, source_raw])
        assert np.all(table.loc[~mask, encoded] == 0.0)
        assert term["missing_policy"] == "OPTIONAL_GUARD_OFF_KEEP_BASE"
        assert raw.startswith("_")


def test_configuration_and_composite_universes_are_finite_and_unique() -> None:
    table, terms = materialize_mhcr_tail_terms(_feature_frame())
    configurations = build_mhcr_guard_configurations(FROZEN_TERM_SPECS)
    assert len(configurations) == 112
    assert len({item["configuration_id"] for item in configurations}) == 112
    assert {len(item["components"]) for item in configurations} == {1, 2}
    extended, composite_terms, mapping = materialize_mhcr_composite_guard_terms(
        features=table,
        eligible_terms=terms,
        configurations=configurations[:7],
    )
    assert len(composite_terms) == len(mapping) == 7
    for term in composite_terms:
        assert np.isfinite(pd.to_numeric(extended[term["feature"]])).all()
        assert term["term_id"] in mapping


def _record(
    key: str,
    *,
    auc: bool,
    safe_cells: int,
    safe: bool,
    scigen_worst: float,
) -> dict[str, object]:
    return {
        "candidate_key": key,
        "base_term_ids_json": json.dumps([f"term_{key}"]),
        "base_weights_json": json.dumps([1.0]),
        "passes_source_auc_gates": auc,
        "safe_passing_cells": safe_cells,
        "passes_safe_all_cells": safe,
        "scigen_pooled_auc": 0.8,
        "scigen_macro_auc": 0.65,
        "scigen_worst_auc": scigen_worst,
        "wyformer_pooled_auc": 0.8,
        "wyformer_macro_auc": 0.7,
        "wyformer_worst_auc": 0.65,
    }


def test_frontier_selection_keeps_all_auc_safe11_and_top_safe_margin() -> None:
    records = pd.DataFrame(
        [
            _record("a", auc=True, safe_cells=11, safe=False, scigen_worst=0.56),
            _record("b", auc=True, safe_cells=10, safe=False, scigen_worst=0.57),
            _record("s1", auc=False, safe_cells=12, safe=True, scigen_worst=0.549),
            _record("s2", auc=False, safe_cells=12, safe=True, scigen_worst=0.545),
            _record("s3", auc=False, safe_cells=12, safe=True, scigen_worst=0.53),
        ]
    )
    selected = select_frontier_bases(records, safe_limit=2)
    assert selected["prior_candidate_key"].tolist() == ["a", "s1", "s2"]
    assert selected["frontier_route"].tolist() == ["auc_safe11", "safe12", "safe12"]


def test_candidate_specs_attach_base_and_all_configurations() -> None:
    records = pd.DataFrame(
        [
            _record("a", auc=True, safe_cells=11, safe=False, scigen_worst=0.56),
            _record("s", auc=False, safe_cells=12, safe=True, scigen_worst=0.549),
        ]
    )
    bases = select_frontier_bases(records, safe_limit=1)
    configurations = build_mhcr_guard_configurations(FROZEN_TERM_SPECS)
    old = {"term_a", "term_s"}
    specs = build_frontier_candidate_specs(
        base_records=bases,
        old_term_ids=old,
        configurations=configurations,
    )
    assert len(specs) == 2 * 113
    assert len({spec["candidate_key"] for spec in specs}) == len(specs)
    assert sum(spec["optional_configuration_id"] is None for spec in specs) == 2


def test_base_reproduction_ignores_guard_variants_but_checks_base_metrics() -> None:
    metrics = {
        "scigen_pooled_auc": 0.80,
        "scigen_macro_auc": 0.65,
        "scigen_worst_auc": 0.56,
        "wyformer_pooled_auc": 0.81,
        "wyformer_macro_auc": 0.70,
        "wyformer_worst_auc": 0.66,
        "passes_source_auc_gates": True,
        "passes_safe_all_cells": False,
        "safe_passing_cells": 11,
    }
    prior = pd.DataFrame(
        [
            {
                "term_ids_json": json.dumps(["base__high"]),
                "weights_json": json.dumps([1.0]),
                "_prior_record": dict(metrics),
            }
        ]
    )
    base = {
        "base_term_ids_json": json.dumps(["base__high"]),
        "base_weights_json": json.dumps([1.0]),
        "optional_configuration_id": None,
        **metrics,
    }
    guarded = {
        **base,
        "optional_configuration_id": "mhcr-config",
        "scigen_worst_auc": 0.90,
    }
    verify_base_reproduction(result_records=[base, guarded], prior=prior)
    broken = {**base, "scigen_worst_auc": 0.55}
    with pytest.raises(RuntimeError, match="base-only diagnostics"):
        verify_base_reproduction(result_records=[broken, guarded], prior=prior)


def test_invalid_feature_values_and_duplicate_terms_fail_closed() -> None:
    frame = _feature_frame()
    frame.loc[0, "mhcr_core_positive_deficit_gain_tau50"] = 1.1
    with pytest.raises(ValueError, match="MHCR value"):
        materialize_mhcr_tail_terms(frame)
    duplicate = [dict(FROZEN_TERM_SPECS[0]), dict(FROZEN_TERM_SPECS[0]), *FROZEN_TERM_SPECS[2:]]
    with pytest.raises(ValueError, match="term identity"):
        build_mhcr_guard_configurations(duplicate)


def test_parallel_search_is_record_and_selection_identical_to_serial() -> None:
    groups: dict[int, str] = {}
    index = 0
    while len(groups) < 5:
        formula = f"parallel-formula-{index}"
        groups.setdefault(int(assign_group_folds(np.asarray([formula]))[0]), formula)
        index += 1
    rows = []
    endpoint = []
    for source in ("scigen", "wyformer"):
        for fold in range(5):
            for severe in (False, True):
                rows.append(
                    {
                        "material_id": f"{source}-{fold}-{int(severe)}",
                        "source_dataset": source,
                        "reduced_formula": groups[fold],
                        "crystal_system": "cubic",
                        "pauling_p2_p5_decision": "KEEP",
                        "base_feature": 5.0 if severe else 0.0,
                        "guard_feature": 1.0 if severe else 0.0,
                        "guard_supported": fold % 2 == 0,
                    }
                )
                endpoint.append(2.0 if severe else 1.0)
    features = pd.DataFrame(rows)
    old_terms = [
        {
            "term_id": "base__high",
            "feature": "base_feature",
            "direction": 1,
            "transform": "asinh",
            "center": 0.0,
            "scale": 1.0,
            "group": "base",
        }
    ]
    optional_terms = [
        {
            "term_id": "guard__high",
            "feature": "guard_feature",
            "direction": 1,
            "transform": "asinh",
            "center": 0.0,
            "scale": 1.0,
            "group": "guard",
            "support_column": "guard_supported",
        }
    ]
    specs = [
        {
            "candidate_key": f"candidate-{weight}",
            "base_term_ids": ["base__high"],
            "base_weights": [1.0],
            "optional_term_id": "guard__high",
            "optional_weight": weight,
        }
        for weight in (0.1, 0.25, 0.5, 1.0)
    ]
    serial = search_optional_guard_laws_parallel(
        features=features,
        endpoint=np.asarray(endpoint),
        old_terms=old_terms,
        optional_terms=optional_terms,
        candidate_specs=specs,
        workers=1,
    )
    parallel = search_optional_guard_laws_parallel(
        features=features,
        endpoint=np.asarray(endpoint),
        old_terms=old_terms,
        optional_terms=optional_terms,
        candidate_specs=specs,
        workers=2,
    )
    assert parallel["candidate_records"] == serial["candidate_records"]
    assert parallel["selected"] == serial["selected"]
    assert parallel["cells"] == serial["cells"]
    assert parallel["pauling_by_cell"] == serial["pauling_by_cell"]
