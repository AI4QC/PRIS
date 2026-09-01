from __future__ import annotations

import inspect
import json

import numpy as np
import pandas as pd

from src.next103_dobvr_optional_guard_search import (
    OPTIONAL_WEIGHT_GRID,
    PROTOCOL,
    build_optional_guard_candidate_specs,
    calibrate_optional_terms,
    compose_optional_guard_score,
    run_dobvr_optional_guard_search,
    select_safe_and_diagnostic_once,
    search_optional_guard_laws,
)
from src.next87_scigen_sparse_law_search import assign_group_folds
from src.next98_cross_source_discovery_search import (
    diagnose_safe_threshold_feasibility,
    select_safe_threshold_across_cells,
)


def test_optional_guard_keeps_base_support_when_guard_is_missing() -> None:
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


def test_protocol_weight_grid_and_runner_boundary_are_frozen() -> None:
    assert PROTOCOL == "2026-08-04-next103-dobvr-optional-guard-search-v1"
    assert OPTIONAL_WEIGHT_GRID == (0.25, 0.5, 1.0, 2.0, 4.0)
    parameters = inspect.signature(run_dobvr_optional_guard_search).parameters
    assert {"scigen_discovery_endpoint_dir", "wyformer_discovery_endpoint_dir"} <= set(
        parameters
    )
    assert not any("validation" in name or "replication" in name for name in parameters)


def test_optional_term_calibration_uses_only_active_finite_rows() -> None:
    features = pd.DataFrame(
        {
            "source_dataset": ["scigen"] * 10 + ["wyformer"] * 10,
            "dobvr_supported": [True] * 4 + [False] * 6 + [True] * 5 + [False] * 5,
            "feature_x": [0.1, 0.2, 0.4, 0.8, *([np.nan] * 6), 0.15, 0.3, 0.6, 1.2, 2.4, *([np.nan] * 5)],
        }
    )
    templates = (
        {
            "term_id": "feature_x__high",
            "feature": "feature_x",
            "direction": 1,
            "transform": "log1p_nonnegative",
            "group": "test",
            "support_column": "dobvr_supported",
        },
    )

    eligible, excluded = calibrate_optional_terms(
        features,
        templates=templates,
        min_source_coverage=0.15,
        min_unique_values=8,
    )

    assert excluded == []
    assert len(eligible) == 1
    term = eligible[0]
    assert term["finite_rows"] == 9
    assert term["source_coverage"] == {"scigen": 0.4, "wyformer": 0.5}
    assert term["center"] == np.median(np.log1p([0.1, 0.2, 0.4, 0.8, 0.15, 0.3, 0.6, 1.2, 2.4]))
    assert term["scale"] > 0.0


def test_candidate_catalogue_adds_zero_or_one_guard_to_auc_passing_bases() -> None:
    base_records = pd.DataFrame(
        {
            "passes_source_auc_gates": [True, False],
            "term_ids_json": [json.dumps(["old_a"]), json.dumps(["old_b"])],
            "weights_json": [json.dumps([2.0]), json.dumps([1.0])],
        }
    )
    optional_terms = [{"term_id": "new_a"}, {"term_id": "new_b"}]

    specs = build_optional_guard_candidate_specs(
        base_records=base_records,
        old_term_ids={"old_a", "old_b"},
        optional_terms=optional_terms,
    )

    assert len(specs) == 1 + 2 * len(OPTIONAL_WEIGHT_GRID)
    assert sum(spec["optional_term_id"] is None for spec in specs) == 1
    assert all(spec["base_term_ids"] == ["old_a"] for spec in specs)
    assert all(spec["base_weights"] == [2.0] for spec in specs)


def test_search_keeps_base_coverage_when_optional_guard_never_activates() -> None:
    groups: dict[int, str] = {}
    index = 0
    while len(groups) < 5:
        candidate = f"formula-{index}"
        fold = int(assign_group_folds(np.asarray([candidate]))[0])
        groups.setdefault(fold, candidate)
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
                        "base_feature": 10.0 if severe else 0.0,
                        "guard_feature": np.nan,
                        "dobvr_supported": False,
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
            "support_column": "dobvr_supported",
        }
    ]
    specs = [
        {
            "candidate_key": "one",
            "base_term_ids": ["base__high"],
            "base_weights": [1.0],
            "optional_term_id": "guard__high",
            "optional_weight": 4.0,
        }
    ]

    result = search_optional_guard_laws(
        features=features,
        endpoint=np.asarray(endpoint),
        old_terms=old_terms,
        optional_terms=optional_terms,
        candidate_specs=specs,
    )

    assert result["candidate_count"] == 1
    record = result["candidate_records"][0]
    assert record["supported_rows"] == len(features)
    assert record["optional_active_rows"] == 0
    assert record["optional_activation_coverage"] == 0.0


def test_single_pass_safe_diagnostic_matches_existing_two_pass_results() -> None:
    score = np.linspace(0.0, 4.0, 120)
    endpoint = np.where(np.arange(120) % 3 == 0, 2.0, 1.0)
    supported = np.ones(120, dtype=bool)
    cells = [
        {
            "cell_id": "left",
            "source_dataset": "test",
            "fold": None,
            "kind": "source_aggregate",
            "mask": np.arange(120) < 60,
        },
        {
            "cell_id": "right",
            "source_dataset": "test",
            "fold": None,
            "kind": "source_aggregate",
            "mask": np.arange(120) >= 60,
        },
    ]
    gates = {
        "coverage_lower": 0.8,
        "protected_recall_lower": 0.8,
        "severe_rejection_precision_lower": 0.95,
        "savings_lower": 0.1,
    }

    safe, diagnostic = select_safe_and_diagnostic_once(
        score=score,
        supported=supported,
        endpoint=endpoint,
        cells=cells,
        gates=gates,
    )
    reference_safe = select_safe_threshold_across_cells(
        score=score,
        supported=supported,
        endpoint=endpoint,
        cells=cells,
        gates=gates,
    )
    reference_diagnostic = diagnose_safe_threshold_feasibility(
        score=score,
        supported=supported,
        endpoint=endpoint,
        cells=cells,
        gates=gates,
    )

    assert (safe is None) == (reference_safe is None)
    if safe is not None:
        assert safe["threshold"] == reference_safe["threshold"]
    assert diagnostic["threshold"] == reference_diagnostic["threshold"]
    assert diagnostic["passing_cells"] == reference_diagnostic["passing_cells"]
    assert diagnostic["failing_cell_ids"] == reference_diagnostic["failing_cell_ids"]
