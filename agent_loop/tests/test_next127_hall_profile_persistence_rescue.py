from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.next127_hall_profile_persistence_rescue import (
    EXPECTED_CANDIDATE_COUNT,
    EXPECTED_FREEZE_SHA256,
    OPTIONAL_TERM_ID,
    build_candidate_specs,
    materialize_virtual_bases,
    select_next125_bases,
    verify_base_reproduction,
)


def _record(key: str, *, auc: bool = True, safe: bool = True) -> dict[str, object]:
    return {
        "candidate_key": key,
        "base_term_ids_json": json.dumps([f"base_{key}"]),
        "base_weights_json": json.dumps([1.0]),
        "optional_term_ids_json": json.dumps([f"mhcr_{key}"]),
        "optional_weights_json": json.dumps([0.5]),
        "passes_source_auc_gates": auc,
        "passes_safe_all_cells": safe,
        "safe_passing_cells": 12 if safe else 11,
        "passes_broad_all_cells": False,
        "scigen_pooled_auc": 0.80,
        "scigen_macro_auc": 0.65,
        "scigen_worst_auc": 0.56,
        "wyformer_pooled_auc": 0.81,
        "wyformer_macro_auc": 0.70,
        "wyformer_worst_auc": 0.66,
    }


def test_freeze_identity_and_candidate_count_are_exact() -> None:
    assert EXPECTED_FREEZE_SHA256 == (
        "b6a925292e9d8d6687bc44ad29bbc83d024ae1d6149a7cbfd697e8650e4d0297"
    )
    assert EXPECTED_CANDIDATE_COUNT == 1300


def test_base_selection_flattens_only_auc_safe12_formulas() -> None:
    records = pd.DataFrame(
        [
            _record("kept"),
            _record("auc_only", safe=False),
            _record("safe_only", auc=False),
        ]
    )
    bases = select_next125_bases(records)
    assert bases["prior_candidate_key"].tolist() == ["kept"]
    assert json.loads(bases.loc[0, "term_ids_json"]) == ["base_kept", "mhcr_kept"]
    assert json.loads(bases.loc[0, "weights_json"]) == [1.0, 0.5]


def test_candidate_specs_are_base_plus_four_frozen_weights() -> None:
    bases = select_next125_bases(pd.DataFrame([_record("a"), _record("b")]))
    specs = build_candidate_specs(
        bases=bases,
        old_term_ids={"base_a", "mhcr_a", "base_b", "mhcr_b"},
    )
    assert len(specs) == 10
    assert len({spec["candidate_key"] for spec in specs}) == 10
    assert sum(spec["optional_term_id"] is None for spec in specs) == 2
    assert {spec["optional_weight"] for spec in specs if spec["optional_term_id"]} == {
        0.1,
        0.25,
        0.5,
        1.0,
    }
    assert {spec["optional_term_id"] for spec in specs if spec["optional_term_id"]} == {
        OPTIONAL_TERM_ID
    }


def test_base_reproduction_ignores_hpp_variants_and_checks_metrics() -> None:
    bases = select_next125_bases(pd.DataFrame([_record("a")]))
    source = bases.loc[0, "_prior_record"]
    base = {
        "candidate_key": json.dumps(
            {
                "base_term_ids": ["base_a", "mhcr_a"],
                "base_weights": [1.0, 0.5],
                "optional_term_id": None,
                "optional_weight": 0.0,
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        "base_term_ids_json": bases.loc[0, "term_ids_json"],
        "base_weights_json": bases.loc[0, "weights_json"],
        "optional_term_id": None,
        **{name: source[name] for name in (
            "scigen_pooled_auc", "scigen_macro_auc", "scigen_worst_auc",
            "wyformer_pooled_auc", "wyformer_macro_auc", "wyformer_worst_auc",
            "passes_source_auc_gates", "passes_safe_all_cells", "safe_passing_cells",
        )},
    }
    guarded = {**base, "optional_term_id": OPTIONAL_TERM_ID, "scigen_worst_auc": 0.9}
    verify_base_reproduction(result_records=[base, guarded], prior=bases)
    with pytest.raises(RuntimeError, match="base-only diagnostics"):
        verify_base_reproduction(
            result_records=[{**base, "scigen_worst_auc": 0.5}, guarded],
            prior=bases,
        )


def test_virtual_base_preserves_nested_guard_fail_open_semantics() -> None:
    features = pd.DataFrame(
        {
            "base_feature": [1.0, 2.0],
            "guard_feature": [10.0, 10.0],
            "guard_supported": [False, True],
        }
    )
    bases = select_next125_bases(pd.DataFrame([_record("a")]))
    old_terms = [
        {
            "term_id": "base_a",
            "feature": "base_feature",
            "direction": 1,
            "transform": "asinh",
            "center": 0.0,
            "scale": 1.0,
        }
    ]
    mhcr_terms = [
        {
            "term_id": "mhcr_a",
            "feature": "guard_feature",
            "direction": 1,
            "transform": "asinh",
            "center": 0.0,
            "scale": 1.0,
            "support_column": "guard_supported",
        }
    ]
    encoded, virtual_terms, mapping = materialize_virtual_bases(
        features=features,
        bases=bases,
        old_terms=old_terms,
        mhcr_terms=mhcr_terms,
    )
    assert len(virtual_terms) == len(mapping) == 1
    term = virtual_terms[0]
    recovered = np.arcsinh(encoded[term["feature"]].to_numpy(float)) / float(
        term["scale"]
    )
    assert recovered[0] == pytest.approx(np.arcsinh(1.0))
    assert recovered[1] == pytest.approx(
        np.arcsinh(2.0) + 0.5 * np.arcsinh(10.0)
    )
