from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.next130_coordination_protection_search import (
    EXPECTED_CANDIDATE_COUNT,
    EXPECTED_CANDIDATE_KEY_SHA256,
    EXPECTED_FREEZE_SHA256,
    PROTECTION_TERM_ID,
    PROTECTION_WEIGHTS,
    apply_protection_score,
    build_candidate_specs,
    materialize_protected_candidates,
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
        "scigen_pooled_auc": 0.80,
        "scigen_macro_auc": 0.65,
        "scigen_worst_auc": 0.56,
        "wyformer_pooled_auc": 0.81,
        "wyformer_macro_auc": 0.70,
        "wyformer_worst_auc": 0.66,
    }


def _bases() -> pd.DataFrame:
    rows = []
    for key in ("a", "b"):
        source = _record(key)
        rows.append(
            {
                "prior_candidate_key": key,
                "term_ids_json": json.dumps([f"base_{key}", f"mhcr_{key}"]),
                "weights_json": json.dumps([1.0, 0.5]),
                "_prior_record": source,
            }
        )
    return pd.DataFrame(rows)


def test_freeze_identity_and_candidate_universe_are_exact() -> None:
    assert EXPECTED_FREEZE_SHA256 == (
        "0636a9075f50ed4e2239a66d069e68443bd31ffe755897ba43947106632a7028"
    )
    assert EXPECTED_CANDIDATE_COUNT == 1560
    assert EXPECTED_CANDIDATE_KEY_SHA256 == (
        "bad1a9c16c54ecb90ae94fc39deec7da2901c98f0f4c6f038810191f1d012730"
    )


def test_protection_only_lowers_supported_base_and_preserves_support() -> None:
    base = np.array([1.0, 0.2, 3.0, np.nan])
    base_supported = np.array([True, True, False, False])
    protection = np.array([0.6, 0.8, 2.0, 1.0])
    active = np.array([True, False, True, True])
    score, supported = apply_protection_score(
        base_score=base,
        base_supported=base_supported,
        protection=protection,
        protection_active=active,
        protection_weight=2.0,
    )
    assert supported.tolist() == base_supported.tolist()
    assert score[0] == pytest.approx(0.0)
    assert score[1] == pytest.approx(base[1])
    assert np.isnan(score[2]) and np.isnan(score[3])
    assert np.all(score[supported] <= base[supported])


def test_candidate_specs_are_base_plus_five_frozen_protection_weights() -> None:
    bases = _bases()
    specs = build_candidate_specs(
        bases=bases,
        old_term_ids={"base_a", "mhcr_a", "base_b", "mhcr_b"},
    )
    assert len(specs) == 12
    assert len({spec["candidate_key"] for spec in specs}) == 12
    assert sum(spec["protection_term_id"] is None for spec in specs) == 2
    assert {
        spec["protection_weight"]
        for spec in specs
        if spec["protection_term_id"] is not None
    } == set(PROTECTION_WEIGHTS)
    assert {
        spec["protection_term_id"]
        for spec in specs
        if spec["protection_term_id"] is not None
    } == {PROTECTION_TERM_ID}


def test_virtual_protected_candidate_recovers_exact_subtractive_score() -> None:
    bases = _bases().iloc[[0]].reset_index(drop=True)
    features = pd.DataFrame(
        {
            "base_virtual": np.sinh(np.array([1.0, 2.0, 0.4])),
            "coordination_protection": [0.5, 0.9, np.nan],
            "coordination_protection_supported": [True, True, False],
        }
    )
    base_terms = [
        {
            "term_id": "virtual_a",
            "feature": "base_virtual",
            "direction": 1,
            "transform": "asinh",
            "center": 0.0,
            "scale": 1.0,
        }
    ]
    physical_specs = build_candidate_specs(
        bases=bases,
        old_term_ids={"base_a", "mhcr_a"},
    )
    extended, terms, runtime, mapping = materialize_protected_candidates(
        features=features,
        bases=bases,
        base_virtual_terms=base_terms,
        base_virtual_by_formula={
            json.dumps(
                {"term_ids": ["base_a", "mhcr_a"], "weights": [1.0, 0.5]},
                sort_keys=True,
                separators=(",", ":"),
            ): "virtual_a"
        },
        physical_specs=physical_specs,
    )
    assert len(terms) == len(runtime) == len(mapping) == 6
    protected = next(
        spec
        for spec in runtime
        if json.loads(spec["candidate_key"])["protection_weight"] == 0.5
    )
    term = next(item for item in terms if item["term_id"] == protected["base_term_ids"][0])
    recovered = np.arcsinh(extended[term["feature"]].to_numpy(float)) / float(
        term["scale"]
    )
    assert recovered.tolist() == pytest.approx([0.75, 1.55, 0.4])
    base_only = next(
        spec
        for spec in runtime
        if json.loads(spec["candidate_key"])["protection_term_id"] is None
    )
    assert base_only["base_term_ids"] == ["virtual_a"]


def test_base_reproduction_ignores_protected_variants_and_checks_metrics() -> None:
    bases = _bases().iloc[[0]].reset_index(drop=True)
    source = bases.loc[0, "_prior_record"]
    base_key = json.dumps(
        {
            "base_term_ids": ["base_a", "mhcr_a"],
            "base_weights": [1.0, 0.5],
            "protection_term_id": None,
            "protection_weight": 0.0,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    protected_key = json.dumps(
        {
            "base_term_ids": ["base_a", "mhcr_a"],
            "base_weights": [1.0, 0.5],
            "protection_term_id": PROTECTION_TERM_ID,
            "protection_weight": 0.1,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    metrics = {
        name: source[name]
        for name in (
            "scigen_pooled_auc",
            "scigen_macro_auc",
            "scigen_worst_auc",
            "wyformer_pooled_auc",
            "wyformer_macro_auc",
            "wyformer_worst_auc",
            "passes_source_auc_gates",
            "passes_safe_all_cells",
            "safe_passing_cells",
        )
    }
    base = {"candidate_key": base_key, **metrics}
    protected = {**base, "candidate_key": protected_key, "scigen_worst_auc": 0.9}
    verify_base_reproduction(result_records=[base, protected], prior=bases)
    with pytest.raises(RuntimeError, match="base-only diagnostics"):
        verify_base_reproduction(
            result_records=[{**base, "scigen_worst_auc": 0.5}, protected],
            prior=bases,
        )
