from __future__ import annotations

import json

import pandas as pd
import pytest

from src.next132_extended_coordination_protection_search import (
    EXTENDED_WEIGHTS,
    PROTECTION_TERM_ID,
    build_extended_candidate_specs,
    select_extended_bases,
    verify_weight2_reproduction,
)


def _next125_base(name: str) -> dict[str, object]:
    return {
        "prior_candidate_key": f"prior-{name}",
        "term_ids_json": json.dumps([f"base-{name}", f"mhcr-{name}"]),
        "weights_json": json.dumps([1.0, 0.5]),
        "_prior_record": {"base_term_ids_json": "[]", "optional_term_ids_json": "[]"},
    }


def _next130_record(name: str, *, weight: float = 2.0, safe: bool = True) -> dict[str, object]:
    key = json.dumps(
        {
            "base_term_ids": [f"base-{name}", f"mhcr-{name}"],
            "base_weights": [1.0, 0.5],
            "protection_term_id": PROTECTION_TERM_ID,
            "protection_weight": weight,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return {
        "candidate_key": key,
        "passes_source_auc_gates": True,
        "passes_safe_all_cells": safe,
        "safe_passing_cells": 12 if safe else 11,
        "protection_term_id": PROTECTION_TERM_ID,
        "protection_weight": weight,
        "scigen_pooled_auc": 0.8,
        "scigen_macro_auc": 0.7,
        "scigen_worst_auc": 0.56,
        "wyformer_pooled_auc": 0.8,
        "wyformer_macro_auc": 0.7,
        "wyformer_worst_auc": 0.66,
    }


def test_select_extended_bases_matches_weight2_safe_formulas_to_next125() -> None:
    next125 = pd.DataFrame([_next125_base("a"), _next125_base("b")])
    next130 = pd.DataFrame(
        [
            _next130_record("a"),
            _next130_record("b", weight=1.0),
            _next130_record("b", safe=False),
        ]
    )
    selected = select_extended_bases(next130, next125)
    assert selected["term_ids_json"].map(json.loads).tolist() == [
        ["base-a", "mhcr-a"]
    ]
    assert selected.loc[0, "_next130_record"]["protection_weight"] == 2.0


def test_extended_specs_have_six_frozen_weights_per_base() -> None:
    bases = pd.DataFrame([_next125_base("a"), _next125_base("b")])
    specs = build_extended_candidate_specs(
        bases=bases,
        old_term_ids={"base-a", "mhcr-a", "base-b", "mhcr-b"},
    )
    assert len(specs) == 12
    assert {spec["protection_weight"] for spec in specs} == set(EXTENDED_WEIGHTS)
    assert {spec["protection_term_id"] for spec in specs} == {PROTECTION_TERM_ID}


def test_weight2_reproduction_checks_published_metrics_and_ignores_new_weights() -> None:
    next125 = pd.DataFrame([_next125_base("a")])
    prior = _next130_record("a")
    bases = select_extended_bases(pd.DataFrame([prior]), next125)
    base = {"candidate_key": prior["candidate_key"], **prior}
    extended_key = json.dumps(
        {**json.loads(prior["candidate_key"]), "protection_weight": 3.0},
        sort_keys=True,
        separators=(",", ":"),
    )
    extended = {**base, "candidate_key": extended_key, "scigen_worst_auc": 0.9}
    verify_weight2_reproduction(result_records=[base, extended], prior=bases)
    with pytest.raises(RuntimeError, match="diagnostics"):
        verify_weight2_reproduction(
            result_records=[{**base, "scigen_worst_auc": 0.5}, extended],
            prior=bases,
        )
