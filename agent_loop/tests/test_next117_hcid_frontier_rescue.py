from __future__ import annotations

import inspect
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.next103_dobvr_optional_guard_search import _optional_term_risk
from src.next117_hcid_frontier_rescue import (
    BASE_REPRODUCTION_AUC_TOLERANCE,
    EXPECTED_CANDIDATE_COUNT,
    EXPECTED_FREEZE_SHA256,
    EXPECTED_INPUT_SHA256,
    FROZEN_TERM_SPECS,
    PROTOCOL,
    _verify_base_reproduction,
    build_frontier_candidate_specs,
    build_hcid_guard_configurations,
    materialize_hcid_tail_terms,
    run_hcid_frontier_rescue,
    select_frontier_bases,
)


def _next114_row(
    *,
    key: str,
    auc: bool,
    safe_cells: int,
    safe_all: bool,
) -> dict[str, object]:
    return {
        "candidate_key": key,
        "base_term_ids_json": '["old_a","old_b","old_c","cmvo"]',
        "base_weights_json": "[1.0,2.0,0.5,1.0]",
        "optional_term_ids_json": '["cmvom"]',
        "optional_weights_json": "[0.5]",
        "passes_source_auc_gates": auc,
        "safe_passing_cells": safe_cells,
        "passes_safe_all_cells": safe_all,
    }


def test_frontier_selection_keeps_auc_best_safe_and_all_safe_routes() -> None:
    records = pd.DataFrame(
        [
            _next114_row(key="auc-best", auc=True, safe_cells=10, safe_all=False),
            _next114_row(key="auc-behind", auc=True, safe_cells=9, safe_all=False),
            _next114_row(key="safe", auc=False, safe_cells=12, safe_all=True),
            _next114_row(key="neither", auc=False, safe_cells=11, safe_all=False),
        ]
    )

    selected = select_frontier_bases(records)

    assert selected["prior_candidate_key"].tolist() == ["auc-best", "safe"]
    assert selected["term_ids_json"].tolist() == [
        '["old_a","old_b","old_c","cmvo","cmvom"]',
        '["old_a","old_b","old_c","cmvo","cmvom"]',
    ]
    assert selected["weights_json"].tolist() == [
        "[1.0,2.0,0.5,1.0,0.5]",
        "[1.0,2.0,0.5,1.0,0.5]",
    ]


def test_frozen_grammar_has_116_configurations_and_117_specs_per_base() -> None:
    configurations = build_hcid_guard_configurations(FROZEN_TERM_SPECS)

    assert PROTOCOL == "2026-08-08-next117-hcid-frontier-rescue-v1"
    assert EXPECTED_FREEZE_SHA256 == (
        "121de88c03b4261f944a1a6665f2244218397cd9c8580ecab8c1353eed1443e1"
    )
    assert len(FROZEN_TERM_SPECS) == 4
    assert len(configurations) == 116
    assert sum(len(item["components"]) == 1 for item in configurations) == 20
    assert sum(len(item["components"]) == 2 for item in configurations) == 96
    base = pd.DataFrame(
        {
            "prior_candidate_key": ["frontier"],
            "term_ids_json": ['["old_a","old_b","old_c","cmvo","cmvom"]'],
            "weights_json": ["[1.0,2.0,0.5,1.0,0.5]"],
        }
    )
    specs = build_frontier_candidate_specs(
        base_records=base,
        old_term_ids={"old_a", "old_b", "old_c", "cmvo", "cmvom"},
        configurations=configurations,
    )
    assert len(specs) == 117
    assert len({spec["candidate_key"] for spec in specs}) == 117
    assert EXPECTED_CANDIDATE_COUNT == 11_349


def test_frozen_next110_wyformer_input_identity_is_exact() -> None:
    assert EXPECTED_INPUT_SHA256["next110_wyformer_features"] == (
        "fd225555c8cadd2219df6fec679c74c78a9a5c15065f23553d7e6d1eec681c94"
    )


def test_hcid_materialization_uses_frozen_derivations_and_missing_is_inactive() -> None:
    rows = pd.DataFrame(
        {
            "hcid_core_supported": [True, True, False],
            "hcid_core_positive_global_deficit": [0.2, 0.2, np.nan],
            "hcid_core_positive_local_density": [0.2, 0.8, np.nan],
            "hcid_core_positive_origin_site_fraction_min": [1.0, 0.75, np.nan],
            "hcid_core_positive_neighbor_site_fraction_min": [1.0, 0.5, np.nan],
        }
    )

    extended, terms = materialize_hcid_tail_terms(rows)

    assert len(terms) == 4
    by_id = {term["term_id"]: term for term in terms}
    expected_raw = {
        "hcid_core_positive_local_density__high": [0.2, 0.8],
        "hcid_core_positive_localization_gain__high": [0.0, 0.6],
        "hcid_core_positive_origin_localization__high": [0.0, 0.25],
        "hcid_core_positive_neighbor_bottleneck__high": [0.0, 0.5],
    }
    for term_id, raw_values in expected_raw.items():
        term = by_id[term_id]
        assert extended[term["raw_feature"]].iloc[:2].tolist() == pytest.approx(raw_values)
        risk, active = _optional_term_risk(extended, term)
        assert active.tolist() == [True, True, False]
        assert risk[2] == 0.0
        assert np.isfinite(risk[:2]).all()
        assert 0.0 <= risk[0] <= risk[1]


def test_runner_hashes_label_free_catalogue_before_endpoint_reads() -> None:
    parameters = inspect.signature(run_hcid_frontier_rescue).parameters
    assert not any(
        "validation" in name or "replication" in name for name in parameters
    )
    source = inspect.getsource(run_hcid_frontier_rescue)
    materialization = source.index("materialize_hcid_tail_terms(")
    configurations = source.index("build_hcid_guard_configurations(")
    catalogue_hash = source.index("label_free_catalogue_sha256")
    scigen_endpoint = source.index('pd.read_parquet(paths["scigen_endpoint"])')
    wyformer_endpoint = source.index('pd.read_parquet(paths["wyformer_endpoint"])')
    assert materialization < configurations < catalogue_hash < scigen_endpoint
    assert catalogue_hash < wyformer_endpoint


def test_base_reproduction_allows_only_frozen_tie_scale_auc_noise() -> None:
    metrics = (
        "scigen_pooled_auc",
        "scigen_macro_auc",
        "scigen_worst_auc",
        "wyformer_pooled_auc",
        "wyformer_macro_auc",
        "wyformer_worst_auc",
    )
    prior_record = {
        **{name: 0.75 for name in metrics},
        "safe_passing_cells": 10,
        "passes_source_auc_gates": True,
        "passes_safe_all_cells": False,
    }
    prior = pd.DataFrame(
        {
            "term_ids_json": ['["a","b","c","d","e"]'],
            "weights_json": ["[1.0,2.0,0.5,1.0,0.5]"],
            "_prior_record": [prior_record],
        }
    )
    observed = {
        **{name: 0.75 + 1.9e-5 for name in metrics},
        "base_term_ids_json": '["a","b","c","d","e"]',
        "base_weights_json": "[1.0,2.0,0.5,1.0,0.5]",
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


def test_runner_fails_closed_before_any_missing_input_is_opened(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    with pytest.raises(FileNotFoundError, match="NEXT117 discovery input"):
        run_hcid_frontier_rescue(
            scigen_feature_dir=missing,
            scigen_discovery_endpoint_dir=missing,
            wyformer_feature_dir=missing,
            wyformer_discovery_endpoint_dir=missing,
            next98_dir=missing,
            next110_dir=missing,
            next111_dir=missing,
            next113_dir=missing,
            next114_dir=missing,
            next116_dir=missing,
            freeze_path=missing / "freeze.json",
            output_dir=tmp_path / "output",
            require_formal_inputs=False,
        )
