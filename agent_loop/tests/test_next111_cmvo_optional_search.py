from __future__ import annotations

import inspect
import math

import numpy as np
import pandas as pd

from src.next103_dobvr_optional_guard_search import _optional_term_risk
from src.next111_cmvo_optional_search import (
    FROZEN_TERM_SPECS,
    PROTOCOL,
    build_cmvo_candidate_specs,
    build_cmvo_guard_configurations,
    materialize_cmvo_tail_terms,
    run_cmvo_optional_search,
)


def _base_row() -> dict[str, object]:
    return {
        "candidate_key": "near-miss",
        "term_ids_json": '["old_a"]',
        "weights_json": "[1.0]",
        "passes_source_auc_gates": True,
    }


def test_frozen_grammar_has_63_configurations_and_64_specs_per_base() -> None:
    configurations = build_cmvo_guard_configurations(FROZEN_TERM_SPECS)

    assert PROTOCOL == "2026-08-08-next111-cmvo-optional-search-v1"
    assert len(FROZEN_TERM_SPECS) == 3
    assert len(configurations) == 63
    assert sum(len(item["components"]) == 1 for item in configurations) == 15
    assert sum(len(item["components"]) == 2 for item in configurations) == 48
    specs = build_cmvo_candidate_specs(
        base_records=pd.DataFrame([_base_row()]),
        old_term_ids={"old_a"},
        configurations=configurations,
    )
    assert len(specs) == 64


def test_tail_materialization_is_exact_and_missing_is_inactive() -> None:
    rows = pd.DataFrame(
        {
            "cmvo_core_supported": [True, True, False],
            "cmvo_core_min_interval_slack": [
                FROZEN_TERM_SPECS[0]["center"],
                FROZEN_TERM_SPECS[0]["center"]
                + FROZEN_TERM_SPECS[0]["scale"],
                np.nan,
            ],
            "cmvo_core_global_balance_gap": [0.0, 1.0, np.nan],
            "cmvo_core_component_balance_gap": [0.0, 1.0, np.nan],
        }
    )

    extended, encoded_terms = materialize_cmvo_tail_terms(rows)

    assert len(encoded_terms) == 3
    by_id = {term["term_id"]: term for term in encoded_terms}
    slack = by_id["cmvo_core_min_interval_slack__high"]
    risk, active = _optional_term_risk(extended, slack)
    assert active.tolist() == [True, True, False]
    assert risk[0] == 0.0
    assert math.isclose(risk[1], 1.0, rel_tol=0.0, abs_tol=1.0e-12)
    assert risk[2] == 0.0

    global_term = by_id["cmvo_core_global_balance_gap__high"]
    global_risk, global_active = _optional_term_risk(extended, global_term)
    assert global_active.tolist() == [True, True, False]
    assert global_risk[0] == 0.0
    assert math.isclose(
        global_risk[1],
        global_term["clip_normalized"],
        rel_tol=0.0,
        abs_tol=1.0e-12,
    )


def test_runner_has_no_validation_or_replication_interface_and_freezes_first() -> None:
    parameters = inspect.signature(run_cmvo_optional_search).parameters
    assert not any(
        "validation" in name or "replication" in name for name in parameters
    )
    source = inspect.getsource(run_cmvo_optional_search)
    materialization = source.index("materialize_cmvo_tail_terms(")
    configurations = source.index("build_cmvo_guard_configurations(")
    catalogue_hash = source.index("label_free_catalogue_sha256")
    scigen_endpoint = source.index('pd.read_parquet(paths["scigen_endpoint"])')
    wyformer_endpoint = source.index('pd.read_parquet(paths["wyformer_endpoint"])')
    assert materialization < configurations < catalogue_hash < scigen_endpoint
    assert catalogue_hash < wyformer_endpoint
