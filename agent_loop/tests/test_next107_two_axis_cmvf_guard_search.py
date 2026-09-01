from __future__ import annotations

import inspect
import math

import numpy as np
import pandas as pd
import pytest

from src.next103_dobvr_optional_guard_search import _optional_term_risk
from src.next107_two_axis_cmvf_guard_search import (
    PROTOCOL,
    build_two_axis_candidate_specs,
    build_two_axis_guard_configurations,
    materialize_composite_guard_terms,
    run_two_axis_cmvf_guard_search,
)


def _eligible_terms() -> list[dict[str, object]]:
    terms: list[dict[str, object]] = []
    for mode in ("core", "expanded"):
        for metric in ("reallocation", "overload", "log_scale_mismatch"):
            feature = f"cmvf_{mode}_{metric}"
            terms.append(
                {
                    "term_id": f"{feature}__high",
                    "feature": feature,
                    "direction": 1,
                    "transform": "log1p_nonnegative",
                    "group": f"cmvf_{mode}",
                    "support_column": f"cmvf_{mode}_supported",
                    "center": 0.0,
                    "scale": 1.0,
                }
            )
    return terms


def test_configuration_grammar_is_frozen_and_never_crosses_catalogues() -> None:
    configurations = build_two_axis_guard_configurations(_eligible_terms())

    assert PROTOCOL == "2026-08-04-next107-two-axis-cmvf-guard-search-v1"
    assert len(configurations) == 180
    assert len({str(config["configuration_id"]) for config in configurations}) == 180
    assert sum(len(config["components"]) == 1 for config in configurations) == 30
    assert sum(len(config["components"]) == 2 for config in configurations) == 150
    for config in configurations:
        groups = {str(component["group"]) for component in config["components"]}
        assert len(groups) == 1


def test_one_base_expands_to_exactly_181_candidates() -> None:
    base_records = pd.DataFrame(
        {
            "passes_source_auc_gates": [True],
            "term_ids_json": ['["old_a"]'],
            "weights_json": ["[1.0]"],
        }
    )
    configurations = build_two_axis_guard_configurations(_eligible_terms())

    specs = build_two_axis_candidate_specs(
        base_records=base_records,
        old_term_ids={"old_a"},
        configurations=configurations,
    )

    assert len(specs) == 181
    assert sum(spec["optional_term_id"] is None for spec in specs) == 1
    assert all(
        spec["optional_weight"] in (0.0, 1.0)
        for spec in specs
    )


def test_composite_encoding_exactly_matches_direct_fail_open_sum() -> None:
    terms = _eligible_terms()[:2]
    features = pd.DataFrame(
        {
            "cmvf_core_reallocation": [math.expm1(1.0), math.expm1(4.0)],
            "cmvf_core_overload": [math.expm1(2.0), math.expm1(5.0)],
            "cmvf_core_supported": [True, False],
        }
    )
    configuration = next(
        config
        for config in build_two_axis_guard_configurations(terms)
        if [component["weight"] for component in config["components"]]
        == [2.0, 0.5]
    )

    extended, composite_terms, mapping = materialize_composite_guard_terms(
        features=features,
        eligible_terms=terms,
        configurations=[configuration],
    )
    encoded_risk, active = _optional_term_risk(extended, composite_terms[0])
    raw_risk = {
        "cmvf_core_reallocation__high": 1.0,
        "cmvf_core_overload__high": 2.0,
    }
    expected = sum(
        float(component["weight"]) * raw_risk[str(component["term_id"])]
        for component in configuration["components"]
    )

    assert encoded_risk.tolist() == pytest.approx([expected, 0.0])
    assert active.tolist() == [True, False]
    assert mapping[composite_terms[0]["term_id"]]["components"] == configuration["components"]
    assert np.isfinite(extended[composite_terms[0]["feature"]]).all()


def test_runner_has_no_validation_or_replication_interface_and_guards_endpoints() -> None:
    parameters = inspect.signature(run_two_axis_cmvf_guard_search).parameters
    assert not any(
        "validation" in name or "replication" in name for name in parameters
    )
    source = inspect.getsource(run_two_axis_cmvf_guard_search)
    calibration = source.index("calibrate_optional_terms(")
    composite = source.index("materialize_composite_guard_terms(")
    scigen_endpoint = source.index('pd.read_parquet(paths["scigen_endpoint"])')
    wyformer_endpoint = source.index('pd.read_parquet(paths["wyformer_endpoint"])')
    assert calibration < composite < scigen_endpoint
    assert composite < wyformer_endpoint
