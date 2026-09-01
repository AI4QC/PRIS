from __future__ import annotations

import inspect
import json

import numpy as np
import pandas as pd
import pytest

from src.next87_scigen_sparse_law_search import _term_risk
from src.next182_local_family_closure_attenuation_search import (
    ATTENUATIONS,
    BROAD_THRESHOLD,
    ELIGIBLE_FEATURES,
    EXPECTED_CANDIDATE_COUNT,
    SAFE_THRESHOLD,
    build_candidate_specs,
    local_family_closure_score,
    materialize_local_family_candidates,
    run_local_family_closure_attenuation_search,
)


def test_operator_only_subtracts_bounded_local_family_contribution() -> None:
    base = np.asarray([BROAD_THRESHOLD - 1e-12, BROAD_THRESHOLD, 0.35, SAFE_THRESHOLD - 1e-12, SAFE_THRESHOLD])
    support = np.ones(len(base), dtype=bool)
    local = np.asarray([0.5, 0.5, 0.2, 0.1, 0.5])
    closure = np.ones(len(base))
    score, corrected_support, active = local_family_closure_score(
        base_score=base, base_support=support, local_family=local, feature=closure, attenuation=0.5
    )
    np.testing.assert_array_equal(active, [False, True, True, True, False])
    np.testing.assert_allclose(score[active], np.maximum(0.0, base[active] - 0.5 * local[active]))
    np.testing.assert_array_equal(score[~active], base[~active])
    np.testing.assert_array_equal(corrected_support, support)


def test_operator_missing_or_unsupported_keeps_base_and_rejects_bad_local_values() -> None:
    base = np.asarray([0.3, 0.3, 0.3])
    support = np.asarray([True, True, False])
    corrected, _, active = local_family_closure_score(
        base_score=base, base_support=support, local_family=[0.2, 0.2, 0.2], feature=[1.0, np.nan, 1.0], attenuation=1.0
    )
    np.testing.assert_array_equal(active, [True, False, False])
    assert corrected[1] == base[1] and corrected[2] == base[2]
    with pytest.raises(ValueError, match="local family"):
        local_family_closure_score(base_score=base, base_support=support, local_family=[0.6, 0.2, 0.2], feature=[1,1,1], attenuation=1.0)


def test_candidate_universe_is_exactly_one_base_plus_six_by_four() -> None:
    assert len(ELIGIBLE_FEATURES) == 6
    assert ATTENUATIONS == (0.25, 0.50, 0.75, 1.00)
    specs = build_candidate_specs(base_candidate_key="frozen-base")
    assert len(specs) == EXPECTED_CANDIDATE_COUNT == 25
    assert len({x["candidate_key"] for x in specs}) == 25
    for spec in specs:
        assert json.dumps(json.loads(spec["candidate_key"]), sort_keys=True, separators=(",", ":")) == spec["candidate_key"]


def test_materialized_terms_recover_every_exact_score() -> None:
    features = pd.DataFrame({name: [0.0, 0.5, 1.0, np.nan] for name in ELIGIBLE_FEATURES})
    base = np.asarray([0.1, 0.3, 0.5, 0.8])
    support = np.asarray([True, True, True, False])
    local = np.asarray([0.5, 0.4, 0.2, 0.1])
    specs = build_candidate_specs(base_candidate_key="frozen-base")
    frame, terms, runtime = materialize_local_family_candidates(features=features, base_score=base, base_support=support, local_family=local, specs=specs)
    assert len(terms) == len(runtime) == EXPECTED_CANDIDATE_COUNT
    for spec, term in zip(specs, terms, strict=True):
        recovered, recovered_support = _term_risk(frame, term)
        raw = np.full(len(frame), np.nan) if spec["strong_closure_feature"] is None else frame[str(spec["strong_closure_feature"])].to_numpy(float)
        expected, expected_support, _ = local_family_closure_score(base_score=base, base_support=support, local_family=local, feature=raw, attenuation=float(spec["attenuation"]))
        np.testing.assert_allclose(recovered[support], expected[support])
        np.testing.assert_array_equal(recovered_support, expected_support)


def test_formal_interface_has_only_discovery_endpoints() -> None:
    parameters = tuple(inspect.signature(run_local_family_closure_attenuation_search).parameters)
    assert "next181_dir" in parameters and "next180_dir" in parameters
    assert not any(token in name for name in parameters for token in ("validation", "replication"))


def test_formal_search_fails_closed_on_missing_input(tmp_path) -> None:
    signature = inspect.signature(run_local_family_closure_attenuation_search)
    kwargs = {name: tmp_path / name for name in signature.parameters if name not in {"search_workers", "require_formal_inputs"}}
    kwargs["search_workers"] = 1
    kwargs["require_formal_inputs"] = False
    with pytest.raises(FileNotFoundError, match="NEXT182 input is missing"):
        run_local_family_closure_attenuation_search(**kwargs)
