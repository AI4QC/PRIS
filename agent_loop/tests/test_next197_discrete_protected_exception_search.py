from __future__ import annotations

import inspect
import json

import numpy as np
import pandas as pd
import pytest

from src.next87_scigen_sparse_law_search import _term_risk
from src.next197_discrete_protected_exception_search import (
    BROAD_THRESHOLD,
    CERTIFICATE_CUTOFFS,
    EXPECTED_CANDIDATE_COUNT,
    EXPECTED_ELIGIBLE_COUNT,
    SAFE_THRESHOLD,
    build_candidate_specs,
    discrete_protected_exception_score,
    materialize_discrete_exception_candidates,
    run_discrete_protected_exception_search,
)


ELIGIBLE = (
    "psndc_crystalnn_closure_min__signed_local_safe__product__high",
    "psndc_crystalnn_closure_min__signed_local_safe__minimum__high",
    "psndc_crystalnn_closure_q10__signed_local_safe__product__high",
    "psndc_crystalnn_closure_q10__signed_local_safe__minimum__high",
    "psndc_crystalnn_volume_q10__signed_local_safe__product__high",
    "psndc_crystalnn_volume_q10__signed_local_safe__minimum__high",
)


def test_interval_fold_uses_cutoff_equality_and_strict_safe_boundary() -> None:
    base = np.asarray(
        [
            BROAD_THRESHOLD - 1.0e-12,
            BROAD_THRESHOLD,
            0.35,
            SAFE_THRESHOLD - 1.0e-12,
            SAFE_THRESHOLD,
        ]
    )
    support = np.ones(len(base), dtype=bool)
    certificate = np.asarray([1.0, 0.25, 0.249999, 1.0, 1.0])
    score, folded_support, active = discrete_protected_exception_score(
        base_score=base,
        base_support=support,
        certificate=certificate,
        certificate_cutoff=0.25,
    )
    np.testing.assert_array_equal(active, [False, True, False, True, False])
    np.testing.assert_allclose(
        score[active], base[active] * (BROAD_THRESHOLD / SAFE_THRESHOLD)
    )
    assert np.all(score[active] < BROAD_THRESHOLD)
    np.testing.assert_array_equal(score[~active], base[~active])
    np.testing.assert_array_equal(folded_support, support)


def test_missing_unsupported_and_bad_values_fail_closed() -> None:
    base = np.asarray([0.3, 0.3, 0.3])
    support = np.asarray([True, True, False])
    score, folded_support, active = discrete_protected_exception_score(
        base_score=base,
        base_support=support,
        certificate=[0.5, np.nan, 1.0],
        certificate_cutoff=0.25,
    )
    np.testing.assert_array_equal(active, [True, False, False])
    assert score[1] == base[1] and score[2] == base[2]
    np.testing.assert_array_equal(folded_support, support)
    with pytest.raises(ValueError, match="certificate"):
        discrete_protected_exception_score(
            base_score=base,
            base_support=support,
            certificate=[1.01, 0.0, 0.0],
            certificate_cutoff=0.25,
        )
    with pytest.raises(ValueError, match="cutoff"):
        discrete_protected_exception_score(
            base_score=base,
            base_support=support,
            certificate=[0.5, 0.5, 0.5],
            certificate_cutoff=0.3,
        )


def test_candidate_universe_is_exact_six_by_nine_plus_base() -> None:
    assert CERTIFICATE_CUTOFFS == (
        1 / 16,
        1 / 8,
        3 / 16,
        1 / 4,
        3 / 8,
        1 / 2,
        5 / 8,
        3 / 4,
        7 / 8,
    )
    specs = build_candidate_specs(
        base_candidate_key="frozen-base", eligible_hypotheses=ELIGIBLE
    )
    assert EXPECTED_ELIGIBLE_COUNT == len(ELIGIBLE) == 6
    assert len(specs) == EXPECTED_CANDIDATE_COUNT == 55
    assert len({str(spec["candidate_key"]) for spec in specs}) == len(specs)
    for spec in specs:
        assert (
            json.dumps(
                json.loads(str(spec["candidate_key"])),
                sort_keys=True,
                separators=(",", ":"),
            )
            == spec["candidate_key"]
        )


def test_materialized_terms_recover_every_exact_folded_score() -> None:
    eligible = ELIGIBLE[:2]
    specs = build_candidate_specs(
        base_candidate_key="frozen-base", eligible_hypotheses=eligible
    )
    base = np.asarray([0.1, 0.3, 0.5, 0.8])
    support = np.asarray([True, True, True, False])
    certificates = {
        eligible[0]: np.asarray([0.0, 0.5, 1.0, 1.0]),
        eligible[1]: np.asarray([1.0, np.nan, 0.25, 1.0]),
    }
    frame, terms, runtime = materialize_discrete_exception_candidates(
        features=pd.DataFrame({"row": range(len(base))}),
        base_score=base,
        base_support=support,
        certificates=certificates,
        specs=specs,
    )
    assert len(terms) == len(runtime) == 1 + 2 * len(CERTIFICATE_CUTOFFS)
    for spec, term in zip(specs, terms, strict=True):
        recovered, recovered_support = _term_risk(frame, term)
        hypothesis = spec["certificate_hypothesis"]
        raw = (
            np.full(len(frame), np.nan)
            if hypothesis is None
            else certificates[str(hypothesis)]
        )
        expected, expected_support, _ = discrete_protected_exception_score(
            base_score=base,
            base_support=support,
            certificate=raw,
            certificate_cutoff=float(spec["certificate_cutoff"]),
        )
        np.testing.assert_allclose(recovered[support], expected[support])
        np.testing.assert_array_equal(recovered_support, expected_support)


def test_formal_interface_is_discovery_only_and_binds_next196() -> None:
    parameters = tuple(
        inspect.signature(run_discrete_protected_exception_search).parameters
    )
    assert "next196_dir" in parameters and "next195_dir" in parameters
    assert not any(
        token in name for name in parameters for token in ("validation", "replication")
    )


def test_formal_search_fails_closed_on_missing_input(tmp_path) -> None:
    signature = inspect.signature(run_discrete_protected_exception_search)
    kwargs = {
        name: tmp_path / name
        for name in signature.parameters
        if name not in {"search_workers", "require_formal_inputs"}
    }
    kwargs["search_workers"] = 1
    kwargs["require_formal_inputs"] = False
    with pytest.raises(FileNotFoundError, match="NEXT197 input is missing"):
        run_discrete_protected_exception_search(**kwargs)
