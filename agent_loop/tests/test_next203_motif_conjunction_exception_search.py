from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

import src.next202_motif_conjunction_audit as n202
from src.next203_motif_conjunction_exception_search import (
    BROAD_THRESHOLD,
    CERTIFICATE_CUTOFFS,
    EXPECTED_CANDIDATE_COUNT,
    EXPECTED_ELIGIBLE_COUNT,
    SAFE_THRESHOLD,
    build_candidate_specs,
    discrete_motif_exception_score,
    materialize_motif_exception_candidates,
    run_motif_conjunction_exception_search,
)


def test_cutoffs_and_formal_candidate_universe_are_exact() -> None:
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
    eligible = tuple(sorted(n202.HYPOTHESES)[:EXPECTED_ELIGIBLE_COUNT])
    specs = build_candidate_specs(
        base_candidate_key="base", eligible_hypotheses=eligible
    )
    assert EXPECTED_ELIGIBLE_COUNT == 21
    assert len(specs) == 1 + 21 * 9 == EXPECTED_CANDIDATE_COUNT == 190
    assert specs[0]["certificate_hypothesis"] is None
    assert len({spec["candidate_key"] for spec in specs}) == 190


def test_discrete_exception_uses_original_interval_and_folds_strictly_below_broad() -> None:
    base = np.array(
        [BROAD_THRESHOLD - 0.01, BROAD_THRESHOLD, 0.30, SAFE_THRESHOLD, 0.30]
    )
    corrected, support, active = discrete_motif_exception_score(
        base_score=base,
        base_support=[True] * 5,
        certificate=[1.0, 0.5, 1.0, 1.0, np.nan],
        certificate_cutoff=0.5,
    )
    assert support.tolist() == [True] * 5
    assert active.tolist() == [False, True, True, False, False]
    assert corrected[0] == pytest.approx(base[0])
    assert corrected[1] < BROAD_THRESHOLD
    assert corrected[2] < BROAD_THRESHOLD
    assert corrected[3] == pytest.approx(base[3])
    assert corrected[4] == pytest.approx(base[4])


def test_materializer_encodes_all_candidates_as_exact_recoverable_virtual_terms() -> None:
    eligible = tuple(sorted(n202.HYPOTHESES)[:EXPECTED_ELIGIBLE_COUNT])
    specs = build_candidate_specs(
        base_candidate_key="base", eligible_hypotheses=eligible
    )
    features = pd.DataFrame({"material_id": ["a", "b"]})
    certificates = {
        name: np.array([1.0, np.nan], dtype=float) for name in eligible
    }
    extended, terms, runtime = materialize_motif_exception_candidates(
        features=features,
        base_score=[0.30, 0.40],
        base_support=[True, True],
        certificates=certificates,
        specs=specs,
    )
    assert len(terms) == len(runtime) == EXPECTED_CANDIDATE_COUNT
    for term in terms:
        encoded = extended[term["feature"]].to_numpy(float)
        recovered = np.arcsinh(encoded) / float(term["scale"])
        assert np.isfinite(recovered).all()


def test_formal_interface_has_discovery_but_no_validation_or_replication_paths() -> None:
    parameters = tuple(
        inspect.signature(run_motif_conjunction_exception_search).parameters
    )
    assert "next202_dir" in parameters
    assert "scigen_discovery_endpoint_dir" in parameters
    assert "wyformer_discovery_endpoint_dir" in parameters
    assert not any(
        token in name for name in parameters for token in ("validation", "replication")
    )


def test_formal_search_fails_closed_on_missing_input(tmp_path) -> None:
    signature = inspect.signature(run_motif_conjunction_exception_search)
    kwargs = {
        name: tmp_path / name
        for name in signature.parameters
        if name not in {"require_formal_inputs", "search_workers"}
    }
    kwargs["require_formal_inputs"] = False
    kwargs["search_workers"] = 1
    with pytest.raises(FileNotFoundError, match="NEXT203 input is missing"):
        run_motif_conjunction_exception_search(**kwargs)
