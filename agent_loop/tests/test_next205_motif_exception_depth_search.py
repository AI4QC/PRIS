from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

import src.next202_motif_conjunction_audit as n202
import src.next203_motif_conjunction_exception_search as n203
from src.next205_motif_exception_depth_search import (
    DEPTH_LEVELS,
    EXPECTED_CANDIDATE_COUNT,
    EXPECTED_ELIGIBLE_COUNT,
    build_candidate_specs,
    motif_exception_depth_score,
    materialize_motif_exception_depth_candidates,
    run_motif_exception_depth_search,
)


def test_depths_and_formal_candidate_universe_are_exact() -> None:
    assert DEPTH_LEVELS == (0.0, 0.25, 0.5, 0.75, 1.0)
    eligible = tuple(sorted(n202.HYPOTHESES)[:EXPECTED_ELIGIBLE_COUNT])
    specs = build_candidate_specs(
        base_candidate_key="base", eligible_hypotheses=eligible
    )
    assert EXPECTED_ELIGIBLE_COUNT == 21
    assert len(specs) == 1 + 21 * 9 * 5 == EXPECTED_CANDIDATE_COUNT == 946
    assert specs[0]["certificate_hypothesis"] is None
    assert specs[0]["pardon_depth"] is None
    assert len({spec["candidate_key"] for spec in specs}) == 946


def test_depth_score_changes_only_certified_original_interval_rows() -> None:
    base = np.array(
        [n203.BROAD_THRESHOLD - 0.01, n203.BROAD_THRESHOLD, 0.30,
         n203.SAFE_THRESHOLD, 0.30]
    )
    corrected, support, active = motif_exception_depth_score(
        base_score=base,
        base_support=[True] * 5,
        certificate=[1.0, 0.5, 1.0, 1.0, np.nan],
        certificate_cutoff=0.5,
        pardon_depth=0.5,
    )
    assert support.tolist() == [True] * 5
    assert active.tolist() == [False, True, True, False, False]
    assert corrected[0] == pytest.approx(base[0])
    assert corrected[1] == pytest.approx(
        base[1] * 0.5 * n203.INTERVAL_FOLD_RATIO
    )
    assert corrected[2] < n203.BROAD_THRESHOLD
    assert corrected[3] == pytest.approx(base[3])
    assert corrected[4] == pytest.approx(base[4])


def test_depth_one_exactly_recovers_next203() -> None:
    base = np.array([n203.BROAD_THRESHOLD, 0.30, n203.SAFE_THRESHOLD])
    kwargs = {
        "base_score": base,
        "base_support": [True, True, True],
        "certificate": [1.0, 1.0, 1.0],
        "certificate_cutoff": 0.5,
    }
    expected = n203.discrete_motif_exception_score(**kwargs)
    actual = motif_exception_depth_score(**kwargs, pardon_depth=1.0)
    for observed, reference in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(observed, reference)


def test_materializer_encodes_every_candidate_as_exact_virtual_term() -> None:
    eligible = tuple(sorted(n202.HYPOTHESES)[:EXPECTED_ELIGIBLE_COUNT])
    specs = build_candidate_specs(
        base_candidate_key="base", eligible_hypotheses=eligible
    )
    features = pd.DataFrame({"material_id": ["a", "b"]})
    certificates = {
        name: np.array([1.0, np.nan], dtype=float) for name in eligible
    }
    extended, terms, runtime = materialize_motif_exception_depth_candidates(
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


def test_formal_interface_has_only_discovery_endpoints() -> None:
    parameters = tuple(inspect.signature(run_motif_exception_depth_search).parameters)
    assert "next203_dir" in parameters and "next202_dir" in parameters
    assert "scigen_discovery_endpoint_dir" in parameters
    assert "wyformer_discovery_endpoint_dir" in parameters
    assert not any(
        token in name for name in parameters for token in ("validation", "replication")
    )


def test_formal_search_fails_closed_on_missing_input(tmp_path) -> None:
    signature = inspect.signature(run_motif_exception_depth_search)
    kwargs = {
        name: tmp_path / name
        for name in signature.parameters
        if name not in {"require_formal_inputs", "search_workers"}
    }
    kwargs["require_formal_inputs"] = False
    kwargs["search_workers"] = 1
    with pytest.raises(FileNotFoundError, match="NEXT205 input is missing"):
        run_motif_exception_depth_search(**kwargs)
