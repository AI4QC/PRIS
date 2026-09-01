from __future__ import annotations

import inspect
import json

import numpy as np
import pandas as pd
import pytest

from src.next87_scigen_sparse_law_search import _term_risk
from src.next194_signed_local_closure_audit import HYPOTHESES
from src.next195_signed_local_closure_search import (
    ATTENUATIONS,
    BROAD_THRESHOLD,
    EXPECTED_CANDIDATE_COUNT,
    EXPECTED_ELIGIBLE_COUNT,
    SAFE_THRESHOLD,
    build_candidate_specs,
    materialize_signed_local_closure_candidates,
    run_signed_local_closure_search,
    signed_local_closure_repair_score,
)


def test_operator_only_changes_supported_finite_certificate_inside_interval() -> None:
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
    certificate = np.asarray([1.0, 1.0, 0.5, 0.25, 1.0])
    score, corrected_support, active = signed_local_closure_repair_score(
        base_score=base,
        base_support=support,
        certificate=certificate,
        attenuation=2.0,
    )
    width = SAFE_THRESHOLD - BROAD_THRESHOLD
    np.testing.assert_array_equal(active, [False, True, True, True, False])
    np.testing.assert_allclose(
        score[active],
        np.maximum(0.0, base[active] - 2.0 * width * certificate[active]),
    )
    np.testing.assert_array_equal(score[~active], base[~active])
    np.testing.assert_array_equal(corrected_support, support)


def test_operator_missing_or_unsupported_keeps_base_and_rejects_bad_certificate() -> None:
    base = np.asarray([0.3, 0.3, 0.3])
    support = np.asarray([True, True, False])
    corrected, corrected_support, active = signed_local_closure_repair_score(
        base_score=base,
        base_support=support,
        certificate=[0.5, np.nan, 1.0],
        attenuation=1.0,
    )
    np.testing.assert_array_equal(active, [True, False, False])
    assert corrected[1] == base[1] and corrected[2] == base[2]
    np.testing.assert_array_equal(corrected_support, support)
    with pytest.raises(ValueError, match="certificate"):
        signed_local_closure_repair_score(
            base_score=base,
            base_support=support,
            certificate=[1.1, 0.0, 0.0],
            attenuation=1.0,
        )


def test_candidate_universe_is_one_base_plus_six_by_six() -> None:
    assert ATTENUATIONS == (0.25, 0.50, 0.75, 1.00, 1.50, 2.00)
    eligible = tuple(sorted(HYPOTHESES)[:EXPECTED_ELIGIBLE_COUNT])
    specs = build_candidate_specs(
        base_candidate_key="frozen-base", eligible_hypotheses=eligible
    )
    assert EXPECTED_ELIGIBLE_COUNT == 6
    assert len(specs) == EXPECTED_CANDIDATE_COUNT == 37
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


def test_materialized_terms_recover_every_exact_score() -> None:
    eligible = tuple(sorted(HYPOTHESES)[:2])
    specs = build_candidate_specs(
        base_candidate_key="frozen-base", eligible_hypotheses=eligible
    )
    base = np.asarray([0.1, 0.3, 0.5, 0.8])
    support = np.asarray([True, True, True, False])
    certificates = {
        eligible[0]: np.asarray([0.0, 0.5, 1.0, 1.0]),
        eligible[1]: np.asarray([1.0, np.nan, 0.25, 1.0]),
    }
    frame, terms, runtime = materialize_signed_local_closure_candidates(
        features=pd.DataFrame({"row": range(len(base))}),
        base_score=base,
        base_support=support,
        certificates=certificates,
        specs=specs,
    )
    assert len(terms) == len(runtime) == 1 + 2 * len(ATTENUATIONS)
    for spec, term in zip(specs, terms, strict=True):
        recovered, recovered_support = _term_risk(frame, term)
        hypothesis = spec["certificate_hypothesis"]
        raw = (
            np.full(len(frame), np.nan)
            if hypothesis is None
            else certificates[str(hypothesis)]
        )
        expected, expected_support, _ = signed_local_closure_repair_score(
            base_score=base,
            base_support=support,
            certificate=raw,
            attenuation=float(spec["attenuation"]),
        )
        np.testing.assert_allclose(recovered[support], expected[support])
        np.testing.assert_array_equal(recovered_support, expected_support)


def test_formal_interface_has_only_discovery_endpoints() -> None:
    parameters = tuple(inspect.signature(run_signed_local_closure_search).parameters)
    assert "next194_dir" in parameters and "next192_dir" in parameters
    assert not any(
        token in name for name in parameters for token in ("validation", "replication")
    )


def test_formal_search_fails_closed_on_missing_input(tmp_path) -> None:
    signature = inspect.signature(run_signed_local_closure_search)
    kwargs = {
        name: tmp_path / name
        for name in signature.parameters
        if name not in {"search_workers", "require_formal_inputs"}
    }
    kwargs["search_workers"] = 1
    kwargs["require_formal_inputs"] = False
    with pytest.raises(FileNotFoundError, match="NEXT195 input is missing"):
        run_signed_local_closure_search(**kwargs)
