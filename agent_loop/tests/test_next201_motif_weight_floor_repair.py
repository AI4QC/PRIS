from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from src.next201_motif_weight_floor_repair import (
    ATTENUATIONS,
    EXPECTED_CANDIDATE_COUNT,
    FLOOR_THRESHOLDS,
    build_candidate_specs,
    materialize_motif_weight_floor_candidates,
    motif_weight_floor_certificate,
    motif_weight_floor_repair_score,
    run_motif_weight_floor_search,
)


def test_floor_and_attenuation_grids_are_exact_and_candidate_count_is_fixed() -> None:
    assert FLOOR_THRESHOLDS == tuple(1.0 - 2.0**-power for power in range(0, 11))
    assert ATTENUATIONS == (0.25, 0.50, 0.75, 1.00, 1.50, 2.00)
    specs = build_candidate_specs(base_candidate_key="base")
    assert len(specs) == EXPECTED_CANDIDATE_COUNT == 67
    assert specs[0]["floor_threshold"] is None
    assert len({spec["candidate_key"] for spec in specs}) == 67


def test_certificate_is_fixed_ramp_to_ideal_and_missing_safe() -> None:
    actual = motif_weight_floor_certificate(
        [np.nan, -0.5, 0.5, 0.75, 1.0, 1.5], floor_threshold=0.5
    )
    assert np.isnan(actual[0])
    assert actual[1:] == pytest.approx([0.0, 0.0, 0.5, 1.0, 1.0])
    with pytest.raises(ValueError, match="NEXT201 floor threshold differs"):
        motif_weight_floor_certificate([1.0], floor_threshold=1.0)


def test_repair_changes_only_original_frozen_interval_and_keeps_missing_base() -> None:
    score, support, active = motif_weight_floor_repair_score(
        base_score=[0.10, 0.20, 0.30, 0.40, 0.30],
        base_support=[True, True, True, True, True],
        certificate=[1.0, 1.0, 1.0, 1.0, np.nan],
        attenuation=1.0,
        broad_threshold=0.20,
        safe_threshold=0.40,
    )
    assert score == pytest.approx([0.10, 0.0, 0.10, 0.40, 0.30])
    assert support.tolist() == [True] * 5
    assert active.tolist() == [False, True, True, False, False]


def test_materializer_encodes_every_candidate_as_exact_recoverable_virtual_term() -> None:
    features = pd.DataFrame({"material_id": ["a", "b"]})
    specs = build_candidate_specs(base_candidate_key="base")
    extended, terms, runtime = materialize_motif_weight_floor_candidates(
        features=features,
        base_score=[0.3, 0.4],
        base_support=[True, True],
        motif_weight_sum_min=[1.0, np.nan],
        specs=specs,
    )
    assert len(terms) == len(runtime) == EXPECTED_CANDIDATE_COUNT
    for term in terms:
        encoded = extended[term["feature"]].to_numpy(float)
        recovered = np.arcsinh(encoded) / float(term["scale"])
        assert np.isfinite(recovered).all()


def test_formal_interface_has_discovery_but_no_validation_or_replication_paths() -> None:
    parameters = tuple(inspect.signature(run_motif_weight_floor_search).parameters)
    assert "next200_dir" in parameters
    assert "scigen_discovery_endpoint_dir" in parameters
    assert "wyformer_discovery_endpoint_dir" in parameters
    assert not any(
        token in name for name in parameters for token in ("validation", "replication")
    )


def test_formal_search_fails_closed_on_missing_input(tmp_path) -> None:
    signature = inspect.signature(run_motif_weight_floor_search)
    kwargs = {
        name: tmp_path / name
        for name in signature.parameters
        if name not in {"require_formal_inputs", "search_workers"}
    }
    kwargs["require_formal_inputs"] = False
    kwargs["search_workers"] = 1
    with pytest.raises(FileNotFoundError, match="NEXT201 input is missing"):
        run_motif_weight_floor_search(**kwargs)
