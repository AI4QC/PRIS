from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from src.next228_margin_local_redistribution_search import (
    AMPLITUDE_FRACTIONS,
    EXPECTED_CANDIDATE_COUNT,
    EXPECTED_ELIGIBLE_COUNT,
    LOCAL_WIDTH_FRACTIONS,
    build_margin_local_candidate_specs,
    margin_local_redistribution_score,
    materialize_margin_local_candidates,
    run_margin_local_redistribution_search,
)


def test_margin_local_score_is_signed_triangular_and_zero_at_band_edge() -> None:
    threshold = 0.5
    repair_width = 0.8
    fraction = 0.25
    amplitude = 0.5
    h = repair_width * fraction
    base = np.array(
        [threshold, threshold, threshold, threshold + h / 2,
         threshold - h / 2, threshold + h, threshold - h]
    )
    score, support, active, local_weight, protection = (
        margin_local_redistribution_score(
            base_score=base,
            base_support=np.ones(len(base), dtype=bool),
            feature_values=np.array([1.0, 0.0, 0.5, 1.0, 0.0, 1.0, 0.0]),
            direction="protected_high",
            q_lo=0.0,
            q_hi=1.0,
            threshold=threshold,
            repair_width=repair_width,
            local_width_fraction=fraction,
            amplitude_fraction=amplitude,
        )
    )
    np.testing.assert_array_equal(support, np.ones(len(base), dtype=bool))
    np.testing.assert_array_equal(active, [True, True, True, True, True, False, False])
    np.testing.assert_allclose(local_weight, [1, 1, 1, 0.5, 0.5, 0, 0])
    np.testing.assert_allclose(protection, [1, 0, 0.5, 1, 0, 1, 0])
    np.testing.assert_allclose(
        score,
        [threshold - amplitude * h, threshold + amplitude * h, threshold,
         threshold + h / 2 - amplitude * h / 2,
         threshold - h / 2 + amplitude * h / 2,
         threshold + h, threshold - h],
    )


def test_margin_local_score_keeps_missing_term_off_and_applies_floor() -> None:
    score, support, active, weight, protection = margin_local_redistribution_score(
        base_score=np.array([0.1, 0.1, 0.1]),
        base_support=np.array([True, True, False]),
        feature_values=np.array([1.0, np.nan, 1.0]),
        direction="protected_high",
        q_lo=0.0,
        q_hi=1.0,
        threshold=0.1,
        repair_width=1.0,
        local_width_fraction=0.25,
        amplitude_fraction=1.0,
    )
    np.testing.assert_array_equal(support, [True, True, False])
    np.testing.assert_array_equal(active, [True, False, False])
    np.testing.assert_allclose(score[:2], [0.0, 0.1])
    assert np.isnan(score[2])
    np.testing.assert_allclose(weight, [1.0, 0.0, 0.0])
    assert protection[0] == 1.0 and np.isnan(protection[1])


def test_candidate_specs_freeze_all_feature_width_amplitude_combinations() -> None:
    features = pd.DataFrame(
        {"a": np.arange(32, dtype=float), "b": np.arange(32, dtype=float)[::-1]}
    )
    specs = build_margin_local_candidate_specs(
        base_candidate_key="next224-frontier",
        eligible_hypotheses=("a__protected_low", "b__protected_high"),
        features=features,
        base_score=np.linspace(0.0, 1.0, len(features)),
        base_support=np.ones(len(features), dtype=bool),
        threshold=0.5,
        repair_width=0.8,
        local_width_fractions=(1 / 64, 1 / 32),
        amplitude_fractions=(0.25, 0.5, 1.0),
    )
    assert len(specs) == 13
    assert specs[0]["hypothesis"] is None
    assert specs[0]["eligible_new_candidate"] is False
    assert all(spec["eligible_new_candidate"] for spec in specs[1:])
    assert len({str(spec["candidate_key"]) for spec in specs}) == len(specs)
    assert all(spec["normalization_population"] == "ALL_FINITE_COMBINED_DISCOVERY" for spec in specs)
    assert specs[1]["q_lo"] == 1.0
    assert specs[1]["q_hi"] == 29.0
    assert LOCAL_WIDTH_FRACTIONS == (1 / 64, 1 / 32, 1 / 16, 1 / 8, 1 / 4)
    assert AMPLITUDE_FRACTIONS == (1 / 4, 1 / 2, 1.0)
    assert EXPECTED_CANDIDATE_COUNT == 391
    assert EXPECTED_ELIGIBLE_COUNT == 390


def test_materializer_encodes_exact_scores_and_preserves_support() -> None:
    features = pd.DataFrame({"source_dataset": ["scigen", "wyformer"], "a": [1.0, 0.0]})
    specs = build_margin_local_candidate_specs(
        base_candidate_key="base",
        eligible_hypotheses=("a__protected_high",),
        features=features,
        base_score=np.array([0.5, 0.5]),
        base_support=np.array([True, True]),
        threshold=0.5,
        repair_width=0.8,
        local_width_fractions=(0.25,),
        amplitude_fractions=(0.5,),
    )
    virtual, terms, runtime, activity = materialize_margin_local_candidates(
        features=features,
        base_score=np.array([0.5, 0.5]),
        base_support=np.array([True, True]),
        specs=specs,
    )
    assert len(terms) == len(runtime) == 2
    no_op = np.arcsinh(virtual[terms[0]["feature"]].to_numpy()) / terms[0]["scale"]
    corrected = np.arcsinh(virtual[terms[1]["feature"]].to_numpy()) / terms[1]["scale"]
    np.testing.assert_allclose(no_op, [0.5, 0.5])
    np.testing.assert_allclose(corrected, [0.4, 0.6])
    assert activity[str(specs[1]["candidate_key"])] == {
        "rows": 2,
        "scigen": 1,
        "wyformer": 1,
    }


def test_formal_interface_has_discovery_but_no_validation_or_replication() -> None:
    parameters = tuple(inspect.signature(run_margin_local_redistribution_search).parameters)
    assert "stage_dirs" in parameters and "design_paths" in parameters
    assert "scigen_discovery_endpoint_dir" in parameters
    assert "wyformer_discovery_endpoint_dir" in parameters
    assert not any(
        token in name for name in parameters for token in ("validation", "replication")
    )


def test_formal_search_fails_closed_on_missing_input(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="NEXT228 input is missing"):
        run_margin_local_redistribution_search(
            scigen_feature_dir=tmp_path / "scigen_features",
            scigen_discovery_endpoint_dir=tmp_path / "scigen_endpoint",
            wyformer_feature_dir=tmp_path / "wyformer_features",
            wyformer_discovery_endpoint_dir=tmp_path / "wyformer_endpoint",
            stage_dirs={stage: tmp_path / f"next{stage}" for stage in range(98, 228)},
            next135_freeze_path=tmp_path / "next135",
            design_paths={stage: tmp_path / f"design{stage}" for stage in range(202, 228)},
            design_path=tmp_path / "design228",
            output_dir=tmp_path / "output",
            require_formal_inputs=False,
        )
