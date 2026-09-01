from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from src.next220_signed_redistribution_search import (
    BETA_FRACTIONS,
    EXPECTED_CANDIDATE_COUNT,
    SCORE_COMPOSITION,
    build_signed_candidate_specs,
    run_signed_redistribution_search,
    signed_redistribution_score,
)


def test_signed_redistribution_is_symmetric_about_half_protection() -> None:
    score = np.full(3, 0.4)
    corrected, support, active = signed_redistribution_score(
        base_score=score,
        base_support=np.ones(3, dtype=bool),
        feature_values=np.array([0.0, 0.5, 1.0]),
        direction="protected_high",
        q_lo=0.0,
        q_hi=1.0,
        lower=0.2,
        upper=0.6,
        beta_fraction=0.25,
    )
    np.testing.assert_array_equal(support, np.ones(3, dtype=bool))
    np.testing.assert_array_equal(active, np.ones(3, dtype=bool))
    np.testing.assert_allclose(corrected, [0.5, 0.4, 0.3])


def test_signed_redistribution_is_band_limited_and_missing_off() -> None:
    score = np.array([0.19, 0.2, 0.4, 0.59, 0.6])
    corrected, _, active = signed_redistribution_score(
        base_score=score,
        base_support=np.ones(5, dtype=bool),
        feature_values=np.array([1.0, 1.0, np.nan, 0.0, 0.0]),
        direction="protected_high",
        q_lo=0.0,
        q_hi=1.0,
        lower=0.2,
        upper=0.6,
        beta_fraction=0.25,
    )
    np.testing.assert_array_equal(active, [False, True, False, True, False])
    assert corrected[0] == score[0]
    assert corrected[1] == pytest.approx(0.1)
    assert corrected[2] == score[2]
    assert corrected[3] == pytest.approx(0.69)
    assert corrected[4] == score[4]
    assert np.all(corrected >= 0.0)


def test_candidate_specs_freeze_exact_111_candidate_grammar() -> None:
    features = pd.DataFrame({"a": np.arange(8.0), "b": np.arange(8.0)[::-1]})
    specs = build_signed_candidate_specs(
        base_candidate_key="next214-final",
        eligible_hypotheses=("a__protected_high", "b__protected_low"),
        features=features,
        base_score=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.59, 0.6, 0.7]),
        base_support=np.ones(8, dtype=bool),
        lower=0.2,
        upper=0.6,
        beta_fractions=(1 / 64, 1 / 32),
    )
    assert len(specs) == 5
    assert specs[0]["hypothesis"] is None
    assert all(spec["score_composition"] == SCORE_COMPOSITION for spec in specs)
    assert len({str(spec["candidate_key"]) for spec in specs}) == 5
    assert BETA_FRACTIONS == (1 / 64, 1 / 32, 1 / 16, 1 / 8, 1 / 4)
    assert EXPECTED_CANDIDATE_COUNT == 111


def test_formal_interface_has_discovery_but_no_validation_or_replication() -> None:
    parameters = tuple(inspect.signature(run_signed_redistribution_search).parameters)
    assert "stage_dirs" in parameters and "design_paths" in parameters
    assert "scigen_discovery_endpoint_dir" in parameters
    assert "wyformer_discovery_endpoint_dir" in parameters
    assert not any(
        token in name for name in parameters for token in ("validation", "replication")
    )


def test_formal_search_fails_closed_on_missing_input(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="NEXT220 input is missing"):
        run_signed_redistribution_search(
            scigen_feature_dir=tmp_path / "scigen_features",
            scigen_discovery_endpoint_dir=tmp_path / "scigen_endpoint",
            wyformer_feature_dir=tmp_path / "wyformer_features",
            wyformer_discovery_endpoint_dir=tmp_path / "wyformer_endpoint",
            stage_dirs={stage: tmp_path / f"next{stage}" for stage in range(98, 220)},
            next135_freeze_path=tmp_path / "next135",
            design_paths={stage: tmp_path / f"design{stage}" for stage in range(202, 220)},
            design_path=tmp_path / "design220",
            output_dir=tmp_path / "output",
            require_formal_inputs=False,
        )
