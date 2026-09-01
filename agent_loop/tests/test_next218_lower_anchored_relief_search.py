from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from src.next218_lower_anchored_relief_search import (
    EXPECTED_CANDIDATE_COUNT,
    SCORE_COMPOSITION,
    build_anchored_candidate_specs,
    lower_anchored_relief_score,
    run_lower_anchored_relief_search,
)


def test_lower_anchored_relief_cannot_cross_repair_boundary() -> None:
    lower = 0.2
    upper = 0.6
    score = np.array(
        [np.nextafter(lower, -np.inf), lower, 0.4, np.nextafter(upper, -np.inf), upper]
    )
    corrected, support, active = lower_anchored_relief_score(
        base_score=score,
        base_support=np.ones(5, dtype=bool),
        feature_values=np.ones(5),
        direction="protected_high",
        q_lo=0.0,
        q_hi=1.0,
        lower=lower,
        upper=upper,
        amplitude_fraction=0.5,
    )
    np.testing.assert_array_equal(support, np.ones(5, dtype=bool))
    np.testing.assert_array_equal(active, [False, True, True, True, False])
    assert corrected[0] == score[0]
    assert corrected[1] == lower
    assert corrected[2] == pytest.approx(0.3)
    assert corrected[3] == pytest.approx(lower + (score[3] - lower) * 0.5)
    assert corrected[4] == score[4]
    assert np.all(corrected[active] >= lower)
    assert np.all(corrected[active] <= score[active])


def test_missing_certificate_and_base_leave_score_exact() -> None:
    score = np.array([0.3, 0.4])
    corrected, _, active = lower_anchored_relief_score(
        base_score=score,
        base_support=np.ones(2, dtype=bool),
        feature_values=np.array([np.nan, 1.0]),
        direction="protected_high",
        q_lo=0.0,
        q_hi=1.0,
        lower=0.2,
        upper=0.6,
        amplitude_fraction=0.5,
    )
    assert corrected[0] == score[0]
    assert active.tolist() == [False, True]
    base, _, base_active = lower_anchored_relief_score(
        base_score=score,
        base_support=np.ones(2, dtype=bool),
        feature_values=np.full(2, np.nan),
        direction=None,
        q_lo=None,
        q_hi=None,
        lower=0.2,
        upper=0.6,
        amplitude_fraction=0.0,
    )
    np.testing.assert_array_equal(base, score)
    assert not base_active.any()


def test_candidate_specs_preserve_frozen_89_candidate_grammar() -> None:
    features = pd.DataFrame({"a": np.arange(8.0), "b": np.arange(8.0)[::-1]})
    specs = build_anchored_candidate_specs(
        base_candidate_key="next214-final",
        eligible_hypotheses=("a__protected_high", "b__protected_low"),
        features=features,
        base_score=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.59, 0.6, 0.7]),
        base_support=np.ones(8, dtype=bool),
        lower=0.2,
        upper=0.6,
        amplitude_fractions=(1 / 16, 1 / 8),
    )
    assert len(specs) == 5
    assert specs[0]["hypothesis"] is None
    assert all(spec["score_composition"] == SCORE_COMPOSITION for spec in specs)
    assert len({str(spec["candidate_key"]) for spec in specs}) == 5
    assert EXPECTED_CANDIDATE_COUNT == 89


def test_formal_interface_has_discovery_but_no_validation_or_replication() -> None:
    parameters = tuple(inspect.signature(run_lower_anchored_relief_search).parameters)
    assert "stage_dirs" in parameters and "design_paths" in parameters
    assert "scigen_discovery_endpoint_dir" in parameters
    assert "wyformer_discovery_endpoint_dir" in parameters
    assert not any(
        token in name for name in parameters for token in ("validation", "replication")
    )


def test_formal_search_fails_closed_on_missing_input(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="NEXT218 input is missing"):
        run_lower_anchored_relief_search(
            scigen_feature_dir=tmp_path / "scigen_features",
            scigen_discovery_endpoint_dir=tmp_path / "scigen_endpoint",
            wyformer_feature_dir=tmp_path / "wyformer_features",
            wyformer_discovery_endpoint_dir=tmp_path / "wyformer_endpoint",
            stage_dirs={stage: tmp_path / f"next{stage}" for stage in range(98, 218)},
            next135_freeze_path=tmp_path / "next135",
            design_paths={stage: tmp_path / f"design{stage}" for stage in range(202, 218)},
            design_path=tmp_path / "design218",
            output_dir=tmp_path / "output",
            require_formal_inputs=False,
        )
