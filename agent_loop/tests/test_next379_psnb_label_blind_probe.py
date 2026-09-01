from __future__ import annotations

import inspect
import json
import math

import numpy as np
import pandas as pd

import experiments.next379_psnb_label_blind_probe as p


def test_novelty_population_includes_all_formal_features_through_next375() -> None:
    assert tuple(p.PRIOR_MODULES)[-5:] == (
        "next347",
        "next355",
        "next359",
        "next367",
        "next375",
    )
    assert len(p.PRIOR_MODULES) == 30
    assert set(p.PRIOR_FILE_NAMES) == set(p.PRIOR_MODULES)
    assert p.MINIMUM_JOINT_FINITE == 40


def test_probe_gates_accept_zero_and_reject_novelty_boundary() -> None:
    sources = {
        source: {
            "supported": 80,
            "unique_rounded_10": 80,
            "minimum": 0.0,
            "maximum": 1.0,
            "maximum_invariance_error": 0.0,
            "maximum_label_free_spearman": {"absolute_correlation": 0.5},
        }
        for source in ("scigen", "wyformer")
    }
    assert all(p.evaluate_probe_gates(sources, count=80).values())
    sources["scigen"]["maximum"] = 1.0 + 1.0e-9
    assert p.evaluate_probe_gates(sources, count=80)["closed_domain"] is False
    sources["scigen"]["maximum"] = 1.0
    sources["scigen"]["maximum_label_free_spearman"][
        "absolute_correlation"
    ] = 0.90
    assert p.evaluate_probe_gates(sources, count=80)["novel"] is False


def test_novelty_requires_forty_joint_finite_observations() -> None:
    candidate = pd.Series(np.arange(80, dtype=float))
    sparse = np.full(80, np.nan)
    sparse[:4] = np.arange(4, dtype=float)
    enough = np.full(80, np.nan)
    enough[:40] = np.arange(40, dtype=float)[::-1]
    result = p.maximum_adequate_label_free_spearman(
        candidate, pd.DataFrame({"sparse": sparse, "enough": enough})
    )
    assert result["feature"] == "enough"
    assert result["joint_finite"] == 40
    assert result["sparse_skipped_control_count"] == 1


def test_json_safe_converts_nonfinite_diagnostic() -> None:
    cleaned = p._json_safe({"missing": math.nan, "finite": 0.5, "gate": False})
    assert cleaned == {"missing": None, "finite": 0.5, "gate": False}
    json.dumps(cleaned, allow_nan=False)


def test_probe_runner_exposes_only_geometry_and_label_free_inputs() -> None:
    parameters = tuple(inspect.signature(p.run_label_blind_probe).parameters)
    assert parameters == (
        "scigen_cohort_dir",
        "wyformer_cohort_dir",
        "scigen_base_feature_dir",
        "wyformer_base_feature_dir",
        "prior_feature_dirs",
        "count",
    )
    assert not any(
        token in name
        for name in parameters
        for token in ("endpoint", "label", "validation", "replication", "relax")
    )


def test_probe_source_hashes_cover_complete_frozen_execution() -> None:
    hashes = p.probe_source_hashes()
    assert tuple(hashes) == (
        "design",
        "src/next267_periodic_radical_voronoi_packing.py",
        "src/next379_periodic_skeletal_net_bottleneck.py",
        "experiments/next379_psnb_label_blind_probe.py",
        "tests/test_next379_periodic_skeletal_net_bottleneck.py",
        "tests/test_next379_psnb_label_blind_probe.py",
    )
    assert hashes["design"] == p.DESIGN_SHA256
    assert all(len(value) == 64 for value in hashes.values())
