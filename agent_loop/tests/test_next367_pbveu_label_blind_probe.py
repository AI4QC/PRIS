from __future__ import annotations

import inspect

import numpy as np
import pandas as pd

from experiments.next367_pbveu_label_blind_probe import (
    BASE_FEATURE_FILES,
    DESIGN_SHA256,
    MINIMUM_JOINT_FINITE,
    PRIOR_FILE_NAMES,
    PRIOR_MODULES,
    evaluate_probe_gates,
    maximum_adequate_label_free_spearman,
    probe_source_hashes,
    run_label_blind_probe,
    select_probe_ids,
)


def test_novelty_population_includes_base_and_all_formal_features_through_next359() -> None:
    assert BASE_FEATURE_FILES == {
        "scigen": "features_discovery.parquet",
        "wyformer": "wyformer_x0_features_discovery.parquet",
    }
    assert tuple(PRIOR_MODULES)[-4:] == ("next323", "next347", "next355", "next359")
    assert len(PRIOR_MODULES) == 28
    assert set(PRIOR_FILE_NAMES) == set(PRIOR_MODULES)
    assert MINIMUM_JOINT_FINITE == 40


def test_probe_selection_is_deterministic_and_unique() -> None:
    frame = pd.DataFrame(
        {
            "material_id": [f"m-{index:03d}" for index in range(24)],
            "natoms": [2 + index % 5 for index in range(24)],
            "chemical_system": ["A-B", "A-C", "B-C"] * 8,
        }
    ).sample(frac=1.0, random_state=7)
    selected = select_probe_ids(frame, count=8)
    assert selected == select_probe_ids(frame.iloc[::-1], count=8)
    assert len(selected) == len(set(selected)) == 8


def test_novelty_ignores_sparse_perfect_control_and_keeps_adequate_control() -> None:
    pbveu = pd.Series(np.arange(80, dtype=float))
    sparse = np.full(80, np.nan)
    sparse[:4] = [0.0, 1.0, 2.0, 3.0]
    adequate = np.sin(np.arange(80, dtype=float))
    result = maximum_adequate_label_free_spearman(
        pbveu,
        pd.DataFrame({"sparse_perfect": sparse, "adequate": adequate}),
    )
    assert result["feature"] == "adequate"
    assert result["joint_finite"] == 80
    assert result["eligible_control_count"] == 1
    assert result["sparse_skipped_control_count"] == 1


def test_novelty_joint_finite_boundary_is_exactly_forty() -> None:
    pbveu = pd.Series(np.arange(80, dtype=float))
    enough = np.full(80, np.nan)
    enough[:40] = np.arange(40, dtype=float)[::-1]
    result = maximum_adequate_label_free_spearman(
        pbveu, pd.DataFrame({"enough": enough})
    )
    assert result["feature"] == "enough"
    assert result["joint_finite"] == 40
    assert result["absolute_correlation"] == 1.0


def test_probe_gates_use_open_zero_domain_and_strict_novelty_boundary() -> None:
    sources = {
        source: {
            "supported": 80,
            "unique_rounded_10": 80,
            "minimum": 0.01,
            "maximum": 1.0,
            "maximum_invariance_error": 0.0,
            "maximum_label_free_spearman": {"absolute_correlation": 0.5},
        }
        for source in ("scigen", "wyformer")
    }
    assert all(evaluate_probe_gates(sources, count=80).values())
    sources["scigen"]["minimum"] = 0.0
    assert evaluate_probe_gates(sources, count=80)["closed_domain"] is False
    sources["scigen"]["minimum"] = 0.01
    sources["scigen"]["maximum_label_free_spearman"]["absolute_correlation"] = 0.90
    assert evaluate_probe_gates(sources, count=80)["novel"] is False


def test_probe_runner_exposes_only_geometry_and_label_free_inputs() -> None:
    assert tuple(inspect.signature(run_label_blind_probe).parameters) == (
        "scigen_cohort_dir",
        "wyformer_cohort_dir",
        "scigen_base_feature_dir",
        "wyformer_base_feature_dir",
        "prior_feature_dirs",
        "count",
    )
    assert not any(
        token in name
        for name in inspect.signature(run_label_blind_probe).parameters
        for token in ("endpoint", "label", "validation", "replication", "relax")
    )


def test_probe_source_hashes_cover_complete_frozen_execution() -> None:
    hashes = probe_source_hashes()
    assert tuple(hashes) == (
        "design",
        "src/next19_valence_transport.py",
        "src/next38_bond_valence_transport_compatibility_features.py",
        "src/next267_periodic_radical_voronoi_packing.py",
        "src/next295_positive_contact_force_closure.py",
        "src/next307_periodic_bond_valence_hodge_loop.py",
        "src/next367_periodic_bond_valence_equal_uniformity.py",
        "experiments/next367_pbveu_label_blind_probe.py",
        "tests/test_next367_periodic_bond_valence_equal_uniformity.py",
        "tests/test_next367_pbveu_label_blind_probe.py",
    )
    assert hashes["design"] == DESIGN_SHA256 == (
        "c63b1042315a6df72a7368de31921f2f8e10cce67aa1e408a581bb5bd197132c"
    )
    assert all(len(value) == 64 for value in hashes.values())
