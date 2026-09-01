from __future__ import annotations

import inspect
import numpy as np
import pandas as pd

import experiments.next387_psvb_label_blind_probe as p


def test_prior_population_extends_through_next383() -> None:
    assert tuple(p.PRIOR_MODULES)[-5:] == ("next359", "next367", "next375", "next379", "next383")
    assert len(p.PRIOR_MODULES) == 32
    assert set(p.PRIOR_FILE_NAMES) == set(p.PRIOR_MODULES)
    assert p.MAXIMUM_BYPASS_LENGTH == 4 and p.MINIMUM_JOINT_FINITE == 40


def test_gates_are_exact() -> None:
    sources = {source: {"supported": 80, "unique_rounded_10": 20, "minimum": 0.0, "maximum": 1.0, "maximum_invariance_error": 0.0, "maximum_label_free_spearman": {"absolute_correlation": 0.89}} for source in ("scigen", "wyformer")}
    assert all(p.evaluate_probe_gates(sources, 80).values())
    sources["scigen"]["unique_rounded_10"] = 19
    assert not p.evaluate_probe_gates(sources, 80)["nondegenerate"]


def test_probe_interface_has_no_outcome_input() -> None:
    parameters = tuple(inspect.signature(p.run_label_blind_probe).parameters)
    assert parameters == ("scigen_cohort_dir", "wyformer_cohort_dir", "scigen_base_feature_dir", "wyformer_base_feature_dir", "prior_feature_dirs", "count")
    assert not any(token in name for name in parameters for token in ("endpoint", "label", "validation", "replication", "relax"))


def test_source_hashes_cover_frozen_execution() -> None:
    hashes = p.probe_source_hashes()
    assert tuple(hashes)[-3:] == (
        "experiments/next387_psvb_label_blind_probe.py",
        "tests/test_next387_periodic_skeletal_vertex_bypass.py",
        "tests/test_next387_psvb_label_blind_probe.py",
    )
    assert hashes["design"] == p.DESIGN_SHA256
    assert all(len(value) == 64 for value in hashes.values())


def test_novelty_helper_is_inherited_and_adequate() -> None:
    candidate = pd.Series(np.arange(80, dtype=float))
    control = pd.DataFrame({"x": np.arange(80, dtype=float)[::-1]})
    result = p.maximum_adequate_label_free_spearman(candidate, control)
    assert result["joint_finite"] == 80 and result["absolute_correlation"] == 1.0
