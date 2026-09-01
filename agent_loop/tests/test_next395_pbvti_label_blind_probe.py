from __future__ import annotations

import inspect

import experiments.next395_pbvti_label_blind_probe as p


def test_prior_universe_and_gates_are_frozen() -> None:
    assert len(p.PRIOR_MODULES) == 32
    assert set(p.PRIOR_MODULES) == set(p.PRIOR_FILE_NAMES)
    sources = {
        source: {
            "supported": 80,
            "minimum": 0.0,
            "maximum": 1.0,
            "unique_rounded_10": 20,
            "maximum_invariance_error": 0.0,
            "maximum_label_free_spearman": {"absolute_correlation": 0.89},
        }
        for source in ("scigen", "wyformer")
    }
    assert all(p.evaluate_probe_gates(sources, 80).values())


def test_probe_interface_has_no_outcome_or_later_geometry() -> None:
    names = tuple(inspect.signature(p.run_label_blind_probe).parameters)
    assert not any(
        token in name
        for name in names
        for token in ("endpoint", "label", "outcome", "validation", "replication", "relax")
    )


def test_hashes_cover_the_frozen_execution_surface() -> None:
    hashes = p.probe_source_hashes()
    assert hashes["design"] == p.DESIGN_SHA256
    assert tuple(hashes)[-2:] == (
        "tests/test_next395_periodic_bond_valence_tensor_isotropy.py",
        "tests/test_next395_pbvti_label_blind_probe.py",
    )
