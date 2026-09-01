from __future__ import annotations

import inspect
from pathlib import Path

import experiments.next425_aprbs_label_blind_probe as p


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
        for token in (
            "endpoint", "label", "outcome", "validation", "replication", "relax"
        )
    )


def test_hashes_cover_frozen_execution_surface_and_p4bss_control() -> None:
    hashes = p.probe_source_hashes()
    assert hashes["design"] == p.DESIGN_SHA256
    assert "src/next420_pauling4_bond_strength_segregation.py" in hashes
    assert "src/next425_apriori_bond_strength_length_order.py" in hashes
    assert tuple(hashes)[-2:] == (
        "tests/test_next425_apriori_bond_strength_length_order.py",
        "tests/test_next425_aprbs_label_blind_probe.py",
    )


def test_formal_prior_root_resolver_is_reused_without_relaxing_identity(
    tmp_path: Path,
) -> None:
    expected = {
        "next166": {
            "scigen": "next166_scigen_periodic_contact_topology.parquet",
            "wyformer": "next166_wyformer_periodic_contact_topology.parquet",
        }
    }
    directory = tmp_path / "next166_periodic_contact_topology_features_v1"
    directory.mkdir()
    for filename in expected["next166"].values():
        (directory / filename).touch()
    assert p._resolve_prior_feature_dirs(tmp_path, expected) == {
        "next166": directory
    }
