from __future__ import annotations

import inspect
from pathlib import Path

import experiments.next470_eccc_label_blind_novelty_probe as p


def test_full_prior_universe_and_sparse_controls_are_frozen() -> None:
    assert len(p.PRIOR_MODULES) == 32
    assert set(p.PRIOR_MODULES) == set(p.PRIOR_FILE_NAMES)
    assert p.SPARSE_CONTROL_NAMES == ("recomputed_clam", "recomputed_mvclam")
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
    names = tuple(inspect.signature(p.run_label_blind_novelty_probe).parameters)
    assert not any(
        token in name
        for name in names
        for token in ("endpoint", "label", "outcome", "validation", "replication", "relax")
    )


def test_hashes_cover_support_certificate_sparse_controls_and_target() -> None:
    hashes = p.probe_source_hashes()
    assert hashes["design"] == p.DESIGN_SHA256
    assert hashes["characteristic_cn_asset"] == p.ASSET_SHA256
    assert "next470_support_certificate" in hashes
    assert "src/next460_characteristic_lewis_anion_matching.py" in hashes
    assert "src/next465_mixed_valence_characteristic_lewis_matching.py" in hashes
    assert "src/next470_element_characteristic_coordination_compatibility.py" in hashes
    assert tuple(hashes)[-1] == "tests/test_next470_eccc_label_blind_novelty_probe.py"


def test_formal_prior_root_resolver_is_reused(tmp_path: Path) -> None:
    expected = {"next166": {"scigen": "a.parquet", "wyformer": "b.parquet"}}
    directory = tmp_path / "next166_periodic_contact_topology_features_v1"
    directory.mkdir()
    for filename in expected["next166"].values():
        (directory / filename).touch()
    assert p._resolve_prior_feature_dirs(tmp_path, expected) == {"next166": directory}
