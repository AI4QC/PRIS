from __future__ import annotations

import inspect
import experiments.next391_psbg_label_blind_probe as p


def test_prior_universe_and_gates() -> None:
    assert len(p.PRIOR_MODULES) == 32
    assert set(p.PRIOR_MODULES) == set(p.PRIOR_FILE_NAMES)
    sources = {s: {"supported": 80, "minimum": 0, "maximum": 0.9, "unique_rounded_10": 20, "maximum_invariance_error": 0, "maximum_label_free_spearman": {"absolute_correlation": 0.89}} for s in ("scigen", "wyformer")}
    assert all(p.evaluate_probe_gates(sources, 80).values())


def test_interface_has_no_outcome() -> None:
    names = tuple(inspect.signature(p.run_label_blind_probe).parameters)
    assert not any(token in name for name in names for token in ("endpoint", "label", "validation", "replication", "relax"))


def test_hashes_cover_execution() -> None:
    hashes = p.probe_source_hashes()
    assert hashes["design"] == p.DESIGN_SHA256
    assert tuple(hashes)[-2:] == ("tests/test_next391_periodic_skeletal_ball_growth.py", "tests/test_next391_psbg_label_blind_probe.py")
