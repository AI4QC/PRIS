"""Pure-geometry contracts for the NEXT26 molecular packing law."""

from __future__ import annotations

from pathlib import Path

from ase import Atoms
import numpy as np
import pandas as pd
import pytest


def _crystal() -> Atoms:
    return Atoms(
        [6, 1, 1, 8, 1],
        positions=[
            [0.0, 0.0, 0.0],
            [1.05, 0.0, 0.0],
            [0.0, 1.05, 0.0],
            [3.0, 3.0, 3.0],
            [3.95, 3.0, 3.0],
        ],
        cell=np.eye(3) * 7.0,
        pbc=True,
    )


def test_packing_features_are_finite_and_invariant_to_wrap_and_permutation() -> None:
    from src.next26_packing import FEATURE_COLUMNS, compute_packing_features

    atoms = _crystal()
    baseline = compute_packing_features(atoms)
    wrapped = atoms.copy()
    wrapped.positions[0] += wrapped.cell.array[0]
    order = [4, 2, 0, 3, 1]
    permuted = wrapped[order]
    observed = compute_packing_features(permuted)

    assert tuple(baseline) == FEATURE_COLUMNS
    assert all(np.isfinite(list(baseline.values())))
    for name in FEATURE_COLUMNS:
        assert observed[name] == pytest.approx(baseline[name], rel=1e-12, abs=1e-12)


def test_denser_cell_increases_packing_and_nonbonded_clash_risk() -> None:
    from src.next26_packing import compute_packing_features

    loose = _crystal()
    dense = loose.copy()
    dense.set_cell(loose.cell.array * 0.72, scale_atoms=True)
    a = compute_packing_features(loose)
    b = compute_packing_features(dense)

    assert b["cov_packing"] > a["cov_packing"]
    assert b["density_proxy"] > a["density_proxy"]
    assert b["volume_pa"] < a["volume_pa"]
    assert b["nonbond_vdw_q01"] < a["nonbond_vdw_q01"]
    assert b["nonbond_clash_frac085"] >= a["nonbond_clash_frac085"]


def test_feature_builder_is_fail_closed_on_non_geometry_metadata(tmp_path: Path) -> None:
    from src.next26_packing import compute_packing_features

    atoms = _crystal()
    atoms.info["energy"] = -1.0
    with pytest.raises(ValueError, match="geometry-only"):
        compute_packing_features(atoms)


def test_feature_schema_contains_no_forbidden_label_tokens() -> None:
    from src.next26_packing import FEATURE_COLUMNS

    forbidden = ("energy", "force", "stress", "relax", "dft", "label", "endpoint")
    assert not [name for name in FEATURE_COLUMNS if any(x in name.lower() for x in forbidden)]

