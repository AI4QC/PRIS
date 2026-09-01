from __future__ import annotations

import io
from pathlib import Path
import zipfile

from ase import Atoms
from ase.io import write
import numpy as np
import pandas as pd
import pytest

from src.next549_omc25_two_sided_contact_freeze import (
    _read_geometry_payload,
    freeze_two_sided_scores,
    two_sided_contact_features,
)


def _extxyz(atoms: Atoms) -> bytes:
    stream = io.StringIO()
    write(stream, atoms, format="extxyz")
    return stream.getvalue().encode()


def test_two_sided_contact_features_are_translation_and_order_invariant() -> None:
    atoms = Atoms(
        symbols=["Li", "Si", "Li"],
        scaled_positions=[[0, 0, 0], [0.4, 0.4, 0.4], [0.8, 0.1, 0.2]],
        cell=[[5, 0, 0], [0.2, 5.5, 0], [0.1, 0.3, 6]],
        pbc=True,
    )
    moved = atoms.copy()
    moved.translate([1.2, -0.7, 0.4])
    permuted = moved[[2, 0, 1]]

    expected = two_sided_contact_features(atoms)
    actual = two_sided_contact_features(permuted)

    assert expected.keys() == actual.keys()
    np.testing.assert_allclose(list(expected.values()), list(actual.values()), rtol=0, atol=1e-12)


def test_frozen_score_is_exact_maximum_of_opposed_midranks() -> None:
    table = pd.DataFrame(
        {
            "material_id": ["a", "b", "c", "d"],
            "contact_ratio_q10": [0.7, 0.8, 0.9, 1.0],
            "contact_ratio_q50": [1.3, 1.2, 1.1, 1.0],
        }
    )

    result = freeze_two_sided_scores(table)

    np.testing.assert_allclose(result["risk_low_q10"], [0.875, 0.625, 0.375, 0.125])
    np.testing.assert_allclose(result["risk_high_q50"], [0.875, 0.625, 0.375, 0.125])
    np.testing.assert_allclose(result["tcse_risk"], [0.875, 0.625, 0.375, 0.125])


def test_geometry_reader_rejects_non_geometric_arrays(tmp_path: Path) -> None:
    clean = Atoms("LiSi", positions=[[0, 0, 0], [1, 1, 1]], cell=[4, 4, 4], pbc=True)
    archive_path = tmp_path / "geometry_only_frames.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("clean.extxyz", _extxyz(clean))
    atoms, digest = _read_geometry_payload(archive_path, "clean.extxyz")
    assert len(atoms) == 2
    assert len(digest) == 64
    assert set(atoms.arrays) == {"numbers", "positions"}

    dirty = clean.copy()
    dirty.new_array("endpoint_energy", np.array([1.0, 2.0]))
    dirty_path = tmp_path / "dirty.zip"
    with zipfile.ZipFile(dirty_path, "w") as archive:
        archive.writestr("dirty.extxyz", _extxyz(dirty))
    with pytest.raises(ValueError, match="non-geometric"):
        _read_geometry_payload(dirty_path, "dirty.extxyz")

