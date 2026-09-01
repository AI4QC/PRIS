from __future__ import annotations

import numpy as np

from src.next60_odac23_robust_scaffold_endpoint import (
    scaffold_condition_key,
    translation_aligned_displacements,
)


def test_translation_alignment_removes_periodic_common_motion() -> None:
    cell = np.diag([10.0, 10.0, 10.0])
    initial = np.asarray([[1.0, 1.0, 1.0], [3.0, 3.0, 3.0], [7.0, 7.0, 7.0]])
    relaxed = initial + [0.4, -0.3, 0.2]
    relaxed[2, 0] += 0.12

    distances, translation = translation_aligned_displacements(
        initial=initial, relaxed=relaxed, cell=cell
    )

    assert np.linalg.norm(translation @ cell) > 0.5
    assert np.isclose(distances[0], 0.0, atol=1e-12)
    assert np.isclose(distances[1], 0.0, atol=1e-12)
    assert np.isclose(distances[2], 0.12, atol=1e-12)


def test_scaffold_key_ignores_positions_but_separates_supercells_and_cells() -> None:
    numbers = np.asarray([29, 8, 8], dtype=int)
    cell = np.diag([10.0, 11.0, 12.0])

    base = scaffold_condition_key(
        framework_name="TEST", supercell=(1, 1, 1), numbers=numbers, cell=cell
    )
    repeated = scaffold_condition_key(
        framework_name="TEST", supercell=(2, 1, 1), numbers=numbers, cell=cell
    )
    changed_cell = scaffold_condition_key(
        framework_name="TEST",
        supercell=(1, 1, 1),
        numbers=numbers,
        cell=cell + np.diag([0.002, 0.0, 0.0]),
    )

    assert base == scaffold_condition_key(
        framework_name="TEST", supercell=(1, 1, 1), numbers=numbers, cell=cell + 0.0001
    )
    assert base != repeated
    assert base != changed_cell
