from __future__ import annotations

import json

from pymatgen.core import Lattice, Structure

from src.next544_lisi_random_relaxation_prediction_freeze import _compute_structure_payload


def test_x0_payload_computes_only_frozen_screening_families() -> None:
    structure = Structure(
        Lattice.cubic(4.2),
        ["Li", "Si"],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )

    row = _compute_structure_payload(("Li1Si1_02__toy", json.dumps(structure.as_dict()).encode()))

    assert row["trajectory_id"] == "Li1Si1_02__toy"
    assert row["contact_supported"] is True
    assert 0.0 < row["cov_q05"]
    assert "energy" not in " ".join(row).lower()
    assert "force" not in " ".join(row).lower()
    assert "stress" not in " ".join(row).lower()
    assert "endpoint" not in " ".join(row).lower()
