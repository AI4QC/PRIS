from __future__ import annotations

import inspect
import io
import json

from ase.io import write
import numpy as np
from pymatgen.core import Lattice, Structure
import pytest

from src.next46_motif_coherence_features import FEATURE_NAMES
from src.next199_cross_source_motif_features import (
    _compute_scigen_payload,
    _compute_wyformer_payload,
    build_cross_source_motif_features,
    compute_motif_row,
)


def _nacl() -> Structure:
    return Structure(
        Lattice.cubic(5.64),
        ["Na", "Cl"],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )


def test_compute_row_has_exact_finite_motif_schema() -> None:
    row = compute_motif_row(_nacl().to_ase_atoms())
    assert tuple(name for name in row if name in FEATURE_NAMES) == FEATURE_NAMES
    assert row["motif_supported"] is True
    assert row["motif_failure"] is None
    assert np.isfinite([row[name] for name in FEATURE_NAMES]).all()


def test_source_payload_parsers_produce_same_geometry_features() -> None:
    structure = _nacl()
    buffer = io.StringIO()
    write(buffer, structure.to_ase_atoms(), format="extxyz")
    scigen_id, scigen = _compute_scigen_payload(
        ("scigen-row", buffer.getvalue().encode())
    )
    wyformer_id, wyformer = _compute_wyformer_payload(
        ("wyformer-row", json.dumps(structure.as_dict()))
    )
    assert scigen_id == "scigen-row" and wyformer_id == "wyformer-row"
    assert scigen["motif_supported"] is True
    assert wyformer["motif_supported"] is True
    np.testing.assert_allclose(
        [scigen[name] for name in FEATURE_NAMES],
        [wyformer[name] for name in FEATURE_NAMES],
        rtol=1.0e-10,
        atol=1.0e-12,
    )


def test_builder_interface_cannot_receive_endpoints_or_other_partitions() -> None:
    parameters = tuple(inspect.signature(build_cross_source_motif_features).parameters)
    assert "scigen_cohort_dir" in parameters and "wyformer_cohort_dir" in parameters
    assert not any(
        token in name
        for name in parameters
        for token in ("endpoint", "validation", "replication")
    )


def test_builder_fails_closed_on_missing_input(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="NEXT199 input is missing"):
        build_cross_source_motif_features(
            scigen_cohort_dir=tmp_path / "scigen",
            wyformer_cohort_dir=tmp_path / "wyformer",
            design_path=tmp_path / "design.md",
            output_dir=tmp_path / "out",
            workers=1,
            require_formal_inputs=False,
        )
