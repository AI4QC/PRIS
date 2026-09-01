from __future__ import annotations

import json
from pathlib import Path
import zipfile

import numpy as np

from src.next541_jarvis_cdvae_initial_prediction_freeze import (
    _atoms_from_initial_record,
    _mask_forbidden_model_values,
    endpoint_inventory,
    mechanism_union_percentile_risk,
)


def test_forbidden_model_values_are_masked_before_json_parsing() -> None:
    raw = (
        b'[{"formula":"LiF","pred":9.87654321,"fenp":-1.23456789,'
        b'"bg":2.34567891,"atoms":{}}]'
    )

    masked, counts = _mask_forbidden_model_values(raw)

    assert b"9.87654321" not in masked
    assert b"-1.23456789" not in masked
    assert b"2.34567891" not in masked
    assert counts == {"bg": 1, "fenp": 1, "pred": 1}
    record = json.loads(masked)[0]
    assert record["pred"] is None
    assert record["fenp"] is None
    assert record["bg"] is None


def test_endpoint_inventory_never_reads_member_payload(tmp_path: Path) -> None:
    archive = tmp_path / "endpoint.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("CDVAE_relax_DFT/POSCAR-LiF.vasp", b"not a POSCAR")
        handle.writestr("__MACOSX/._POSCAR-LiF.vasp", b"metadata")

    table = endpoint_inventory(archive)

    assert table.to_dict("records") == [
        {
            "endpoint_member": "CDVAE_relax_DFT/POSCAR-LiF.vasp",
            "endpoint_filename": "POSCAR-LiF.vasp",
            "endpoint_formula_token": "LiF",
            "composition_key": "F1 Li1",
        }
    ]


def test_mechanism_union_percentile_risk_has_frozen_midrank_semantics() -> None:
    contact = np.array([-0.9, -0.7, -0.7, -0.5])
    sssp = np.array([-0.8, np.nan, -0.4, -0.2])
    pbaaa = np.array([0.1, 0.5, 0.9, np.nan])

    result = mechanism_union_percentile_risk(contact, sssp, pbaaa)

    np.testing.assert_allclose(result["contact_percentile"], [0.125, 0.5, 0.5, 0.875])
    np.testing.assert_allclose(
        result["sssp_percentile"], [1 / 6, np.nan, 0.5, 5 / 6], equal_nan=True
    )
    np.testing.assert_allclose(
        result["pbaaa_percentile"], [1 / 6, 0.5, 5 / 6, np.nan], equal_nan=True
    )
    expected = [
        1 - (1 - 0.125) * (1 - 1 / 6) * (1 - 1 / 6),
        1 - (1 - 0.5) * (1 - 0.5),
        1 - (1 - 0.5) * (1 - 0.5) * (1 - 5 / 6),
        1 - (1 - 0.875) * (1 - 5 / 6),
    ]
    np.testing.assert_allclose(result["mupr_risk"], expected)
    np.testing.assert_array_equal(result["mechanism_count"], [3, 2, 3, 2])


def test_initial_atoms_conversion_uses_only_raw_periodic_geometry() -> None:
    record = {
        "formula": "LiF",
        "pred": None,
        "fenp": None,
        "bg": None,
        "atoms": {
            "lattice_mat": [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
            "coords": [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
            "elements": ["Li", "F"],
            "cartesian": False,
            "abc": [4.0, 4.0, 4.0],
            "angles": [90.0, 90.0, 90.0],
            "props": ["", ""],
        },
    }

    atoms = _atoms_from_initial_record(record)

    assert atoms.get_chemical_symbols() == ["Li", "F"]
    assert atoms.pbc.tolist() == [True, True, True]
    np.testing.assert_allclose(atoms.get_scaled_positions(), record["atoms"]["coords"])
    assert not atoms.info
