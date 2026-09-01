import hashlib
import json
import zipfile

import numpy as np
import pandas as pd
import pytest

from src import next6_elementa_p9 as p9


pytestmark = pytest.mark.filterwarnings("error")


FRAME = """2
Lattice=\"4 0 0 0 4 0 0 0 4\" Properties=species:S:1:pos:R:3 material_id=test ionic_step=0
Na 0 0 0
Cl 2 2 2
"""


def test_corrected_p9_row_uses_unified_valence_cascade_and_opposite_sign_graph(monkeypatch):
    calls = {}

    def fake_infer(structure):
        calls["species"] = [site.specie.symbol for site in structure]
        return np.array([1.0, -1.0]), "frac_oxi"

    fake_neighbors = [[{"site_index": 1}], [{"site_index": 0}]]
    monkeypatch.setattr(p9, "infer_formal_valences", fake_infer)
    monkeypatch.setattr(p9, "_crystal_nn_info", lambda structure: fake_neighbors)

    def fake_p9(structure, valences, *, neighbors):
        calls["valences"] = list(valences)
        calls["neighbors"] = neighbors
        return {
            "p9c_bond_mismatch_mean": 0.1,
            "p9c_bond_mismatch_q95": 0.2,
            "p9c_bond_mismatch_max": 0.3,
            "p9c_cat_site_mismatch_max": 0.4,
        }

    monkeypatch.setattr(p9, "p9c_lewis_features", fake_p9)
    got = p9.corrected_p9_initial_features(
        {"sid": "elem-1", "rk": "Cl1|Na1", "material": "NaCl_01", "text": FRAME}
    )

    assert calls == {
        "species": ["Na", "Cl"],
        "valences": [1.0, -1.0],
        "neighbors": fake_neighbors,
    }
    assert got["input_role"] == "unrelaxed_x0_only"
    assert got["initial_ionic_step"] == 0
    assert got["strict_x0_ok"] is True
    assert got["p9c_feature_ok"] is True
    assert got["p9c_valence_method"] == "frac_oxi"
    assert got["p9c_bond_mismatch_max"] == 0.3
    assert "energy" not in got and "e_per_atom" not in got


def test_corrected_p9_row_abstains_on_feature_failure(monkeypatch):
    def fail(_structure):
        raise ValueError("no charge balance")

    monkeypatch.setattr(p9, "infer_formal_valences", fail)
    got = p9.corrected_p9_initial_features(
        {"sid": "elem-2", "rk": "Cl1|Na1", "material": "NaCl_02", "text": FRAME}
    )

    assert got["p9c_feature_ok"] is False
    assert got["p9c_feature_error"] == "ValueError:no charge balance"
    assert all(np.isnan(got[column]) for column in p9.P9_COLUMNS)


def test_nonzero_ionic_step_is_abstained_before_feature_calculation(monkeypatch):
    later = FRAME.replace("ionic_step=0", "ionic_step=3")

    def forbidden(_structure):
        raise AssertionError("post-initial frame must not reach valence inference")

    monkeypatch.setattr(p9, "infer_formal_valences", forbidden)
    got = p9.corrected_p9_initial_features(
        {"sid": "elem-3", "rk": "Cl1|Na1", "material": "NaCl_03", "text": later}
    )

    assert got["input_role"] == "trajectory_earliest_available"
    assert got["initial_ionic_step"] == 3
    assert got["strict_x0_ok"] is False
    assert got["p9c_feature_ok"] is False
    assert got["p9c_feature_error"] == "nonzero_initial_ionic_step"


def test_charge_ensemble_keeps_all_balanced_assignments_and_reports_envelope(monkeypatch):
    from pymatgen.core import Lattice, Structure

    structure = Structure(
        Lattice.cubic(5),
        ["Fe", "Mn", "O", "O", "O"],
        [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]],
    )
    assignments = p9.enumerate_uniform_charge_assignments(structure)
    assert len(assignments) == 2
    assert sorted((values[0], values[1]) for values in assignments) == [
        (2.0, 4.0),
        (3.0, 3.0),
    ]

    monkeypatch.setattr(p9, "_crystal_nn_info", lambda _structure: [[{}]] * 5)

    def fake_p9(_structure, valences, *, neighbors):
        value = float(valences[0])
        return {column: value for column in p9.P9_COLUMNS}

    monkeypatch.setattr(p9, "p9c_lewis_features", fake_p9)
    got = p9.robust_p9_charge_envelope(structure)

    assert got["p9r_assignment_count"] == 2
    assert got["p9r_feature_ok"] is True
    assert got["p9r_bond_mismatch_q95_min"] == 2.0
    assert got["p9r_bond_mismatch_q95_max"] == 3.0


def test_run_p9_extraction_reads_only_x0_inputs_and_writes_hashed_artifact(
    tmp_path, monkeypatch
):
    source = tmp_path / "x0"
    source.mkdir()
    metadata = pd.DataFrame(
        {
            "sid": ["elem-1", "elem-2"],
            "rk": ["Cl1|Na1", "Cl1|Na1"],
            "material": ["NaCl_01", "NaCl_02"],
            "input_role": ["unrelaxed_x0_only", "unrelaxed_x0_only"],
        }
    )
    metadata.to_parquet(source / "elementa_x0_features.parquet", index=False)
    with zipfile.ZipFile(source / "elementa_initial_frames.zip", "w") as archive:
        archive.writestr("elem-1.extxyz", FRAME)
        archive.writestr("elem-2.extxyz", FRAME.replace("material_id=test", "material_id=test2"))

    def fake_row(row):
        return {
            "sid": row["sid"],
            "rk": row["rk"],
            "material": row["material"],
            "input_role": "unrelaxed_x0_only",
            "p9c_feature_ok": True,
            "p9c_feature_error": "",
            "p9c_valence_method": "guess_oxi",
            **{column: float(row["sid"][-1]) for column in p9.P9_COLUMNS},
        }

    monkeypatch.setattr(p9, "corrected_p9_initial_features", fake_row)
    output = tmp_path / "result"
    manifest = p9.run_p9_extraction(source, output, workers=1)

    table = pd.read_parquet(output / "elementa_x0_p9_features.parquet")
    assert table.sid.tolist() == ["elem-1", "elem-2"]
    assert table.p9c_feature_ok.tolist() == [True, True]
    assert not {"energy", "e_per_atom", "final_ionic_step"} & set(table.columns)
    loaded = json.loads((output / "MANIFEST.json").read_text())
    assert loaded == manifest
    digest = hashlib.sha256(
        (output / "elementa_x0_p9_features.parquet").read_bytes()
    ).hexdigest()
    assert manifest["outputs_sha256"]["elementa_x0_p9_features.parquet"] == digest
    assert manifest["counts"] == {"input_frames": 2, "output_rows": 2}
