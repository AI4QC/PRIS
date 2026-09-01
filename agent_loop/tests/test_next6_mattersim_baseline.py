import hashlib
import json
import zipfile

import numpy as np
import pandas as pd
import pytest

from src.next6_mattersim_baseline import frame_to_atoms, run_mattersim_baseline


pytestmark = pytest.mark.filterwarnings("error")


FRAME = """2
Lattice=\"4 0 0 0 4 0 0 0 4\" Properties=species:S:1:pos:R:3:forces:R:3 material_id=test ionic_step=0 energy=-99 stress=\"1 2 3 4 5 6\"
Na 0 0 0 9 9 9
Cl 2 2 2 8 8 8
"""


def test_frame_to_atoms_uses_only_species_cell_and_coordinates():
    atoms = frame_to_atoms(FRAME)
    assert atoms.get_chemical_symbols() == ["Na", "Cl"]
    assert atoms.get_positions().tolist() == [[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]]
    assert atoms.get_cell().array.tolist() == [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]]
    assert atoms.pbc.tolist() == [True, True, True]
    assert atoms.info == {}
    assert "forces" not in atoms.arrays


def test_run_baseline_abstains_nonzero_step_and_hashes_predictions(tmp_path):
    elementa = tmp_path / "elementa"
    p9_dir = tmp_path / "p9"
    output = tmp_path / "result"
    elementa.mkdir()
    p9_dir.mkdir()
    with zipfile.ZipFile(elementa / "elementa_initial_frames.zip", "w") as archive:
        archive.writestr("a.extxyz", FRAME)
        archive.writestr("b.extxyz", FRAME.replace("ionic_step=0", "ionic_step=3"))
    pd.DataFrame(
        {
            "sid": ["b", "a"],
            "rk": ["Cl1|Na1", "Cl1|Na1"],
            "material": ["NaCl_02", "NaCl_01"],
            "strict_x0_ok": [False, True],
            "initial_ionic_step": [3, 0],
        }
    ).to_parquet(p9_dir / "elementa_x0_p9_features.parquet", index=False)
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"frozen-model")

    calls = []

    def predictor(atoms):
        calls.append(len(atoms))
        assert len(atoms) == 1
        return [-10.0]

    manifest = run_mattersim_baseline(
        elementa,
        p9_dir,
        output,
        checkpoint=checkpoint,
        device="cpu",
        batch_size=4,
        chunk_size=8,
        predictor=predictor,
    )

    assert calls == [1]
    table = pd.read_parquet(output / "mattersim_x0_predictions.parquet").set_index("sid")
    assert table.loc["a", "mattersim_feature_ok"]
    assert table.loc["a", "mattersim_energy_total"] == -10.0
    assert table.loc["a", "mattersim_energy_per_atom"] == -5.0
    assert not table.loc["b", "mattersim_feature_ok"]
    assert table.loc["b", "mattersim_feature_error"] == "nonzero_initial_ionic_step"
    assert np.isnan(table.loc["b", "mattersim_energy_per_atom"])
    loaded = json.loads((output / "MANIFEST.json").read_text())
    assert loaded == manifest
    digest = hashlib.sha256(
        (output / "mattersim_x0_predictions.parquet").read_bytes()
    ).hexdigest()
    assert manifest["outputs_sha256"]["mattersim_x0_predictions.parquet"] == digest
