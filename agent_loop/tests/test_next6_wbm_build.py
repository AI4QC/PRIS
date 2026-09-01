import pandas as pd
import pytest
import zipfile

from src.next6_wbm_build import build_wbm_artifacts, prepare_wbm_inventory, sha256_file


def test_inventory_uses_unique_prototypes_and_corrected_hull_label():
    # Break caught: keeping duplicate prototypes or labelling with e_above_hull_wbm
    # instead of the frozen corrected MP2020 hull changes the benchmark population.
    raw = pd.DataFrame(
        {
            "material_id": ["a", "b", "c"],
            "formula": ["Li2O2", "LiO", "NaCl"],
            "unique_prototype": [True, False, True],
            "e_above_hull_wbm": [-1.0, -1.0, -1.0],
            "e_above_hull_mp2020_corrected_ppd_mp": [0.1, -0.2, 0.0],
            "site_stats_fingerprint_init_final_norm_diff": [0.2, 0.3, 0.4],
        }
    )

    got = prepare_wbm_inventory(raw)

    assert got.material_id.tolist() == ["a", "c"]
    assert got.formula_key.tolist() == ["LiO", "ClNa"]
    assert got.stable.tolist() == [False, True]
    assert set(got.stage) <= {"formula_selection", "threshold_calibration", "test"}


def test_inventory_rejects_duplicate_material_ids():
    # Break caught: duplicate IDs would make a zip frame join to multiple labels.
    raw = pd.DataFrame(
        {
            "material_id": ["a", "a"],
            "formula": ["LiO", "NaCl"],
            "unique_prototype": [True, True],
            "e_above_hull_mp2020_corrected_ppd_mp": [0.1, -0.1],
            "site_stats_fingerprint_init_final_norm_diff": [0.2, 0.3],
        }
    )

    with pytest.raises(ValueError, match="duplicate material_id"):
        prepare_wbm_inventory(raw)


def test_sha256_file_hashes_actual_bytes(tmp_path):
    # Break caught: hashing a path string rather than content breaks provenance.
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"abc")
    assert sha256_file(artifact) == (
        "ba7816bf8f01cfea414140de5dae2223"
        "b00361a396177a9cb410ff61f20015ad"
    )


def test_build_artifacts_keeps_x0_features_and_labels_in_split_files(tmp_path):
    # Break caught: joining all labels and features into one repeatedly readable table
    # defeats the calibration/test isolation contract.
    summary = pd.DataFrame(
        {
            "material_id": ["select", "cal", "test"],
            "formula": ["CH4", "H2O", "LiO"],
            "unique_prototype": [True, True, True],
            "e_above_hull_wbm": [-0.4, -0.3, -0.2],
            "e_above_hull_mp2020_corrected_ppd_mp": [-0.2, -0.1, 0.2],
            "site_stats_fingerprint_init_final_norm_diff": [0.2, 0.1, 0.9],
        }
    )
    summary_path = tmp_path / "summary.csv.gz"
    summary.to_csv(summary_path, index=False, compression="gzip")
    archive_path = tmp_path / "initial.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(
            "select.extxyz",
            "5\nLattice=\"10 0 0 0 10 0 0 0 10\" Properties=species:S:1:pos:R:3 material_id=select pbc=\"T T T\"\n"
            "C 0 0 0\nH 1 0 0\nH 0 1 0\nH 0 0 1\nH 1 1 1\n",
        )
        archive.writestr(
            "cal.extxyz",
            "3\nLattice=\"10 0 0 0 10 0 0 0 10\" Properties=species:S:1:pos:R:3 material_id=cal pbc=\"T T T\"\n"
            "H 0 0 0\nH 1 0 0\nO 0 1 0\n",
        )
        archive.writestr(
            "test.extxyz",
            "2\nLattice=\"10 0 0 0 10 0 0 0 10\" Properties=species:S:1:pos:R:3 material_id=test pbc=\"T T T\"\n"
            "Li 0 0 0\nO 1 0 0\n",
        )

    out_dir = tmp_path / "out"
    manifest = build_wbm_artifacts(
        summary_path=summary_path,
        initial_zip=archive_path,
        output_dir=out_dir,
        workers=1,
    )

    select_y = pd.read_parquet(out_dir / "formula_selection_labels.parquet")
    cal_y = pd.read_parquet(out_dir / "threshold_calibration_labels.parquet")
    test_y = pd.read_parquet(out_dir / "test_labels.parquet")
    select_x = pd.read_parquet(out_dir / "formula_selection_x0_features.parquet")
    cal_x = pd.read_parquet(out_dir / "threshold_calibration_x0_features.parquet")
    test_x = pd.read_parquet(out_dir / "test_x0_features.parquet")
    assert select_y.material_id.tolist() == ["select"]
    assert cal_y.material_id.tolist() == ["cal"]
    assert test_y.material_id.tolist() == ["test"]
    assert select_x.material_id.tolist() == ["select"]
    assert cal_x.material_id.tolist() == ["cal"]
    assert test_x.material_id.tolist() == ["test"]
    assert cal_y.stable.tolist() == [True]
    assert test_y.stable.tolist() == [False]
    assert "stable" not in cal_x.columns
    assert "e_above_hull_mp2020_corrected_ppd_mp" not in test_x.columns
    assert manifest["counts"] == {
        "formula_selection": 1,
        "threshold_calibration": 1,
        "test": 1,
        "total": 3,
    }
