from __future__ import annotations

import csv
import zlib

import pandas as pd
import pytest

from experiments.pu_synthesizability_20260821.data import (
    decode_blob_cif,
    iter_negative_records,
    normalize_pool_id,
    read_clscore_csv,
    validate_frozen_positive_frame,
)


def test_validate_frozen_positive_frame_requires_exact_contiguous_snapshot():
    frame = pd.DataFrame(
        {
            "source_index": [2, 0, 1],
            "blob_offset": [20, 0, 10],
            "blob_length": [5, 5, 5],
            "source": ["cod", "icsd", "icsd"],
        }
    )
    got = validate_frozen_positive_frame(frame, expected_rows=3)
    assert got.source_index.tolist() == [0, 1, 2]

    with pytest.raises(ValueError, match="contiguous"):
        validate_frozen_positive_frame(frame[frame.source_index != 1], expected_rows=2)


def test_decode_blob_cif_checks_length_and_decompresses(tmp_path):
    raw = b"data_demo\n_cell_length_a 3.0\n"
    payload = zlib.compress(raw)
    blob = tmp_path / "structures.blob"
    blob.write_bytes(b"prefix" + payload + b"suffix")

    assert decode_blob_cif(blob, offset=6, length=len(payload)) == raw.decode()
    with pytest.raises(ValueError, match="outside blob"):
        decode_blob_cif(blob, offset=6, length=len(payload) + 99)


def test_negative_release_join_is_one_to_one_and_streaming(tmp_path):
    meta = pd.DataFrame(
        {
            "index": [0, 1],
            "orig_index": [10, 11],
            "CLscore_A": [0.01, 0.02],
            "CLscore_B": [0.03, 0.04],
            "confidence": ["high", "medium"],
            "provenance": ["lemat", "elementa"],
        }
    ).set_index("index", drop=False)
    csv_path = tmp_path / "neg.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["index", "material_id", "cif", "pbes_gap", "provenance"]
        )
        writer.writeheader()
        writer.writerow(
            {"index": 0, "material_id": "m0", "cif": "data_0", "pbes_gap": 0, "provenance": "lemat"}
        )
        writer.writerow(
            {"index": 1, "material_id": "m1", "cif": "data_1", "pbes_gap": 1, "provenance": "elementa"}
        )

    rows = list(iter_negative_records([csv_path], meta, expected_rows=2))
    assert [row["orig_index"] for row in rows] == [10, 11]
    assert rows[1]["CLscore_A"] == pytest.approx(0.02)

    with pytest.raises(ValueError, match="one-to-one"):
        list(iter_negative_records([csv_path, csv_path], meta, expected_rows=2))


def test_negative_reader_accepts_large_serialized_cif_fields(tmp_path):
    meta = pd.DataFrame(
        {
            "index": [0],
            "orig_index": [10],
            "CLscore_A": [0.01],
            "CLscore_B": [0.02],
            "confidence": ["high"],
            "provenance": ["lemat"],
        }
    )
    csv_path = tmp_path / "large.csv"
    large_cif = "data_large\n" + "x" * 200_000
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["index", "material_id", "cif"])
        writer.writeheader()
        writer.writerow({"index": 0, "material_id": "m0", "cif": large_cif})
    got = list(iter_negative_records([csv_path], meta, expected_rows=1))
    assert got[0]["cif"] == large_cif


def test_negative_train_and_validation_streams_are_merged_by_release_index(tmp_path):
    meta = pd.DataFrame(
        {
            "index": [0, 1, 2, 3],
            "orig_index": [10, 11, 12, 13],
            "CLscore_A": [0.01] * 4,
            "CLscore_B": [0.02] * 4,
            "confidence": ["high"] * 4,
            "provenance": ["lemat"] * 4,
        }
    )
    paths = []
    for name, indices in [("train.csv", [0, 2]), ("val.csv", [1, 3])]:
        path = tmp_path / name
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["index", "material_id", "cif"])
            writer.writeheader()
            for index in indices:
                writer.writerow({"index": index, "material_id": f"m{index}", "cif": f"data_{index}"})
        paths.append(path)
    got = list(iter_negative_records(paths, meta, expected_rows=4))
    assert [row["record_index"] for row in got] == [0, 1, 2, 3]


def test_clscore_reader_repairs_known_three_column_merged_header(tmp_path):
    path = tmp_path / "clscore_all.csv"
    path.write_text(
        "id,CLscore,bagging\n"
        "u10,0.125000,31,0.050000\n"
        "u11,0.000000,29,0.000000\n"
    )
    got = read_clscore_csv(path, score_name="CLscore_A")
    assert got.columns.tolist() == ["pool_id", "orig_index", "CLscore_A", "bagging_A", "CLstd_A"]
    assert got.orig_index.tolist() == [10, 11]
    assert got.CLstd_A.tolist() == [0.05, 0.0]


@pytest.mark.parametrize("bad", ["10", "x10", "u-1", "u1.5"])
def test_normalize_pool_id_fails_closed_on_unexpected_ids(bad):
    with pytest.raises(ValueError):
        normalize_pool_id(bad)
