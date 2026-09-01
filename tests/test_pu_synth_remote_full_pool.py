from __future__ import annotations

import csv
import json
import importlib.util
import io
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def test_remote_full_pool_module_exists() -> None:
    assert (
        importlib.util.find_spec(
            "experiments.pu_synthesizability_20260821.remote_full_pool"
        )
        is not None
    )


def test_task_bounds_tile_the_pool_and_clip_the_last_task() -> None:
    from experiments.pu_synthesizability_20260821.remote_full_pool import task_bounds

    assert task_bounds(0, total_rows=8_125_976, chunk_size=50_000) == (0, 50_000)
    assert task_bounds(154, total_rows=8_125_976, chunk_size=50_000) == (
        7_700_000,
        7_750_000,
    )
    assert task_bounds(162, total_rows=8_125_976, chunk_size=50_000) == (
        8_100_000,
        8_125_976,
    )
    with pytest.raises(IndexError, match="outside 0..162"):
        task_bounds(163, total_rows=8_125_976, chunk_size=50_000)


def _write_indexed_csv(path: Path, rows: list[list[str]]) -> Path:
    def render(row: list[str]) -> bytes:
        stream = io.StringIO(newline="")
        csv.writer(stream, lineterminator="\n").writerow(row)
        return stream.getvalue().encode()

    chunks = [render(["index", "material_id", "cif", "pbes_gap"])]
    offsets: list[int] = []
    for row in rows:
        offsets.append(sum(map(len, chunks)))
        chunks.append(render(row))
    path.write_bytes(b"".join(chunks))
    offset_path = path.with_suffix(".offsets.npy")
    np.save(offset_path, np.asarray(offsets, dtype=np.int64))
    return offset_path


def _build_tiny_manifest(
    tmp_path: Path,
    *,
    chunk_size: int = 2,
    runtime_context: dict[str, Path | str] | None = None,
) -> tuple[dict, Path]:
    from experiments.pu_synthesizability_20260821.remote_full_pool import (
        build_input_manifest,
    )

    train = tmp_path / "train.csv"
    val = tmp_path / "val.csv"
    train_offsets = _write_indexed_csv(
        train,
        [
            ["1", "train-a", "data_a", "0.1"],
            ["0", "train-b", "data_b", "0.2"],
        ],
    )
    val_offsets = _write_indexed_csv(
        val,
        [["2", "val-a", "data_c", "0.3"]],
    )
    bvparm = tmp_path / "bvparm.cif"
    bvparm.write_text("test\n")
    manifest = build_input_manifest(
        [
            {"split": "train", "path": str(train), "offset_path": str(train_offsets)},
            {"split": "val", "path": str(val), "offset_path": str(val_offsets)},
        ],
        expected_counts={"train": 2, "val": 1},
        chunk_size=chunk_size,
        implementation_paths=[Path(__file__)],
        bvparm_path=bvparm,
        **(runtime_context or {}),
    )
    return manifest, bvparm


def test_read_global_slice_seeks_across_train_val_boundary(tmp_path: Path) -> None:
    from experiments.pu_synthesizability_20260821.remote_full_pool import (
        read_global_slice,
    )

    train = tmp_path / "train.csv"
    val = tmp_path / "val.csv"
    train_offsets = _write_indexed_csv(
        train,
        [
            ["3", "train-a", "data_a\n_cell_length_a 1", "0.1"],
            ["1", "train-b", "data_b\n_cell_length_a 2", "0.2"],
        ],
    )
    val_offsets = _write_indexed_csv(
        val,
        [
            ["0", "val-a", "data_c\n_cell_length_a 3", ""],
            ["2", "val-b", "data_d\n_cell_length_a 4", "0.4"],
        ],
    )
    sources = [
        {"split": "train", "path": str(train), "rows": 2, "offset_path": str(train_offsets)},
        {"split": "val", "path": str(val), "rows": 2, "offset_path": str(val_offsets)},
    ]

    records = list(read_global_slice(sources, start=1, stop=3))

    assert [(row["source_split"], row["split_row"]) for row in records] == [
        ("train", 1),
        ("val", 0),
    ]
    assert [row["pool_row"] for row in records] == [1, 2]
    assert [row["orig_index"] for row in records] == [1, 0]
    assert [row["pool_id"] for row in records] == ["u1", "u0"]
    assert records[1]["cif"] == "data_c\n_cell_length_a 3"


def test_offset_driven_index_audit_freezes_noncontiguous_unique_indices(
    tmp_path: Path,
) -> None:
    from experiments.pu_synthesizability_20260821.remote_full_pool import (
        audit_source_indices,
    )

    train = tmp_path / "train.csv"
    val = tmp_path / "val.csv"
    train_offsets = _write_indexed_csv(
        train,
        [["10", "a", "data_a", ""], ["30", "b", "data_b", ""]],
    )
    val_offsets = _write_indexed_csv(val, [["20", "c", "data_c", ""]])
    sources = [
        {"split": "train", "path": str(train), "rows": 2, "offset_path": str(train_offsets)},
        {"split": "val", "path": str(val), "rows": 1, "offset_path": str(val_offsets)},
    ]

    audit = audit_source_indices(sources)

    ordered = np.asarray([10, 30, 20], dtype="<i8")
    assert audit["rows"] == 3
    assert audit["unique"] == 3
    assert audit["minimum"] == 10
    assert audit["maximum"] == 30
    assert audit["duplicates"] == 0
    assert audit["out_of_range"] == 3
    assert audit["missing_in_range"] == 3
    assert audit["contiguous_zero_based"] is False
    import hashlib

    assert audit["ordered_int64_le_sha256"] == hashlib.sha256(
        ordered.tobytes()
    ).hexdigest()
    assert audit["sorted_int64_le_sha256"] == hashlib.sha256(
        np.sort(ordered).tobytes()
    ).hexdigest()


def test_offset_driven_index_audit_rejects_duplicate_orig_indices(
    tmp_path: Path,
) -> None:
    from experiments.pu_synthesizability_20260821.remote_full_pool import (
        audit_source_indices,
    )

    train = tmp_path / "train.csv"
    val = tmp_path / "val.csv"
    train_offsets = _write_indexed_csv(train, [["7", "a", "data_a", ""]])
    val_offsets = _write_indexed_csv(val, [["7", "b", "data_b", ""]])
    with pytest.raises(ValueError, match="orig_index values are not unique"):
        audit_source_indices(
            [
                {
                    "split": "train",
                    "path": str(train),
                    "rows": 1,
                    "offset_path": str(train_offsets),
                },
                {
                    "split": "val",
                    "path": str(val),
                    "rows": 1,
                    "offset_path": str(val_offsets),
                },
            ]
        )


def test_build_manifest_enforces_source_counts_and_full_sha_gate(tmp_path: Path) -> None:
    from experiments.pu_synthesizability_20260821.remote_full_pool import (
        build_input_manifest,
        validate_frozen_inputs,
    )

    train = tmp_path / "train.csv"
    val = tmp_path / "val.csv"
    train_offsets = _write_indexed_csv(
        train,
        [["0", "aaa", "data_a\n_cell_length_a 1", "0.1"]],
    )
    val_offsets = _write_indexed_csv(
        val,
        [["1", "bbb", "data_b\n_cell_length_a 2", "0.2"]],
    )
    bvparm = tmp_path / "bvparm.cif"
    bvparm.write_text("bond-valence-test\n")
    sources = [
        {"split": "train", "path": str(train), "offset_path": str(train_offsets)},
        {"split": "val", "path": str(val), "offset_path": str(val_offsets)},
    ]

    bad_hash_sources = [dict(source) for source in sources]
    bad_hash_sources[0]["expected_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="train source SHA256 does not match frozen hash"):
        build_input_manifest(
            bad_hash_sources,
            expected_counts={"train": 1, "val": 1},
            chunk_size=1,
            implementation_paths=[Path(__file__)],
            bvparm_path=bvparm,
        )

    with pytest.raises(ValueError, match="train offsets contain 1 rows, expected 2"):
        build_input_manifest(
            sources,
            expected_counts={"train": 2, "val": 1},
            chunk_size=1,
            implementation_paths=[Path(__file__)],
            bvparm_path=bvparm,
        )

    manifest = build_input_manifest(
        sources,
        expected_counts={"train": 1, "val": 1},
        chunk_size=1,
        implementation_paths=[Path(__file__)],
        bvparm_path=bvparm,
    )
    validate_frozen_inputs(manifest, full_hash=True)
    assert manifest["total_rows"] == 2
    assert manifest["task_count"] == 2
    assert manifest["index_audit"]["contiguous_zero_based"] is True
    assert len(manifest["input_fingerprint"]) == 64

    original_stat = train.stat()
    train.write_bytes(train.read_bytes().replace(b"aaa", b"ccc"))
    # Restore the frozen timestamp and byte count so only the full SHA gate can catch it.
    assert train.stat().st_size == manifest["sources"][0]["bytes"]
    train.touch()
    import os

    os.utime(
        train,
        ns=(original_stat.st_atime_ns, manifest["sources"][0]["mtime_ns"]),
    )
    with pytest.raises(ValueError, match="train source SHA256 changed"):
        validate_frozen_inputs(manifest, full_hash=True)


def test_array_task_quick_gate_does_not_scan_entire_offset_arrays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import experiments.pu_synthesizability_20260821.remote_full_pool as full_pool

    manifest, _ = _build_tiny_manifest(tmp_path)

    def refuse_diff(*args: object, **kwargs: object) -> None:
        raise AssertionError("quick task gate must not scan every offset")

    source_paths = {
        str(Path(source["path"]).resolve()) for source in manifest["sources"]
    }
    real_sha256_file = full_pool.sha256_file

    def refuse_source_sha(path: str | Path) -> str:
        if str(Path(path).resolve()) in source_paths:
            raise AssertionError("quick task gate must not hash candidate CSVs")
        return real_sha256_file(path)

    monkeypatch.setattr(full_pool.np, "diff", refuse_diff)
    monkeypatch.setattr(full_pool, "sha256_file", refuse_source_sha)
    full_pool.validate_frozen_inputs(manifest, full_hash=False)


def test_array_task_quick_gate_hashes_offsets_without_hashing_source(
    tmp_path: Path,
) -> None:
    from experiments.pu_synthesizability_20260821.remote_full_pool import (
        validate_frozen_inputs,
    )

    manifest, _ = _build_tiny_manifest(tmp_path)
    offsets = Path(manifest["sources"][0]["offset_path"])
    frozen_stat = offsets.stat()
    values = np.load(offsets, allow_pickle=False)
    values[1] -= 1
    np.save(offsets, values)
    assert offsets.stat().st_size == manifest["sources"][0]["offset"]["bytes"]
    os.utime(
        offsets,
        ns=(frozen_stat.st_atime_ns, manifest["sources"][0]["offset"]["mtime_ns"]),
    )

    with pytest.raises(ValueError, match="train offsets SHA256 changed"):
        validate_frozen_inputs(manifest, full_hash=False)


def test_input_manifest_sha_file_is_checked_before_array_work(tmp_path: Path) -> None:
    from experiments.pu_synthesizability_20260821.remote_full_pool import (
        load_input_manifest,
        publish_input_manifest,
    )

    manifest, _ = _build_tiny_manifest(tmp_path)
    manifest_path = tmp_path / "input_manifest.json"
    sha_path = tmp_path / "input_manifest.sha256"
    publish_input_manifest(manifest, manifest_path, sha_path)

    assert load_input_manifest(manifest_path, sha_path) == manifest
    manifest_path.write_text(manifest_path.read_text().replace('"chunk_size": 2', '"chunk_size": 3'))
    with pytest.raises(ValueError, match="input manifest SHA256 mismatch"):
        load_input_manifest(manifest_path, sha_path)


def test_prepare_run_freezes_sources_offsets_counts_code_and_bvparm(
    tmp_path: Path,
) -> None:
    from experiments.pu_synthesizability_20260821.remote_full_pool import (
        load_input_manifest,
        prepare_run,
        sha256_file,
    )

    inputs = tmp_path / "inputs"
    inputs.mkdir()
    train = inputs / "train.csv"
    val = inputs / "val.csv"
    train_offsets = _write_indexed_csv(
        train,
        [["0", "train-a", "data_a", ""], ["1", "train-b", "data_b", ""]],
    )
    val_offsets = _write_indexed_csv(val, [["2", "val-a", "data_c", ""]])
    counts = inputs / "counts.npy"
    np.save(counts, np.asarray([10, 5, 2, 1], dtype=np.int64))
    bundle = tmp_path / "bundle"
    (bundle / "data").mkdir(parents=True)
    (bundle / "implementation.py").write_text("VALUE = 1\n")
    bvparm = bundle / "data/test-bvparm.cif"
    bvparm.write_text("Na 1 Cl -1 2.15 0.37\n")
    frozen = {
        "train": {
            "path": str(train),
            "rows": 2,
            "sha256": sha256_file(train),
            "offset_path": str(train_offsets),
            "offset_sha256": sha256_file(train_offsets),
        },
        "val": {
            "path": str(val),
            "rows": 1,
            "sha256": sha256_file(val),
            "offset_path": str(val_offsets),
            "offset_sha256": sha256_file(val_offsets),
        },
        "counts_path": str(counts),
        "counts_sha256": sha256_file(counts),
        "counts": [10, 5, 2, 1],
    }
    run_root = tmp_path / "run"

    prepared = prepare_run(
        run_root=run_root,
        bundle_root=bundle,
        chunk_size=2,
        frozen_inputs=frozen,
        implementation_relative_paths=["implementation.py"],
        bvparm_relative_path="data/test-bvparm.cif",
        expected_bvparm_sha256=sha256_file(bvparm),
        expected_bvparm_rows=1,
    )

    loaded = load_input_manifest(
        run_root / "input_manifest.json", run_root / "input_manifest.sha256"
    )
    assert prepared == loaded
    assert loaded["task_count"] == 2
    assert loaded["bvparm"]["table_rows"] == 1
    assert set(loaded["runtime"]["packages"]) == {
        "numpy",
        "pandas",
        "pyarrow",
        "pymatgen",
        "spglib",
        "scipy",
    }
    assert loaded["runtime"]["python_version"]
    assert len(loaded["runtime"]["executable"]["sha256"]) == 64
    assert (run_root / "logs").is_dir()
    assert (run_root / "results").is_dir()
    with pytest.raises(FileExistsError, match="published input manifest"):
        prepare_run(
            run_root=run_root,
            bundle_root=bundle,
            chunk_size=2,
            frozen_inputs=frozen,
            implementation_relative_paths=["implementation.py"],
            bvparm_relative_path="data/test-bvparm.cif",
            expected_bvparm_sha256=sha256_file(bvparm),
            expected_bvparm_rows=1,
        )


def test_production_bundle_list_and_bvparm_are_complete() -> None:
    from experiments.pu_synthesizability_20260821.remote_full_pool import (
        EXPECTED_BVPARM_ROWS,
        EXPECTED_BVPARM_SHA256,
        MINIMAL_BUNDLE_RELATIVE_PATHS,
        count_bvparm_rows,
        sha256_file,
    )

    root = Path(__file__).resolve().parents[1]
    assert all((root / relative).is_file() for relative in MINIMAL_BUNDLE_RELATIVE_PATHS)
    bvparm = root / "data/bvparm2020.cif"
    assert sha256_file(bvparm) == EXPECTED_BVPARM_SHA256
    assert count_bvparm_rows(bvparm) == EXPECTED_BVPARM_ROWS
    upload_list = (
        root
        / "experiments/pu_synthesizability_20260821/remote_bundle_files.txt"
    ).read_text().splitlines()
    assert upload_list == [*MINIMAL_BUNDLE_RELATIVE_PATHS, "data/bvparm2020.cif"]
    assert (
        "experiments/pu_synthesizability_20260821/cross_env_calibration.py"
        in MINIMAL_BUNDLE_RELATIVE_PATHS
    )


def test_cli_exposes_prepare_run_task_and_final_verify() -> None:
    from experiments.pu_synthesizability_20260821.remote_full_pool import build_parser

    parser = build_parser()
    prepare = parser.parse_args(
        ["prepare", "--run-root", "/run", "--bundle-root", "/bundle"]
    )
    task = parser.parse_args(
        [
            "run-task",
            "--input-manifest",
            "/run/input_manifest.json",
            "--input-manifest-sha-file",
            "/run/input_manifest.sha256",
            "--task-id",
            "7",
            "--output-dir",
            "/run/results",
            "--workers",
            "47",
            "--src-dir",
            "/bundle/src",
            "--bvparm",
            "/bundle/data/bvparm2020.cif",
        ]
    )
    verify = parser.parse_args(
        [
            "verify",
            "--input-manifest",
            "/run/input_manifest.json",
            "--input-manifest-sha-file",
            "/run/input_manifest.sha256",
            "--output-dir",
            "/run/results",
        ]
    )

    assert prepare.command == "prepare"
    assert (task.command, task.task_id, task.workers) == ("run-task", 7, 47)
    assert verify.command == "verify"
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "prepare",
                "--run-root",
                "/run",
                "--bundle-root",
                "/bundle",
                "--chunk-size",
                "1000",
            ]
        )


def test_production_prepare_cli_freezes_the_exact_container_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import experiments.pu_synthesizability_20260821.remote_full_pool as full_pool

    captured: dict[str, object] = {}

    def fake_prepare(**kwargs):
        captured.update(kwargs)
        return {"prepared": True}

    monkeypatch.setattr(full_pool, "prepare_run", fake_prepare)
    assert (
        full_pool.main(
            [
                "prepare",
                "--run-root",
                str(tmp_path / "run"),
                "--bundle-root",
                str(tmp_path / "bundle"),
            ]
        )
        == 0
    )
    assert captured["container_sif_path"] == full_pool.PRODUCTION_CONTAINER_SIF
    assert captured["requirements_lock_path"] == full_pool.PRODUCTION_REQUIREMENTS_LOCK
    assert captured["site_packages_path"] == full_pool.PRODUCTION_SITE_PACKAGES
    assert (
        captured["site_packages_manifest_path"]
        == full_pool.PRODUCTION_SITE_PACKAGES_MANIFEST
    )
    assert (
        captured["expected_container_sif_sha256"]
        == full_pool.PRODUCTION_CONTAINER_SIF_SHA256
    )
    assert (
        captured["expected_requirements_lock_sha256"]
        == full_pool.PRODUCTION_REQUIREMENTS_LOCK_SHA256
    )
    assert (
        captured["expected_site_packages_manifest_sha256"]
        == full_pool.PRODUCTION_SITE_PACKAGES_MANIFEST_SHA256
    )


def test_runtime_versions_are_a_quick_gate_and_executable_sha_is_a_final_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import experiments.pu_synthesizability_20260821.remote_full_pool as full_pool

    manifest, _ = _build_tiny_manifest(tmp_path)
    executable = str(Path(manifest["runtime"]["executable"]["path"]).resolve())
    real_sha256_file = full_pool.sha256_file
    executable_hash_calls = 0

    def count_executable_hash(path: str | Path) -> str:
        nonlocal executable_hash_calls
        if str(Path(path).resolve()) == executable:
            executable_hash_calls += 1
        return real_sha256_file(path)

    monkeypatch.setattr(full_pool, "sha256_file", count_executable_hash)
    full_pool.validate_frozen_inputs(manifest, full_hash=False)
    assert executable_hash_calls == 0
    full_pool.validate_frozen_inputs(manifest, full_hash=True)
    assert executable_hash_calls == 1

    manifest["runtime"]["packages"]["pymatgen"] = "drifted-version"
    manifest["input_fingerprint"] = full_pool._fingerprint_payload(manifest)
    with pytest.raises(ValueError, match="runtime package versions changed"):
        full_pool.validate_frozen_inputs(manifest, full_hash=False)


def test_container_runtime_freezes_sif_lock_and_site_tree_with_quick_and_full_gates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import experiments.pu_synthesizability_20260821.remote_full_pool as full_pool

    sif = tmp_path / "python.sif"
    lock = tmp_path / "exact_requirements.freeze"
    site = tmp_path / "site-packages"
    site_manifest = tmp_path / "site-packages.sha256"
    (site / "pkg").mkdir(parents=True)
    sif.write_bytes(b"frozen-sif")
    lock.write_text("numpy==1.26.4\n")
    (site / "pkg/__init__.py").write_text("VERSION = '1'\n")
    (site / "pkg/data.bin").write_bytes(b"data")
    site_manifest.write_text("fixture site manifest\n")
    manifest, _ = _build_tiny_manifest(
        tmp_path,
        runtime_context={
            "container_sif_path": sif,
            "requirements_lock_path": lock,
            "site_packages_path": site,
            "site_packages_manifest_path": site_manifest,
            "expected_container_sif_sha256": full_pool.sha256_file(sif),
            "expected_requirements_lock_sha256": full_pool.sha256_file(lock),
            "expected_site_packages_manifest_sha256": full_pool.sha256_file(
                site_manifest
            ),
        },
    )
    container = manifest["runtime"]["container_environment"]
    assert container["sif"]["sha256"] == full_pool.sha256_file(sif)
    assert container["requirements_lock"]["sha256"] == full_pool.sha256_file(lock)
    assert container["site_packages_manifest"]["sha256"] == full_pool.sha256_file(
        site_manifest
    )
    assert container["site_packages"]["files"] == 2
    assert container["site_packages"]["file_bytes"] == 18
    assert len(container["site_packages"]["sha256"]) == 64

    real_sha = full_pool.sha256_file
    real_tree = full_pool.site_packages_tree_identity
    file_hash_calls: list[str] = []
    tree_calls = 0

    def count_hash(path: str | Path) -> str:
        resolved = str(Path(path).resolve())
        if resolved in {
            str(sif.resolve()),
            str(lock.resolve()),
            str(site_manifest.resolve()),
        }:
            file_hash_calls.append(resolved)
        return real_sha(path)

    def count_tree(path: str | Path) -> dict[str, object]:
        nonlocal tree_calls
        tree_calls += 1
        return real_tree(path)

    monkeypatch.setattr(full_pool, "sha256_file", count_hash)
    monkeypatch.setattr(full_pool, "site_packages_tree_identity", count_tree)
    full_pool.validate_frozen_inputs(manifest, full_hash=False)
    assert file_hash_calls == []
    assert tree_calls == 0
    full_pool.validate_frozen_inputs(manifest, full_hash=True)
    assert file_hash_calls == [
        str(sif.resolve()),
        str(lock.resolve()),
        str(site_manifest.resolve()),
    ]
    assert tree_calls == 1

    (site / "pkg/data.bin").write_bytes(b"drift")
    with pytest.raises(ValueError, match="site-packages tree identity changed"):
        full_pool.validate_frozen_inputs(manifest, full_hash=True)


def test_container_runtime_rejects_wrong_expected_sif_or_lock_hash(
    tmp_path: Path,
) -> None:
    sif = tmp_path / "python.sif"
    lock = tmp_path / "exact_requirements.freeze"
    site = tmp_path / "site-packages"
    site_manifest = tmp_path / "site-packages.sha256"
    site.mkdir()
    sif.write_bytes(b"frozen-sif")
    lock.write_text("numpy==1.26.4\n")
    site_manifest.write_text("fixture\n")

    with pytest.raises(ValueError, match="container SIF SHA256"):
        _build_tiny_manifest(
            tmp_path,
            runtime_context={
                "container_sif_path": sif,
                "requirements_lock_path": lock,
                "site_packages_path": site,
                "site_packages_manifest_path": site_manifest,
                "expected_container_sif_sha256": "0" * 64,
            },
        )


def test_atomic_shard_manifest_supports_resume_and_detects_tampering(
    tmp_path: Path,
) -> None:
    from experiments.pu_synthesizability_20260821.remote_full_pool import (
        resume_shard,
        write_shard_atomic,
    )

    output = tmp_path / "part-00000-000000000-000000001.parquet"
    expected = {
        "input_fingerprint": "a" * 64,
        "task_id": 0,
        "start": 0,
        "stop": 2,
        "rows": 2,
    }
    frame = pd.DataFrame(
        {
            "pool_row": [0, 1],
            "orig_index": [1, 0],
            "pool_id": ["u1", "u0"],
            "parse_ok": [True, False],
            "cif_parse_route": ["default", "failed"],
        }
    )

    assert resume_shard(output, expected) is None
    summary = write_shard_atomic(frame, output, expected)
    assert summary["output_sha256"]
    assert summary["parse_failures"] == 1
    assert resume_shard(output, expected) == summary

    output.write_bytes(output.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="output SHA256 mismatch"):
        resume_shard(output, expected)


def test_resume_refuses_an_incomplete_output_manifest_pair(tmp_path: Path) -> None:
    from experiments.pu_synthesizability_20260821.remote_full_pool import resume_shard

    output = tmp_path / "part.parquet"
    output.write_bytes(b"orphan")
    with pytest.raises(ValueError, match="incomplete shard pair"):
        resume_shard(
            output,
            {
                "input_fingerprint": "a" * 64,
                "task_id": 0,
                "start": 0,
                "stop": 1,
                "rows": 1,
            },
        )


def test_frozen_remote_hashes_are_encoded_in_the_runner() -> None:
    from experiments.pu_synthesizability_20260821.remote_full_pool import (
        FROZEN_REMOTE_INPUTS,
    )

    assert FROZEN_REMOTE_INPUTS["train"]["sha256"] == (
        "eb6c89dfa3c15b4edbfcc701a985dfe51e5da51cce480f50b1ede404cbf45461"
    )
    assert FROZEN_REMOTE_INPUTS["val"]["sha256"] == (
        "d9f6a4719437bc827e0840152d576c93d3e514d664f0ecada18e2c47c4254fc2"
    )
    assert FROZEN_REMOTE_INPUTS["train"]["offset_sha256"] == (
        "e274985f66e7e3fe8b1168425028fd13ce70f825badf9cea0ff8a399807cb9e8"
    )
    assert FROZEN_REMOTE_INPUTS["val"]["offset_sha256"] == (
        "1645392531a6742309938789e424793315c5db97502da31d183f572aa1a4a7a4"
    )
    assert FROZEN_REMOTE_INPUTS["counts_sha256"] == (
        "a8d4400cb83293a1242bad67d9aaa05f04984e65f59312c9056d1e8420cb9b8c"
    )


def test_counts_file_is_frozen_and_must_match_candidate_counts(tmp_path: Path) -> None:
    from experiments.pu_synthesizability_20260821.remote_full_pool import (
        build_input_manifest,
        sha256_file,
    )

    train = tmp_path / "train.csv"
    val = tmp_path / "val.csv"
    train_offsets = _write_indexed_csv(train, [["0", "a", "data_a", ""]])
    val_offsets = _write_indexed_csv(val, [["1", "b", "data_b", ""]])
    counts = tmp_path / "counts.npy"
    np.save(counts, np.asarray([10, 2, 1, 1], dtype=np.int64))
    bvparm = tmp_path / "bvparm.cif"
    bvparm.write_text("test\n")

    manifest = build_input_manifest(
        [
            {"split": "train", "path": str(train), "offset_path": str(train_offsets)},
            {"split": "val", "path": str(val), "offset_path": str(val_offsets)},
        ],
        expected_counts={"train": 1, "val": 1},
        chunk_size=1,
        implementation_paths=[Path(__file__)],
        bvparm_path=bvparm,
        counts_path=counts,
        expected_counts_vector=[10, 2, 1, 1],
        expected_counts_sha256=sha256_file(counts),
    )
    assert manifest["counts"]["values"] == [10, 2, 1, 1]

    with pytest.raises(ValueError, match="counts.npy values do not match frozen counts"):
        build_input_manifest(
            [
                {"split": "train", "path": str(train), "offset_path": str(train_offsets)},
                {"split": "val", "path": str(val), "offset_path": str(val_offsets)},
            ],
            expected_counts={"train": 1, "val": 1},
            chunk_size=1,
            implementation_paths=[Path(__file__)],
            bvparm_path=bvparm,
            counts_path=counts,
            expected_counts_vector=[10, 2, 2, 0],
            expected_counts_sha256=sha256_file(counts),
        )


def test_run_task_uses_fixed_range_and_resumes_without_re_evaluation(
    tmp_path: Path,
) -> None:
    from experiments.pu_synthesizability_20260821.remote_full_pool import (
        build_input_manifest,
        run_task,
    )

    train = tmp_path / "train.csv"
    val = tmp_path / "val.csv"
    train_offsets = _write_indexed_csv(
        train,
        [
            ["1", "a", "data_a", ""],
            ["0", "b", "data_b", ""],
        ],
    )
    val_offsets = _write_indexed_csv(val, [["2", "c", "data_c", ""]])
    bvparm = tmp_path / "bvparm.cif"
    bvparm.write_text("test\n")
    manifest = build_input_manifest(
        [
            {"split": "train", "path": str(train), "offset_path": str(train_offsets)},
            {"split": "val", "path": str(val), "offset_path": str(val_offsets)},
        ],
        expected_counts={"train": 2, "val": 1},
        chunk_size=2,
        implementation_paths=[Path(__file__)],
        bvparm_path=bvparm,
    )
    calls = 0

    def evaluate(records: list[dict[str, object]]) -> pd.DataFrame:
        nonlocal calls
        calls += 1
        return pd.DataFrame(
            [
                {
                    **{key: row[key] for key in ("pool_row", "orig_index", "pool_id")},
                    "parse_ok": True,
                    "cif_parse_route": "default",
                }
                for row in records
            ]
        )

    first = run_task(
        manifest,
        task_id=0,
        output_dir=tmp_path / "results",
        workers=1,
        src_dir=tmp_path,
        bvparm_path=bvparm,
        batch_evaluator=evaluate,
    )
    second = run_task(
        manifest,
        task_id=0,
        output_dir=tmp_path / "results",
        workers=1,
        src_dir=tmp_path,
        bvparm_path=bvparm,
        batch_evaluator=evaluate,
    )

    assert calls == 1
    assert first == second
    assert (first["start"], first["stop"], first["rows"]) == (0, 2, 2)


def test_final_verifier_requires_every_row_once_and_supports_success_resume(
    tmp_path: Path,
) -> None:
    from experiments.pu_synthesizability_20260821.remote_full_pool import (
        run_task,
        sha256_file,
        verify_full_pool,
    )

    manifest, bvparm = _build_tiny_manifest(tmp_path)
    results = tmp_path / "results"

    def evaluate(records: list[dict[str, object]]) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    **{key: row[key] for key in ("pool_row", "orig_index", "pool_id")},
                    "parse_ok": True,
                    "cif_parse_route": "default",
                }
                for row in records
            ]
        )

    run_task(
        manifest,
        task_id=0,
        output_dir=results,
        workers=1,
        src_dir=tmp_path,
        bvparm_path=bvparm,
        batch_evaluator=evaluate,
    )
    with pytest.raises(FileNotFoundError, match="missing result shard for task 1"):
        verify_full_pool(manifest, output_dir=results)

    run_task(
        manifest,
        task_id=1,
        output_dir=results,
        workers=1,
        src_dir=tmp_path,
        bvparm_path=bvparm,
        batch_evaluator=evaluate,
    )
    duplicate_results = tmp_path / "duplicate-results"
    shutil.copytree(results, duplicate_results)
    for output in sorted(duplicate_results.glob("part-*.parquet")):
        summary_path = output.with_suffix(".manifest.json")
        summary = json.loads(summary_path.read_text())
        summary["output"] = str(output.resolve())
        if summary["task_id"] == 1:
            frame = pd.read_parquet(output)
            frame.loc[:, "orig_index"] = 1
            frame.loc[:, "pool_id"] = "u1"
            frame.to_parquet(output, index=False, engine="pyarrow", compression="zstd")
        summary["output_bytes"] = output.stat().st_size
        summary["output_sha256"] = sha256_file(output)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="duplicate orig_index"):
        verify_full_pool(manifest, output_dir=duplicate_results)

    success_path = tmp_path / "_SUCCESS.json"
    first = verify_full_pool(manifest, output_dir=results, success_path=success_path)
    second = verify_full_pool(manifest, output_dir=results, success_path=success_path)
    assert first == second
    assert first["total_rows"] == 3
    assert first["task_count"] == 2
    assert success_path.exists()


def test_final_verifier_accepts_a_frozen_unique_noncontiguous_index_set(
    tmp_path: Path,
) -> None:
    from experiments.pu_synthesizability_20260821.remote_full_pool import (
        build_input_manifest,
        run_task,
        verify_full_pool,
    )

    train = tmp_path / "train.csv"
    val = tmp_path / "val.csv"
    train_offsets = _write_indexed_csv(
        train,
        [["10", "a", "data_a", ""], ["30", "b", "data_b", ""]],
    )
    val_offsets = _write_indexed_csv(val, [["20", "c", "data_c", ""]])
    bvparm = tmp_path / "bvparm.cif"
    bvparm.write_text("test\n")
    manifest = build_input_manifest(
        [
            {"split": "train", "path": str(train), "offset_path": str(train_offsets)},
            {"split": "val", "path": str(val), "offset_path": str(val_offsets)},
        ],
        expected_counts={"train": 2, "val": 1},
        chunk_size=2,
        implementation_paths=[Path(__file__)],
        bvparm_path=bvparm,
    )

    def evaluate(records: list[dict[str, object]]) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    **{key: row[key] for key in ("pool_row", "orig_index", "pool_id")},
                    "parse_ok": True,
                    "cif_parse_route": "default",
                }
                for row in records
            ]
        )

    for task_id in range(2):
        run_task(
            manifest,
            task_id=task_id,
            output_dir=tmp_path / "results",
            workers=1,
            src_dir=tmp_path,
            bvparm_path=bvparm,
            batch_evaluator=evaluate,
        )
    summary = verify_full_pool(manifest, output_dir=tmp_path / "results")
    assert manifest["index_audit"]["contiguous_zero_based"] is False
    assert summary["orig_index_audit"] == manifest["index_audit"]


def test_production_batch_evaluator_uses_portable_bvparm_path() -> None:
    from pymatgen.core import Lattice, Structure

    from experiments.pu_synthesizability_20260821.remote_full_pool import (
        evaluate_records,
    )

    root = Path(__file__).resolve().parents[1]
    structure = Structure(
        Lattice.cubic(5.64),
        ["Na", "Cl"],
        [[0, 0, 0], [0.5, 0.5, 0.5]],
    )
    frame = evaluate_records(
        [
            {
                "cohort": "candidate_pool",
                "record_index": 0,
                "index": 0,
                "orig_index": 0,
                "pool_id": "u0",
                "pool_row": 0,
                "source_split": "train",
                "split_row": 0,
                "material_id": "NaCl",
                "cif": structure.to(fmt="cif"),
                "pbes_gap": None,
            }
        ],
        workers=1,
        chunksize=1,
        src_dir=root / "src",
        bvparm_path=root / "data/bvparm2020.cif",
    )

    assert len(frame) == 1
    assert frame.loc[0, "pool_row"] == 0
    assert frame.loc[0, "parse_ok"]
    assert frame.loc[0, "D8_verdict"] in {
        "pass",
        "explicit_violation",
        "no_verdict",
    }


def test_slurm_array_uses_full_cpu_nodes_without_thread_oversubscription() -> None:
    root = Path(__file__).resolve().parents[1]
    script = (
        root
        / "experiments/pu_synthesizability_20260821/pris_full_pool_remote.sbatch"
    ).read_text()

    assert "#SBATCH --partition=hcpu48,hcpu48y" in script
    assert "#SBATCH --cpus-per-task=48" in script
    assert "#SBATCH --array=0-162%12" in script
    assert "#SBATCH --mem" not in script
    assert "export OMP_NUM_THREADS=1" in script
    assert "export OPENBLAS_NUM_THREADS=1" in script
    assert "export MKL_NUM_THREADS=1" in script
    assert 'WORKERS="$((SLURM_CPUS_PER_TASK - 1))"' in script
    assert "python311-bookworm.sif" in script
    assert "env/site-packages" in script
    assert "source /etc/profile" in script
    assert "module load sigularity/3.8.7" in script
    assert script.index("source /etc/profile") < script.index("set -u")
    assert "srun --cpu-bind=cores /usr/bin/env" in script
    assert "singularity exec --bind /data1:/data1" in script
    assert "PYTHONDONTWRITEBYTECODE=1" in script
    assert "/envs/pu/bin/python" not in script
    assert " run-task " in script
