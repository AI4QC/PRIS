from __future__ import annotations

import inspect
import math
from pathlib import Path

import pandas as pd
from pymatgen.core import Lattice, Structure
import pytest

from src.next84_scigen_geometry_lockbox import (
    GEOMETRY_NAMES as SCIGEN_GEOMETRY_NAMES,
    MANIFEST_NAME as SCIGEN_MANIFEST_NAME,
    METADATA_NAME as SCIGEN_METADATA_NAME,
)
from src.next93b_wyformer_blind_lockbox import (
    GEOMETRY_NAMES as WYFORMER_GEOMETRY_NAMES,
    MANIFEST_NAME as WYFORMER_MANIFEST_NAME,
    METADATA_NAME as WYFORMER_METADATA_NAME,
)
from src.next124_cross_source_mhcr_features import (
    EXPECTED_INPUT_SHA256,
    FEATURE_COLUMNS,
    FEATURE_FILES,
    MANIFEST_NAME,
    NUMERIC_FEATURE_NAMES,
    PROTOCOL,
    build_cross_source_discovery_mhcr_features,
    compute_mhcr_feature_row,
)
import src.next124_cross_source_mhcr_features as next124


def test_formal_identity_and_schema_are_label_free() -> None:
    assert PROTOCOL == "2026-08-08-next124-cross-source-discovery-mhcr-v1"
    assert EXPECTED_INPUT_SHA256["design"] == (
        "681f32fd5e4c5e5c795db128c49695ae6237a2e92edf0987125b54da58a4ca1a"
    )
    assert len(NUMERIC_FEATURE_NAMES) == 16
    forbidden = ("energy", "force", "stress", "relax", "dft", "model", "proxy", "endpoint")
    assert not any(
        token in name.lower() for name in FEATURE_COLUMNS for token in forbidden
    )


def test_row_computes_core_and_expanded_mhcr_independently() -> None:
    structure = Structure(
        Lattice.cubic(4.2),
        ["Na", "O"],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )
    row = compute_mhcr_feature_row(structure, graph_mode="voronoi")
    assert tuple(row) == FEATURE_COLUMNS
    for mode in ("core", "expanded"):
        assert row[f"mhcr_{mode}_supported"] is True
        assert row[f"mhcr_{mode}_failure"] is None
        assert len(str(row[f"mhcr_{mode}_catalogue_sha256"])) == 64
    assert all(math.isfinite(float(row[name])) for name in NUMERIC_FEATURE_NAMES)
    assert all(0.0 <= float(row[name]) <= 1.0 for name in NUMERIC_FEATURE_NAMES)


def _dummy_input_tree(root: Path) -> tuple[Path, Path, Path]:
    scigen = root / "scigen"
    wyformer = root / "wyformer"
    scigen.mkdir()
    wyformer.mkdir()
    for path in (
        scigen / SCIGEN_MANIFEST_NAME,
        scigen / SCIGEN_METADATA_NAME,
        scigen / SCIGEN_GEOMETRY_NAMES["discovery"],
        wyformer / WYFORMER_MANIFEST_NAME,
        wyformer / WYFORMER_METADATA_NAME,
        wyformer / WYFORMER_GEOMETRY_NAMES["discovery"],
    ):
        path.write_bytes(b"frozen synthetic input")
    design = root / "design.md"
    design.write_text("frozen\n", encoding="utf-8")
    return scigen, wyformer, design


def test_builder_reads_only_discovery_publishes_atomically_and_refuses_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parameters = inspect.signature(
        build_cross_source_discovery_mhcr_features
    ).parameters
    assert not any(
        token in name
        for name in parameters
        for token in ("endpoint", "validation", "replication")
    )
    scigen, wyformer, design = _dummy_input_tree(tmp_path)
    metadata = {
        "scigen": pd.DataFrame({"material_id": ["s1"], "partition_role": ["discovery"]}),
        "wyformer": pd.DataFrame({"material_id": ["w1"], "partition_role": ["discovery"]}),
    }
    monkeypatch.setattr(next124, "_validate_scigen", lambda *args, **kwargs: None)
    monkeypatch.setattr(next124, "_validate_wyformer", lambda *args, **kwargs: None)
    monkeypatch.setattr(next124, "_read_json", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        next124,
        "_discovery_metadata",
        lambda path, *, source: metadata[source].copy(),
    )
    monkeypatch.setattr(next124, "_scigen_payloads", lambda *args, **kwargs: [("s1", b"")])
    monkeypatch.setattr(next124, "_wyformer_payloads", lambda *args, **kwargs: [("w1", "")])

    def fake_many(payloads, *, source, workers):
        row = {name: 0.0 for name in NUMERIC_FEATURE_NAMES}
        for mode in ("core", "expanded"):
            row.update(
                {
                    f"mhcr_{mode}_supported": True,
                    f"mhcr_{mode}_failure": None,
                    f"mhcr_{mode}_catalogue_sha256": "a" * 64,
                    f"mhcr_{mode}_pymatgen_version": "test",
                    f"mhcr_{mode}_scipy_version": "test",
                }
            )
        return [(payloads[0][0], row)]

    monkeypatch.setattr(next124, "_compute_many", fake_many)
    output = tmp_path / "features"
    manifest = build_cross_source_discovery_mhcr_features(
        scigen_cohort_dir=scigen,
        wyformer_cohort_dir=wyformer,
        design_path=design,
        output_dir=output,
        workers=1,
        require_formal_inputs=False,
    )
    assert manifest["partitions_read"] == ["discovery"]
    for name in (
        "labels_opened",
        "endpoint_payloads_opened",
        "validation_geometry_opened",
        "replication_geometry_opened",
        "dft_calculation_executed",
        "dft_values_used_by_features",
        "learned_energy_force_stress_proxy_used",
        "physical_relaxation_executed",
    ):
        assert manifest[name] is False
    assert (output / MANIFEST_NAME).is_file()
    for source in ("scigen", "wyformer"):
        table = pd.read_parquet(output / FEATURE_FILES[source])
        assert len(table) == 1
        assert set(FEATURE_COLUMNS) <= set(table.columns)
    with pytest.raises(FileExistsError):
        build_cross_source_discovery_mhcr_features(
            scigen_cohort_dir=scigen,
            wyformer_cohort_dir=wyformer,
            design_path=design,
            output_dir=output,
            workers=1,
            require_formal_inputs=False,
        )


def test_formal_solver_environment_is_single_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = {
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    for name, value in expected.items():
        monkeypatch.setenv(name, value)
    assert next124._solver_thread_environment(require_formal_inputs=True) == expected
    monkeypatch.setenv("MKL_NUM_THREADS", "4")
    with pytest.raises(ValueError, match="single-thread solver environment"):
        next124._solver_thread_environment(require_formal_inputs=True)
