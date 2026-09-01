#!/usr/bin/env python3
"""Cross-source materialization of raw-structure MHCR features."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Sequence

import numpy as np
import pandas as pd
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from src.next11_geometry_only_frames import _parse_frame
from src.next84_scigen_geometry_lockbox import (
    GEOMETRY_NAMES as SCIGEN_GEOMETRY_NAMES,
    MANIFEST_NAME as SCIGEN_COHORT_MANIFEST_NAME,
    METADATA_NAME as SCIGEN_METADATA_NAME,
)
from src.next93b_wyformer_blind_lockbox import (
    GEOMETRY_NAMES as WYFORMER_GEOMETRY_NAMES,
    MANIFEST_NAME as WYFORMER_COHORT_MANIFEST_NAME,
    METADATA_NAME as WYFORMER_METADATA_NAME,
)
from src.next102_cross_source_dobvr_features import (
    _discovery_metadata,
    _read_json,
    _scigen_payloads,
    _sha256_file,
    _validate_scigen,
    _validate_wyformer,
    _write_json,
    _wyformer_payloads,
)
from src.next123_multiscale_hall_contact_robustness import (
    FEATURE_NAMES as BASE_FEATURE_NAMES,
    STRENGTH_THRESHOLDS,
    compute_multiscale_hall_contact_robustness,
)


PROTOCOL = "2026-08-08-next124-cross-source-discovery-mhcr-v1"
GRAPH_MODE = "voronoi"
DISCOVERY_ROLE = "discovery"
MANIFEST_NAME = "MANIFEST.json"
CATALOGUE_NAME = "NEXT124_MHCR_FEATURE_CATALOGUE.json"
FEATURE_FILES = {
    "scigen": "scigen_discovery_mhcr_features.parquet",
    "wyformer": "wyformer_discovery_mhcr_features.parquet",
}
CATALOGUE_MODES = ("core", "expanded")
NUMERIC_FEATURE_NAMES = tuple(
    name.replace("mhcr_", f"mhcr_{mode}_", 1)
    for mode in CATALOGUE_MODES
    for name in BASE_FEATURE_NAMES
)
STATUS_COLUMNS = tuple(
    name
    for mode in CATALOGUE_MODES
    for name in (
        f"mhcr_{mode}_supported",
        f"mhcr_{mode}_failure",
        f"mhcr_{mode}_catalogue_sha256",
        f"mhcr_{mode}_pymatgen_version",
        f"mhcr_{mode}_scipy_version",
    )
)
FEATURE_COLUMNS = (*NUMERIC_FEATURE_NAMES, *STATUS_COLUMNS)
EXPECTED_INPUT_SHA256 = {
    "scigen_cohort_manifest": "dc5bf33c6ce6dc2c10bcd3704688055058145fbe7269ada23ffbe4b141d75fe7",
    "scigen_metadata": "f91455f23b0a96f60fd1c779249e2be46a7ecf94fcdde2b146426a95aac05bde",
    "scigen_geometry_discovery": "e561ef12343c66dcc72bcabf6b8719ad727e01c9582a094e281da73b862ab575",
    "wyformer_cohort_manifest": "e0539d556538cb4c052431bc6a1e5c1663bc3de427677dbc8a446dcc3b4fbc54",
    "wyformer_metadata": "3b152b4b84c8d3f7ff5e85611dc1fd2728296f150e907ac4578ce55d2b27dd2b",
    "wyformer_geometry_discovery": "f1ce5ae4fba8c13fcbf3e25de4f596b919d9b41da5b072d9a28eefeaffc69784",
    "design": "681f32fd5e4c5e5c795db128c49695ae6237a2e92edf0987125b54da58a4ca1a",
}


def compute_mhcr_feature_row(structure, *, graph_mode: str) -> dict[str, object]:
    """Compute core and expanded MHCR with independent abstention."""

    row: dict[str, object] = {name: math.nan for name in NUMERIC_FEATURE_NAMES}
    statuses: dict[str, object] = {}
    for mode in CATALOGUE_MODES:
        try:
            result = compute_multiscale_hall_contact_robustness(
                structure,
                graph_mode=graph_mode,
                catalogue_mode=mode,
            )
        except Exception as exc:
            statuses.update(
                {
                    f"mhcr_{mode}_supported": False,
                    f"mhcr_{mode}_failure": f"{type(exc).__name__}: {exc}",
                    f"mhcr_{mode}_catalogue_sha256": None,
                    f"mhcr_{mode}_pymatgen_version": None,
                    f"mhcr_{mode}_scipy_version": None,
                }
            )
            continue
        for name, value in result.features.items():
            row[name.replace("mhcr_", f"mhcr_{mode}_", 1)] = float(value)
        statuses.update(
            {
                f"mhcr_{mode}_supported": bool(result.supported),
                f"mhcr_{mode}_failure": result.failure_reason,
                f"mhcr_{mode}_catalogue_sha256": result.catalogue_sha256,
                f"mhcr_{mode}_pymatgen_version": result.pymatgen_version,
                f"mhcr_{mode}_scipy_version": result.scipy_version,
            }
        )
    row.update(statuses)
    if tuple(row) != FEATURE_COLUMNS:
        raise RuntimeError("NEXT124 feature row schema differs")
    return row


def _parse_error_row(message: str) -> dict[str, object]:
    row: dict[str, object] = {name: math.nan for name in NUMERIC_FEATURE_NAMES}
    for mode in CATALOGUE_MODES:
        row.update(
            {
                f"mhcr_{mode}_supported": False,
                f"mhcr_{mode}_failure": message,
                f"mhcr_{mode}_catalogue_sha256": None,
                f"mhcr_{mode}_pymatgen_version": None,
                f"mhcr_{mode}_scipy_version": None,
            }
        )
    return row


def _compute_scigen_payload(item: tuple[str, bytes]) -> tuple[str, dict[str, object]]:
    material_id, payload = item
    try:
        atoms = _parse_frame(payload, strict_output=True).atoms
        structure = AseAtomsAdaptor.get_structure(atoms)
        row = compute_mhcr_feature_row(structure, graph_mode=GRAPH_MODE)
    except Exception as exc:
        row = _parse_error_row(f"structure_parse: {type(exc).__name__}: {exc}")
    return material_id, row


def _compute_wyformer_payload(item: tuple[str, str]) -> tuple[str, dict[str, object]]:
    material_id, payload = item
    try:
        structure = Structure.from_dict(json.loads(payload))
        row = compute_mhcr_feature_row(structure, graph_mode=GRAPH_MODE)
    except Exception as exc:
        row = _parse_error_row(f"structure_parse: {type(exc).__name__}: {exc}")
    return material_id, row


def _compute_many(payloads: Sequence[tuple], *, source: str, workers: int):
    function = _compute_scigen_payload if source == "scigen" else _compute_wyformer_payload
    if workers == 1:
        return [function(item) for item in payloads]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(function, payloads, chunksize=8))


def _source_counts(table: pd.DataFrame) -> dict[str, object]:
    counts: dict[str, object] = {"rows": int(len(table))}
    for mode in CATALOGUE_MODES:
        supported = f"mhcr_{mode}_supported"
        failure = f"mhcr_{mode}_failure"
        counts[f"{mode}_supported"] = int(table[supported].eq(True).sum())
        counts[f"{mode}_failures"] = dict(
            Counter(table.loc[table[supported].eq(False), failure])
        )
    counts["finite_numeric_features"] = {
        name: int(np.isfinite(pd.to_numeric(table[name], errors="coerce")).sum())
        for name in NUMERIC_FEATURE_NAMES
    }
    return counts


def _solver_thread_environment(
    *, require_formal_inputs: bool
) -> dict[str, str | None]:
    names = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
    values = {name: os.environ.get(name) for name in names}
    if require_formal_inputs and any(value != "1" for value in values.values()):
        raise ValueError("NEXT124 formal run requires a single-thread solver environment")
    return values


def build_cross_source_discovery_mhcr_features(
    *,
    scigen_cohort_dir: Path,
    wyformer_cohort_dir: Path,
    design_path: Path,
    output_dir: Path,
    workers: int = 12,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    """Materialize MHCR only from physically split discovery geometries."""

    scigen = Path(scigen_cohort_dir).resolve()
    wyformer = Path(wyformer_cohort_dir).resolve()
    target = Path(output_dir).resolve()
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if type(workers) is not int or workers <= 0:
        raise ValueError("workers must be a positive exact integer")
    solver_thread_environment = _solver_thread_environment(
        require_formal_inputs=require_formal_inputs
    )
    paths = {
        "scigen_cohort_manifest": scigen / SCIGEN_COHORT_MANIFEST_NAME,
        "scigen_metadata": scigen / SCIGEN_METADATA_NAME,
        "scigen_geometry_discovery": scigen / SCIGEN_GEOMETRY_NAMES[DISCOVERY_ROLE],
        "wyformer_cohort_manifest": wyformer / WYFORMER_COHORT_MANIFEST_NAME,
        "wyformer_metadata": wyformer / WYFORMER_METADATA_NAME,
        "wyformer_geometry_discovery": wyformer / WYFORMER_GEOMETRY_NAMES[DISCOVERY_ROLE],
        "design": Path(design_path).resolve(),
    }
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT124 discovery input is missing")
    input_hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if require_formal_inputs and input_hashes != EXPECTED_INPUT_SHA256:
        raise ValueError("NEXT124 formal input identity differs")
    _validate_scigen(_read_json(paths["scigen_cohort_manifest"]), input_hashes)
    _validate_wyformer(_read_json(paths["wyformer_cohort_manifest"]), input_hashes)
    metadata = {
        "scigen": _discovery_metadata(paths["scigen_metadata"], source="scigen"),
        "wyformer": _discovery_metadata(paths["wyformer_metadata"], source="wyformer"),
    }
    payloads = {
        "scigen": _scigen_payloads(
            paths["scigen_geometry_discovery"],
            metadata["scigen"]["material_id"].astype(str).tolist(),
        ),
        "wyformer": _wyformer_payloads(
            paths["wyformer_geometry_discovery"],
            metadata["wyformer"]["material_id"].astype(str).tolist(),
        ),
    }

    repository_root = Path(__file__).resolve().parents[1]
    source_paths = {
        "src/next19_valence_transport.py": repository_root / "src/next19_valence_transport.py",
        "src/next102_cross_source_dobvr_features.py": repository_root / "src/next102_cross_source_dobvr_features.py",
        "src/next104_convex_mixed_valence_flow.py": repository_root / "src/next104_convex_mixed_valence_flow.py",
        "src/next109_convex_mixed_valence_obstruction.py": repository_root / "src/next109_convex_mixed_valence_obstruction.py",
        "src/next123_multiscale_hall_contact_robustness.py": repository_root / "src/next123_multiscale_hall_contact_robustness.py",
        "src/next124_cross_source_mhcr_features.py": Path(__file__).resolve(),
    }
    source_hashes = {name: _sha256_file(path) for name, path in source_paths.items()}
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    output_paths: list[Path] = []
    counts: dict[str, object] = {}
    started = time.perf_counter()
    try:
        for source in ("scigen", "wyformer"):
            computed = _compute_many(payloads[source], source=source, workers=workers)
            frame = pd.DataFrame(
                [{"material_id": material_id, **row} for material_id, row in computed]
            )
            table = metadata[source].merge(
                frame, on="material_id", how="left", validate="one_to_one"
            )
            if len(table) != len(metadata[source]) or any(
                name not in table for name in FEATURE_COLUMNS
            ):
                raise RuntimeError(f"NEXT124 {source} row accounting differs")
            feature_path = staging / FEATURE_FILES[source]
            table.to_parquet(feature_path, index=False)
            output_paths.append(feature_path)
            counts[source] = _source_counts(table)

        catalogue = {
            "protocol": PROTOCOL,
            "graph_mode": GRAPH_MODE,
            "strength_thresholds": list(STRENGTH_THRESHOLDS),
            "partitions_read": [DISCOVERY_ROLE],
            "numeric_feature_names": list(NUMERIC_FEATURE_NAMES),
            "status_columns": list(STATUS_COLUMNS),
            "catalogue_modes": list(CATALOGUE_MODES),
            "endpoint_columns_present": False,
            "labels_opened": False,
        }
        catalogue_path = staging / CATALOGUE_NAME
        _write_json(catalogue_path, catalogue)
        output_paths.append(catalogue_path)
        manifest: dict[str, object] = {
            "protocol": PROTOCOL,
            "mode": "cross_source_discovery_only_raw_x0_mhcr_feature_freeze",
            "graph_mode": GRAPH_MODE,
            "strength_thresholds": list(STRENGTH_THRESHOLDS),
            "workers": workers,
            "solver_thread_environment": solver_thread_environment,
            "elapsed_seconds": time.perf_counter() - started,
            "partitions_read": [DISCOVERY_ROLE],
            "counts": counts,
            "labels_opened": False,
            "endpoint_payloads_opened": False,
            "validation_geometry_opened": False,
            "replication_geometry_opened": False,
            "relaxed_structures_opened": False,
            "dft_calculation_executed": False,
            "dft_values_used_by_features": False,
            "learned_energy_force_stress_proxy_used": False,
            "model_or_proxy_potential_used": False,
            "physical_relaxation_executed": False,
            "inputs_sha256": {
                name: {"path": str(paths[name]), "sha256": value}
                for name, value in input_hashes.items()
            },
            "executed_source_sha256": source_hashes,
            "outputs_sha256": {
                path.name: _sha256_file(path) for path in output_paths
            },
            "scientific_improvement_claim": False,
        }
        _write_json(staging / MANIFEST_NAME, manifest)
        if any(_sha256_file(path) != input_hashes[name] for name, path in paths.items()):
            raise RuntimeError("NEXT124 input changed before publication")
        if any(_sha256_file(path) != source_hashes[name] for name, path in source_paths.items()):
            raise RuntimeError("NEXT124 source changed before publication")
        os.replace(staging, target)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scigen-cohort-dir", type=Path, required=True)
    parser.add_argument("--wyformer-cohort-dir", type=Path, required=True)
    parser.add_argument("--design-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    manifest = build_cross_source_discovery_mhcr_features(
        scigen_cohort_dir=args.scigen_cohort_dir,
        wyformer_cohort_dir=args.wyformer_cohort_dir,
        design_path=args.design_path,
        output_dir=args.output_dir,
        workers=args.workers,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EXPECTED_INPUT_SHA256",
    "FEATURE_COLUMNS",
    "FEATURE_FILES",
    "MANIFEST_NAME",
    "NUMERIC_FEATURE_NAMES",
    "PROTOCOL",
    "build_cross_source_discovery_mhcr_features",
    "compute_mhcr_feature_row",
]
