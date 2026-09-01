#!/usr/bin/env python3
"""Discovery-only materialization of NEXT101 and NEXT101b features."""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Mapping, Sequence
import zipfile

import numpy as np
import pandas as pd
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from src.next11_geometry_only_frames import _parse_frame
from src.next101_discrete_oxidation_bv_realizability import (
    FEATURE_NAMES as DOBVR_FEATURE_NAMES,
    compute_discrete_oxidation_bv_realizability,
)
from src.next101b_expanded_oxidation_bv_realizability import (
    FEATURE_NAMES as DOBVRB_FEATURE_NAMES,
    compute_expanded_discrete_oxidation_bv_realizability,
)
from src.next84_scigen_geometry_lockbox import (
    GEOMETRY_NAMES as SCIGEN_GEOMETRY_NAMES,
    MANIFEST_NAME as SCIGEN_COHORT_MANIFEST_NAME,
    METADATA_NAME as SCIGEN_METADATA_NAME,
    PROTOCOL as SCIGEN_COHORT_PROTOCOL,
)
from src.next93b_wyformer_blind_lockbox import (
    GEOMETRY_NAMES as WYFORMER_GEOMETRY_NAMES,
    INPUT_ROLE as WYFORMER_INPUT_ROLE,
    MANIFEST_NAME as WYFORMER_COHORT_MANIFEST_NAME,
    METADATA_NAME as WYFORMER_METADATA_NAME,
    PROTOCOL as WYFORMER_COHORT_PROTOCOL,
)


PROTOCOL = "2026-08-04-next102-cross-source-discovery-dobvr-v1"
GRAPH_MODE = "voronoi"
DISCOVERY_ROLE = "discovery"
MANIFEST_NAME = "MANIFEST.json"
CATALOGUE_NAME = "NEXT102_DOBVR_FEATURE_CATALOGUE.json"
FEATURE_NAMES = {
    "scigen": "scigen_discovery_dobvr_features.parquet",
    "wyformer": "wyformer_discovery_dobvr_features.parquet",
}
EXPECTED_INPUT_SHA256 = {
    "scigen_cohort_manifest": "dc5bf33c6ce6dc2c10bcd3704688055058145fbe7269ada23ffbe4b141d75fe7",
    "scigen_metadata": "f91455f23b0a96f60fd1c779249e2be46a7ecf94fcdde2b146426a95aac05bde",
    "scigen_geometry_discovery": "e561ef12343c66dcc72bcabf6b8719ad727e01c9582a094e281da73b862ab575",
    "wyformer_cohort_manifest": "e0539d556538cb4c052431bc6a1e5c1663bc3de427677dbc8a446dcc3b4fbc54",
    "wyformer_metadata": "3b152b4b84c8d3f7ff5e85611dc1fd2728296f150e907ac4578ce55d2b27dd2b",
    "wyformer_geometry_discovery": "f1ce5ae4fba8c13fcbf3e25de4f596b919d9b41da5b072d9a28eefeaffc69784",
    "design": "d25bec2b21574f41158eda2b03f31ea5285a1d2fe7c72cfbad2fcbc27a05ac9d",
    "amendment": "ad60c2b12edd2ff9c099b5cbcbdf3d840187f6a6a7e269e73d5f48b1e9096b4f",
}
STATUS_COLUMNS = (
    "dobvr_supported",
    "dobvr_failure",
    "dobvr_catalogue_sha256",
    "dobvr_pymatgen_version",
    "dobvr_best_assignment_json",
    "dobvrb_supported",
    "dobvrb_failure",
    "dobvrb_catalogue_sha256",
    "dobvrb_pymatgen_version",
    "dobvrb_best_assignment_json",
)
FEATURE_COLUMNS = (*DOBVR_FEATURE_NAMES, *DOBVRB_FEATURE_NAMES, *STATUS_COLUMNS)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    Path(path).write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _assignment_json(assignment) -> str | None:
    if assignment is None:
        return None
    payload: dict[str, object] = {
        "element_states": [list(item) for item in assignment.element_states],
    }
    if hasattr(assignment, "catalogue_tier"):
        payload["catalogue_tier"] = int(assignment.catalogue_tier)
    if hasattr(assignment, "electronegativity_margin"):
        payload["electronegativity_margin"] = float(
            assignment.electronegativity_margin
        )
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def compute_dobvr_feature_row(structure, *, graph_mode: str) -> dict[str, object]:
    """Compute both additive DOBVR families with independent fail-open status."""

    row: dict[str, object] = {
        name: math.nan for name in (*DOBVR_FEATURE_NAMES, *DOBVRB_FEATURE_NAMES)
    }
    try:
        strict = compute_discrete_oxidation_bv_realizability(
            structure, graph_mode=graph_mode
        )
    except Exception as exc:
        strict_status = {
            "dobvr_supported": False,
            "dobvr_failure": f"{type(exc).__name__}: {exc}",
            "dobvr_catalogue_sha256": None,
            "dobvr_pymatgen_version": None,
            "dobvr_best_assignment_json": None,
        }
    else:
        row.update(strict.features)
        strict_status = {
            "dobvr_supported": bool(strict.supported),
            "dobvr_failure": strict.failure_reason,
            "dobvr_catalogue_sha256": strict.catalogue_sha256,
            "dobvr_pymatgen_version": strict.pymatgen_version,
            "dobvr_best_assignment_json": _assignment_json(strict.best_assignment),
        }
    try:
        expanded = compute_expanded_discrete_oxidation_bv_realizability(
            structure, graph_mode=graph_mode
        )
    except Exception as exc:
        expanded_status = {
            "dobvrb_supported": False,
            "dobvrb_failure": f"{type(exc).__name__}: {exc}",
            "dobvrb_catalogue_sha256": None,
            "dobvrb_pymatgen_version": None,
            "dobvrb_best_assignment_json": None,
        }
    else:
        row.update(expanded.features)
        expanded_status = {
            "dobvrb_supported": bool(expanded.supported),
            "dobvrb_failure": expanded.failure_reason,
            "dobvrb_catalogue_sha256": expanded.catalogue_sha256,
            "dobvrb_pymatgen_version": expanded.pymatgen_version,
            "dobvrb_best_assignment_json": _assignment_json(
                expanded.best_assignment
            ),
        }
    row.update(strict_status)
    row.update(expanded_status)
    if tuple(row) != FEATURE_COLUMNS:
        raise RuntimeError("NEXT102 feature row schema differs")
    return row


def _parse_error_row(message: str) -> dict[str, object]:
    row: dict[str, object] = {
        name: math.nan for name in (*DOBVR_FEATURE_NAMES, *DOBVRB_FEATURE_NAMES)
    }
    row.update(
        {
            "dobvr_supported": False,
            "dobvr_failure": message,
            "dobvr_catalogue_sha256": None,
            "dobvr_pymatgen_version": None,
            "dobvr_best_assignment_json": None,
            "dobvrb_supported": False,
            "dobvrb_failure": message,
            "dobvrb_catalogue_sha256": None,
            "dobvrb_pymatgen_version": None,
            "dobvrb_best_assignment_json": None,
        }
    )
    return row


def _compute_scigen_payload(item: tuple[str, bytes]) -> tuple[str, dict[str, object]]:
    material_id, payload = item
    try:
        atoms = _parse_frame(payload, strict_output=True).atoms
        structure = AseAtomsAdaptor.get_structure(atoms)
        row = compute_dobvr_feature_row(structure, graph_mode=GRAPH_MODE)
    except Exception as exc:
        row = _parse_error_row(f"structure_parse: {type(exc).__name__}: {exc}")
    return material_id, row


def _compute_wyformer_payload(item: tuple[str, str]) -> tuple[str, dict[str, object]]:
    material_id, payload = item
    try:
        structure = Structure.from_dict(json.loads(payload))
        row = compute_dobvr_feature_row(structure, graph_mode=GRAPH_MODE)
    except Exception as exc:
        row = _parse_error_row(f"structure_parse: {type(exc).__name__}: {exc}")
    return material_id, row


def _compute_many(payloads: Sequence[tuple], *, source: str, workers: int):
    function = _compute_scigen_payload if source == "scigen" else _compute_wyformer_payload
    if workers == 1:
        return [function(item) for item in payloads]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(function, payloads, chunksize=8))


def _scigen_payloads(
    path: Path, expected_ids: Sequence[str]
) -> list[tuple[str, bytes]]:
    expected = tuple(sorted(str(value) for value in expected_ids))
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if names != sorted(names) or any(not name.endswith(".extxyz") for name in names):
            raise ValueError("SCIGEN discovery geometry inventory differs")
        ids = tuple(Path(name).stem for name in names)
        if ids != expected:
            raise ValueError("SCIGEN discovery geometry identity differs")
        return [
            (material_id, archive.read(name))
            for material_id, name in zip(ids, names, strict=True)
        ]


def _wyformer_payloads(
    path: Path, expected_ids: Sequence[str]
) -> list[tuple[str, str]]:
    frame = pd.read_parquet(path)
    if set(frame.columns) != {"material_id", "structure_json"}:
        raise ValueError("WyFormer discovery geometry columns differ")
    if frame["material_id"].duplicated().any():
        raise ValueError("WyFormer discovery geometry identities are duplicated")
    mapping = dict(
        zip(frame["material_id"].astype(str), frame["structure_json"].astype(str))
    )
    expected = tuple(str(value) for value in expected_ids)
    if set(mapping) != set(expected):
        raise ValueError("WyFormer discovery geometry identity differs")
    return [(material_id, mapping[material_id]) for material_id in expected]


def _validate_scigen(
    manifest: Mapping[str, object], hashes: Mapping[str, str]
) -> None:
    outputs = manifest.get("outputs_sha256")
    if (
        manifest.get("protocol") != SCIGEN_COHORT_PROTOCOL
        or manifest.get("labels_opened") is not False
        or manifest.get("endpoint_payloads_opened") is not False
        or manifest.get("relaxed_structures_opened") is not False
        or not isinstance(outputs, Mapping)
        or outputs.get(SCIGEN_METADATA_NAME) != hashes["scigen_metadata"]
        or outputs.get(SCIGEN_GEOMETRY_NAMES[DISCOVERY_ROLE])
        != hashes["scigen_geometry_discovery"]
    ):
        raise ValueError("SCIGEN discovery cohort provenance differs")


def _validate_wyformer(
    manifest: Mapping[str, object], hashes: Mapping[str, str]
) -> None:
    outputs = manifest.get("outputs_sha256")
    if (
        manifest.get("protocol") != WYFORMER_COHORT_PROTOCOL
        or manifest.get("labels_opened_by_feature_builder") is not False
        or manifest.get("discovery_endpoint_opened") is not False
        or manifest.get("validation_endpoint_opened") is not False
        or manifest.get("replication_endpoint_opened") is not False
        or manifest.get("relaxed_structures_published") is not False
        or manifest.get("learned_proxy_execution_input") is not False
        or not isinstance(outputs, Mapping)
        or outputs.get(WYFORMER_METADATA_NAME) != hashes["wyformer_metadata"]
        or outputs.get(WYFORMER_GEOMETRY_NAMES[DISCOVERY_ROLE])
        != hashes["wyformer_geometry_discovery"]
    ):
        raise ValueError("WyFormer discovery cohort provenance differs")


def _discovery_metadata(path: Path, *, source: str) -> pd.DataFrame:
    table = pd.read_parquet(path)
    common = {
        "material_id",
        "reduced_formula",
        "chemical_system",
        "natoms",
        "partition_role",
        "input_role",
    }
    required = (
        common | {"lattice_class"}
        if source == "scigen"
        else common | {"generated_space_group", "crystal_system"}
    )
    if required - set(table.columns) or table["material_id"].duplicated().any():
        raise ValueError(f"{source} metadata schema or identity differs")
    expected_role = (
        "raw_generated_pre_dft_unrelaxed_x0"
        if source == "scigen"
        else WYFORMER_INPUT_ROLE
    )
    if not table["input_role"].eq(expected_role).all():
        raise ValueError(f"{source} metadata input role differs")
    part = table[table["partition_role"].eq(DISCOVERY_ROLE)].copy()
    if part.empty:
        raise ValueError(f"{source} discovery metadata is empty")
    return part.sort_values("material_id", kind="stable", ignore_index=True)


def _source_counts(table: pd.DataFrame) -> dict[str, object]:
    return {
        "rows": int(len(table)),
        "dobvr_supported": int(table["dobvr_supported"].eq(True).sum()),
        "dobvrb_supported": int(table["dobvrb_supported"].eq(True).sum()),
        "dobvr_failures": dict(
            Counter(table.loc[table["dobvr_supported"].eq(False), "dobvr_failure"])
        ),
        "dobvrb_failures": dict(
            Counter(table.loc[table["dobvrb_supported"].eq(False), "dobvrb_failure"])
        ),
        "finite_numeric_features": {
            name: int(np.isfinite(pd.to_numeric(table[name], errors="coerce")).sum())
            for name in (*DOBVR_FEATURE_NAMES, *DOBVRB_FEATURE_NAMES)
        },
    }


def build_cross_source_discovery_dobvr_features(
    *,
    scigen_cohort_dir: Path,
    wyformer_cohort_dir: Path,
    design_path: Path,
    amendment_path: Path,
    output_dir: Path,
    workers: int = 16,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    """Materialize DOBVR only from both physically split discovery geometries."""

    scigen = Path(scigen_cohort_dir).resolve()
    wyformer = Path(wyformer_cohort_dir).resolve()
    target = Path(output_dir).resolve()
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if type(workers) is not int or workers <= 0:
        raise ValueError("workers must be a positive exact integer")
    paths = {
        "scigen_cohort_manifest": scigen / SCIGEN_COHORT_MANIFEST_NAME,
        "scigen_metadata": scigen / SCIGEN_METADATA_NAME,
        "scigen_geometry_discovery": scigen
        / SCIGEN_GEOMETRY_NAMES[DISCOVERY_ROLE],
        "wyformer_cohort_manifest": wyformer / WYFORMER_COHORT_MANIFEST_NAME,
        "wyformer_metadata": wyformer / WYFORMER_METADATA_NAME,
        "wyformer_geometry_discovery": wyformer
        / WYFORMER_GEOMETRY_NAMES[DISCOVERY_ROLE],
        "design": Path(design_path).resolve(),
        "amendment": Path(amendment_path).resolve(),
    }
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT102 discovery input is missing")
    input_hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if require_formal_inputs and input_hashes != EXPECTED_INPUT_SHA256:
        raise ValueError("NEXT102 formal input identity differs")
    _validate_scigen(_read_json(paths["scigen_cohort_manifest"]), input_hashes)
    _validate_wyformer(_read_json(paths["wyformer_cohort_manifest"]), input_hashes)

    metadata = {
        "scigen": _discovery_metadata(paths["scigen_metadata"], source="scigen"),
        "wyformer": _discovery_metadata(
            paths["wyformer_metadata"], source="wyformer"
        ),
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
        "src/next101_discrete_oxidation_bv_realizability.py": repository_root
        / "src/next101_discrete_oxidation_bv_realizability.py",
        "src/next101b_expanded_oxidation_bv_realizability.py": repository_root
        / "src/next101b_expanded_oxidation_bv_realizability.py",
        "src/next102_cross_source_dobvr_features.py": Path(__file__).resolve(),
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
            computed_frame = pd.DataFrame(
                [{"material_id": material_id, **row} for material_id, row in computed]
            )
            table = metadata[source].merge(
                computed_frame,
                on="material_id",
                how="left",
                validate="one_to_one",
            )
            if len(table) != len(metadata[source]) or tuple(
                name for name in FEATURE_COLUMNS if name not in table
            ):
                raise RuntimeError(f"NEXT102 {source} row accounting differs")
            feature_path = staging / FEATURE_NAMES[source]
            table.to_parquet(feature_path, index=False)
            output_paths.append(feature_path)
            counts[source] = _source_counts(table)

        catalogue = {
            "protocol": PROTOCOL,
            "graph_mode": GRAPH_MODE,
            "partitions_read": [DISCOVERY_ROLE],
            "numeric_feature_names": [
                *DOBVR_FEATURE_NAMES,
                *DOBVRB_FEATURE_NAMES,
            ],
            "status_columns": list(STATUS_COLUMNS),
            "endpoint_columns_present": False,
            "labels_opened": False,
            "families": {
                "NEXT101_common_icsd_uniform": list(DOBVR_FEATURE_NAMES),
                "NEXT101b_all_table_oriented_uniform": list(DOBVRB_FEATURE_NAMES),
            },
        }
        catalogue_path = staging / CATALOGUE_NAME
        _write_json(catalogue_path, catalogue)
        output_paths.append(catalogue_path)
        manifest: dict[str, object] = {
            "protocol": PROTOCOL,
            "mode": "cross_source_discovery_only_raw_x0_dobvr_feature_freeze",
            "graph_mode": GRAPH_MODE,
            "workers": workers,
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
            raise RuntimeError("NEXT102 input changed before publication")
        if any(_sha256_file(path) != source_hashes[name] for name, path in source_paths.items()):
            raise RuntimeError("NEXT102 source changed before publication")
        os.replace(staging, target)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


__all__ = [
    "FEATURE_COLUMNS",
    "FEATURE_NAMES",
    "MANIFEST_NAME",
    "PROTOCOL",
    "STATUS_COLUMNS",
    "build_cross_source_discovery_dobvr_features",
    "compute_dobvr_feature_row",
]
