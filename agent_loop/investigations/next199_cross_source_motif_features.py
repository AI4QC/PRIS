#!/usr/bin/env python3
"""Freeze cross-source motif coherence features from discovery x0 only."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Mapping, Sequence

from ase import Atoms
import numpy as np
import pandas as pd
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

import src.next46_motif_coherence_features as n46
import src.next85_scigen_label_free_features as n85
import src.next94_wyformer_label_free_features as n94
from src.next102_cross_source_dobvr_features import _sha256_file, _write_json


PROTOCOL = "2026-08-08-next199-cross-source-motif-features-v1"
MANIFEST_NAME = "MANIFEST.json"
CATALOGUE_NAME = "NEXT199_CROSS_SOURCE_MOTIF_CATALOGUE.json"
FEATURE_FILES = {
    "scigen": "next199_scigen_discovery_motif_features.parquet",
    "wyformer": "next199_wyformer_discovery_motif_features.parquet",
}
FEATURE_NAMES = n46.FEATURE_NAMES
EXPECTED_DESIGN_SHA256 = (
    "6d7d8a3cacc609c089ea60b8863683f852c72a9f843d3de6ca90d4dd0a7e4703"
)
EXPECTED_INPUT_SHA256 = {
    "scigen_manifest": "dc5bf33c6ce6dc2c10bcd3704688055058145fbe7269ada23ffbe4b141d75fe7",
    "scigen_metadata": "f91455f23b0a96f60fd1c779249e2be46a7ecf94fcdde2b146426a95aac05bde",
    "scigen_discovery_geometry": "e561ef12343c66dcc72bcabf6b8719ad727e01c9582a094e281da73b862ab575",
    "wyformer_manifest": "e0539d556538cb4c052431bc6a1e5c1663bc3de427677dbc8a446dcc3b4fbc54",
    "wyformer_metadata": "3b152b4b84c8d3f7ff5e85611dc1fd2728296f150e907ac4578ce55d2b27dd2b",
    "wyformer_discovery_geometry": "f1ce5ae4fba8c13fcbf3e25de4f596b919d9b41da5b072d9a28eefeaffc69784",
    "design": EXPECTED_DESIGN_SHA256,
}
EXPECTED_UPSTREAM_SOURCE_SHA256 = {
    "src/next46_motif_coherence_features.py": "d59b9882c8bf1b56ec6b0d45e971b83499a91db4328334f91dd0850cb8e8dbac",
    "src/next85_scigen_label_free_features.py": "2caf0fa0aafe6df6732c3b8ed02cd19d96076314273331f32a449b6bd3b41335",
    "src/next94_wyformer_label_free_features.py": "ccb04a9387b4fad9ea3b8e7e7cd54fb69965f98a3c44342c198a8511b17702a9",
}


def compute_motif_row(atoms: Atoms) -> dict[str, object]:
    """Return the exact NEXT46 schema with fail-closed status metadata."""

    result = n46.compute_motif_coherence_features(atoms)
    row: dict[str, object] = {
        name: math.nan for name in FEATURE_NAMES
    }
    row["motif_supported"] = bool(result.supported)
    row["motif_failure"] = result.failure_reason
    if result.supported:
        if tuple(result.features) != FEATURE_NAMES:
            raise ValueError("NEXT199 motif feature schema differs")
        for name in FEATURE_NAMES:
            value = float(result.features[name])
            row[name] = value if math.isfinite(value) else math.nan
        if not np.isfinite([row[name] for name in FEATURE_NAMES]).all():
            row.update({name: math.nan for name in FEATURE_NAMES})
            row["motif_supported"] = False
            row["motif_failure"] = "nonfinite_motif_feature"
    return row


def _error_row(exc: Exception) -> dict[str, object]:
    return {
        **{name: math.nan for name in FEATURE_NAMES},
        "motif_supported": False,
        "motif_failure": f"{type(exc).__name__}: {exc}",
    }


def _compute_scigen_payload(
    item: tuple[str, bytes]
) -> tuple[str, dict[str, object]]:
    material_id, payload = item
    try:
        parsed = n85._parse_frame(payload, strict_output=True)
        return material_id, compute_motif_row(parsed.atoms)
    except Exception as exc:
        return material_id, _error_row(exc)


def _compute_wyformer_payload(
    item: tuple[str, str]
) -> tuple[str, dict[str, object]]:
    material_id, payload = item
    try:
        structure = Structure.from_dict(json.loads(payload))
        atoms = AseAtomsAdaptor.get_atoms(structure)
        return material_id, compute_motif_row(atoms)
    except Exception as exc:
        return material_id, _error_row(exc)


def _compute_scigen_many(
    payloads: Sequence[tuple[str, bytes]], *, workers: int
) -> list[tuple[str, dict[str, object]]]:
    if workers == 1:
        return [_compute_scigen_payload(item) for item in payloads]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(_compute_scigen_payload, payloads, chunksize=8))


def _compute_wyformer_many(
    payloads: Sequence[tuple[str, str]], *, workers: int
) -> list[tuple[str, dict[str, object]]]:
    if workers == 1:
        return [_compute_wyformer_payload(item) for item in payloads]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(_compute_wyformer_payload, payloads, chunksize=8))


def _read_manifest(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} must contain an object")
    return payload


def build_cross_source_motif_features(
    *,
    scigen_cohort_dir: Path,
    wyformer_cohort_dir: Path,
    design_path: Path,
    output_dir: Path,
    workers: int = 16,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    """Build discovery-only motif features without any endpoint path."""

    scigen = Path(scigen_cohort_dir).resolve()
    wyformer = Path(wyformer_cohort_dir).resolve()
    target = Path(output_dir).resolve()
    paths = {
        "scigen_manifest": scigen / n85.COHORT_MANIFEST_NAME,
        "scigen_metadata": scigen / n85.COHORT_METADATA_NAME,
        "scigen_discovery_geometry": scigen / n85.GEOMETRY_NAMES["discovery"],
        "wyformer_manifest": wyformer / n94.COHORT_MANIFEST_NAME,
        "wyformer_metadata": wyformer / n94.COHORT_METADATA_NAME,
        "wyformer_discovery_geometry": wyformer / n94.GEOMETRY_NAMES["discovery"],
        "design": Path(design_path).resolve(),
    }
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if type(workers) is not int or workers <= 0:
        raise ValueError("workers must be a positive exact integer")
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT199 input is missing")
    input_hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if require_formal_inputs and input_hashes != EXPECTED_INPUT_SHA256:
        differing = sorted(
            name
            for name in set(input_hashes) | set(EXPECTED_INPUT_SHA256)
            if input_hashes.get(name) != EXPECTED_INPUT_SHA256.get(name)
        )
        raise ValueError(f"NEXT199 formal input identity differs: {differing}")

    scigen_manifest = _read_manifest(paths["scigen_manifest"])
    scigen_outputs = scigen_manifest.get("outputs_sha256", {})
    if (
        scigen_manifest.get("protocol") != n85.COHORT_PROTOCOL
        or scigen_manifest.get("labels_opened") is not False
        or scigen_manifest.get("endpoint_payloads_opened") is not False
        or scigen_manifest.get("relaxed_structures_opened") is not False
        or not isinstance(scigen_outputs, Mapping)
        or scigen_outputs.get(n85.COHORT_METADATA_NAME)
        != input_hashes["scigen_metadata"]
        or scigen_outputs.get(n85.GEOMETRY_NAMES["discovery"])
        != input_hashes["scigen_discovery_geometry"]
    ):
        raise ValueError("NEXT199 NEXT84 provenance differs")
    wyformer_manifest = _read_manifest(paths["wyformer_manifest"])
    wyformer_outputs = wyformer_manifest.get("outputs_sha256", {})
    if (
        wyformer_manifest.get("protocol") != n94.COHORT_PROTOCOL
        or wyformer_manifest.get("discovery_endpoint_opened") is not False
        or wyformer_manifest.get("validation_endpoint_opened") is not False
        or wyformer_manifest.get("replication_endpoint_opened") is not False
        or wyformer_manifest.get("relaxed_structures_published") is not False
        or not isinstance(wyformer_outputs, Mapping)
        or wyformer_outputs.get(n94.COHORT_METADATA_NAME)
        != input_hashes["wyformer_metadata"]
        or wyformer_outputs.get(n94.GEOMETRY_NAMES["discovery"])
        != input_hashes["wyformer_discovery_geometry"]
    ):
        raise ValueError("NEXT199 NEXT93b provenance differs")

    repository = Path(__file__).resolve().parents[1]
    upstream_hashes = {
        name: _sha256_file(repository / name)
        for name in EXPECTED_UPSTREAM_SOURCE_SHA256
    }
    if require_formal_inputs and upstream_hashes != EXPECTED_UPSTREAM_SOURCE_SHA256:
        raise ValueError("NEXT199 frozen upstream source differs")

    metadata = {
        "scigen": pd.read_parquet(paths["scigen_metadata"]),
        "wyformer": pd.read_parquet(paths["wyformer_metadata"]),
    }
    discovery_metadata = {}
    for source, frame in metadata.items():
        required = {
            "material_id",
            "reduced_formula",
            "chemical_system",
            "natoms",
            "partition_role",
            "input_role",
        }
        if required - set(frame.columns) or frame["material_id"].astype(str).duplicated().any():
            raise ValueError(f"NEXT199 {source} metadata differs")
        selected = frame.loc[frame["partition_role"].eq("discovery")].copy()
        selected = selected.sort_values("material_id", kind="stable", ignore_index=True)
        if selected.empty or not selected["partition_role"].eq("discovery").all():
            raise ValueError(f"NEXT199 {source} discovery identity differs")
        discovery_metadata[source] = selected

    scigen_ids = discovery_metadata["scigen"]["material_id"].astype(str).tolist()
    wyformer_ids = discovery_metadata["wyformer"]["material_id"].astype(str).tolist()
    scigen_payloads = n85._archive_payloads(
        paths["scigen_discovery_geometry"], scigen_ids
    )
    wyformer_payloads = n94._payloads(
        paths["wyformer_discovery_geometry"], wyformer_ids
    )

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent)
    )
    started = time.perf_counter()
    source_path = Path(__file__).resolve()
    source_hash = _sha256_file(source_path)
    try:
        computed = {
            "scigen": _compute_scigen_many(scigen_payloads, workers=workers),
            "wyformer": _compute_wyformer_many(wyformer_payloads, workers=workers),
        }
        output_paths: list[Path] = []
        counts = {}
        for source in ("scigen", "wyformer"):
            computed_frame = pd.DataFrame(
                [
                    {"material_id": material_id, **row}
                    for material_id, row in computed[source]
                ]
            )
            table = discovery_metadata[source].merge(
                computed_frame, on="material_id", how="left", validate="one_to_one"
            )
            if len(table) != len(discovery_metadata[source]):
                raise RuntimeError(f"NEXT199 {source} row accounting differs")
            output = staging / FEATURE_FILES[source]
            table.to_parquet(output, index=False)
            output_paths.append(output)
            counts[source] = {
                "rows": int(len(table)),
                "supported": int(table["motif_supported"].fillna(False).sum()),
                "failures": int((~table["motif_supported"].fillna(False)).sum()),
                "finite_feature_counts": {
                    name: int(
                        np.isfinite(pd.to_numeric(table[name], errors="coerce")).sum()
                    )
                    for name in FEATURE_NAMES
                },
            }
        catalogue = {
            "protocol": PROTOCOL,
            "feature_names": list(FEATURE_NAMES),
            "feature_count": len(FEATURE_NAMES),
            "source_partitions_read": {"scigen": ["discovery"], "wyformer": ["discovery"]},
            "endpoint_columns_present": False,
            "labels_opened": False,
        }
        catalogue_path = staging / CATALOGUE_NAME
        _write_json(catalogue_path, catalogue)
        output_paths.append(catalogue_path)
        manifest = {
            "protocol": PROTOCOL,
            "mode": "physically_isolated_discovery_x0_motif_feature_freeze",
            "workers": workers,
            "elapsed_seconds": time.perf_counter() - started,
            "counts": counts,
            "source_partitions_read": {"scigen": ["discovery"], "wyformer": ["discovery"]},
            "internal_validation_geometry_opened": False,
            "internal_replication_geometry_opened": False,
            "discovery_endpoints_opened": False,
            "validation_endpoints_opened": False,
            "replication_endpoints_opened": False,
            "labels_opened": False,
            "dft_calculation_executed": False,
            "dft_values_used_by_features": False,
            "learned_energy_force_stress_proxy_used": False,
            "model_or_proxy_potential_used": False,
            "physical_relaxation_executed": False,
            "inputs_sha256": input_hashes,
            "upstream_source_sha256": upstream_hashes,
            "executed_source_sha256": {
                "src/next199_cross_source_motif_features.py": source_hash
            },
            "outputs_sha256": {path.name: _sha256_file(path) for path in output_paths},
            "scientific_improvement_claim": False,
        }
        _write_json(staging / MANIFEST_NAME, manifest)
        if any(_sha256_file(path) != input_hashes[name] for name, path in paths.items()):
            raise RuntimeError("NEXT199 input changed before publication")
        if _sha256_file(source_path) != source_hash:
            raise RuntimeError("NEXT199 source changed before publication")
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
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--allow-nonformal-inputs", action="store_true")
    args = parser.parse_args()
    manifest = build_cross_source_motif_features(
        scigen_cohort_dir=args.scigen_cohort_dir,
        wyformer_cohort_dir=args.wyformer_cohort_dir,
        design_path=args.design_path,
        output_dir=args.output_dir,
        workers=args.workers,
        require_formal_inputs=not args.allow_nonformal_inputs,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


__all__ = [
    "FEATURE_NAMES",
    "build_cross_source_motif_features",
    "compute_motif_row",
]


if __name__ == "__main__":
    raise SystemExit(main())
