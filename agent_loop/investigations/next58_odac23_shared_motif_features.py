#!/usr/bin/env python3
"""One-pass global and metal/donor CrystalNN features for selected ODAC23 x0."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

from ase import Atoms
import numpy as np
import pandas as pd
from pymatgen.core import Element

from src.next19_feature_build import _publish_directory_no_replace, _sha256
from src.next46_motif_coherence_features import FEATURE_NAMES as GLOBAL_MOTIF_NAMES
from src.next49_framework_topology import _environment_versions
from src.next52_site_resolved_motif_features import (
    DONOR_NUMBERS,
    SITE_MOTIF_FEATURE_NAMES,
    _fingerprint_matrix,
    _site_quantities,
)
from src.next54_odac23_train_selection import (
    GEOMETRY_NAME as SOURCE_GEOMETRY_NAME,
    MANIFEST_NAME as SELECTION_MANIFEST_NAME,
    PROTOCOL as SELECTION_PROTOCOL,
)
from src.next55_odac23_analytic_features import (
    FEATURES_NAME as ANALYTIC_FEATURES_NAME,
    MANIFEST_NAME as ANALYTIC_MANIFEST_NAME,
    NEXT55_FEATURE_NAMES,
    PROTOCOL as ANALYTIC_PROTOCOL,
    _load_archive,
)


PROTOCOL = "2026-08-03-next58-odac23-shared-local-motif-features-v1"
DESIGN_SHA256 = "38a60a64caa827ac579fd54410515170d37557a66e62b417e583a93724557639"
EXPECTED_SELECTION_MANIFEST_SHA256 = (
    "9ea1f0e6c04c8619dd295aa1579da15b51d8241971b3adacb716fdbf93290927"
)
EXPECTED_ANALYTIC_MANIFEST_SHA256 = (
    "bc4a05b70c0b4d84723c83e4a79e2f20ffce7f55f9ddf3a32cb8b5f3bcef6f1e"
)
FEATURES_NAME = "next58_odac23_shared_motif_features.parquet"
MANIFEST_NAME = "MANIFEST.json"
EXTRA_MOTIF_FEATURE_NAMES = (
    "metal_motif_cn_dominance_q10",
    "donor_motif_cn_dominance_q10",
    "metal_motif_order_strength_q10",
    "donor_motif_order_strength_q10",
    "metal_motif_cn_entropy_max",
    "donor_motif_cn_entropy_max",
    "metal_motif_effective_cn_min",
    "metal_motif_effective_cn_max",
    "donor_motif_effective_cn_min",
    "donor_motif_effective_cn_max",
    "donor_motif_dispersion_rms",
    "donor_motif_dispersion_q95",
    "metal_motif_low_clarity_fraction",
    "donor_motif_low_clarity_fraction",
)
SHARED_MOTIF_FEATURE_NAMES = (
    tuple(GLOBAL_MOTIF_NAMES)
    + tuple(SITE_MOTIF_FEATURE_NAMES)
    + EXTRA_MOTIF_FEATURE_NAMES
)
NEXT58_FEATURE_NAMES = tuple(NEXT55_FEATURE_NAMES) + SHARED_MOTIF_FEATURE_NAMES


@dataclass(frozen=True)
class SharedMotifResult:
    supported: bool
    failure_reason: str | None
    features: Mapping[str, float]


def _global_values(matrix: np.ndarray, numbers: np.ndarray) -> dict[str, float]:
    quantities = _site_quantities(matrix)
    same_residuals = []
    species_centroids = []
    for number in sorted(set(numbers.tolist())):
        group = matrix[numbers == number]
        centroid = np.mean(group, axis=0)
        species_centroids.append(centroid)
        same_residuals.extend(np.linalg.norm(group - centroid, axis=1).tolist())
    same = np.asarray(same_residuals, dtype=float)
    global_centroid = np.mean(matrix, axis=0)
    global_residual = np.linalg.norm(matrix - global_centroid, axis=1)
    separations = [
        float(np.linalg.norm(species_centroids[left] - species_centroids[right]))
        for left in range(len(species_centroids))
        for right in range(left + 1, len(species_centroids))
    ]
    weight_sum = quantities["weight_sum"]
    dominance = quantities["dominance"]
    entropy = quantities["entropy"]
    effective_cn = quantities["effective_cn"]
    order_strength = quantities["order_strength"]
    norms = quantities["norm"]
    return {
        "motif_weight_sum_mean": float(np.mean(weight_sum)),
        "motif_weight_sum_min": float(np.min(weight_sum)),
        "motif_weight_sum_std": float(np.std(weight_sum)),
        "motif_cn_dominance_mean": float(np.mean(dominance)),
        "motif_cn_dominance_min": float(np.min(dominance)),
        "motif_cn_dominance_std": float(np.std(dominance)),
        "motif_cn_entropy_mean": float(np.mean(entropy)),
        "motif_cn_entropy_q95": float(np.quantile(entropy, 0.95)),
        "motif_effective_cn_mean": float(np.mean(effective_cn)),
        "motif_effective_cn_std": float(np.std(effective_cn)),
        "motif_effective_cn_range": float(np.ptp(effective_cn)),
        "motif_order_strength_mean": float(np.mean(order_strength)),
        "motif_order_strength_min": float(np.min(order_strength)),
        "motif_order_strength_std": float(np.std(order_strength)),
        "motif_fingerprint_norm_mean": float(np.mean(norms)),
        "motif_fingerprint_norm_std": float(np.std(norms)),
        "motif_same_element_dispersion_rms": float(np.sqrt(np.mean(same**2))),
        "motif_same_element_dispersion_q95": float(np.quantile(same, 0.95)),
        "motif_same_element_dispersion_max": float(np.max(same)),
        "motif_global_dispersion_rms": float(np.sqrt(np.mean(global_residual**2))),
        "motif_species_centroid_separation_mean": float(
            np.mean(separations) if separations else 0.0
        ),
    }


def _site_values(
    matrix: np.ndarray, numbers: np.ndarray
) -> tuple[dict[str, float], dict[str, float]]:
    metal = np.asarray(
        [bool(Element.from_Z(int(number)).is_metal) for number in numbers], dtype=bool
    )
    donor = np.asarray([int(number) in DONOR_NUMBERS for number in numbers], dtype=bool)
    if not metal.any():
        raise ValueError("periodic framework has no recognized metal site")
    quantities = _site_quantities(matrix)

    def values(mask: np.ndarray, name: str, fallback: float = 0.0) -> np.ndarray:
        selected = quantities[name][mask]
        return selected if len(selected) else np.asarray([fallback], dtype=float)

    metal_matrix = matrix[metal]
    metal_centroid = np.mean(metal_matrix, axis=0)
    metal_residual = np.linalg.norm(metal_matrix - metal_centroid, axis=1)
    donor_matrix = matrix[donor]
    donor_centroid = np.mean(donor_matrix, axis=0) if donor.any() else np.zeros(matrix.shape[1])
    donor_residual = (
        np.linalg.norm(donor_matrix - donor_centroid, axis=1)
        if donor.any()
        else np.asarray([0.0])
    )
    site = {
        "metal_motif_site_fraction": float(np.mean(metal)),
        "metal_motif_weight_sum_mean": float(np.mean(values(metal, "weight_sum"))),
        "metal_motif_weight_sum_min": float(np.min(values(metal, "weight_sum"))),
        "metal_motif_weight_sum_std": float(np.std(values(metal, "weight_sum"))),
        "metal_motif_cn_dominance_mean": float(np.mean(values(metal, "dominance"))),
        "metal_motif_cn_dominance_min": float(np.min(values(metal, "dominance"))),
        "metal_motif_cn_dominance_std": float(np.std(values(metal, "dominance"))),
        "metal_motif_cn_entropy_mean": float(np.mean(values(metal, "entropy"))),
        "metal_motif_cn_entropy_q95": float(np.quantile(values(metal, "entropy"), 0.95)),
        "metal_motif_effective_cn_mean": float(np.mean(values(metal, "effective_cn"))),
        "metal_motif_effective_cn_std": float(np.std(values(metal, "effective_cn"))),
        "metal_motif_effective_cn_range": float(np.ptp(values(metal, "effective_cn"))),
        "metal_motif_order_strength_mean": float(np.mean(values(metal, "order_strength"))),
        "metal_motif_order_strength_min": float(np.min(values(metal, "order_strength"))),
        "metal_motif_order_strength_std": float(np.std(values(metal, "order_strength"))),
        "metal_motif_fingerprint_norm_mean": float(np.mean(values(metal, "norm"))),
        "metal_motif_fingerprint_norm_std": float(np.std(values(metal, "norm"))),
        "metal_motif_dispersion_rms": float(np.sqrt(np.mean(metal_residual**2))),
        "metal_motif_dispersion_q95": float(np.quantile(metal_residual, 0.95)),
        "donor_motif_site_fraction": float(np.mean(donor)),
        "donor_motif_cn_dominance_mean": float(np.mean(values(donor, "dominance"))),
        "donor_motif_cn_dominance_min": float(np.min(values(donor, "dominance"))),
        "donor_motif_cn_entropy_mean": float(np.mean(values(donor, "entropy"))),
        "donor_motif_cn_entropy_q95": float(np.quantile(values(donor, "entropy"), 0.95)),
        "donor_motif_effective_cn_mean": float(np.mean(values(donor, "effective_cn"))),
        "donor_motif_effective_cn_std": float(np.std(values(donor, "effective_cn"))),
        "donor_motif_order_strength_mean": float(np.mean(values(donor, "order_strength"))),
        "donor_motif_order_strength_min": float(np.min(values(donor, "order_strength"))),
        "donor_motif_fingerprint_norm_mean": float(np.mean(values(donor, "norm"))),
        "metal_donor_centroid_separation": float(np.linalg.norm(metal_centroid - donor_centroid)),
    }
    extra = {
        "metal_motif_cn_dominance_q10": float(np.quantile(values(metal, "dominance"), 0.10)),
        "donor_motif_cn_dominance_q10": float(np.quantile(values(donor, "dominance"), 0.10)),
        "metal_motif_order_strength_q10": float(np.quantile(values(metal, "order_strength"), 0.10)),
        "donor_motif_order_strength_q10": float(np.quantile(values(donor, "order_strength"), 0.10)),
        "metal_motif_cn_entropy_max": float(np.max(values(metal, "entropy"))),
        "donor_motif_cn_entropy_max": float(np.max(values(donor, "entropy"))),
        "metal_motif_effective_cn_min": float(np.min(values(metal, "effective_cn"))),
        "metal_motif_effective_cn_max": float(np.max(values(metal, "effective_cn"))),
        "donor_motif_effective_cn_min": float(np.min(values(donor, "effective_cn"))),
        "donor_motif_effective_cn_max": float(np.max(values(donor, "effective_cn"))),
        "donor_motif_dispersion_rms": float(np.sqrt(np.mean(donor_residual**2))),
        "donor_motif_dispersion_q95": float(np.quantile(donor_residual, 0.95)),
        "metal_motif_low_clarity_fraction": float(np.mean(values(metal, "dominance") < 0.5)),
        "donor_motif_low_clarity_fraction": float(np.mean(values(donor, "dominance") < 0.5)),
    }
    return site, extra


def compute_shared_motif_features(atoms: Atoms) -> SharedMotifResult:
    """Compute global/site/tail features from a single CrystalNN matrix."""

    try:
        if (
            len(atoms) < 1
            or not np.all(atoms.pbc)
            or atoms.calc is not None
            or bool(atoms.info)
            or set(atoms.arrays) != {"numbers", "positions"}
        ):
            raise ValueError("shared motifs require exact geometry-only Atoms")
        matrix = _fingerprint_matrix(atoms)
        numbers = np.asarray(atoms.numbers, dtype=int)
        global_values = _global_values(matrix, numbers)
        site_values, extra_values = _site_values(matrix, numbers)
        result = {**global_values, **site_values, **extra_values}
        if tuple(result) != SHARED_MOTIF_FEATURE_NAMES or not np.isfinite(
            list(result.values())
        ).all():
            raise ValueError("NEXT58 shared motif schema differs")
        return SharedMotifResult(True, None, result)
    except Exception as exc:
        return SharedMotifResult(False, f"{type(exc).__name__}: {exc}", {})


def _feature_record(atoms: Atoms) -> dict[str, object]:
    result = compute_shared_motif_features(atoms)
    row: dict[str, object] = {
        "shared_motif_supported": result.supported,
        "shared_motif_failure": result.failure_reason,
    }
    row.update(
        {
            name: float(result.features[name]) if result.supported else math.nan
            for name in SHARED_MOTIF_FEATURE_NAMES
        }
    )
    return row


def _strict_json(path: Path, role: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {role}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{role} must be an object")
    return value


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def build_shared_motif_batch(
    *, selection_dir: Path, analytic_dir: Path, design_path: Path, output_dir: Path, workers: int = 1
) -> dict[str, object]:
    selection_dir = Path(selection_dir).resolve()
    analytic_dir = Path(analytic_dir).resolve()
    target = Path(output_dir).resolve()
    if type(workers) is not int or not 1 <= workers <= 64:
        raise ValueError("NEXT58 workers must be 1 through 64")
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    paths = {
        "geometry": selection_dir / SOURCE_GEOMETRY_NAME,
        "selection_manifest": selection_dir / SELECTION_MANIFEST_NAME,
        "analytic_features": analytic_dir / ANALYTIC_FEATURES_NAME,
        "analytic_manifest": analytic_dir / ANALYTIC_MANIFEST_NAME,
        "design": Path(design_path).resolve(),
    }
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT58 input is missing")
    hashes = {name: _sha256(path) for name, path in paths.items()}
    if (
        hashes["selection_manifest"] != EXPECTED_SELECTION_MANIFEST_SHA256
        or hashes["analytic_manifest"] != EXPECTED_ANALYTIC_MANIFEST_SHA256
        or hashes["design"] != DESIGN_SHA256
    ):
        raise ValueError("NEXT58 frozen input hash differs")
    selection_manifest = _strict_json(paths["selection_manifest"], "NEXT54 manifest")
    analytic_manifest = _strict_json(paths["analytic_manifest"], "NEXT55 manifest")
    selection_outputs = selection_manifest.get("outputs_sha256")
    analytic_outputs = analytic_manifest.get("outputs_sha256")
    if (
        selection_manifest.get("protocol") != SELECTION_PROTOCOL
        or not isinstance(selection_outputs, Mapping)
        or selection_outputs.get(paths["geometry"].name) != hashes["geometry"]
        or analytic_manifest.get("protocol") != ANALYTIC_PROTOCOL
        or analytic_manifest.get("labels_opened") is not False
        or not isinstance(analytic_outputs, Mapping)
        or analytic_outputs.get(paths["analytic_features"].name) != hashes["analytic_features"]
    ):
        raise ValueError("NEXT58 x0-only provenance differs")

    analytic = pd.read_parquet(paths["analytic_features"])
    if analytic.empty or analytic["material_id"].duplicated().any():
        raise ValueError("NEXT58 analytic feature identity differs")
    material_ids = tuple(analytic["material_id"].astype(str))
    structures = _load_archive(paths["geometry"], material_ids)
    rows = []
    failures: Counter[str] = Counter()
    if workers == 1:
        iterator = map(_feature_record, structures)
        executor = None
    else:
        executor = ProcessPoolExecutor(max_workers=workers)
        iterator = executor.map(_feature_record, structures, chunksize=1)
    try:
        for index, row in enumerate(iterator, start=1):
            rows.append(row)
            if not bool(row["shared_motif_supported"]):
                failures[str(row["shared_motif_failure"] or "unsupported")] += 1
            if index % 25 == 0 or index == len(structures):
                print(f"NEXT58 ODAC23 shared motifs: {index}/{len(structures)}", flush=True)
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    table = pd.concat([analytic.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    table["next58_supported"] = (
        table["combined_supported"].astype(bool)
        & table["shared_motif_supported"].astype(bool)
        & np.isfinite(table.loc[:, NEXT58_FEATURE_NAMES]).all(axis=1)
    )

    source_path = Path(__file__).resolve()
    matminer_source = Path(
        __import__("matminer.featurizers.site.fingerprint", fromlist=["x"]).__file__
    ).resolve()
    manifest: dict[str, object] = {
        "protocol": PROTOCOL,
        "mode": "selected_odac23_train_label_free_shared_crystalnn_motifs",
        "input_role": "one_raw_unrelaxed_framework_x0_geometry_only",
        "labels_opened": False,
        "relaxed_coordinate_payloads_opened": False,
        "endpoint_columns_selected": False,
        "model_or_proxy_potential_used": False,
        "dft_or_energy_proxy_used_at_execution": False,
        "physical_relaxation_executed": False,
        "same_composition_candidates_used": False,
        "feature_columns": list(NEXT58_FEATURE_NAMES),
        "shared_motif_feature_columns": list(SHARED_MOTIF_FEATURE_NAMES),
        "worker_processes": workers,
        "counts": {
            "rows": len(table),
            "shared_motif_supported": int(table["shared_motif_supported"].sum()),
            "next58_supported": int(table["next58_supported"].sum()),
            "failures": dict(sorted(failures.items())),
        },
        "inputs_sha256": {
            name: {"path": str(path), "sha256": hashes[name]}
            for name, path in paths.items()
        },
        "executed_source_sha256": {
            "src/next58_odac23_shared_motif_features.py": _sha256(source_path),
            str(matminer_source): _sha256(matminer_source),
        },
        "environment_versions": {
            **_environment_versions(),
            "matminer": importlib.metadata.version("matminer"),
        },
        "scientific_improvement_claim": False,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        feature_path = staging / FEATURES_NAME
        table.to_parquet(feature_path, index=False)
        manifest["outputs_sha256"] = {FEATURES_NAME: _sha256(feature_path)}
        (staging / MANIFEST_NAME).write_bytes(_json_bytes(manifest))
        if _sha256(source_path) != manifest["executed_source_sha256"][
            "src/next58_odac23_shared_motif_features.py"
        ]:
            raise RuntimeError("NEXT58 source changed before publication")
        if any(_sha256(path) != hashes[name] for name, path in paths.items()):
            raise RuntimeError("NEXT58 input changed before publication")
        _publish_directory_no_replace(staging, target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-dir", type=Path, required=True)
    parser.add_argument("--analytic-dir", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = build_shared_motif_batch(
        selection_dir=args.selection_dir,
        analytic_dir=args.analytic_dir,
        design_path=args.design,
        output_dir=args.output_dir,
        workers=args.workers,
    )
    print(json.dumps(manifest["counts"], indent=2, sort_keys=True))


__all__ = [
    "EXTRA_MOTIF_FEATURE_NAMES",
    "FEATURES_NAME",
    "NEXT58_FEATURE_NAMES",
    "PROTOCOL",
    "SHARED_MOTIF_FEATURE_NAMES",
    "SharedMotifResult",
    "build_shared_motif_batch",
    "compute_shared_motif_features",
]


if __name__ == "__main__":
    main()
