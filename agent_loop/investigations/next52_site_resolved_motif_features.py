#!/usr/bin/env python3
"""Metal- and donor-site CrystalNN coherence descriptors from one raw x0."""

from __future__ import annotations

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
import warnings

from ase import Atoms
import numpy as np
import pandas as pd
from pymatgen.core import Element
from pymatgen.io.ase import AseAtomsAdaptor

from src.next11_geometry_only_frames import _load_archive_only
from src.next19_feature_build import _publish_directory_no_replace, _sha256
from src.next46_motif_coherence_features import (
    _COORDINATION,
    _FEATURIZER,
    _LABELS,
    _ORDER_INDICES,
    _WEIGHT_INDICES,
)
from src.next49_framework_topology import _environment_versions
from src.next50_framework_motif_features import (
    COMBINED_FEATURE_NAMES,
    FEATURES_NAME as NEXT50_FEATURES_NAME,
    PROTOCOL as NEXT50_PROTOCOL,
)


PROTOCOL = "2026-08-03-next52-site-resolved-framework-motif-features-v1"
FEATURES_NAME = "next52_qmof_site_resolved_motif_features.parquet"
MANIFEST_NAME = "MANIFEST.json"
DONOR_NUMBERS = frozenset((7, 8, 9, 15, 16, 17, 35, 53))
SITE_MOTIF_FEATURE_NAMES = (
    "metal_motif_site_fraction",
    "metal_motif_weight_sum_mean",
    "metal_motif_weight_sum_min",
    "metal_motif_weight_sum_std",
    "metal_motif_cn_dominance_mean",
    "metal_motif_cn_dominance_min",
    "metal_motif_cn_dominance_std",
    "metal_motif_cn_entropy_mean",
    "metal_motif_cn_entropy_q95",
    "metal_motif_effective_cn_mean",
    "metal_motif_effective_cn_std",
    "metal_motif_effective_cn_range",
    "metal_motif_order_strength_mean",
    "metal_motif_order_strength_min",
    "metal_motif_order_strength_std",
    "metal_motif_fingerprint_norm_mean",
    "metal_motif_fingerprint_norm_std",
    "metal_motif_dispersion_rms",
    "metal_motif_dispersion_q95",
    "donor_motif_site_fraction",
    "donor_motif_cn_dominance_mean",
    "donor_motif_cn_dominance_min",
    "donor_motif_cn_entropy_mean",
    "donor_motif_cn_entropy_q95",
    "donor_motif_effective_cn_mean",
    "donor_motif_effective_cn_std",
    "donor_motif_order_strength_mean",
    "donor_motif_order_strength_min",
    "donor_motif_fingerprint_norm_mean",
    "metal_donor_centroid_separation",
)
ALL_FEATURE_NAMES = tuple(COMBINED_FEATURE_NAMES) + SITE_MOTIF_FEATURE_NAMES


@dataclass(frozen=True)
class SiteResolvedMotifResult:
    supported: bool
    failure_reason: str | None
    features: Mapping[str, float]


def _fingerprint_matrix(atoms: Atoms) -> np.ndarray:
    structure = AseAtomsAdaptor.get_structure(atoms)
    rows: list[np.ndarray] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for index in range(len(structure)):
            rows.append(np.asarray(_FEATURIZER.featurize(structure, index), dtype=float))
    matrix = np.vstack(rows)
    if matrix.shape != (len(atoms), len(_LABELS)) or not np.isfinite(matrix).all():
        raise ValueError("CrystalNN ops fingerprint is non-finite or changed shape")
    return matrix


def _site_quantities(matrix: np.ndarray) -> dict[str, np.ndarray]:
    weights = np.clip(matrix[:, _WEIGHT_INDICES], 0.0, None)
    weight_sum = np.sum(weights, axis=1)
    normalized = np.zeros_like(weights)
    positive = weight_sum > 1.0e-15
    normalized[positive] = weights[positive] / weight_sum[positive, None]
    dominance = np.max(normalized, axis=1)
    selected = normalized > 0.0
    log_values = np.zeros_like(normalized)
    log_values[selected] = normalized[selected] * np.log(normalized[selected])
    entropy = -np.sum(log_values, axis=1) / math.log(len(_WEIGHT_INDICES))
    return {
        "weight_sum": weight_sum,
        "dominance": dominance,
        "entropy": entropy,
        "effective_cn": normalized @ _COORDINATION,
        "order_strength": np.max(
            np.clip(matrix[:, _ORDER_INDICES], 0.0, None), axis=1
        ),
        "norm": np.linalg.norm(matrix, axis=1),
    }


def compute_site_resolved_motif_features(atoms: Atoms) -> SiteResolvedMotifResult:
    """Aggregate CrystalNN coherence separately over metal and donor atoms."""

    try:
        if (
            len(atoms) < 1
            or not np.all(atoms.pbc)
            or atoms.calc is not None
            or bool(atoms.info)
            or set(atoms.arrays) != {"numbers", "positions"}
        ):
            raise ValueError("site-resolved motifs require exact geometry-only Atoms")
        matrix = _fingerprint_matrix(atoms)
        numbers = np.asarray(atoms.numbers, dtype=int)
        metal = np.asarray(
            [bool(Element.from_Z(int(number)).is_metal) for number in numbers],
            dtype=bool,
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
        donor_centroid = np.mean(matrix[donor], axis=0) if donor.any() else np.zeros(matrix.shape[1])
        feature_values = {
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
            "metal_motif_effective_cn_range": float(
                np.ptp(values(metal, "effective_cn"))
            ),
            "metal_motif_order_strength_mean": float(
                np.mean(values(metal, "order_strength"))
            ),
            "metal_motif_order_strength_min": float(
                np.min(values(metal, "order_strength"))
            ),
            "metal_motif_order_strength_std": float(
                np.std(values(metal, "order_strength"))
            ),
            "metal_motif_fingerprint_norm_mean": float(np.mean(values(metal, "norm"))),
            "metal_motif_fingerprint_norm_std": float(np.std(values(metal, "norm"))),
            "metal_motif_dispersion_rms": float(
                np.sqrt(np.mean(metal_residual**2))
            ),
            "metal_motif_dispersion_q95": float(np.quantile(metal_residual, 0.95)),
            "donor_motif_site_fraction": float(np.mean(donor)),
            "donor_motif_cn_dominance_mean": float(np.mean(values(donor, "dominance"))),
            "donor_motif_cn_dominance_min": float(np.min(values(donor, "dominance"))),
            "donor_motif_cn_entropy_mean": float(np.mean(values(donor, "entropy"))),
            "donor_motif_cn_entropy_q95": float(np.quantile(values(donor, "entropy"), 0.95)),
            "donor_motif_effective_cn_mean": float(np.mean(values(donor, "effective_cn"))),
            "donor_motif_effective_cn_std": float(np.std(values(donor, "effective_cn"))),
            "donor_motif_order_strength_mean": float(
                np.mean(values(donor, "order_strength"))
            ),
            "donor_motif_order_strength_min": float(
                np.min(values(donor, "order_strength"))
            ),
            "donor_motif_fingerprint_norm_mean": float(np.mean(values(donor, "norm"))),
            "metal_donor_centroid_separation": float(
                np.linalg.norm(metal_centroid - donor_centroid)
            ),
        }
        if tuple(feature_values) != SITE_MOTIF_FEATURE_NAMES or not np.isfinite(
            list(feature_values.values())
        ).all():
            raise ValueError("site-resolved motif feature schema differs")
        return SiteResolvedMotifResult(True, None, feature_values)
    except Exception as exc:
        return SiteResolvedMotifResult(False, f"{type(exc).__name__}: {exc}", {})


def _site_record(atoms: Atoms) -> dict[str, object]:
    result = compute_site_resolved_motif_features(atoms)
    row: dict[str, object] = {
        "site_motif_supported": result.supported,
        "site_motif_failure": result.failure_reason,
    }
    row.update(
        {
            name: float(result.features[name]) if result.supported else math.nan
            for name in SITE_MOTIF_FEATURE_NAMES
        }
    )
    return row


def _strict_json(path: Path, *, role: str) -> dict[str, object]:
    try:
        value = json.loads(Path(path).read_text("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {role}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{role} must be a JSON object")
    return value


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def build_qmof_site_resolved_motif_batch(
    *,
    geometry_path: Path,
    next50_features_path: Path,
    next50_manifest_path: Path,
    output_dir: Path,
    workers: int = 1,
) -> dict[str, object]:
    """Seal site-resolved descriptors without accessing a QMOF endpoint."""

    if type(workers) is not int or not 1 <= workers <= 64:
        raise ValueError("NEXT52 workers must be an integer from 1 through 64")
    target = Path(output_dir).resolve()
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    paths = {
        "geometry": Path(geometry_path).resolve(),
        "next50_features": Path(next50_features_path).resolve(),
        "next50_manifest": Path(next50_manifest_path).resolve(),
    }
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT52 input is missing")
    hashes = {name: _sha256(path) for name, path in paths.items()}
    manifest50 = _strict_json(paths["next50_manifest"], role="NEXT50 manifest")
    outputs50 = manifest50.get("outputs_sha256")
    inputs50 = manifest50.get("inputs_sha256")
    geometry50 = inputs50.get("geometry") if isinstance(inputs50, Mapping) else None
    if (
        manifest50.get("protocol") != NEXT50_PROTOCOL
        or manifest50.get("labels_opened") is not False
        or manifest50.get("relaxed_coordinate_payloads_opened") is not False
        or manifest50.get("model_or_proxy_potential_used") is not False
        or not isinstance(outputs50, Mapping)
        or outputs50.get(NEXT50_FEATURES_NAME) != hashes["next50_features"]
        or not isinstance(geometry50, Mapping)
        or geometry50.get("sha256") != hashes["geometry"]
    ):
        raise ValueError("NEXT52 input crossed the x0-only boundary")
    base = pd.read_parquet(paths["next50_features"])
    if (
        base.empty
        or not {"material_id", "source_family", *COMBINED_FEATURE_NAMES}.issubset(base)
        or base["material_id"].isna().any()
        or base["material_id"].duplicated().any()
    ):
        raise ValueError("NEXT52 base feature table differs")
    material_ids = base["material_id"].astype(str).tolist()
    archive_ids, frames = _load_archive_only(paths["geometry"], tuple(material_ids))
    if archive_ids != material_ids:
        raise ValueError("NEXT52 geometry order differs")

    executor = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    iterator = (
        executor.map(_site_record, frames, chunksize=1)
        if executor is not None
        else map(_site_record, frames)
    )
    rows: list[dict[str, object]] = []
    failures: Counter[str] = Counter()
    try:
        for index, row in enumerate(iterator, start=1):
            rows.append(row)
            if not bool(row["site_motif_supported"]):
                failures[str(row["site_motif_failure"] or "unsupported")] += 1
            if index % 50 == 0 or index == len(frames):
                print(f"NEXT52 QMOF site motifs: {index}/{len(frames)}", flush=True)
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    table = pd.concat([base.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    table["extended_supported"] = (
        table["combined_supported"].astype(bool)
        & table["site_motif_supported"].astype(bool)
        & np.isfinite(table.loc[:, ALL_FEATURE_NAMES]).all(axis=1)
    )

    source_path = Path(__file__).resolve()
    source_hash = _sha256(source_path)
    manifest: dict[str, object] = {
        "protocol": PROTOCOL,
        "mode": "exposed_qmof_label_free_site_resolved_motif_build",
        "input_role": "unrelaxed_x0_geometry_only",
        "labels_opened": False,
        "relaxed_coordinate_payloads_opened": False,
        "endpoint_columns_selected": False,
        "model_or_proxy_potential_used": False,
        "dft_or_energy_proxy_used_at_execution": False,
        "physical_relaxation_executed": False,
        "same_composition_candidates_used": False,
        "feature_columns": list(ALL_FEATURE_NAMES),
        "worker_processes": workers,
        "counts": {
            "rows": len(table),
            "site_motif_supported": int(table["site_motif_supported"].sum()),
            "extended_supported": int(table["extended_supported"].sum()),
            "failure_counts": dict(sorted(failures.items())),
        },
        "environment_versions": {
            **_environment_versions(),
            "matminer": importlib.metadata.version("matminer"),
        },
        "inputs_sha256": {
            name: {"path": str(path), "sha256": hashes[name]}
            for name, path in paths.items()
        },
        "executed_source_sha256": {
            "src/next52_site_resolved_motif_features.py": source_hash
        },
        "scientific_improvement_claim": False,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        output_path = staging / FEATURES_NAME
        table.to_parquet(output_path, index=False)
        manifest["outputs_sha256"] = {FEATURES_NAME: _sha256(output_path)}
        (staging / MANIFEST_NAME).write_bytes(_json_bytes(manifest))
        for name, path in paths.items():
            if _sha256(path) != hashes[name]:
                raise RuntimeError(f"NEXT52 input {name} changed before publication")
        if _sha256(source_path) != source_hash:
            raise RuntimeError("NEXT52 source changed before publication")
        _publish_directory_no_replace(staging, target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


__all__ = [
    "ALL_FEATURE_NAMES",
    "FEATURES_NAME",
    "MANIFEST_NAME",
    "PROTOCOL",
    "SITE_MOTIF_FEATURE_NAMES",
    "SiteResolvedMotifResult",
    "build_qmof_site_resolved_motif_batch",
    "compute_site_resolved_motif_features",
]
