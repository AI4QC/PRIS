#!/usr/bin/env python3
"""Deterministic local-motif coherence descriptors from one raw crystal x0."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
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
from matminer.featurizers.site import CrystalNNFingerprint
from pymatgen.io.ase import AseAtomsAdaptor

from src.next19_feature_build import _publish_directory_no_replace, _sha256
from src.next43_analytic_feature_bank import _validated_inputs


PROTOCOL = "2026-08-03-next46-local-motif-coherence-features-v1"
FEATURE_NAME = "next46_motif_coherence_features.parquet"
MANIFEST_NAME = "MANIFEST.json"
FEATURE_NAMES = (
    "motif_weight_sum_mean",
    "motif_weight_sum_min",
    "motif_weight_sum_std",
    "motif_cn_dominance_mean",
    "motif_cn_dominance_min",
    "motif_cn_dominance_std",
    "motif_cn_entropy_mean",
    "motif_cn_entropy_q95",
    "motif_effective_cn_mean",
    "motif_effective_cn_std",
    "motif_effective_cn_range",
    "motif_order_strength_mean",
    "motif_order_strength_min",
    "motif_order_strength_std",
    "motif_fingerprint_norm_mean",
    "motif_fingerprint_norm_std",
    "motif_same_element_dispersion_rms",
    "motif_same_element_dispersion_q95",
    "motif_same_element_dispersion_max",
    "motif_global_dispersion_rms",
    "motif_species_centroid_separation_mean",
)


@dataclass(frozen=True)
class MotifFeatureResult:
    supported: bool
    failure_reason: str | None
    features: Mapping[str, float]


def _failure(exc: Exception | str) -> MotifFeatureResult:
    reason = f"{type(exc).__name__}: {exc}" if isinstance(exc, Exception) else str(exc)
    return MotifFeatureResult(False, reason, {})


def _featurizer_schema() -> tuple[CrystalNNFingerprint, list[str], np.ndarray, np.ndarray, np.ndarray]:
    featurizer = CrystalNNFingerprint.from_preset("ops")
    labels = list(featurizer.feature_labels())
    weight_indices = np.asarray(
        [index for index, label in enumerate(labels) if label.startswith("wt CN_")],
        dtype=int,
    )
    order_indices = np.asarray(
        [index for index, label in enumerate(labels) if not label.startswith("wt CN_")],
        dtype=int,
    )
    coordination = np.asarray(
        [float(labels[index].split("CN_")[-1]) for index in weight_indices],
        dtype=float,
    )
    if not len(weight_indices) or not len(order_indices) or len(coordination) != len(weight_indices):
        raise RuntimeError("CrystalNN ops fingerprint schema differs")
    return featurizer, labels, weight_indices, order_indices, coordination


_FEATURIZER, _LABELS, _WEIGHT_INDICES, _ORDER_INDICES, _COORDINATION = _featurizer_schema()


def compute_motif_coherence_features(atoms: Atoms) -> MotifFeatureResult:
    """Aggregate local Voronoi/CN order parameters without changing x0."""

    try:
        if (
            len(atoms) < 1
            or not np.all(atoms.pbc)
            or atoms.calc is not None
            or bool(atoms.info)
            or set(atoms.arrays) != {"numbers", "positions"}
        ):
            raise ValueError("motif features require exact periodic geometry-only Atoms")
        structure = AseAtomsAdaptor.get_structure(atoms)
        rows: list[np.ndarray] = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for index in range(len(structure)):
                rows.append(np.asarray(_FEATURIZER.featurize(structure, index), dtype=float))
        matrix = np.vstack(rows)
        if matrix.shape != (len(atoms), len(_LABELS)) or not np.isfinite(matrix).all():
            raise ValueError("CrystalNN ops fingerprint is non-finite or changed shape")
        weights = np.clip(matrix[:, _WEIGHT_INDICES], 0.0, None)
        weight_sum = np.sum(weights, axis=1)
        positive = weight_sum > 1.0e-15
        normalized = np.zeros_like(weights)
        normalized[positive] = weights[positive] / weight_sum[positive, None]
        dominance = np.max(normalized, axis=1)
        log_values = np.zeros_like(normalized)
        selected = normalized > 0.0
        log_values[selected] = normalized[selected] * np.log(normalized[selected])
        entropy = -np.sum(log_values, axis=1) / math.log(len(_WEIGHT_INDICES))
        effective_cn = normalized @ _COORDINATION
        order_strength = np.max(np.clip(matrix[:, _ORDER_INDICES], 0.0, None), axis=1)
        norms = np.linalg.norm(matrix, axis=1)

        same_residuals: list[float] = []
        species_centroids: list[np.ndarray] = []
        numbers = np.asarray(atoms.numbers, dtype=int)
        for number in sorted(set(numbers.tolist())):
            group = matrix[numbers == number]
            centroid = np.mean(group, axis=0)
            species_centroids.append(centroid)
            same_residuals.extend(np.linalg.norm(group - centroid, axis=1).tolist())
        same = np.asarray(same_residuals, dtype=float)
        global_centroid = np.mean(matrix, axis=0)
        global_residual = np.linalg.norm(matrix - global_centroid, axis=1)
        separations: list[float] = []
        for left in range(len(species_centroids)):
            for right in range(left + 1, len(species_centroids)):
                separations.append(
                    float(np.linalg.norm(species_centroids[left] - species_centroids[right]))
                )
        features = {
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
            "motif_effective_cn_range": float(np.max(effective_cn) - np.min(effective_cn)),
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
        if tuple(features) != FEATURE_NAMES or not np.isfinite(list(features.values())).all():
            raise ValueError("motif coherence feature schema differs")
        return MotifFeatureResult(True, None, features)
    except Exception as exc:
        return _failure(exc)


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def build_motif_feature_bank(
    *, metadata_path: Path, geometry_path: Path, cohort_manifest_path: Path, output_dir: Path
) -> dict[str, object]:
    """Build and seal NEXT46 without opening any endpoint artifact."""

    metadata_path = Path(metadata_path).resolve()
    geometry_path = Path(geometry_path).resolve()
    cohort_manifest_path = Path(cohort_manifest_path).resolve()
    target = Path(output_dir).resolve()
    if os.path.lexists(target):
        raise FileExistsError(f"refusing existing output: {target}")
    metadata, structures = _validated_inputs(
        metadata_path=metadata_path,
        geometry_path=geometry_path,
        cohort_manifest_path=cohort_manifest_path,
    )
    rows: list[dict[str, object]] = []
    failures: Counter[str] = Counter()
    supported = 0
    for index, atoms in enumerate(structures, start=1):
        result = compute_motif_coherence_features(atoms)
        row: dict[str, object] = {name: math.nan for name in FEATURE_NAMES}
        if result.supported:
            supported += 1
            row.update({name: float(result.features[name]) for name in FEATURE_NAMES})
        else:
            failures[result.failure_reason or "unsupported"] += 1
        row["motif_supported"] = bool(result.supported)
        row["motif_failure"] = result.failure_reason
        rows.append(row)
        if index % 50 == 0 or index == len(structures):
            print(f"NEXT46 motif coherence: {index}/{len(structures)}", flush=True)
    table = pd.concat([metadata.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        feature_path = staging / FEATURE_NAME
        table.to_parquet(feature_path, index=False)
        repository = Path(__file__).resolve().parents[1]
        matminer_source = Path(__import__("matminer.featurizers.site.fingerprint", fromlist=["x"]).__file__).resolve()
        manifest: dict[str, object] = {
            "protocol": PROTOCOL,
            "evidence_role": "local motif coherence from one raw x0",
            "input_role": "one_raw_pre_dft_pre_mlip_x0_only",
            "rows": len(table),
            "candidate_feature_count": len(FEATURE_NAMES),
            "candidate_features": list(FEATURE_NAMES),
            "supported_rows": supported,
            "failure_counts": dict(sorted(failures.items())),
            "labels_opened": False,
            "endpoint_fields_read": False,
            "dft_values_used": False,
            "mlip_or_model_potential_used": False,
            "learned_energy_force_stress_proxy_used": False,
            "physical_relaxation_executed": False,
            "same_composition_alternatives_used": False,
            "inputs_sha256": {
                "metadata": _sha256(metadata_path),
                "geometry": _sha256(geometry_path),
                "cohort_manifest": _sha256(cohort_manifest_path),
            },
            "executed_source_sha256": {
                "src/next46_motif_coherence_features.py": _sha256(
                    repository / "src/next46_motif_coherence_features.py"
                ),
                str(matminer_source): _sha256(matminer_source),
            },
            "scientific_improvement_claim": False,
        }
        manifest["outputs_sha256"] = {FEATURE_NAME: _sha256(feature_path)}
        (staging / MANIFEST_NAME).write_bytes(_json_bytes(manifest))
        if any(
            _sha256(path) != manifest["inputs_sha256"][role]
            for role, path in (
                ("metadata", metadata_path),
                ("geometry", geometry_path),
                ("cohort_manifest", cohort_manifest_path),
            )
        ):
            raise RuntimeError("NEXT46 input changed during feature build")
        _publish_directory_no_replace(staging, target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--geometry", required=True, type=Path)
    parser.add_argument("--cohort-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = build_motif_feature_bank(
        metadata_path=args.metadata,
        geometry_path=args.geometry,
        cohort_manifest_path=args.cohort_manifest,
        output_dir=args.output_dir,
    )
    print(json.dumps({"rows": manifest["rows"], "output": str(args.output_dir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
