#!/usr/bin/env python3
"""Build discovery-only Delaunay void-cage isotropy features."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Mapping, Sequence
import warnings

from ase import Atoms
import numpy as np
import pandas as pd
from pymatgen.analysis.local_env import VoronoiNN
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

import src.next85_scigen_label_free_features as n85
import src.next94_wyformer_label_free_features as n94
from src.next102_cross_source_dobvr_features import _sha256_file, _write_json


PROTOCOL = "2026-08-09-next255-delaunay-void-cage-isotropy-v1"
MANIFEST_NAME = "MANIFEST.json"
CATALOGUE_NAME = "NEXT255_DVCI_CATALOGUE.json"
FEATURE_FILES = {
    "scigen": "next255_scigen_discovery_dvci_features.parquet",
    "wyformer": "next255_wyformer_discovery_dvci_features.parquet",
}
METRIC_NAMES = ("tightness", "volume", "eigenratio", "closure")
STATISTIC_NAMES = ("mean", "q10", "q25", "lower_quartile_mean")
FEATURE_NAMES = tuple(
    f"dvci_{metric}_{statistic}"
    for metric in METRIC_NAMES
    for statistic in STATISTIC_NAMES
)
EXPECTED_DESIGN_SHA256 = (
    "869b8ca8f9307d978330678f4d7c12a34d70bca08b38a8ba8205344760124ded"
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
    "src/next85_scigen_label_free_features.py": (
        "2caf0fa0aafe6df6732c3b8ed02cd19d96076314273331f32a449b6bd3b41335"
    ),
    "src/next94_wyformer_label_free_features.py": (
        "ccb04a9387b4fad9ea3b8e7e7cd54fb69965f98a3c44342c198a8511b17702a9"
    ),
}
NUMERICAL_TOLERANCE = 1.0e-12
BISECTOR_RESIDUAL_TOLERANCE = 1.0e-8
RAY_NORM_TOLERANCE = 1.0e-12
LOWER_QUARTILE_INCLUSION_TOLERANCE = 1.0e-12
BOUNDARY_FLAGS = {
    "dft_calculation_executed": False,
    "dft_values_used_by_executable_formula": False,
    "learned_energy_force_stress_proxy_used": False,
    "model_or_proxy_potential_used": False,
    "physical_relaxation_executed": False,
    "opened_validation_outputs_used": False,
    "scigen_replication_endpoint_opened": False,
    "wyformer_replication_endpoint_opened": False,
}


@dataclass(frozen=True)
class DVCIFeatureResult:
    supported: bool
    failure_reason: str | None
    incidence_count: int
    features: Mapping[str, float]


def _failure(exc: Exception | str) -> DVCIFeatureResult:
    reason = f"{type(exc).__name__}: {exc}" if isinstance(exc, Exception) else str(exc)
    return DVCIFeatureResult(False, reason, 0, {})


def _bounded(value: float, *, label: str) -> float:
    if (
        not math.isfinite(value)
        or value < -NUMERICAL_TOLERANCE
        or value > 1.0 + NUMERICAL_TOLERANCE
    ):
        raise ValueError(f"{label} is outside the frozen [0,1] guard")
    return float(np.clip(value, 0.0, 1.0))


def void_cage_metrics(neighbor_displacements: object) -> dict[str, float]:
    """Solve one complete co-spherical cage and return frozen angular metrics."""

    rows = np.asarray(neighbor_displacements, dtype=float)
    if (
        rows.ndim != 2
        or rows.shape[1:] != (3,)
        or len(rows) < 3
        or np.any(~np.isfinite(rows))
    ):
        raise ValueError("NEXT255 incident face population differs")
    rhs = np.einsum("ij,ij->i", rows, rows) / 2.0
    center, _, rank, _ = np.linalg.lstsq(rows, rhs, rcond=None)
    if rank != 3 or np.any(~np.isfinite(center)):
        raise ValueError("NEXT255 bisector rank differs")
    residual = float(np.max(np.abs(rows @ center - rhs)))
    denominator = max(1.0, float(np.max(np.abs(rhs))))
    if not math.isfinite(residual) or residual / denominator > BISECTOR_RESIDUAL_TOLERANCE:
        raise ValueError("NEXT255 bisector residual differs")
    rays = np.vstack((-center, rows - center))
    norms = np.linalg.norm(rays, axis=1)
    if np.any(~np.isfinite(norms)) or np.any(norms <= RAY_NORM_TOLERANCE):
        raise ValueError("NEXT255 void-cage ray differs")
    directions = rays / norms[:, None]
    gram = directions.T @ directions / len(directions)
    eigenvalues = np.linalg.eigvalsh(gram)
    if (
        eigenvalues.shape != (3,)
        or np.any(~np.isfinite(eigenvalues))
        or eigenvalues[0] < -NUMERICAL_TOLERANCE
        or eigenvalues[-1] <= 0.0
        or eigenvalues[-1] > 1.0 + NUMERICAL_TOLERANCE
    ):
        raise ValueError("NEXT255 direction Gram spectrum differs")
    eigenvalues = np.clip(eigenvalues, 0.0, 1.0)
    result = {
        "tightness": _bounded(3.0 * eigenvalues[0], label="tightness"),
        "volume": _bounded(27.0 * float(np.prod(eigenvalues)), label="volume"),
        "eigenratio": _bounded(
            float(eigenvalues[0] / eigenvalues[-1]), label="eigenratio"
        ),
        "closure": _bounded(
            1.0 - float(np.linalg.norm(np.mean(directions, axis=0))),
            label="closure",
        ),
    }
    if tuple(result) != METRIC_NAMES:
        raise RuntimeError("NEXT255 metric schema differs")
    return result


def _population_mean(values: np.ndarray) -> float:
    return float(math.fsum(float(value) for value in values) / len(values))


def aggregate_dvci_features(metric_populations: Mapping[str, object]) -> dict[str, float]:
    """Aggregate complete-cage metrics into the frozen sixteen features."""

    if tuple(metric_populations) != METRIC_NAMES:
        raise ValueError("NEXT255 metric population schema differs")
    features: dict[str, float] = {}
    expected_count: int | None = None
    for metric in METRIC_NAMES:
        values = np.asarray(metric_populations[metric], dtype=float)
        if (
            values.ndim != 1
            or len(values) == 0
            or np.any(~np.isfinite(values))
            or np.any(values < -NUMERICAL_TOLERANCE)
            or np.any(values > 1.0 + NUMERICAL_TOLERANCE)
        ):
            raise ValueError("NEXT255 metric population differs")
        if expected_count is None:
            expected_count = len(values)
        elif len(values) != expected_count:
            raise ValueError("NEXT255 incidence accounting differs")
        values = np.clip(values, 0.0, 1.0)
        q10 = float(np.quantile(values, 0.10, method="inverted_cdf"))
        q25 = float(np.quantile(values, 0.25, method="inverted_cdf"))
        lower = values[values <= q25 + LOWER_QUARTILE_INCLUSION_TOLERANCE]
        if len(lower) == 0:
            raise RuntimeError("NEXT255 lower-quartile population differs")
        features[f"dvci_{metric}_mean"] = _population_mean(values)
        features[f"dvci_{metric}_q10"] = q10
        features[f"dvci_{metric}_q25"] = q25
        features[f"dvci_{metric}_lower_quartile_mean"] = _population_mean(lower)
    if tuple(features) != FEATURE_NAMES or not np.isfinite(list(features.values())).all():
        raise ValueError("NEXT255 aggregate feature schema differs")
    return features


def _face_records(
    *, structure: Structure, center: int, info: Sequence[Mapping[str, object]]
) -> list[tuple[tuple[int, tuple[int, int, int]], tuple[int, ...], np.ndarray]]:
    best: dict[
        tuple[int, tuple[int, int, int]],
        tuple[float, tuple[tuple[int, ...], tuple[float, ...]], np.ndarray, tuple[int, ...]],
    ] = {}
    center_coords = np.asarray(structure[center].coords, dtype=float)
    for item in info:
        try:
            site_index = int(item["site_index"])
            image = tuple(int(round(float(value))) for value in item["image"])
            poly = item["poly_info"]
            if not isinstance(poly, Mapping):
                continue
            area = float(poly["area"])
            vertices = tuple(sorted(set(int(value) for value in poly["verts"])))
            neighbor = item["site"]
            displacement = np.asarray(neighbor.coords, dtype=float) - center_coords
        except (AttributeError, KeyError, TypeError, ValueError):
            continue
        if (
            len(image) != 3
            or site_index < 0
            or site_index >= len(structure)
            or not math.isfinite(area)
            or area <= 0.0
            or len(vertices) < 3
            or displacement.shape != (3,)
            or np.any(~np.isfinite(displacement))
        ):
            continue
        key = (site_index, image)
        tie = (vertices, tuple(float(value) for value in np.round(displacement, 12)))
        previous = best.get(key)
        if previous is None or area > previous[0] or (area == previous[0] and tie < previous[1]):
            best[key] = (area, tie, displacement, vertices)
    if not best:
        raise ValueError("NEXT255 site has no valid Voronoi face")
    return [
        (key, best[key][3], best[key][2])
        for key in sorted(best)
    ]


def compute_dvci_features(atoms: Atoms) -> DVCIFeatureResult:
    """Compute frozen Delaunay void-cage isotropy from raw x0 geometry."""

    try:
        if (
            len(atoms) < 1
            or not np.all(atoms.pbc)
            or atoms.calc is not None
            or bool(atoms.info)
            or set(atoms.arrays) != {"numbers", "positions"}
        ):
            raise ValueError("DVCI features require exact periodic geometry-only Atoms")
        structure = AseAtomsAdaptor.get_structure(atoms)
        finder = VoronoiNN(
            weight="solid_angle", tol=0, cutoff=13, compute_adj_neighbors=True
        )
        populations: dict[str, list[float]] = {name: [] for name in METRIC_NAMES}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for center in range(len(structure)):
                records = _face_records(
                    structure=structure,
                    center=center,
                    info=finder.get_nn_info(structure, center),
                )
                incident: dict[int, list[np.ndarray]] = defaultdict(list)
                for _, vertices, displacement in records:
                    for vertex in vertices:
                        incident[vertex].append(displacement)
                if not incident:
                    raise ValueError("NEXT255 site has no valid Voronoi vertex")
                for vertex in sorted(incident):
                    metrics = void_cage_metrics(incident[vertex])
                    for name in METRIC_NAMES:
                        populations[name].append(metrics[name])
        features = aggregate_dvci_features(populations)
        incidence_count = len(populations[METRIC_NAMES[0]])
        return DVCIFeatureResult(True, None, incidence_count, features)
    except Exception as exc:
        return _failure(exc)


def compute_dvci_row(atoms: Atoms) -> dict[str, object]:
    result = compute_dvci_features(atoms)
    row: dict[str, object] = {name: math.nan for name in FEATURE_NAMES}
    row["dvci_supported"] = bool(result.supported)
    row["dvci_failure"] = result.failure_reason
    row["dvci_incidence_count"] = int(result.incidence_count)
    if result.supported:
        row.update({name: float(result.features[name]) for name in FEATURE_NAMES})
    return row


def _error_row(exc: Exception) -> dict[str, object]:
    return {
        **{name: math.nan for name in FEATURE_NAMES},
        "dvci_supported": False,
        "dvci_failure": f"{type(exc).__name__}: {exc}",
        "dvci_incidence_count": 0,
    }


def _compute_scigen_payload(item: tuple[str, bytes]) -> tuple[str, dict[str, object]]:
    material_id, payload = item
    try:
        parsed = n85._parse_frame(payload, strict_output=True)
        return material_id, compute_dvci_row(parsed.atoms)
    except Exception as exc:
        return material_id, _error_row(exc)


def _compute_wyformer_payload(item: tuple[str, str]) -> tuple[str, dict[str, object]]:
    material_id, payload = item
    try:
        structure = Structure.from_dict(json.loads(payload))
        return material_id, compute_dvci_row(AseAtomsAdaptor.get_atoms(structure))
    except Exception as exc:
        return material_id, _error_row(exc)


def _compute_many(
    payloads: Sequence[tuple[str, bytes]] | Sequence[tuple[str, str]],
    *,
    source: str,
    workers: int,
) -> list[tuple[str, dict[str, object]]]:
    worker = _compute_scigen_payload if source == "scigen" else _compute_wyformer_payload
    if workers == 1:
        return [worker(item) for item in payloads]  # type: ignore[arg-type]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(worker, payloads, chunksize=8))  # type: ignore[arg-type]


def _read_manifest(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} must contain an object")
    return payload


def build_cross_source_dvci_features(
    *,
    scigen_cohort_dir: Path,
    wyformer_cohort_dir: Path,
    design_path: Path,
    output_dir: Path,
    workers: int = 16,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    """Build NEXT255 from physically isolated discovery geometry only."""

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
        raise ValueError("NEXT255 workers must be a positive exact integer")
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT255 input is missing")
    input_hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if require_formal_inputs and input_hashes != EXPECTED_INPUT_SHA256:
        differing = sorted(
            name
            for name in set(input_hashes) | set(EXPECTED_INPUT_SHA256)
            if input_hashes.get(name) != EXPECTED_INPUT_SHA256.get(name)
        )
        raise ValueError(f"NEXT255 formal input identity differs: {differing}")
    repository = Path(__file__).resolve().parents[1]
    upstream_hashes = {
        name: _sha256_file(repository / name)
        for name in EXPECTED_UPSTREAM_SOURCE_SHA256
    }
    if require_formal_inputs and upstream_hashes != EXPECTED_UPSTREAM_SOURCE_SHA256:
        raise ValueError("NEXT255 frozen upstream source differs")
    scigen_manifest = _read_manifest(paths["scigen_manifest"])
    wyformer_manifest = _read_manifest(paths["wyformer_manifest"])
    if (
        scigen_manifest.get("protocol") != n85.COHORT_PROTOCOL
        or scigen_manifest.get("labels_opened") is not False
        or scigen_manifest.get("endpoint_payloads_opened") is not False
        or scigen_manifest.get("relaxed_structures_opened") is not False
        or wyformer_manifest.get("protocol") != n94.COHORT_PROTOCOL
        or wyformer_manifest.get("discovery_endpoint_opened") is not False
        or wyformer_manifest.get("validation_endpoint_opened") is not False
        or wyformer_manifest.get("replication_endpoint_opened") is not False
        or wyformer_manifest.get("relaxed_structures_published") is not False
    ):
        raise ValueError("NEXT255 discovery geometry provenance differs")
    metadata = {
        "scigen": pd.read_parquet(paths["scigen_metadata"]),
        "wyformer": pd.read_parquet(paths["wyformer_metadata"]),
    }
    discovery: dict[str, pd.DataFrame] = {}
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
            raise ValueError(f"NEXT255 {source} metadata differs")
        selected = frame.loc[frame["partition_role"].eq("discovery")].copy()
        selected = selected.sort_values("material_id", kind="mergesort").reset_index(drop=True)
        if selected.empty:
            raise ValueError(f"NEXT255 {source} discovery identity differs")
        discovery[source] = selected
    scigen_ids = discovery["scigen"]["material_id"].astype(str).tolist()
    wyformer_ids = discovery["wyformer"]["material_id"].astype(str).tolist()
    payloads = {
        "scigen": n85._archive_payloads(paths["scigen_discovery_geometry"], scigen_ids),
        "wyformer": n94._payloads(paths["wyformer_discovery_geometry"], wyformer_ids),
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    source_path = Path(__file__).resolve()
    source_hash = _sha256_file(source_path)
    started = time.perf_counter()
    try:
        computed = {
            source: _compute_many(payloads[source], source=source, workers=workers)
            for source in ("scigen", "wyformer")
        }
        counts = {}
        output_paths: list[Path] = []
        for source in ("scigen", "wyformer"):
            computed_frame = pd.DataFrame(
                [{"material_id": material_id, **row} for material_id, row in computed[source]]
            )
            if (
                computed_frame["material_id"].astype(str).duplicated().any()
                or set(computed_frame["material_id"].astype(str))
                != set(discovery[source]["material_id"].astype(str))
            ):
                raise RuntimeError(f"NEXT255 {source} material identity differs")
            table = discovery[source].merge(
                computed_frame, on="material_id", how="left", validate="one_to_one"
            )
            if len(table) != len(discovery[source]):
                raise RuntimeError(f"NEXT255 {source} row accounting differs")
            supported = table["dvci_supported"].fillna(False).astype(bool)
            finite_counts = {
                name: int(np.isfinite(pd.to_numeric(table[name], errors="coerce")).sum())
                for name in FEATURE_NAMES
            }
            if not supported.all() or any(count != len(table) for count in finite_counts.values()):
                failures = Counter(table.loc[~supported, "dvci_failure"].astype(str))
                raise RuntimeError(
                    f"NEXT255 {source} formal support differs: {dict(sorted(failures.items()))}"
                )
            incidence = pd.to_numeric(table["dvci_incidence_count"], errors="coerce")
            if not np.isfinite(incidence).all() or (incidence <= 0).any():
                raise RuntimeError(f"NEXT255 {source} incidence accounting differs")
            output = staging / FEATURE_FILES[source]
            table.to_parquet(output, index=False)
            output_paths.append(output)
            counts[source] = {
                "rows": int(len(table)),
                "supported": int(supported.sum()),
                "failures": int((~supported).sum()),
                "failure_counts": {},
                "finite_feature_counts": finite_counts,
                "incidence_count_min": int(incidence.min()),
                "incidence_count_max": int(incidence.max()),
                "incidence_count_sum": int(incidence.sum()),
            }
        if counts["scigen"]["rows"] != 13_470 or counts["wyformer"]["rows"] != 5_232:
            raise RuntimeError("NEXT255 frozen discovery row counts differ")
        catalogue = {
            "protocol": PROTOCOL,
            "feature_names": list(FEATURE_NAMES),
            "feature_count": len(FEATURE_NAMES),
            "metric_names": list(METRIC_NAMES),
            "statistics": list(STATISTIC_NAMES),
            "voronoi": {
                "weight": "solid_angle",
                "tol": 0,
                "cutoff": 13,
                "compute_adj_neighbors": True,
            },
            "cage_population": "atom_void_incidence_weighted_complete_cospherical_cages",
            "forced_tetrahedralization": False,
            "quantile_method": "inverted_cdf",
            "quantiles": [0.10, 0.25],
            "lower_quartile_inclusion_tolerance": LOWER_QUARTILE_INCLUSION_TOLERANCE,
            "bisector_residual_tolerance": BISECTOR_RESIDUAL_TOLERANCE,
            "source_partitions_read": {"scigen": ["discovery"], "wyformer": ["discovery"]},
            "endpoint_columns_present": False,
            "labels_opened": False,
        }
        catalogue_path = staging / CATALOGUE_NAME
        _write_json(catalogue_path, catalogue)
        output_paths.append(catalogue_path)
        manifest = {
            "protocol": PROTOCOL,
            "mode": "physically_isolated_discovery_x0_dvci_feature_freeze",
            "workers": workers,
            "elapsed_seconds": time.perf_counter() - started,
            "counts": counts,
            "source_partitions_read": {"scigen": ["discovery"], "wyformer": ["discovery"]},
            "labels_opened": False,
            "endpoint_fields_read": False,
            "internal_validation_geometry_opened": False,
            "internal_replication_geometry_opened": False,
            **BOUNDARY_FLAGS,
            "scientific_improvement_claim": False,
            "inputs_sha256": input_hashes,
            "upstream_source_sha256": upstream_hashes,
            "executed_source_sha256": {
                "src/next255_delaunay_void_cage_isotropy.py": source_hash
            },
            "outputs_sha256": {path.name: _sha256_file(path) for path in output_paths},
        }
        _write_json(staging / MANIFEST_NAME, manifest)
        if any(_sha256_file(path) != input_hashes[name] for name, path in paths.items()):
            raise RuntimeError("NEXT255 input changed before publication")
        if _sha256_file(source_path) != source_hash:
            raise RuntimeError("NEXT255 source changed before publication")
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
    manifest = build_cross_source_dvci_features(
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
    "DVCIFeatureResult",
    "FEATURE_NAMES",
    "METRIC_NAMES",
    "PROTOCOL",
    "aggregate_dvci_features",
    "build_cross_source_dvci_features",
    "compute_dvci_features",
    "void_cage_metrics",
]


if __name__ == "__main__":
    raise SystemExit(main())
