#!/usr/bin/env python3
"""Build discovery-only morphometric Voronoi bond-order features."""

from __future__ import annotations

import argparse
from collections import Counter
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


PROTOCOL = "2026-08-09-next239-morphometric-voronoi-bond-order-v1"
MANIFEST_NAME = "MANIFEST.json"
CATALOGUE_NAME = "NEXT239_MVBO_CATALOGUE.json"
FEATURE_FILES = {
    "scigen": "next239_scigen_discovery_mvbo_features.parquet",
    "wyformer": "next239_wyformer_discovery_mvbo_features.parquet",
}
FEATURE_NAMES = (
    "mvbo_q4_mean",
    "mvbo_q4_std",
    "mvbo_q6_mean",
    "mvbo_q6_std",
    "mvbo_facet_evenness_min",
    "mvbo_facet_evenness_q10",
    "mvbo_facet_evenness_mean",
    "mvbo_facet_evenness_std",
    "mvbo_same_element_q46_dispersion_rms",
    "mvbo_same_element_q46_dispersion_q95",
    "mvbo_same_element_q46_dispersion_max",
)
EXPECTED_DESIGN_SHA256 = (
    "31e524b62cf466a3cb1a47ddba127bc71b739bf22114ad8e2896bdfb4f3b7011"
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
LEGENDRE_ORDERS = (4, 6)
NUMERICAL_TOLERANCE = 1.0e-10
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
class MVBOFeatureResult:
    supported: bool
    failure_reason: str | None
    features: Mapping[str, float]


def _failure(exc: Exception | str) -> MVBOFeatureResult:
    reason = f"{type(exc).__name__}: {exc}" if isinstance(exc, Exception) else str(exc)
    return MVBOFeatureResult(False, reason, {})


def _bounded(value: float, *, label: str) -> float:
    if (
        not math.isfinite(value)
        or value < -NUMERICAL_TOLERANCE
        or value > 1.0 + NUMERICAL_TOLERANCE
    ):
        raise ValueError(f"{label} is outside the frozen [0,1] guard")
    return float(np.clip(value, 0.0, 1.0))


def morphometric_site_invariants(
    *, normals: object, areas: object
) -> tuple[float, float, float]:
    """Return area-weighted q4, q6, and facet evenness for one site."""

    direction = np.asarray(normals, dtype=float)
    raw_area = np.asarray(areas, dtype=float)
    if (
        direction.ndim != 2
        or direction.shape[1:] != (3,)
        or len(direction) == 0
        or raw_area.shape != (len(direction),)
        or np.any(~np.isfinite(direction))
        or np.any(~np.isfinite(raw_area))
        or np.any(raw_area <= 0.0)
    ):
        raise ValueError("NEXT239 site facet population differs")
    norms = np.linalg.norm(direction, axis=1)
    if np.any(~np.isfinite(norms)) or np.any(np.abs(norms - 1.0) > 1.0e-7):
        raise ValueError("NEXT239 Voronoi facet normal differs")
    unit = direction / norms[:, None]
    weights = raw_area / float(np.sum(raw_area))
    dots = np.clip(unit @ unit.T, -1.0, 1.0)
    values: list[float] = []
    for order in LEGENDRE_ORDERS:
        coefficients = np.zeros(order + 1, dtype=float)
        coefficients[order] = 1.0
        polynomial = np.polynomial.legendre.legval(dots, coefficients)
        squared = float(weights @ polynomial @ weights)
        if squared < -NUMERICAL_TOLERANCE or squared > 1.0 + NUMERICAL_TOLERANCE:
            raise ValueError(f"NEXT239 q{order} squared differs")
        values.append(_bounded(math.sqrt(max(0.0, squared)), label=f"q{order}"))
    evenness = 1.0 / (len(weights) * float(np.sum(weights**2)))
    values.append(_bounded(evenness, label="facet evenness"))
    return tuple(values)  # type: ignore[return-value]


def aggregate_mvbo_features(
    *, q4: object, q6: object, evenness: object, atomic_numbers: object
) -> dict[str, float]:
    """Aggregate site morphometric invariants into the frozen schema."""

    q4_values = np.asarray(q4, dtype=float)
    q6_values = np.asarray(q6, dtype=float)
    evenness_values = np.asarray(evenness, dtype=float)
    numbers = np.asarray(atomic_numbers, dtype=int)
    if (
        q4_values.ndim != 1
        or len(q4_values) == 0
        or q6_values.shape != q4_values.shape
        or evenness_values.shape != q4_values.shape
        or numbers.shape != q4_values.shape
        or np.any(~np.isfinite(q4_values))
        or np.any(~np.isfinite(q6_values))
        or np.any(~np.isfinite(evenness_values))
        or np.any(numbers <= 0)
    ):
        raise ValueError("NEXT239 aggregate population differs")
    for name, values in (
        ("q4", q4_values),
        ("q6", q6_values),
        ("evenness", evenness_values),
    ):
        if np.any(values < -NUMERICAL_TOLERANCE) or np.any(
            values > 1.0 + NUMERICAL_TOLERANCE
        ):
            raise ValueError(f"NEXT239 aggregate {name} bounds differ")
    vectors = np.column_stack((q4_values, q6_values))
    residuals: list[float] = []
    for number in sorted(set(numbers.tolist())):
        group = vectors[numbers == number]
        centroid = np.mean(group, axis=0)
        residuals.extend(np.linalg.norm(group - centroid, axis=1).tolist())
    same = np.asarray(residuals, dtype=float)
    features = {
        "mvbo_q4_mean": float(np.mean(q4_values)),
        "mvbo_q4_std": float(np.std(q4_values)),
        "mvbo_q6_mean": float(np.mean(q6_values)),
        "mvbo_q6_std": float(np.std(q6_values)),
        "mvbo_facet_evenness_min": float(np.min(evenness_values)),
        "mvbo_facet_evenness_q10": float(
            np.quantile(evenness_values, 0.10, method="inverted_cdf")
        ),
        "mvbo_facet_evenness_mean": float(np.mean(evenness_values)),
        "mvbo_facet_evenness_std": float(np.std(evenness_values)),
        "mvbo_same_element_q46_dispersion_rms": float(
            np.sqrt(np.mean(same**2))
        ),
        "mvbo_same_element_q46_dispersion_q95": float(
            np.quantile(same, 0.95, method="inverted_cdf")
        ),
        "mvbo_same_element_q46_dispersion_max": float(np.max(same)),
    }
    if tuple(features) != FEATURE_NAMES or not np.isfinite(list(features.values())).all():
        raise ValueError("NEXT239 aggregate feature schema differs")
    return features


def compute_mvbo_features(atoms: Atoms) -> MVBOFeatureResult:
    """Compute the frozen morphometric features from one raw x0 structure."""

    try:
        if (
            len(atoms) < 1
            or not np.all(atoms.pbc)
            or atoms.calc is not None
            or bool(atoms.info)
            or set(atoms.arrays) != {"numbers", "positions"}
        ):
            raise ValueError("MVBO features require exact periodic geometry-only Atoms")
        structure = AseAtomsAdaptor.get_structure(atoms)
        finder = VoronoiNN(weight="solid_angle", tol=0, cutoff=13)
        site_values: list[tuple[float, float, float]] = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for index in range(len(structure)):
                info = finder.get_nn_info(structure, index)
                facets: dict[
                    tuple[int, tuple[int, int, int]], tuple[np.ndarray, float]
                ] = {}
                for item in info:
                    try:
                        site_index = int(item["site_index"])
                        image = tuple(
                            int(round(float(value))) for value in item["image"]
                        )
                        poly = item["poly_info"]
                        normal = np.asarray(poly["normal"], dtype=float)
                        area = float(poly["area"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    if len(image) != 3 or normal.shape != (3,):
                        continue
                    key = (site_index, image)
                    previous = facets.get(key)
                    if previous is None or area > previous[1]:
                        facets[key] = (normal, area)
                ordered = [facets[key] for key in sorted(facets)]
                if not ordered:
                    raise ValueError("site has no valid Voronoi facet")
                site_values.append(
                    morphometric_site_invariants(
                        normals=np.asarray([value[0] for value in ordered]),
                        areas=np.asarray([value[1] for value in ordered]),
                    )
                )
        matrix = np.asarray(site_values, dtype=float)
        if matrix.shape != (len(atoms), 3):
            raise ValueError("NEXT239 site invariant schema differs")
        features = aggregate_mvbo_features(
            q4=matrix[:, 0],
            q6=matrix[:, 1],
            evenness=matrix[:, 2],
            atomic_numbers=np.asarray(atoms.numbers, dtype=int),
        )
        return MVBOFeatureResult(True, None, features)
    except Exception as exc:
        return _failure(exc)


def compute_mvbo_row(atoms: Atoms) -> dict[str, object]:
    result = compute_mvbo_features(atoms)
    row: dict[str, object] = {name: math.nan for name in FEATURE_NAMES}
    row["mvbo_supported"] = bool(result.supported)
    row["mvbo_failure"] = result.failure_reason
    if result.supported:
        row.update({name: float(result.features[name]) for name in FEATURE_NAMES})
    return row


def _error_row(exc: Exception) -> dict[str, object]:
    return {
        **{name: math.nan for name in FEATURE_NAMES},
        "mvbo_supported": False,
        "mvbo_failure": f"{type(exc).__name__}: {exc}",
    }


def _compute_scigen_payload(item: tuple[str, bytes]) -> tuple[str, dict[str, object]]:
    material_id, payload = item
    try:
        parsed = n85._parse_frame(payload, strict_output=True)
        return material_id, compute_mvbo_row(parsed.atoms)
    except Exception as exc:
        return material_id, _error_row(exc)


def _compute_wyformer_payload(item: tuple[str, str]) -> tuple[str, dict[str, object]]:
    material_id, payload = item
    try:
        structure = Structure.from_dict(json.loads(payload))
        return material_id, compute_mvbo_row(AseAtomsAdaptor.get_atoms(structure))
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


def build_cross_source_mvbo_features(
    *,
    scigen_cohort_dir: Path,
    wyformer_cohort_dir: Path,
    design_path: Path,
    output_dir: Path,
    workers: int = 16,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    """Build NEXT239 from physically isolated discovery geometry only."""

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
        raise ValueError("NEXT239 workers must be a positive exact integer")
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT239 input is missing")
    input_hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if require_formal_inputs and input_hashes != EXPECTED_INPUT_SHA256:
        differing = sorted(
            name
            for name in set(input_hashes) | set(EXPECTED_INPUT_SHA256)
            if input_hashes.get(name) != EXPECTED_INPUT_SHA256.get(name)
        )
        raise ValueError(f"NEXT239 formal input identity differs: {differing}")
    repository = Path(__file__).resolve().parents[1]
    upstream_hashes = {
        name: _sha256_file(repository / name)
        for name in EXPECTED_UPSTREAM_SOURCE_SHA256
    }
    if require_formal_inputs and upstream_hashes != EXPECTED_UPSTREAM_SOURCE_SHA256:
        raise ValueError("NEXT239 frozen upstream source differs")
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
        raise ValueError("NEXT239 discovery geometry provenance differs")
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
            raise ValueError(f"NEXT239 {source} metadata differs")
        selected = frame.loc[frame["partition_role"].eq("discovery")].copy()
        selected = selected.sort_values("material_id", kind="mergesort").reset_index(drop=True)
        if selected.empty:
            raise ValueError(f"NEXT239 {source} discovery identity differs")
        discovery[source] = selected
    scigen_ids = discovery["scigen"]["material_id"].astype(str).tolist()
    wyformer_ids = discovery["wyformer"]["material_id"].astype(str).tolist()
    payloads = {
        "scigen": n85._archive_payloads(paths["scigen_discovery_geometry"], scigen_ids),
        "wyformer": n94._payloads(paths["wyformer_discovery_geometry"], wyformer_ids),
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent)
    )
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
                [
                    {"material_id": material_id, **row}
                    for material_id, row in computed[source]
                ]
            )
            table = discovery[source].merge(
                computed_frame, on="material_id", how="left", validate="one_to_one"
            )
            if len(table) != len(discovery[source]):
                raise RuntimeError(f"NEXT239 {source} row accounting differs")
            output = staging / FEATURE_FILES[source]
            table.to_parquet(output, index=False)
            output_paths.append(output)
            supported = table["mvbo_supported"].fillna(False).astype(bool)
            failures = Counter(table.loc[~supported, "mvbo_failure"].astype(str))
            counts[source] = {
                "rows": int(len(table)),
                "supported": int(supported.sum()),
                "failures": int((~supported).sum()),
                "failure_counts": dict(sorted(failures.items())),
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
            "legendre_orders": list(LEGENDRE_ORDERS),
            "voronoi": {"weight": "solid_angle", "tol": 0, "cutoff": 13},
            "facet_weight": "area_over_site_total_area",
            "quantile_method": "inverted_cdf",
            "source_partitions_read": {
                "scigen": ["discovery"],
                "wyformer": ["discovery"],
            },
            "endpoint_columns_present": False,
            "labels_opened": False,
        }
        catalogue_path = staging / CATALOGUE_NAME
        _write_json(catalogue_path, catalogue)
        output_paths.append(catalogue_path)
        manifest = {
            "protocol": PROTOCOL,
            "mode": "physically_isolated_discovery_x0_mvbo_feature_freeze",
            "workers": workers,
            "elapsed_seconds": time.perf_counter() - started,
            "counts": counts,
            "source_partitions_read": {
                "scigen": ["discovery"],
                "wyformer": ["discovery"],
            },
            "labels_opened": False,
            "endpoint_fields_read": False,
            "internal_validation_geometry_opened": False,
            "internal_replication_geometry_opened": False,
            **BOUNDARY_FLAGS,
            "scientific_improvement_claim": False,
            "inputs_sha256": input_hashes,
            "upstream_source_sha256": upstream_hashes,
            "executed_source_sha256": {
                "src/next239_morphometric_voronoi_bond_order.py": source_hash
            },
            "outputs_sha256": {
                path.name: _sha256_file(path) for path in output_paths
            },
        }
        _write_json(staging / MANIFEST_NAME, manifest)
        if any(
            _sha256_file(path) != input_hashes[name]
            for name, path in paths.items()
        ):
            raise RuntimeError("NEXT239 input changed before publication")
        if _sha256_file(source_path) != source_hash:
            raise RuntimeError("NEXT239 source changed before publication")
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
    manifest = build_cross_source_mvbo_features(
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
    "MVBOFeatureResult",
    "aggregate_mvbo_features",
    "build_cross_source_mvbo_features",
    "compute_mvbo_features",
    "morphometric_site_invariants",
]


if __name__ == "__main__":
    raise SystemExit(main())
