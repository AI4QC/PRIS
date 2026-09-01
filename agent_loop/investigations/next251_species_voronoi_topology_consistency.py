#!/usr/bin/env python3
"""Build discovery-only species-conditioned Voronoi topology features."""

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


PROTOCOL = "2026-08-09-next251-species-voronoi-topology-consistency-v1"
MANIFEST_NAME = "MANIFEST.json"
CATALOGUE_NAME = "NEXT251_SVTC_CATALOGUE.json"
FEATURE_FILES = {
    "scigen": "next251_scigen_discovery_svtc_features.parquet",
    "wyformer": "next251_wyformer_discovery_svtc_features.parquet",
}
FEATURE_NAMES = (
    "svtc_raw_odd_area_mean",
    "svtc_raw_odd_area_q90",
    "svtc_raw_degree_entropy_mean",
    "svtc_raw_degree_entropy_q90",
    "svtc_raw_species_modal_agreement",
    "svtc_raw_species_signature_entropy",
    "svtc_raw_species_signature_gini",
    "svtc_raw_species_effective_signature_excess",
    "svtc_robust_odd_area_mean",
    "svtc_robust_odd_area_q90",
    "svtc_robust_degree_entropy_mean",
    "svtc_robust_degree_entropy_q90",
    "svtc_robust_species_modal_agreement",
    "svtc_robust_species_signature_entropy",
    "svtc_robust_species_signature_gini",
    "svtc_robust_species_effective_signature_excess",
)
EXPECTED_PARENT_DESIGN_SHA256 = (
    "dea5e9391cdfcd38d8485e7115c13c99963ccc77cb8c8b4b15463f0903e1b8b3"
)
EXPECTED_FIRST_AMENDMENT_SHA256 = (
    "de29b931e7493d6f650ba106c19e87fed6c7990c99855978f9cfbc82b733b146"
)
EXPECTED_DESIGN_SHA256 = (
    "fe613871d60f1d6bd2d5babd463e48d2d6b189de17a17a382b3c260df4b29974"
)
EXPECTED_INPUT_SHA256 = {
    "scigen_manifest": "dc5bf33c6ce6dc2c10bcd3704688055058145fbe7269ada23ffbe4b141d75fe7",
    "scigen_metadata": "f91455f23b0a96f60fd1c779249e2be46a7ecf94fcdde2b146426a95aac05bde",
    "scigen_discovery_geometry": "e561ef12343c66dcc72bcabf6b8719ad727e01c9582a094e281da73b862ab575",
    "wyformer_manifest": "e0539d556538cb4c052431bc6a1e5c1663bc3de427677dbc8a446dcc3b4fbc54",
    "wyformer_metadata": "3b152b4b84c8d3f7ff5e85611dc1fd2728296f150e907ac4578ce55d2b27dd2b",
    "wyformer_discovery_geometry": "f1ce5ae4fba8c13fcbf3e25de4f596b919d9b41da5b072d9a28eefeaffc69784",
    "parent_design": EXPECTED_PARENT_DESIGN_SHA256,
    "first_amendment": EXPECTED_FIRST_AMENDMENT_SHA256,
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
TOPOLOGY_MODES = ("raw", "robust")
FACE_DEGREE_BINS = (3, 4, 5, 6, 7, 8, 9)
ROBUST_AREA_FRACTION = 1 / 64
NUMERICAL_TOLERANCE = 1.0e-12
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
class SiteTopology:
    signature: tuple[int, int, int, int, int, int, int]
    odd_area_fraction: float
    degree_entropy: float


@dataclass(frozen=True)
class SVTCFeatureResult:
    supported: bool
    failure_reason: str | None
    features: Mapping[str, float]


def _failure(exc: Exception | str) -> SVTCFeatureResult:
    reason = f"{type(exc).__name__}: {exc}" if isinstance(exc, Exception) else str(exc)
    return SVTCFeatureResult(False, reason, {})


def _bounded(value: float, *, label: str) -> float:
    if (
        not math.isfinite(value)
        or value < -NUMERICAL_TOLERANCE
        or value > 1.0 + NUMERICAL_TOLERANCE
    ):
        raise ValueError(f"{label} is outside the frozen [0,1] guard")
    return float(np.clip(value, 0.0, 1.0))


def _degree_bin_index(degree: int) -> int:
    if type(degree) is not int or degree < 3:
        raise ValueError("NEXT251 Voronoi face degree differs")
    return min(degree, 9) - 3


def face_topology(*, areas: object, degrees: object, mode: str) -> SiteTopology:
    """Return one site's frozen raw or robust face-degree topology."""

    raw_area = np.asarray(areas, dtype=float)
    raw_degree_float = np.asarray(degrees, dtype=float)
    if mode not in TOPOLOGY_MODES:
        raise ValueError("NEXT251 topology mode differs")
    if (
        raw_area.ndim != 1
        or len(raw_area) == 0
        or raw_degree_float.shape != raw_area.shape
        or np.any(~np.isfinite(raw_area))
        or np.any(raw_area <= 0.0)
        or np.any(~np.isfinite(raw_degree_float))
        or np.any(raw_degree_float != np.floor(raw_degree_float))
        or np.any(raw_degree_float < 3)
    ):
        raise ValueError("NEXT251 face population differs")
    raw_degree = raw_degree_float.astype(int)
    total = float(np.sum(raw_area))
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("NEXT251 face population differs")
    keep = np.ones(len(raw_area), dtype=bool)
    if mode == "robust":
        keep = raw_area / total >= ROBUST_AREA_FRACTION
    if not keep.any():
        raise ValueError("NEXT251 robust face population is empty")
    selected_area = raw_area[keep]
    selected_degree = raw_degree[keep]
    weights = selected_area / float(np.sum(selected_area))
    signature = np.zeros(len(FACE_DEGREE_BINS), dtype=int)
    area_by_bin = np.zeros(len(FACE_DEGREE_BINS), dtype=float)
    for degree, weight in zip(selected_degree.tolist(), weights.tolist(), strict=True):
        index = _degree_bin_index(int(degree))
        signature[index] += 1
        area_by_bin[index] += float(weight)
    nonzero = area_by_bin > 0.0
    entropy = -float(np.sum(area_by_bin[nonzero] * np.log(area_by_bin[nonzero])))
    entropy /= math.log(len(FACE_DEGREE_BINS))
    odd = float(np.sum(area_by_bin[[0, 2, 4, 6]]))
    result = SiteTopology(
        tuple(int(value) for value in signature),
        _bounded(odd, label="odd face-area fraction"),
        _bounded(entropy, label="face-degree entropy"),
    )
    if sum(result.signature) != int(keep.sum()):
        raise RuntimeError("NEXT251 face signature accounting differs")
    return result


def species_topology_consistency(
    *, signatures: Sequence[object], atomic_numbers: object
) -> dict[str, float]:
    """Return frozen same-element topology-consistency statistics."""

    numbers_float = np.asarray(atomic_numbers, dtype=float)
    if (
        len(signatures) == 0
        or numbers_float.shape != (len(signatures),)
        or np.any(~np.isfinite(numbers_float))
        or np.any(numbers_float != np.floor(numbers_float))
        or np.any(numbers_float <= 0)
    ):
        raise ValueError("NEXT251 species topology population differs")
    numbers = numbers_float.astype(int)
    normalized: list[tuple[int, ...]] = []
    for signature in signatures:
        values_float = np.asarray(signature, dtype=float)
        if (
            values_float.shape != (len(FACE_DEGREE_BINS),)
            or np.any(~np.isfinite(values_float))
            or np.any(values_float != np.floor(values_float))
            or np.any(values_float < 0)
            or int(np.sum(values_float)) <= 0
        ):
            raise ValueError("NEXT251 species signature differs")
        normalized.append(tuple(int(value) for value in values_float))
    total_sites = len(normalized)
    modal = 0
    entropy_sum = 0.0
    gini_sum = 0.0
    effective_excess_sum = 0.0
    for number in sorted(set(numbers.tolist())):
        indices = np.flatnonzero(numbers == number)
        counts = Counter(normalized[index] for index in indices.tolist())
        species_sites = len(indices)
        unique = len(counts)
        modal += max(counts.values())
        if unique > 1:
            probabilities = np.asarray(list(counts.values()), dtype=float) / species_sites
            raw_entropy = -float(np.sum(probabilities * np.log(probabilities)))
            entropy_sum += species_sites * raw_entropy / math.log(unique)
            gini_sum += species_sites * (1.0 - float(np.sum(probabilities**2)))
            effective_excess_sum += species_sites * (1.0 - math.exp(-raw_entropy))
    result = {
        "species_modal_agreement": _bounded(modal / total_sites, label="modal agreement"),
        "species_signature_entropy": _bounded(
            entropy_sum / total_sites, label="species signature entropy"
        ),
        "species_signature_gini": _bounded(
            gini_sum / total_sites, label="species signature Gini impurity"
        ),
        "species_effective_signature_excess": _bounded(
            effective_excess_sum / total_sites,
            label="effective signature excess",
        ),
    }
    return result


def aggregate_svtc_features(
    *,
    raw_sites: Sequence[SiteTopology],
    robust_sites: Sequence[SiteTopology],
    atomic_numbers: object,
) -> dict[str, float]:
    """Aggregate site topology into the frozen sixteen-feature schema."""

    if len(raw_sites) == 0 or len(raw_sites) != len(robust_sites):
        raise ValueError("NEXT251 aggregate topology population differs")
    numbers = np.asarray(atomic_numbers, dtype=float)
    if numbers.shape != (len(raw_sites),):
        raise ValueError("NEXT251 aggregate atomic-number population differs")
    features: dict[str, float] = {}
    for mode, sites in (("raw", raw_sites), ("robust", robust_sites)):
        odd = np.asarray([site.odd_area_fraction for site in sites], dtype=float)
        entropy = np.asarray([site.degree_entropy for site in sites], dtype=float)
        if (
            np.any(~np.isfinite(odd))
            or np.any(~np.isfinite(entropy))
            or np.any(odd < -NUMERICAL_TOLERANCE)
            or np.any(odd > 1.0 + NUMERICAL_TOLERANCE)
            or np.any(entropy < -NUMERICAL_TOLERANCE)
            or np.any(entropy > 1.0 + NUMERICAL_TOLERANCE)
        ):
            raise ValueError("NEXT251 aggregate site bounds differ")
        features[f"svtc_{mode}_odd_area_mean"] = float(np.mean(odd))
        features[f"svtc_{mode}_odd_area_q90"] = float(
            np.quantile(odd, 0.90, method="linear")
        )
        features[f"svtc_{mode}_degree_entropy_mean"] = float(np.mean(entropy))
        features[f"svtc_{mode}_degree_entropy_q90"] = float(
            np.quantile(entropy, 0.90, method="linear")
        )
        consistency = species_topology_consistency(
            signatures=[site.signature for site in sites],
            atomic_numbers=numbers,
        )
        for suffix, value in consistency.items():
            features[f"svtc_{mode}_{suffix}"] = value
    if tuple(features) != FEATURE_NAMES or not np.isfinite(list(features.values())).all():
        raise ValueError("NEXT251 aggregate feature schema differs")
    return features


def compute_svtc_features(atoms: Atoms) -> SVTCFeatureResult:
    """Compute frozen species-conditioned Voronoi topology from raw x0 geometry."""

    try:
        if (
            len(atoms) < 1
            or not np.all(atoms.pbc)
            or atoms.calc is not None
            or bool(atoms.info)
            or set(atoms.arrays) != {"numbers", "positions"}
        ):
            raise ValueError("SVTC features require exact periodic geometry-only Atoms")
        structure = AseAtomsAdaptor.get_structure(atoms)
        finder = VoronoiNN(weight="solid_angle", tol=0, cutoff=13)
        raw_sites: list[SiteTopology] = []
        robust_sites: list[SiteTopology] = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for center in range(len(structure)):
                info = finder.get_nn_info(structure, center)
                facets: dict[tuple[int, tuple[int, int, int]], tuple[float, int]] = {}
                for item in info:
                    try:
                        site_index = int(item["site_index"])
                        image = tuple(int(round(float(value))) for value in item["image"])
                        poly = item["poly_info"]
                        area = float(poly["area"])
                        degree = int(poly["n_verts"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    if (
                        len(image) != 3
                        or site_index < 0
                        or site_index >= len(structure)
                        or not math.isfinite(area)
                        or area <= 0.0
                        or degree < 3
                    ):
                        continue
                    key = (site_index, image)
                    candidate = (area, degree)
                    previous = facets.get(key)
                    if (
                        previous is None
                        or candidate[0] > previous[0]
                        or (candidate[0] == previous[0] and candidate[1] < previous[1])
                    ):
                        facets[key] = candidate
                keys = sorted(facets)
                if not keys:
                    raise ValueError("site has no valid Voronoi face")
                areas = np.asarray([facets[key][0] for key in keys], dtype=float)
                degrees = np.asarray([facets[key][1] for key in keys], dtype=int)
                raw_sites.append(face_topology(areas=areas, degrees=degrees, mode="raw"))
                robust_sites.append(
                    face_topology(areas=areas, degrees=degrees, mode="robust")
                )
        features = aggregate_svtc_features(
            raw_sites=raw_sites,
            robust_sites=robust_sites,
            atomic_numbers=atoms.numbers,
        )
        return SVTCFeatureResult(True, None, features)
    except Exception as exc:
        return _failure(exc)


def compute_svtc_row(atoms: Atoms) -> dict[str, object]:
    result = compute_svtc_features(atoms)
    row: dict[str, object] = {name: math.nan for name in FEATURE_NAMES}
    row["svtc_supported"] = bool(result.supported)
    row["svtc_failure"] = result.failure_reason
    if result.supported:
        row.update({name: float(result.features[name]) for name in FEATURE_NAMES})
    return row


def _error_row(exc: Exception) -> dict[str, object]:
    return {
        **{name: math.nan for name in FEATURE_NAMES},
        "svtc_supported": False,
        "svtc_failure": f"{type(exc).__name__}: {exc}",
    }


def _compute_scigen_payload(item: tuple[str, bytes]) -> tuple[str, dict[str, object]]:
    material_id, payload = item
    try:
        parsed = n85._parse_frame(payload, strict_output=True)
        return material_id, compute_svtc_row(parsed.atoms)
    except Exception as exc:
        return material_id, _error_row(exc)


def _compute_wyformer_payload(item: tuple[str, str]) -> tuple[str, dict[str, object]]:
    material_id, payload = item
    try:
        structure = Structure.from_dict(json.loads(payload))
        return material_id, compute_svtc_row(AseAtomsAdaptor.get_atoms(structure))
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


def build_cross_source_svtc_features(
    *,
    scigen_cohort_dir: Path,
    wyformer_cohort_dir: Path,
    parent_design_path: Path,
    first_amendment_path: Path,
    design_path: Path,
    output_dir: Path,
    workers: int = 16,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    """Build NEXT251 from physically isolated discovery geometry only."""

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
        "parent_design": Path(parent_design_path).resolve(),
        "first_amendment": Path(first_amendment_path).resolve(),
        "design": Path(design_path).resolve(),
    }
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if type(workers) is not int or workers <= 0:
        raise ValueError("NEXT251 workers must be a positive exact integer")
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT251 input is missing")
    input_hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if require_formal_inputs and input_hashes != EXPECTED_INPUT_SHA256:
        differing = sorted(
            name
            for name in set(input_hashes) | set(EXPECTED_INPUT_SHA256)
            if input_hashes.get(name) != EXPECTED_INPUT_SHA256.get(name)
        )
        raise ValueError(f"NEXT251 formal input identity differs: {differing}")
    repository = Path(__file__).resolve().parents[1]
    upstream_hashes = {
        name: _sha256_file(repository / name)
        for name in EXPECTED_UPSTREAM_SOURCE_SHA256
    }
    if require_formal_inputs and upstream_hashes != EXPECTED_UPSTREAM_SOURCE_SHA256:
        raise ValueError("NEXT251 frozen upstream source differs")
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
        raise ValueError("NEXT251 discovery geometry provenance differs")
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
            raise ValueError(f"NEXT251 {source} metadata differs")
        selected = frame.loc[frame["partition_role"].eq("discovery")].copy()
        selected = selected.sort_values("material_id", kind="mergesort").reset_index(drop=True)
        if selected.empty:
            raise ValueError(f"NEXT251 {source} discovery identity differs")
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
                [
                    {"material_id": material_id, **row}
                    for material_id, row in computed[source]
                ]
            )
            if (
                computed_frame["material_id"].astype(str).duplicated().any()
                or set(computed_frame["material_id"].astype(str))
                != set(discovery[source]["material_id"].astype(str))
            ):
                raise RuntimeError(f"NEXT251 {source} material identity differs")
            table = discovery[source].merge(
                computed_frame, on="material_id", how="left", validate="one_to_one"
            )
            if len(table) != len(discovery[source]):
                raise RuntimeError(f"NEXT251 {source} row accounting differs")
            supported = table["svtc_supported"].fillna(False).astype(bool)
            finite_counts = {
                name: int(np.isfinite(pd.to_numeric(table[name], errors="coerce")).sum())
                for name in FEATURE_NAMES
            }
            if not supported.all() or any(count != len(table) for count in finite_counts.values()):
                failures = Counter(table.loc[~supported, "svtc_failure"].astype(str))
                raise RuntimeError(
                    f"NEXT251 {source} formal support differs: {dict(sorted(failures.items()))}"
                )
            output = staging / FEATURE_FILES[source]
            table.to_parquet(output, index=False)
            output_paths.append(output)
            counts[source] = {
                "rows": int(len(table)),
                "supported": int(supported.sum()),
                "failures": int((~supported).sum()),
                "failure_counts": {},
                "finite_feature_counts": finite_counts,
            }
        if counts["scigen"]["rows"] != 13_470 or counts["wyformer"]["rows"] != 5_232:
            raise RuntimeError("NEXT251 frozen discovery row counts differ")
        catalogue = {
            "protocol": PROTOCOL,
            "feature_names": list(FEATURE_NAMES),
            "feature_count": len(FEATURE_NAMES),
            "topology_modes": list(TOPOLOGY_MODES),
            "face_degree_bins": ["3", "4", "5", "6", "7", "8", "9+"],
            "robust_area_fraction": ROBUST_AREA_FRACTION,
            "voronoi": {"weight": "solid_angle", "tol": 0, "cutoff": 13},
            "face_weight": "area_over_retained_site_total_area",
            "signature_conditioning": "atomic_number",
            "species_entropy_normalization": "log_observed_unique_signature_count",
            "species_dispersion": [
                "signature_gini_impurity",
                "one_minus_inverse_effective_signature_count",
            ],
            "quantile_method": "linear",
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
            "mode": "physically_isolated_discovery_x0_svtc_feature_freeze",
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
                "src/next251_species_voronoi_topology_consistency.py": source_hash
            },
            "outputs_sha256": {path.name: _sha256_file(path) for path in output_paths},
        }
        _write_json(staging / MANIFEST_NAME, manifest)
        if any(_sha256_file(path) != input_hashes[name] for name, path in paths.items()):
            raise RuntimeError("NEXT251 input changed before publication")
        if _sha256_file(source_path) != source_hash:
            raise RuntimeError("NEXT251 source changed before publication")
        os.replace(staging, target)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scigen-cohort-dir", type=Path, required=True)
    parser.add_argument("--wyformer-cohort-dir", type=Path, required=True)
    parser.add_argument("--parent-design-path", type=Path, required=True)
    parser.add_argument("--first-amendment-path", type=Path, required=True)
    parser.add_argument("--design-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--allow-nonformal-inputs", action="store_true")
    args = parser.parse_args()
    manifest = build_cross_source_svtc_features(
        scigen_cohort_dir=args.scigen_cohort_dir,
        wyformer_cohort_dir=args.wyformer_cohort_dir,
        parent_design_path=args.parent_design_path,
        first_amendment_path=args.first_amendment_path,
        design_path=args.design_path,
        output_dir=args.output_dir,
        workers=args.workers,
        require_formal_inputs=not args.allow_nonformal_inputs,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


__all__ = [
    "FEATURE_NAMES",
    "PROTOCOL",
    "ROBUST_AREA_FRACTION",
    "SVTCFeatureResult",
    "SiteTopology",
    "aggregate_svtc_features",
    "build_cross_source_svtc_features",
    "compute_svtc_features",
    "face_topology",
    "species_topology_consistency",
]


if __name__ == "__main__":
    raise SystemExit(main())
