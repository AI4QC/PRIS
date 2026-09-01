#!/usr/bin/env python3
"""Materialize species-resolved periodic radical-packing descriptors."""

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

import numpy as np
import pandas as pd

import src.next267_periodic_radical_voronoi_packing as n267
from src.next102_cross_source_dobvr_features import _sha256_file, _write_json


PROTOCOL = "2026-08-09-next275-species-resolved-radical-packing-v1"
MANIFEST_NAME = "MANIFEST.json"
CATALOGUE_NAME = "NEXT275_PRVS_FEATURE_CATALOGUE.json"
FEATURE_FILES = {
    "scigen": "next275_scigen_species_resolved_radical_features.parquet",
    "wyformer": "next275_wyformer_species_resolved_radical_features.parquet",
}
FEATURE_NAMES = (
    "prvs_volume_within_cv",
    "prvs_volume_between_cv",
    "prvs_volume_within_variance_fraction",
    "prvs_volume_weighted_species_cv",
    "prvs_volume_max_species_cv",
    "prvs_chebyshev_within_cv",
    "prvs_chebyshev_between_cv",
    "prvs_chebyshev_within_variance_fraction",
    "prvs_chebyshev_weighted_species_cv",
    "prvs_chebyshev_max_species_cv",
)
EXPECTED_DESIGN_SHA256 = (
    "ce27df015380f6a6011c03c664525166d4b33701ff9dde5f4ea969ded1a4b3c0"
)
EXPECTED_INPUT_SHA256 = {
    **{key: value for key, value in n267.EXPECTED_INPUT_SHA256.items() if key != "design"},
    "design": EXPECTED_DESIGN_SHA256,
}
EXPECTED_NEXT267_SOURCE_SHA256 = (
    "8f1e7ed9eb73a81a5755d455ffc05aab6f539cbd66afbbbfc384ca88391adca1"
)
BOUNDARY_FLAGS = n267.BOUNDARY_FLAGS
OUTPUT_GRID = n267.OUTPUT_GRID


@dataclass(frozen=True)
class PRVSFeatureResult:
    supported: bool
    failure_reason: str | None
    site_count: int
    empty_cell_count: int
    min_facet_count: int
    max_facet_count: int
    volume_tiling_relative_error: float
    features: Mapping[str, float]


def _quantize(value: float) -> float:
    return float(np.rint(float(value) * OUTPUT_GRID) / OUTPUT_GRID)


def species_variance_features(
    *, values: object, species: Sequence[str]
) -> dict[str, float]:
    """Return the frozen exact within/between species decomposition."""

    array = np.asarray(values, dtype=float)
    labels = np.asarray([str(value) for value in species], dtype=object)
    if (
        array.ndim != 1
        or labels.ndim != 1
        or len(array) == 0
        or len(array) != len(labels)
        or not np.isfinite(array).all()
        or np.any(array <= 0.0)
        or any(not label for label in labels)
    ):
        raise ValueError("NEXT275 species population differs")
    center = float(math.fsum(float(value) for value in array) / len(array))
    within_ss = 0.0
    between_ss = 0.0
    weighted_species_cv = 0.0
    max_species_cv = 0.0
    for label in sorted(set(labels.tolist())):
        group = array[labels == label]
        group_center = float(math.fsum(float(value) for value in group) / len(group))
        group_ss = math.fsum((float(value) - group_center) ** 2 for value in group)
        within_ss += group_ss
        between_ss += len(group) * (group_center - center) ** 2
        group_cv = 0.0 if len(group) == 1 else math.sqrt(group_ss / len(group)) / group_center
        weighted_species_cv += len(group) / len(array) * group_cv
        max_species_cv = max(max_species_cv, group_cv)
    total_ss = within_ss + between_ss
    result = {
        "within_cv": math.sqrt(max(0.0, within_ss) / len(array)) / center,
        "between_cv": math.sqrt(max(0.0, between_ss) / len(array)) / center,
        "within_variance_fraction": 0.0 if total_ss == 0.0 else within_ss / total_ss,
        "weighted_species_cv": weighted_species_cv,
        "max_species_cv": max_species_cv,
    }
    if not np.isfinite(list(result.values())).all():
        raise RuntimeError("NEXT275 species variance result is non-finite")
    return result


def _failure(exc: Exception) -> PRVSFeatureResult:
    return PRVSFeatureResult(
        False,
        f"{type(exc).__name__}: {exc}",
        0,
        0,
        0,
        0,
        math.nan,
        {name: math.nan for name in FEATURE_NAMES},
    )


def compute_species_resolved_prv_features(atoms) -> PRVSFeatureResult:
    """Compute the frozen species-resolved PRV summaries without labels."""

    try:
        symbols = np.asarray(atoms.get_chemical_symbols(), dtype=object)
        radii = np.asarray([n267._tabulated_radius(str(value)) for value in symbols])
        cells = n267.periodic_radical_cells(atoms, radii=radii)
        volume = abs(float(np.linalg.det(np.asarray(atoms.cell.array, dtype=float))))
        cell_volumes = np.asarray([value.volume for value in cells], dtype=float)
        relative_error = abs(float(cell_volumes.sum()) - volume) / volume
        empty = np.asarray([value.empty for value in cells], dtype=bool)
        nonempty = ~empty
        if not nonempty.any():
            raise ValueError("NEXT275 structure has no positive-volume labelled cell")
        volume_ratio = cell_volumes[nonempty] / (
            (4.0 * math.pi / 3.0) * radii[nonempty] ** 3
        )
        chebyshev_ratio = np.asarray(
            [value.chebyshev_radius for value in cells], dtype=float
        )[nonempty] / radii[nonempty]
        kept_species = symbols[nonempty].astype(str).tolist()
        volume_stats = species_variance_features(
            values=volume_ratio, species=kept_species
        )
        chebyshev_stats = species_variance_features(
            values=chebyshev_ratio, species=kept_species
        )
        features = {
            "prvs_volume_within_cv": _quantize(volume_stats["within_cv"]),
            "prvs_volume_between_cv": _quantize(volume_stats["between_cv"]),
            "prvs_volume_within_variance_fraction": _quantize(
                volume_stats["within_variance_fraction"]
            ),
            "prvs_volume_weighted_species_cv": _quantize(
                volume_stats["weighted_species_cv"]
            ),
            "prvs_volume_max_species_cv": _quantize(volume_stats["max_species_cv"]),
            "prvs_chebyshev_within_cv": _quantize(chebyshev_stats["within_cv"]),
            "prvs_chebyshev_between_cv": _quantize(chebyshev_stats["between_cv"]),
            "prvs_chebyshev_within_variance_fraction": _quantize(
                chebyshev_stats["within_variance_fraction"]
            ),
            "prvs_chebyshev_weighted_species_cv": _quantize(
                chebyshev_stats["weighted_species_cv"]
            ),
            "prvs_chebyshev_max_species_cv": _quantize(
                chebyshev_stats["max_species_cv"]
            ),
        }
        if tuple(features) != FEATURE_NAMES or not np.isfinite(list(features.values())).all():
            raise ValueError("NEXT275 feature schema or values differ")
        facets = np.asarray([value.facet_count for value in cells if not value.empty])
        return PRVSFeatureResult(
            True,
            None,
            len(cells),
            int(empty.sum()),
            int(facets.min()),
            int(facets.max()),
            float(relative_error),
            features,
        )
    except Exception as exc:
        return _failure(exc)


def compute_prvs_row(atoms) -> dict[str, object]:
    result = compute_species_resolved_prv_features(atoms)
    row: dict[str, object] = {name: math.nan for name in FEATURE_NAMES}
    row.update(
        {
            "prvs_supported": bool(result.supported),
            "prvs_failure": result.failure_reason,
            "prvs_site_count": result.site_count,
            "prvs_empty_cell_count": result.empty_cell_count,
            "prvs_min_facet_count": result.min_facet_count,
            "prvs_max_facet_count": result.max_facet_count,
            "prvs_volume_tiling_relative_error": result.volume_tiling_relative_error,
        }
    )
    if result.supported:
        row.update({name: float(result.features[name]) for name in FEATURE_NAMES})
    return row


def _error_row(exc: Exception) -> dict[str, object]:
    return {
        **{name: math.nan for name in FEATURE_NAMES},
        "prvs_supported": False,
        "prvs_failure": f"{type(exc).__name__}: {exc}",
        "prvs_site_count": 0,
        "prvs_empty_cell_count": 0,
        "prvs_min_facet_count": 0,
        "prvs_max_facet_count": 0,
        "prvs_volume_tiling_relative_error": math.nan,
    }


def _compute_scigen_payload(item: tuple[str, bytes]):
    material_id, payload = item
    try:
        parsed = n267.n85._parse_frame(payload, strict_output=True)
        return material_id, compute_prvs_row(parsed.atoms)
    except Exception as exc:
        return material_id, _error_row(exc)


def _compute_wyformer_payload(item: tuple[str, str]):
    material_id, payload = item
    try:
        structure = n267.Structure.from_dict(json.loads(payload))
        atoms = n267.AseAtomsAdaptor.get_atoms(structure)
        return material_id, compute_prvs_row(atoms)
    except Exception as exc:
        return material_id, _error_row(exc)


def _compute_many(payloads, *, source: str, workers: int):
    worker = _compute_scigen_payload if source == "scigen" else _compute_wyformer_payload
    if workers == 1:
        return [worker(item) for item in payloads]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(worker, payloads, chunksize=4))


def build_species_resolved_prv_features(
    *,
    scigen_cohort_dir: Path,
    wyformer_cohort_dir: Path,
    design_path: Path,
    output_dir: Path,
    workers: int = 16,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    """Build NEXT275 from physically isolated discovery geometry only."""

    scigen = Path(scigen_cohort_dir).resolve()
    wyformer = Path(wyformer_cohort_dir).resolve()
    target = Path(output_dir).resolve()
    paths = {
        "scigen_manifest": scigen / n267.n85.COHORT_MANIFEST_NAME,
        "scigen_metadata": scigen / n267.n85.COHORT_METADATA_NAME,
        "scigen_discovery_geometry": scigen / n267.n85.GEOMETRY_NAMES["discovery"],
        "wyformer_manifest": wyformer / n267.n94.COHORT_MANIFEST_NAME,
        "wyformer_metadata": wyformer / n267.n94.COHORT_METADATA_NAME,
        "wyformer_discovery_geometry": wyformer / n267.n94.GEOMETRY_NAMES["discovery"],
        "design": Path(design_path).resolve(),
    }
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if type(workers) is not int or workers <= 0:
        raise ValueError("NEXT275 workers must be a positive exact integer")
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT275 input is missing")
    input_hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if require_formal_inputs and input_hashes != EXPECTED_INPUT_SHA256:
        differing = sorted(
            name for name in set(input_hashes) | set(EXPECTED_INPUT_SHA256)
            if input_hashes.get(name) != EXPECTED_INPUT_SHA256.get(name)
        )
        raise ValueError(f"NEXT275 formal input identity differs: {differing}")
    if _sha256_file(Path(n267.__file__).resolve()) != EXPECTED_NEXT267_SOURCE_SHA256:
        raise ValueError("NEXT275 frozen NEXT267 source differs")
    scigen_manifest = n267._read_manifest(paths["scigen_manifest"])
    wyformer_manifest = n267._read_manifest(paths["wyformer_manifest"])
    if (
        scigen_manifest.get("protocol") != n267.n85.COHORT_PROTOCOL
        or scigen_manifest.get("labels_opened") is not False
        or scigen_manifest.get("endpoint_payloads_opened") is not False
        or scigen_manifest.get("relaxed_structures_opened") is not False
        or wyformer_manifest.get("protocol") != n267.n94.COHORT_PROTOCOL
        or wyformer_manifest.get("discovery_endpoint_opened") is not False
        or wyformer_manifest.get("validation_endpoint_opened") is not False
        or wyformer_manifest.get("replication_endpoint_opened") is not False
        or wyformer_manifest.get("relaxed_structures_published") is not False
    ):
        raise ValueError("NEXT275 discovery geometry provenance differs")
    metadata = {
        "scigen": pd.read_parquet(paths["scigen_metadata"]),
        "wyformer": pd.read_parquet(paths["wyformer_metadata"]),
    }
    discovery: dict[str, pd.DataFrame] = {}
    for source, frame in metadata.items():
        required = {"material_id", "reduced_formula", "chemical_system", "natoms", "partition_role", "input_role"}
        if required - set(frame.columns) or frame["material_id"].astype(str).duplicated().any():
            raise ValueError(f"NEXT275 {source} metadata differs")
        selected = frame.loc[frame["partition_role"].eq("discovery")].copy()
        selected = selected.sort_values("material_id", kind="mergesort").reset_index(drop=True)
        if selected.empty:
            raise ValueError(f"NEXT275 {source} discovery identity differs")
        discovery[source] = selected
    payloads = {
        "scigen": n267.n85._archive_payloads(
            paths["scigen_discovery_geometry"], discovery["scigen"]["material_id"].astype(str).tolist()
        ),
        "wyformer": n267.n94._payloads(
            paths["wyformer_discovery_geometry"], discovery["wyformer"]["material_id"].astype(str).tolist()
        ),
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
        counts: dict[str, object] = {}
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
                raise RuntimeError(f"NEXT275 {source} material identity differs")
            table = discovery[source].merge(computed_frame, on="material_id", how="left", validate="one_to_one")
            supported = table["prvs_supported"].fillna(False).astype(bool)
            finite = np.column_stack([
                np.isfinite(pd.to_numeric(table[name], errors="coerce").to_numpy(float))
                for name in FEATURE_NAMES
            ])
            tiling = pd.to_numeric(table["prvs_volume_tiling_relative_error"], errors="coerce")
            sites = pd.to_numeric(table["prvs_site_count"], errors="coerce")
            minimum = pd.to_numeric(table["prvs_min_facet_count"], errors="coerce")
            maximum = pd.to_numeric(table["prvs_max_facet_count"], errors="coerce")
            if (
                len(table) != len(discovery[source])
                or not finite[supported].all()
                or finite[~supported].any()
                or not (sites[supported] > 0).all()
                or not (minimum[supported] >= 4).all()
                or not (maximum[supported] >= minimum[supported]).all()
                or not (tiling[supported] <= n267.VOLUME_TILING_RELATIVE_TOLERANCE).all()
            ):
                raise RuntimeError(f"NEXT275 {source} support certificate differs")
            output = staging / FEATURE_FILES[source]
            table.to_parquet(output, index=False)
            output_paths.append(output)
            failures = Counter(table.loc[~supported, "prvs_failure"].astype(str))
            counts[source] = {
                "rows": int(len(table)),
                "supported": int(supported.sum()),
                "failures": int((~supported).sum()),
                "failure_counts": dict(sorted(failures.items())),
                "finite_feature_counts": {name: int(finite[:, index].sum()) for index, name in enumerate(FEATURE_NAMES)},
                "site_count": int(sites[supported].sum()),
                "empty_cell_count": int(pd.to_numeric(table.loc[supported, "prvs_empty_cell_count"]).sum()),
                "maximum_volume_tiling_relative_error": float(tiling[supported].max()),
            }
        if counts["scigen"]["rows"] != 13_470 or counts["wyformer"]["rows"] != 5_232:  # type: ignore[index]
            raise RuntimeError("NEXT275 frozen discovery row counts differ")
        catalogue = {
            "protocol": PROTOCOL,
            "feature_names": list(FEATURE_NAMES),
            "feature_count": len(FEATURE_NAMES),
            "base_geometry_protocol": n267.PROTOCOL,
            "decomposition": "exact_population_within_between_species_variance",
            "singleton_species_cv": 0.0,
            "radius_policy": ["atomic_radius_calculated", "atomic_radius_fallback"],
            "power_distance": "squared_euclidean_minus_radius_squared",
            "output_grid": OUTPUT_GRID,
            "volume_tiling_relative_tolerance": n267.VOLUME_TILING_RELATIVE_TOLERANCE,
            "source_partitions_read": {"scigen": ["discovery"], "wyformer": ["discovery"]},
            "endpoint_columns_present": False,
            "labels_opened": False,
        }
        catalogue_path = staging / CATALOGUE_NAME
        _write_json(catalogue_path, catalogue)
        output_paths.append(catalogue_path)
        manifest = {
            "protocol": PROTOCOL,
            "mode": "physically_isolated_discovery_x0_species_resolved_radical_feature_freeze",
            "workers": workers,
            "elapsed_seconds": time.perf_counter() - started,
            "counts": counts,
            "next276_audit_authorized": True,
            "source_partitions_read": {"scigen": ["discovery"], "wyformer": ["discovery"]},
            "labels_opened": False,
            "endpoint_fields_read": False,
            "internal_validation_geometry_opened": False,
            "internal_replication_geometry_opened": False,
            **BOUNDARY_FLAGS,
            "scientific_improvement_claim": False,
            "inputs_sha256": input_hashes,
            "upstream_source_sha256": {"src/next267_periodic_radical_voronoi_packing.py": EXPECTED_NEXT267_SOURCE_SHA256},
            "executed_source_sha256": {"src/next275_species_resolved_radical_packing.py": source_hash},
            "outputs_sha256": {path.name: _sha256_file(path) for path in output_paths},
        }
        _write_json(staging / MANIFEST_NAME, manifest)
        if any(_sha256_file(path) != input_hashes[name] for name, path in paths.items()):
            raise RuntimeError("NEXT275 input changed before publication")
        if _sha256_file(source_path) != source_hash:
            raise RuntimeError("NEXT275 source changed before publication")
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
    manifest = build_species_resolved_prv_features(
        scigen_cohort_dir=args.scigen_cohort_dir,
        wyformer_cohort_dir=args.wyformer_cohort_dir,
        design_path=args.design_path,
        output_dir=args.output_dir,
        workers=args.workers,
        require_formal_inputs=not args.allow_nonformal_inputs,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
