#!/usr/bin/env python3
"""Infinite-periodic capacity ratio for raw radius-allocation redistribution."""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import time

from ase import Atoms
import numpy as np

import src.next267_periodic_radical_voronoi_packing as n267
import src.next331_radical_facet_minimum_participation as n331
import src.next339_periodic_geometric_homogenized_transmissivity as n339


PROTOCOL = "2026-08-13-next347-periodic-allocation-redistribution-capacity-v1"
MANIFEST_NAME = "MANIFEST.json"
CATALOGUE_NAME = "NEXT347_PARC_FEATURE_CATALOGUE.json"
FEATURE_FILES = {
    "scigen": "next347_scigen_periodic_allocation_redistribution_capacity.parquet",
    "wyformer": "next347_wyformer_periodic_allocation_redistribution_capacity.parquet",
}
FEATURE_NAMES = ("parc_allocation_redistribution_protection",)
FEATURE_DIRECTIONS = {FEATURE_NAMES[0]: "protected_high"}
EXPECTED_DESIGN_SHA256 = (
    "b6815af26a012ac27b04341dab03b26c0b0c78f5bb0a3b10feec7d2fc8d1e5e2"
)
EXPECTED_PROBE_SHA256 = (
    "0520bb7c824f7b9d2d4fba0ecc61eb3c5336db366f08c24e28b29063645cd851"
)
OUTPUT_GRID = 10_000_000_000
ALLOCATION_TOLERANCE = 1.0e-12
POISSON_RESIDUAL_TOLERANCE = 1.0e-8
ENERGY_TOLERANCE = 1.0e-12
BOUNDARY_FLAGS = dict(n331.BOUNDARY_FLAGS)
EXPECTED_ROWS = {"scigen": 13_470, "wyformer": 5_232}
MINIMUM_FORMAL_COVERAGE = 0.90
EXPECTED_INPUT_SHA256 = {
    "design": EXPECTED_DESIGN_SHA256,
    "probe_result": EXPECTED_PROBE_SHA256,
    "next267_source": "8f1e7ed9eb73a81a5755d455ffc05aab6f539cbd66afbbbfc384ca88391adca1",
    "next331_source": "08dfe2a8f2fc4f3d518e072ea2a81bd55661aaec49dcc23eb409e803b3786cf0",
    "next339_source": "bef33e641991f5b993d6ef77484c9c575b766e799d0ca09d562de648455a7bbe",
    "scigen_manifest": "dc5bf33c6ce6dc2c10bcd3704688055058145fbe7269ada23ffbe4b141d75fe7",
    "scigen_metadata": "f91455f23b0a96f60fd1c779249e2be46a7ecf94fcdde2b146426a95aac05bde",
    "scigen_discovery_geometry": "e561ef12343c66dcc72bcabf6b8719ad727e01c9582a094e281da73b862ab575",
    "wyformer_manifest": "e0539d556538cb4c052431bc6a1e5c1663bc3de427677dbc8a446dcc3b4fbc54",
    "wyformer_metadata": "3b152b4b84c8d3f7ff5e85611dc1fd2728296f150e907ac4578ce55d2b27dd2b",
    "wyformer_discovery_geometry": "f1ce5ae4fba8c13fcbf3e25de4f596b919d9b41da5b072d9a28eefeaffc69784",
}


@dataclass(frozen=True)
class AllocationCapacityResult:
    protection: float
    capacity_ratio: float
    global_energy: float
    capacity_energy: float
    allocation_total_variation: float
    maximum_poisson_residual: float
    periodic_capacities: tuple[float, ...]


@dataclass(frozen=True)
class PARCFeatureResult:
    supported: bool
    failure_reason: str | None
    site_count: int
    edge_count: int
    allocation_total_variation: float
    capacity_ratio: float
    minimum_periodic_capacity: float
    maximum_poisson_residual: float
    volume_tiling_relative_error: float
    features: dict[str, float]


def _failure(exc: Exception | str) -> PARCFeatureResult:
    reason = str(exc) if isinstance(exc, str) else f"{type(exc).__name__}: {exc}"
    return PARCFeatureResult(
        False, reason, 0, 0, math.nan, math.nan, math.nan, math.nan, math.nan, {}
    )


def _quantize(value: float) -> float:
    return float(np.rint(float(value) * OUTPUT_GRID) / OUTPUT_GRID)


def _validated_edges(
    *, site_count: int, endpoints: object, conductances: object
) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(site_count, (int, np.integer)) or int(site_count) < 1:
        raise ValueError("NEXT347 edge population differs")
    pair_raw = np.asarray(endpoints)
    conductance = np.asarray(conductances, dtype=float)
    if pair_raw.ndim != 2 or pair_raw.shape[1:] != (2,) or len(pair_raw) < 1:
        raise ValueError("NEXT347 edge population differs")
    try:
        pair_numeric = np.asarray(pair_raw, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("NEXT347 edge population differs") from exc
    if (
        not np.isfinite(pair_numeric).all()
        or not np.equal(pair_numeric, np.rint(pair_numeric)).all()
    ):
        raise ValueError("NEXT347 edge population differs")
    pair = pair_numeric.astype(int)
    if (
        conductance.shape != (len(pair),)
        or not np.isfinite(conductance).all()
        or np.any(conductance <= 0.0)
        or np.any(pair < 0)
        or np.any(pair >= int(site_count))
    ):
        raise ValueError("NEXT347 edge population differs")
    return pair, conductance


def periodic_incident_capacity(
    *, site_count: int, endpoints: object, conductances: object
) -> np.ndarray:
    """Return lifted periodic degree, counting each self-image orbit twice."""

    pair, conductance = _validated_edges(
        site_count=site_count, endpoints=endpoints, conductances=conductances
    )
    capacity = np.zeros(int(site_count), dtype=float)
    for (left, right), value in zip(pair, conductance, strict=True):
        if left == right:
            capacity[left] += 2.0 * value
        else:
            capacity[left] += value
            capacity[right] += value
    return capacity


def allocation_redistribution_capacity_protection(
    *,
    observed: object,
    target: object,
    endpoints: object,
    conductances: object,
) -> AllocationCapacityResult:
    """Compare periodic Green energy with the frozen incident-capacity reference."""

    actual = np.asarray(observed, dtype=float)
    desired = np.asarray(target, dtype=float)
    if (
        actual.ndim != 1
        or desired.shape != actual.shape
        or len(actual) < 1
        or not np.isfinite(actual).all()
        or not np.isfinite(desired).all()
        or np.any(actual < 0.0)
        or np.any(desired < 0.0)
        or not math.isclose(
            math.fsum(actual.tolist()), 1.0, abs_tol=ALLOCATION_TOLERANCE
        )
        or not math.isclose(
            math.fsum(desired.tolist()), 1.0, abs_tol=ALLOCATION_TOLERANCE
        )
    ):
        raise ValueError("NEXT347 allocation population differs")
    pair, conductance = _validated_edges(
        site_count=len(actual), endpoints=endpoints, conductances=conductances
    )
    capacity = periodic_incident_capacity(
        site_count=len(actual), endpoints=pair, conductances=conductance
    )

    source = actual - desired
    total_variation = 0.5 * math.fsum(np.abs(source).tolist())
    if float(np.max(np.abs(source))) <= ALLOCATION_TOLERANCE:
        return AllocationCapacityResult(
            1.0, 0.0, 0.0, 0.0, total_variation, 0.0, tuple(capacity.tolist())
        )
    if np.any((np.abs(source) > ALLOCATION_TOLERANCE) & (capacity <= 0.0)):
        raise ValueError("NEXT347 nonzero allocation source has zero periodic capacity")

    incidence = np.zeros((len(pair), len(actual)), dtype=float)
    rows = np.arange(len(pair), dtype=int)
    nonself = pair[:, 0] != pair[:, 1]
    incidence[rows[nonself], pair[nonself, 0]] = -1.0
    incidence[rows[nonself], pair[nonself, 1]] = 1.0
    laplacian = incidence.T @ (conductance[:, None] * incidence)
    potential = np.linalg.pinv(laplacian, rcond=1.0e-12, hermitian=True) @ source
    residual = laplacian @ potential - source
    residual_scale = max(1.0, float(np.max(np.abs(source))))
    maximum_residual = float(np.max(np.abs(residual)) / residual_scale)
    if not math.isfinite(maximum_residual) or maximum_residual > POISSON_RESIDUAL_TOLERANCE:
        raise ValueError(f"NEXT347 Poisson residual differs: {maximum_residual:.12g}")

    global_energy = float(source @ potential)
    capacity_energy = math.fsum(
        float(source[index]) ** 2 / float(capacity[index])
        for index in range(len(source))
        if abs(float(source[index])) > ALLOCATION_TOLERANCE
    )
    energy_scale = max(1.0, abs(global_energy), abs(capacity_energy))
    if (
        not math.isfinite(global_energy)
        or not math.isfinite(capacity_energy)
        or global_energy < -ENERGY_TOLERANCE * energy_scale
        or capacity_energy <= 0.0
    ):
        raise ValueError("NEXT347 redistribution energy differs")
    global_energy = max(0.0, global_energy)
    ratio = global_energy / capacity_energy
    protection = 1.0 / (1.0 + ratio)
    if (
        not math.isfinite(ratio)
        or ratio < 0.0
        or not math.isfinite(protection)
        or protection <= 0.0
        or protection > 1.0
    ):
        raise ValueError("NEXT347 redistribution capacity differs")
    return AllocationCapacityResult(
        protection,
        ratio,
        global_energy,
        capacity_energy,
        total_variation,
        maximum_residual,
        tuple(capacity.tolist()),
    )


def compute_parc_features(atoms: Atoms) -> PARCFeatureResult:
    try:
        work = n331._geometry_only_atoms(atoms)
        radii = np.asarray(
            [n267._tabulated_radius(str(symbol)) for symbol in work.get_chemical_symbols()],
            dtype=float,
        )
        graph = n339.periodic_power_facet_graph(work, radii=radii)
        cells = n267.periodic_radical_cells(work, radii=radii)
        if len(cells) != len(work) or any(cell.empty for cell in cells):
            raise ValueError("NEXT347 requires every labelled power cell to be nonempty")
        volume = abs(float(np.linalg.det(np.asarray(work.cell.array, dtype=float))))
        cell_volumes = np.asarray([cell.volume for cell in cells], dtype=float)
        tiling_error = abs(math.fsum(cell_volumes.tolist()) - volume) / volume
        if tiling_error > n267.VOLUME_TILING_RELATIVE_TOLERANCE:
            raise ValueError(f"NEXT347 volume-tiling certificate differs: {tiling_error:.12g}")
        observed = cell_volumes / volume
        target = radii**3 / math.fsum((radii**3).tolist())
        result = allocation_redistribution_capacity_protection(
            observed=observed,
            target=target,
            endpoints=graph.endpoints,
            conductances=graph.conductances,
        )
        value = _quantize(result.protection)
        if value <= 0.0:
            raise ValueError("NEXT347 positive protection quantized to zero")
        return PARCFeatureResult(
            True,
            None,
            len(work),
            len(graph.endpoints),
            result.allocation_total_variation,
            result.capacity_ratio,
            min(result.periodic_capacities),
            result.maximum_poisson_residual,
            max(tiling_error, graph.volume_tiling_relative_error),
            {FEATURE_NAMES[0]: value},
        )
    except Exception as exc:
        return _failure(exc)


def compute_parc_row(atoms: Atoms) -> dict[str, object]:
    result = compute_parc_features(atoms)
    row: dict[str, object] = {FEATURE_NAMES[0]: math.nan}
    row.update(
        {
            "parc_supported": bool(result.supported),
            "parc_failure": result.failure_reason,
            "parc_site_count": result.site_count,
            "parc_edge_count": result.edge_count,
            "parc_allocation_total_variation": result.allocation_total_variation,
            "parc_capacity_ratio": result.capacity_ratio,
            "parc_minimum_periodic_capacity": result.minimum_periodic_capacity,
            "parc_maximum_poisson_residual": result.maximum_poisson_residual,
            "parc_volume_tiling_relative_error": result.volume_tiling_relative_error,
        }
    )
    if result.supported:
        row[FEATURE_NAMES[0]] = float(result.features[FEATURE_NAMES[0]])
    return row


def _error_row(exc: Exception) -> dict[str, object]:
    row: dict[str, object] = {FEATURE_NAMES[0]: math.nan}
    row.update(
        {
            "parc_supported": False,
            "parc_failure": f"{type(exc).__name__}: {exc}",
            "parc_site_count": 0,
            "parc_edge_count": 0,
            "parc_allocation_total_variation": math.nan,
            "parc_capacity_ratio": math.nan,
            "parc_minimum_periodic_capacity": math.nan,
            "parc_maximum_poisson_residual": math.nan,
            "parc_volume_tiling_relative_error": math.nan,
        }
    )
    return row


def _compute_scigen_payload(
    item: tuple[str, bytes],
) -> tuple[str, dict[str, object]]:
    material_id, payload = item
    try:
        parsed = n267.n85._parse_frame(payload, strict_output=True)
        return material_id, compute_parc_row(parsed.atoms)
    except Exception as exc:
        return material_id, _error_row(exc)


def _compute_wyformer_payload(
    item: tuple[str, str],
) -> tuple[str, dict[str, object]]:
    material_id, payload = item
    try:
        structure = n267.Structure.from_dict(json.loads(payload))
        atoms = n267.AseAtomsAdaptor.get_atoms(structure)
        return material_id, compute_parc_row(atoms)
    except Exception as exc:
        return material_id, _error_row(exc)


def _compute_many(payloads, *, source: str, workers: int):
    worker = _compute_scigen_payload if source == "scigen" else _compute_wyformer_payload
    if workers == 1:
        return [worker(item) for item in payloads]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(worker, payloads, chunksize=4))


def _label_free_statistics(table) -> dict[str, object]:
    import pandas as pd

    statistics: dict[str, object] = {}
    for name in FEATURE_NAMES:
        values = pd.to_numeric(table[name], errors="coerce").dropna().to_numpy(float)
        if len(values) < 1:
            raise RuntimeError("NEXT347 label-free feature population is empty")
        statistics[name] = {
            "minimum": float(np.min(values)),
            "q10": float(np.quantile(values, 0.10, method="inverted_cdf")),
            "median": float(np.median(values)),
            "q90": float(np.quantile(values, 0.90, method="inverted_cdf")),
            "maximum": float(np.max(values)),
            "unique_rounded_10": int(np.unique(np.round(values, 10)).size),
        }
    return statistics


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    Path(path).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def build_cross_source_parc_features(
    *,
    scigen_cohort_dir: Path,
    wyformer_cohort_dir: Path,
    design_path: Path,
    probe_result_path: Path,
    output_dir: Path,
    workers: int = 16,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    """Build NEXT347 from physically isolated discovery geometry only."""

    import pandas as pd

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
        "probe_result": Path(probe_result_path).resolve(),
        "next267_source": Path(n267.__file__).resolve(),
        "next331_source": Path(n331.__file__).resolve(),
        "next339_source": Path(n339.__file__).resolve(),
    }
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if type(workers) is not int or workers <= 0:
        raise ValueError("NEXT347 workers must be a positive exact integer")
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT347 input is missing")
    input_hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if require_formal_inputs and input_hashes != EXPECTED_INPUT_SHA256:
        differing = sorted(
            name
            for name in set(input_hashes) | set(EXPECTED_INPUT_SHA256)
            if input_hashes.get(name) != EXPECTED_INPUT_SHA256.get(name)
        )
        raise ValueError(f"NEXT347 formal input identity differs: {differing}")

    with paths["probe_result"].open(encoding="utf-8") as handle:
        probe = json.load(handle)
    boundary_names = (
        "labels_opened",
        "endpoint_fields_read",
        "validation_geometry_opened",
        "replication_geometry_opened",
        "dft_calculation_executed",
        "dft_values_used",
        "learned_energy_force_stress_proxy_used",
        "model_or_proxy_potential_used",
        "physical_relaxation_executed",
    )
    if (
        probe.get("protocol") != "2026-08-13-next347-parc-label-blind-probe-v1"
        or probe.get("design_sha256") != EXPECTED_DESIGN_SHA256
        or probe.get("probe_passed") is not True
        or set(probe.get("gates", {}).values()) != {True}
        or any(probe.get(name) is not False for name in boundary_names)
    ):
        raise ValueError("NEXT347 label-blind probe authorization differs")

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
        raise ValueError("NEXT347 discovery geometry provenance differs")

    metadata = {
        "scigen": pd.read_parquet(paths["scigen_metadata"]),
        "wyformer": pd.read_parquet(paths["wyformer_metadata"]),
    }
    discovery: dict[str, object] = {}
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
            raise ValueError(f"NEXT347 {source} metadata differs")
        selected = frame.loc[frame["partition_role"].eq("discovery")].copy()
        selected = selected.sort_values("material_id", kind="mergesort").reset_index(drop=True)
        if len(selected) != EXPECTED_ROWS[source]:
            raise ValueError(f"NEXT347 {source} discovery identity differs")
        discovery[source] = selected
    payloads = {
        "scigen": n267.n85._archive_payloads(
            paths["scigen_discovery_geometry"],
            discovery["scigen"]["material_id"].astype(str).tolist(),
        ),
        "wyformer": n267.n94._payloads(
            paths["wyformer_discovery_geometry"],
            discovery["wyformer"]["material_id"].astype(str).tolist(),
        ),
    }

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    source_path = Path(__file__).resolve()
    root = source_path.parent.parent
    executed_paths = {
        "src/next347_periodic_allocation_redistribution_capacity.py": source_path,
        "experiments/next347_parc_label_blind_probe.py": root / "experiments/next347_parc_label_blind_probe.py",
        "tests/test_next347_periodic_allocation_redistribution_capacity.py": root / "tests/test_next347_periodic_allocation_redistribution_capacity.py",
        "tests/test_next347_parc_label_blind_probe.py": root / "tests/test_next347_parc_label_blind_probe.py",
    }
    executed_hashes = {name: _sha256_file(path) for name, path in executed_paths.items()}
    started = time.perf_counter()
    try:
        computed = {
            source: _compute_many(payloads[source], source=source, workers=workers)
            for source in ("scigen", "wyformer")
        }
        counts: dict[str, object] = {}
        statistics: dict[str, object] = {}
        outputs: list[Path] = []
        for source in ("scigen", "wyformer"):
            computed_frame = pd.DataFrame(
                [{"material_id": material_id, **row} for material_id, row in computed[source]]
            )
            if (
                computed_frame["material_id"].astype(str).duplicated().any()
                or set(computed_frame["material_id"].astype(str))
                != set(discovery[source]["material_id"].astype(str))
            ):
                raise RuntimeError(f"NEXT347 {source} material identity differs")
            table = discovery[source].merge(
                computed_frame, on="material_id", how="left", validate="one_to_one"
            )
            supported = table["parc_supported"].fillna(False).astype(bool)
            finite = np.isfinite(
                pd.to_numeric(table[FEATURE_NAMES[0]], errors="coerce").to_numpy(float)
            )
            if not finite[supported].all() or finite[~supported].any():
                raise RuntimeError(f"NEXT347 {source} finite support semantics differ")
            coverage = float(supported.mean())
            if coverage < MINIMUM_FORMAL_COVERAGE:
                raise RuntimeError(f"NEXT347 {source} coverage below frozen minimum")
            source_statistics = _label_free_statistics(table)
            if source_statistics[FEATURE_NAMES[0]]["unique_rounded_10"] < 20:
                raise RuntimeError(f"NEXT347 {source} feature is degenerate")
            statistics[source] = source_statistics
            failures = Counter(table.loc[~supported, "parc_failure"].fillna("unknown"))
            finite_residual = pd.to_numeric(
                table.loc[supported, "parc_maximum_poisson_residual"], errors="coerce"
            ).to_numpy(float)
            counts[source] = {
                "rows": int(len(table)),
                "supported": int(supported.sum()),
                "failures": int((~supported).sum()),
                "coverage": coverage,
                "site_count": int(pd.to_numeric(table.loc[supported, "parc_site_count"]).sum()),
                "edge_count": int(pd.to_numeric(table.loc[supported, "parc_edge_count"]).sum()),
                "maximum_finite_poisson_residual": float(
                    np.max(finite_residual[np.isfinite(finite_residual)])
                ) if np.isfinite(finite_residual).any() else math.nan,
                "finite_feature_count": int(finite.sum()),
                "failure_counts": {str(key): int(value) for key, value in failures.items()},
            }
            output_path = staging / FEATURE_FILES[source]
            table.to_parquet(output_path, index=False)
            outputs.append(output_path)

        catalogue = {
            "protocol": PROTOCOL,
            "feature_count": 1,
            "feature_names": list(FEATURE_NAMES),
            "feature_directions": FEATURE_DIRECTIONS,
            "directions_frozen_before_outcome": True,
            "graph": "NEXT339 reciprocal A/d periodic power-facet graph",
            "source": "power-cell volume share minus tabulated-radius cubed share",
            "formula": "1/(1+(b.T L^+ b)/sum_i(b_i^2/c_i))",
            "periodic_capacity": "nonself endpoint g plus twice each self-image orbit g",
            "output_grid": 1.0 / OUTPUT_GRID,
            "minimum_formal_coverage": MINIMUM_FORMAL_COVERAGE,
            "label_free_statistics": statistics,
            "next348_audit_authorized": True,
            **BOUNDARY_FLAGS,
        }
        catalogue_path = staging / CATALOGUE_NAME
        _write_json(catalogue_path, catalogue)
        outputs.append(catalogue_path)
        manifest = {
            "protocol": PROTOCOL,
            "mode": "physically_isolated_discovery_x0_periodic_allocation_redistribution_capacity_freeze",
            "workers": workers,
            "elapsed_seconds": float(time.perf_counter() - started),
            "counts": counts,
            "source_partitions_read": {"scigen": ["discovery"], "wyformer": ["discovery"]},
            "labels_opened": False,
            "endpoint_fields_read": False,
            "internal_validation_geometry_opened": False,
            "internal_replication_geometry_opened": False,
            "scientific_improvement_claim": False,
            "next348_audit_authorized": True,
            **BOUNDARY_FLAGS,
            "inputs_sha256": input_hashes,
            "executed_source_sha256": executed_hashes,
            "outputs_sha256": {path.name: _sha256_file(path) for path in outputs},
        }
        manifest_path = staging / MANIFEST_NAME
        _write_json(manifest_path, manifest)
        if any(_sha256_file(path) != executed_hashes[name] for name, path in executed_paths.items()):
            raise RuntimeError("NEXT347 executed artifact changed before publication")
        os.replace(staging, target)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


__all__ = [
    "BOUNDARY_FLAGS",
    "CATALOGUE_NAME",
    "EXPECTED_ROWS",
    "FEATURE_DIRECTIONS",
    "FEATURE_FILES",
    "FEATURE_NAMES",
    "MANIFEST_NAME",
    "MINIMUM_FORMAL_COVERAGE",
    "POISSON_RESIDUAL_TOLERANCE",
    "PROTOCOL",
    "AllocationCapacityResult",
    "PARCFeatureResult",
    "allocation_redistribution_capacity_protection",
    "build_cross_source_parc_features",
    "compute_parc_features",
    "compute_parc_row",
    "periodic_incident_capacity",
]
