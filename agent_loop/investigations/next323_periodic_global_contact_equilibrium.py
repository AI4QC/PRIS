"""Prospective no-DFT periodic global contact-equilibrium certificate."""

from __future__ import annotations

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
from typing import Sequence

import numpy as np
from ase import Atoms
import pandas as pd
from scipy.optimize import linprog

import src.next267_periodic_radical_voronoi_packing as n267
import src.next279_radical_packing_autocorrelation as n279
import src.next295_positive_contact_force_closure as n295
import src.next319_periodic_contact_shell_neutralization as n319


PROTOCOL = "2026-08-09-next323-periodic-global-contact-equilibrium-v1"
MANIFEST_NAME = "MANIFEST.json"
CATALOGUE_NAME = "NEXT323_PGCE_FEATURE_CATALOGUE.json"
FEATURE_FILES = {
    "scigen": "next323_scigen_periodic_global_contact_equilibrium.parquet",
    "wyformer": "next323_wyformer_periodic_global_contact_equilibrium.parquet",
}
FEATURE_NAMES = ("pgce_all_facet_participation_floor",)
FEATURE_DIRECTIONS = {FEATURE_NAMES[0]: "protected_high"}
EXPECTED_DESIGN_SHA256 = (
    "2b0398622aa1cded50bef87ff22f6a5301df05f39517fdc9a66c0ad3f7103567"
)
RECIPROCAL_VECTOR_TOLERANCE = 1.0e-8
LP_FEASIBILITY_TOLERANCE = 1.0e-10
PUBLICATION_RESIDUAL_TOLERANCE = 1.0e-9
QUANTIZATION_DECIMALS = 10
EXPECTED_ROWS = {"scigen": 13_470, "wyformer": 5_232}
MINIMUM_FORMAL_COVERAGE = 0.90
EXPECTED_INPUT_SHA256 = {
    "design": EXPECTED_DESIGN_SHA256,
    "probe_result": "7e5de860c34e21de5949065e7dc90a802c8f33addf035fa782e32bb700b0700d",
    "next267_source": "8f1e7ed9eb73a81a5755d455ffc05aab6f539cbd66afbbbfc384ca88391adca1",
    "next279_source": "b224e46424de199122b8f90061612edfcfd0318fb85178d404b6119f967df5c2",
    "next295_source": "4b92811e7f3c7ac60c1506104a18d2bd9d0fe06c6202e7f34cb996b32cd649a3",
    "next319_source": "cac92a55bef6991cc3a7e6fe650b3df79a7cebffedef7a9499bc77fe01745ed5",
    "scigen_manifest": "dc5bf33c6ce6dc2c10bcd3704688055058145fbe7269ada23ffbe4b141d75fe7",
    "scigen_metadata": "f91455f23b0a96f60fd1c779249e2be46a7ecf94fcdde2b146426a95aac05bde",
    "scigen_discovery_geometry": "e561ef12343c66dcc72bcabf6b8719ad727e01c9582a094e281da73b862ab575",
    "wyformer_manifest": "e0539d556538cb4c052431bc6a1e5c1663bc3de427677dbc8a446dcc3b4fbc54",
    "wyformer_metadata": "3b152b4b84c8d3f7ff5e85611dc1fd2728296f150e907ac4578ce55d2b27dd2b",
    "wyformer_discovery_geometry": "f1ce5ae4fba8c13fcbf3e25de4f596b919d9b41da5b072d9a28eefeaffc69784",
}
BOUNDARY_FLAGS = {
    "dft_calculation_executed": False,
    "dft_values_used_by_executable_formula": False,
    "learned_energy_force_stress_proxy_used": False,
    "model_or_proxy_potential_used": False,
    "physical_relaxation_executed": False,
}


@dataclass(frozen=True)
class PGCEKernelResult:
    supported: bool
    error: str | None
    participation_floor: float
    maximum_equilibrium_residual: float
    reciprocal_pair_count: int
    directed_contact_count: int


@dataclass(frozen=True)
class PGCEFeatureResult:
    supported: bool
    failure_reason: str | None
    site_count: int
    directed_contact_count: int
    reciprocal_pair_count: int
    maximum_equilibrium_residual: float
    features: dict[str, float]


def _integer_array(values: object, *, shape: tuple[int, ...], label: str) -> np.ndarray:
    raw = np.asarray(values)
    if raw.shape != shape:
        raise ValueError(f"PGCE {label} shape differs")
    try:
        numeric = np.asarray(raw, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"PGCE {label} differs") from exc
    if not np.isfinite(numeric).all() or not np.equal(numeric, np.rint(numeric)).all():
        raise ValueError(f"PGCE {label} must be finite integers")
    return np.rint(numeric).astype(np.int64)


def _reciprocal_pairs(
    *,
    centers: np.ndarray,
    neighbors: np.ndarray,
    translations: np.ndarray,
    vectors: np.ndarray,
) -> tuple[tuple[int, int], ...]:
    keys = [
        (
            int(centers[index]),
            int(neighbors[index]),
            int(translations[index, 0]),
            int(translations[index, 1]),
            int(translations[index, 2]),
        )
        for index in range(len(centers))
    ]
    if len(set(keys)) != len(keys):
        raise ValueError("PGCE reciprocal incidences must be unique")
    lookup = {key: index for index, key in enumerate(keys)}
    pairs: list[tuple[int, int]] = []
    for index, key in enumerate(keys):
        reverse_key = (key[1], key[0], -key[2], -key[3], -key[4])
        reverse = lookup.get(reverse_key)
        if reverse is None or reverse == index:
            raise ValueError("PGCE reciprocal incidence is missing")
        scale = max(
            1.0,
            float(np.linalg.norm(vectors[index])),
            float(np.linalg.norm(vectors[reverse])),
        )
        if float(np.max(np.abs(vectors[index] + vectors[reverse]))) > (
            RECIPROCAL_VECTOR_TOLERANCE * scale
        ):
            raise ValueError("PGCE reciprocal vectors are not opposite")
        if index < reverse:
            pairs.append((index, reverse))
    if 2 * len(pairs) != len(keys):
        raise ValueError("PGCE reciprocal pairing differs")
    return tuple(pairs)


def _zero_result(*, pair_count: int, directed_count: int) -> PGCEKernelResult:
    return PGCEKernelResult(
        True,
        None,
        0.0,
        math.nan,
        pair_count,
        directed_count,
    )


def positive_contact_equilibrium_floor(
    *,
    n_sites: int,
    centers: Sequence[int] | np.ndarray,
    neighbors: Sequence[int] | np.ndarray,
    translations: Sequence[Sequence[int]] | np.ndarray,
    vectors: Sequence[Sequence[float]] | np.ndarray,
) -> PGCEKernelResult:
    """Return the frozen all-incidence positive-equilibrium participation floor."""

    if not isinstance(n_sites, (int, np.integer)) or int(n_sites) < 1:
        raise ValueError("PGCE n_sites differs")
    n_sites = int(n_sites)
    raw_centers = np.asarray(centers)
    if raw_centers.ndim != 1 or len(raw_centers) < 2:
        raise ValueError("PGCE directed contact population differs")
    count = len(raw_centers)
    center = _integer_array(raw_centers, shape=(count,), label="centers")
    neighbor = _integer_array(neighbors, shape=(count,), label="neighbors")
    translation = _integer_array(
        translations, shape=(count, 3), label="translations"
    )
    vector = np.asarray(vectors, dtype=float)
    if vector.shape != (count, 3) or not np.isfinite(vector).all():
        raise ValueError("PGCE vectors differ")
    if np.any(center < 0) or np.any(center >= n_sites):
        raise ValueError("PGCE centers differ")
    if np.any(neighbor < 0) or np.any(neighbor >= n_sites):
        raise ValueError("PGCE neighbors differ")
    lengths = np.linalg.norm(vector, axis=1)
    if not np.isfinite(lengths).all() or np.any(lengths <= 0.0):
        raise ValueError("PGCE vectors must be nonzero")
    pairs = _reciprocal_pairs(
        centers=center,
        neighbors=neighbor,
        translations=translation,
        vectors=vector,
    )

    direction = vector / lengths[:, None]
    atomic = np.zeros((3 * n_sites, count), dtype=float)
    for incidence in range(count):
        site = int(center[incidence])
        atomic[3 * site : 3 * site + 3, incidence] = direction[incidence]

    equality = np.zeros((3 * n_sites + len(pairs) + 1, count + 1), dtype=float)
    equality[: 3 * n_sites, :count] = atomic
    row = 3 * n_sites
    for left, right in pairs:
        equality[row, left] = 1.0
        equality[row, right] = -1.0
        row += 1
    equality[row, :count] = 1.0
    target = np.zeros(len(equality), dtype=float)
    target[-1] = 1.0

    lower_floor = np.zeros((count, count + 1), dtype=float)
    lower_floor[np.arange(count), np.arange(count)] = -1.0
    lower_floor[:, -1] = 1.0 / count
    objective = np.zeros(count + 1, dtype=float)
    objective[-1] = -1.0
    result = linprog(
        objective,
        A_ub=lower_floor,
        b_ub=np.zeros(count, dtype=float),
        A_eq=equality,
        b_eq=target,
        bounds=[(0.0, None)] * count + [(0.0, 1.0)],
        method="highs",
        options={
            "primal_feasibility_tolerance": LP_FEASIBILITY_TOLERANCE,
            "dual_feasibility_tolerance": LP_FEASIBILITY_TOLERANCE,
        },
    )
    if result.status == 2:
        return _zero_result(pair_count=len(pairs), directed_count=count)
    if not result.success or result.x is None or not np.isfinite(result.x).all():
        raise RuntimeError(f"PGCE linear program failed: {result.message}")

    solution = np.asarray(result.x, dtype=float)
    coefficients = solution[:count]
    participation = float(np.clip(solution[-1], 0.0, 1.0))
    equality_residual = float(np.max(np.abs(equality @ solution - target)))
    floor_residual = float(
        max(0.0, participation / count - float(coefficients.min()))
    )
    residual = max(equality_residual, floor_residual)
    if residual > PUBLICATION_RESIDUAL_TOLERANCE:
        raise RuntimeError("PGCE independent LP residual exceeds tolerance")
    participation = float(round(participation, QUANTIZATION_DECIMALS))
    return PGCEKernelResult(
        True,
        None,
        participation,
        residual,
        len(pairs),
        count,
    )


def _feature_failure(exc: Exception | str) -> PGCEFeatureResult:
    message = str(exc) if isinstance(exc, str) else f"{type(exc).__name__}: {exc}"
    return PGCEFeatureResult(False, message, 0, 0, 0, math.nan, {})


def compute_pgce_features(atoms: Atoms) -> PGCEFeatureResult:
    """Compute PGCE from elements and one raw unrelaxed periodic geometry."""

    try:
        strict = n295._geometry_only_atoms(atoms)
        work = n267._validated_reduced_atoms(strict)
        radii = np.asarray(
            [n267._tabulated_radius(str(symbol)) for symbol in work.get_chemical_symbols()],
            dtype=float,
        )
        cells, contacts = n279.periodic_radical_cells_and_contacts(work, radii=radii)
        if len(cells) != len(work) or any(cell.empty for cell in cells):
            raise ValueError("PGCE graph requires every labelled power cell to be nonempty")
        if not contacts:
            raise ValueError("PGCE graph has no active-facet contact incidence")
        if not n279.contacts_are_reciprocal(contacts):
            raise ValueError("PGCE active-facet contacts are not reciprocal")
        translated = n319._translated_contacts(work=work, contacts=contacts)
        vectors = np.asarray([contact.displacement for contact in contacts], dtype=float)
        kernel = positive_contact_equilibrium_floor(
            n_sites=len(work),
            centers=translated[:, 0],
            neighbors=translated[:, 1],
            translations=translated[:, 2:],
            vectors=vectors,
        )
        if not kernel.supported:
            raise RuntimeError(kernel.error or "PGCE kernel failed")
        features = {FEATURE_NAMES[0]: kernel.participation_floor}
        if tuple(features) != FEATURE_NAMES or not np.isfinite(list(features.values())).all():
            raise RuntimeError("PGCE feature schema or values differ")
        return PGCEFeatureResult(
            True,
            None,
            len(work),
            kernel.directed_contact_count,
            kernel.reciprocal_pair_count,
            kernel.maximum_equilibrium_residual,
            features,
        )
    except Exception as exc:
        return _feature_failure(exc)


def compute_pgce_row(atoms: Atoms) -> dict[str, object]:
    """Return the exact additive table row for one initial geometry."""

    result = compute_pgce_features(atoms)
    return {
        "pgce_supported": bool(result.supported),
        "pgce_failure": result.failure_reason,
        "pgce_site_count": int(result.site_count),
        "pgce_directed_contact_count": int(result.directed_contact_count),
        "pgce_reciprocal_pair_count": int(result.reciprocal_pair_count),
        "pgce_maximum_equilibrium_residual": float(
            result.maximum_equilibrium_residual
        ),
        **{
            name: float(result.features.get(name, math.nan))
            for name in FEATURE_NAMES
        },
    }


def _error_row(exc: Exception) -> dict[str, object]:
    return {
        "pgce_supported": False,
        "pgce_failure": f"{type(exc).__name__}: {exc}",
        "pgce_site_count": 0,
        "pgce_directed_contact_count": 0,
        "pgce_reciprocal_pair_count": 0,
        "pgce_maximum_equilibrium_residual": math.nan,
        **{name: math.nan for name in FEATURE_NAMES},
    }


def _compute_scigen_payload(item: tuple[str, bytes]) -> tuple[str, dict[str, object]]:
    material_id, payload = item
    try:
        parsed = n267.n85._parse_frame(payload, strict_output=True)
        return material_id, compute_pgce_row(parsed.atoms)
    except Exception as exc:
        return material_id, _error_row(exc)


def _compute_wyformer_payload(item: tuple[str, str]) -> tuple[str, dict[str, object]]:
    material_id, payload = item
    try:
        structure = n267.Structure.from_dict(json.loads(payload))
        atoms = n267.AseAtomsAdaptor.get_atoms(structure)
        return material_id, compute_pgce_row(atoms)
    except Exception as exc:
        return material_id, _error_row(exc)


def _compute_many(payloads, *, source: str, workers: int):
    worker = _compute_scigen_payload if source == "scigen" else _compute_wyformer_payload
    if workers == 1:
        return [worker(item) for item in payloads]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(worker, payloads, chunksize=4))


def _label_free_statistics(table: pd.DataFrame) -> dict[str, object]:
    statistics: dict[str, object] = {}
    for name in FEATURE_NAMES:
        values = pd.to_numeric(table[name], errors="coerce").dropna().to_numpy(float)
        statistics[name] = {
            "minimum": float(np.min(values)),
            "q10": float(np.quantile(values, 0.10, method="inverted_cdf")),
            "median": float(np.median(values)),
            "q90": float(np.quantile(values, 0.90, method="inverted_cdf")),
            "maximum": float(np.max(values)),
            "unique_rounded_10": int(np.unique(np.round(values, 10)).size),
        }
    return statistics


def build_cross_source_pgce_features(
    *,
    scigen_cohort_dir: Path,
    wyformer_cohort_dir: Path,
    design_path: Path,
    probe_result_path: Path,
    output_dir: Path,
    workers: int = 16,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    """Build NEXT323 from physically isolated discovery geometry only."""

    scigen = Path(scigen_cohort_dir).resolve()
    wyformer = Path(wyformer_cohort_dir).resolve()
    target = Path(output_dir).resolve()
    paths = {
        "scigen_manifest": scigen / n267.n85.COHORT_MANIFEST_NAME,
        "scigen_metadata": scigen / n267.n85.COHORT_METADATA_NAME,
        "scigen_discovery_geometry": scigen
        / n267.n85.GEOMETRY_NAMES["discovery"],
        "wyformer_manifest": wyformer / n267.n94.COHORT_MANIFEST_NAME,
        "wyformer_metadata": wyformer / n267.n94.COHORT_METADATA_NAME,
        "wyformer_discovery_geometry": wyformer
        / n267.n94.GEOMETRY_NAMES["discovery"],
        "design": Path(design_path).resolve(),
        "probe_result": Path(probe_result_path).resolve(),
        "next267_source": Path(n267.__file__).resolve(),
        "next279_source": Path(n279.__file__).resolve(),
        "next295_source": Path(n295.__file__).resolve(),
        "next319_source": Path(n319.__file__).resolve(),
    }
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if type(workers) is not int or workers <= 0:
        raise ValueError("NEXT323 workers must be a positive exact integer")
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT323 input is missing")
    input_hashes = {name: n319._sha256_file(path) for name, path in paths.items()}
    if require_formal_inputs and input_hashes != EXPECTED_INPUT_SHA256:
        differing = sorted(
            name
            for name in set(input_hashes) | set(EXPECTED_INPUT_SHA256)
            if input_hashes.get(name) != EXPECTED_INPUT_SHA256.get(name)
        )
        raise ValueError(f"NEXT323 formal input identity differs: {differing}")

    with paths["probe_result"].open(encoding="utf-8") as handle:
        probe = json.load(handle)
    if (
        probe.get("design_sha256") != EXPECTED_DESIGN_SHA256
        or probe.get("probe_passed") is not True
        or any(probe.get(name) is not False for name in (
            "labels_opened",
            "endpoint_fields_read",
            "validation_geometry_opened",
            "replication_geometry_opened",
            "dft_calculation_executed",
            "dft_values_used",
            "learned_energy_force_stress_proxy_used",
            "model_or_proxy_potential_used",
            "physical_relaxation_executed",
        ))
    ):
        raise ValueError("NEXT323 label-blind probe authorization differs")

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
        raise ValueError("NEXT323 discovery geometry provenance differs")

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
            raise ValueError(f"NEXT323 {source} metadata differs")
        selected = frame.loc[frame["partition_role"].eq("discovery")].copy()
        selected = selected.sort_values("material_id", kind="mergesort").reset_index(drop=True)
        if len(selected) != EXPECTED_ROWS[source]:
            raise ValueError(f"NEXT323 {source} discovery identity differs")
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
    test_path = source_path.parent.parent / "tests/test_next323_periodic_global_contact_equilibrium.py"
    executed_hashes = {
        "src/next323_periodic_global_contact_equilibrium.py": n319._sha256_file(source_path),
        "tests/test_next323_periodic_global_contact_equilibrium.py": n319._sha256_file(test_path),
    }
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
                raise RuntimeError(f"NEXT323 {source} material identity differs")
            table = discovery[source].merge(
                computed_frame, on="material_id", how="left", validate="one_to_one"
            )
            supported = table["pgce_supported"].fillna(False).astype(bool)
            finite = np.isfinite(
                pd.to_numeric(table[FEATURE_NAMES[0]], errors="coerce").to_numpy(float)
            )
            if not finite[supported].all() or finite[~supported].any():
                raise RuntimeError(f"NEXT323 {source} finite support semantics differ")
            coverage = float(supported.mean())
            if coverage < MINIMUM_FORMAL_COVERAGE:
                raise RuntimeError(f"NEXT323 {source} coverage below frozen minimum")
            source_statistics = _label_free_statistics(table)
            if source_statistics[FEATURE_NAMES[0]]["unique_rounded_10"] < 20:
                raise RuntimeError(f"NEXT323 {source} feature is degenerate")
            statistics[source] = source_statistics
            failures = Counter(table.loc[~supported, "pgce_failure"].fillna("unknown"))
            finite_residual = pd.to_numeric(
                table.loc[supported, "pgce_maximum_equilibrium_residual"], errors="coerce"
            ).to_numpy(float)
            counts[source] = {
                "rows": int(len(table)),
                "supported": int(supported.sum()),
                "failures": int((~supported).sum()),
                "coverage": coverage,
                "site_count": int(pd.to_numeric(table.loc[supported, "pgce_site_count"]).sum()),
                "directed_contact_count": int(
                    pd.to_numeric(table.loc[supported, "pgce_directed_contact_count"]).sum()
                ),
                "reciprocal_pair_count": int(
                    pd.to_numeric(table.loc[supported, "pgce_reciprocal_pair_count"]).sum()
                ),
                "maximum_finite_equilibrium_residual": float(
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
            "graph": "NEXT279 reciprocal directed active-facet periodic contact incidences",
            "formula": "max alpha subject to site equilibrium, reciprocal equality, sum(f)=1, f_c>=alpha/M",
            "coefficients": "dimensionless dual contact certificates, not physical forces or stresses",
            "output_grid": 10.0 ** (-QUANTIZATION_DECIMALS),
            "minimum_formal_coverage": MINIMUM_FORMAL_COVERAGE,
            "label_free_statistics": statistics,
            "next324_audit_authorized": True,
            **BOUNDARY_FLAGS,
        }
        catalogue_path = staging / CATALOGUE_NAME
        n319._write_json(catalogue_path, catalogue)
        outputs.append(catalogue_path)
        manifest = {
            "protocol": PROTOCOL,
            "mode": "physically_isolated_discovery_x0_periodic_global_contact_equilibrium_freeze",
            "workers": workers,
            "elapsed_seconds": float(time.perf_counter() - started),
            "counts": counts,
            "source_partitions_read": {"scigen": ["discovery"], "wyformer": ["discovery"]},
            "labels_opened": False,
            "endpoint_fields_read": False,
            "internal_validation_geometry_opened": False,
            "internal_replication_geometry_opened": False,
            "opened_validation_outputs_used": False,
            "scigen_replication_endpoint_opened": False,
            "wyformer_replication_endpoint_opened": False,
            "scientific_improvement_claim": False,
            "next324_audit_authorized": True,
            **BOUNDARY_FLAGS,
            "inputs_sha256": input_hashes,
            "executed_source_sha256": executed_hashes,
            "outputs_sha256": {path.name: n319._sha256_file(path) for path in outputs},
        }
        manifest_path = staging / MANIFEST_NAME
        n319._write_json(manifest_path, manifest)
        if (
            n319._sha256_file(source_path)
            != executed_hashes["src/next323_periodic_global_contact_equilibrium.py"]
            or n319._sha256_file(test_path)
            != executed_hashes["tests/test_next323_periodic_global_contact_equilibrium.py"]
        ):
            raise RuntimeError("NEXT323 executed artifact changed before publication")
        os.replace(staging, target)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


__all__ = [
    "BOUNDARY_FLAGS",
    "CATALOGUE_NAME",
    "EXPECTED_DESIGN_SHA256",
    "FEATURE_DIRECTIONS",
    "FEATURE_FILES",
    "FEATURE_NAMES",
    "MANIFEST_NAME",
    "PGCEFeatureResult",
    "PGCEKernelResult",
    "PROTOCOL",
    "build_cross_source_pgce_features",
    "compute_pgce_features",
    "compute_pgce_row",
    "positive_contact_equilibrium_floor",
]
