#!/usr/bin/env python3
"""Periodic formal-charge Green-resistance features from raw geometry."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import time

from ase import Atoms
import numpy as np
import pandas as pd
from pymatgen.io.ase import AseAtomsAdaptor

import src.next19_valence_transport as n19
import src.next267_periodic_radical_voronoi_packing as n267
import src.next279_radical_packing_autocorrelation as n279
import src.next295_positive_contact_force_closure as n295
from src.next102_cross_source_dobvr_features import _sha256_file, _write_json


PROTOCOL = "2026-08-09-next315-periodic-charge-green-resistance-v1"
MANIFEST_NAME = "MANIFEST.json"
CATALOGUE_NAME = "NEXT315_PCGR_FEATURE_CATALOGUE.json"
FEATURE_FILES = {
    "scigen": "next315_scigen_periodic_charge_green_resistance.parquet",
    "wyformer": "next315_wyformer_periodic_charge_green_resistance.parquet",
}
FEATURE_NAMES = (
    "pcgr_charge_resistance",
    "pcgr_voltage_drop_q90",
    "pcgr_voltage_drop_max",
)
FEATURE_DIRECTIONS = {name: "protected_low" for name in FEATURE_NAMES}
EXPECTED_ROWS = {"scigen": 13_470, "wyformer": 5_232}
MINIMUM_FORMAL_COVERAGE = 0.90
EXPECTED_DESIGN_SHA256 = (
    "5a0204c00a478a05ebc699a9d5f8a29b50b453da43bd3b25ce96eee9ee8fb49f"
)
EXPECTED_INPUT_SHA256 = {
    "design": EXPECTED_DESIGN_SHA256,
    "next19_source": "f1195a7ef519827f8da1704b9abe773bcee105eff1bdf6dfd5b8eabba1b94712",
    "next267_source": "8f1e7ed9eb73a81a5755d455ffc05aab6f539cbd66afbbbfc384ca88391adca1",
    "next279_source": "b224e46424de199122b8f90061612edfcfd0318fb85178d404b6119f967df5c2",
    "next295_source": "4b92811e7f3c7ac60c1506104a18d2bd9d0fe06c6202e7f34cb996b32cd649a3",
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
class PCGRFeatureResult:
    """Fail-open result for one periodic charge Green-resistance solve."""

    supported: bool
    failure_reason: str | None
    site_count: int
    directed_contact_count: int
    intersite_contact_count: int
    positive_mode_count: int
    nullity: int
    minimum_positive_eigenvalue: float
    maximum_eigenvalue: float
    valence_policy: str | None
    features: Mapping[str, float]


def _failure(exc: Exception | str) -> PCGRFeatureResult:
    reason = str(exc) if isinstance(exc, str) else f"{type(exc).__name__}: {exc}"
    return PCGRFeatureResult(
        False,
        reason,
        0,
        0,
        0,
        0,
        0,
        math.nan,
        math.nan,
        None,
        {},
    )


def _inverted_cdf(values: object, quantile: float) -> float:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError("PCGR quantile population differs")
    ordered = np.sort(array, kind="stable")
    index = max(0, int(math.ceil(float(quantile) * len(ordered))) - 1)
    return float(ordered[min(index, len(ordered) - 1)])


def _quantize(value: float) -> float:
    return float(np.rint(float(value) * n267.OUTPUT_GRID) / n267.OUTPUT_GRID)


def charge_green_resistance_features(
    *, charges: object, directed_contacts: object
) -> PCGRFeatureResult:
    """Solve ``L phi = q`` on a reciprocal unit-conductance contact graph."""

    try:
        charge = np.asarray(charges, dtype=float)
        if charge.ndim != 1 or len(charge) < 2 or not np.isfinite(charge).all():
            raise ValueError("PCGR charges must be a finite vector with at least two sites")
        magnitude = float(np.abs(charge).sum())
        if magnitude <= 0.0 or abs(float(charge.sum())) > 1.0e-8 * max(1.0, magnitude):
            raise ValueError("PCGR formal charges must be neutral")
        if not np.any(charge > 0.0) or not np.any(charge < 0.0):
            raise ValueError("PCGR formal charges must contain both signs")

        raw = np.asarray(directed_contacts)
        if raw.ndim != 2 or raw.shape[1:] != (2,) or len(raw) == 0:
            raise ValueError("PCGR directed contact population differs")
        try:
            pair = raw.astype(np.int64)
            numeric = raw.astype(float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("PCGR directed contact population differs") from exc
        if (
            not np.isfinite(numeric).all()
            or not np.equal(numeric, pair).all()
            or np.any(pair < 0)
            or np.any(pair >= len(charge))
        ):
            raise ValueError("PCGR directed contact population differs")
        counts = Counter((int(left), int(right)) for left, right in pair)
        reversed_counts = Counter(
            {(right, left): count for (left, right), count in counts.items()}
        )
        if counts != reversed_counts:
            raise ValueError("PCGR directed contacts are not reciprocal")

        adjacency = np.zeros((len(charge), len(charge)), dtype=float)
        np.add.at(adjacency, (pair[:, 0], pair[:, 1]), 1.0)
        if not np.array_equal(adjacency, adjacency.T):
            raise ValueError("PCGR quotient adjacency is not symmetric")
        laplacian = np.diag(adjacency.sum(axis=1)) - adjacency
        if not np.isfinite(laplacian).all() or not np.array_equal(
            laplacian, laplacian.T
        ):
            raise ValueError("PCGR graph Laplacian differs")

        eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
        if not np.isfinite(eigenvalues).all() or not np.isfinite(eigenvectors).all():
            raise ValueError("PCGR Laplacian eigensystem differs")
        eigen_scale = max(1.0, float(np.max(np.abs(eigenvalues))))
        eigen_tolerance = (
            np.finfo(float).eps * max(laplacian.shape) * eigen_scale
        )
        if np.any(eigenvalues < -eigen_tolerance):
            raise ValueError("PCGR Laplacian has a negative mode")
        positive = eigenvalues > eigen_tolerance
        null = ~positive
        if not np.any(positive):
            raise ValueError("PCGR graph has no positive Laplacian mode")
        null_projection = eigenvectors[:, null].T @ charge
        projection_limit = 1.0e-8 * max(1.0, float(np.linalg.norm(charge)))
        if len(null_projection) and float(np.max(np.abs(null_projection))) > projection_limit:
            raise ValueError("PCGR charge is not balanced on every graph component")

        intersite = pair[:, 0] != pair[:, 1]
        if not np.any(intersite):
            raise ValueError("PCGR graph has no non-self contact incidence")
        positive_vectors = eigenvectors[:, positive]
        potential = positive_vectors @ (
            (positive_vectors.T @ charge) / eigenvalues[positive]
        )
        if not np.isfinite(potential).all():
            raise ValueError("PCGR auxiliary graph potential differs")
        dissipation = float(charge @ potential)
        numerical_tolerance = 1.0e-10 * max(
            1.0, float(np.linalg.norm(charge) * np.linalg.norm(potential))
        )
        if not math.isfinite(dissipation) or dissipation < -numerical_tolerance:
            raise ValueError("PCGR graph dissipation differs")
        dissipation = max(0.0, dissipation)
        charge_norm_squared = float(charge @ charge)
        charge_rms = math.sqrt(float(np.mean(charge**2)))
        if charge_norm_squared <= 0.0 or not math.isfinite(charge_rms) or charge_rms <= 0.0:
            raise ValueError("PCGR charge normalization differs")
        drops = np.abs(
            potential[pair[intersite, 0]] - potential[pair[intersite, 1]]
        ) / charge_rms
        values = {
            "pcgr_charge_resistance": _quantize(
                dissipation / charge_norm_squared
            ),
            "pcgr_voltage_drop_q90": _quantize(_inverted_cdf(drops, 0.90)),
            "pcgr_voltage_drop_max": _quantize(float(np.max(drops))),
        }
        if tuple(values) != FEATURE_NAMES or not np.isfinite(list(values.values())).all():
            raise ValueError("PCGR feature schema or values differ")
        if any(value < 0.0 for value in values.values()):
            raise ValueError("PCGR features must be nonnegative")
        return PCGRFeatureResult(
            True,
            None,
            len(charge),
            len(pair),
            int(np.sum(intersite)),
            int(np.sum(positive)),
            int(np.sum(null)),
            float(np.min(eigenvalues[positive])),
            float(np.max(eigenvalues)),
            None,
            values,
        )
    except Exception as exc:
        return _failure(exc)


def compute_pcgr_features(atoms: Atoms) -> PCGRFeatureResult:
    """Compute PCGR from elements and one raw unrelaxed periodic geometry."""

    try:
        strict = n295._geometry_only_atoms(atoms)
        work = n267._validated_reduced_atoms(strict)
        structure = AseAtomsAdaptor.get_structure(work)
        assignment = n19.infer_valence_assignment(structure)
        if not assignment.supported or assignment.values is None:
            raise ValueError(assignment.failure_reason or "PCGR valence assignment failed")
        charges = np.asarray(assignment.values, dtype=float)
        radii = np.asarray(
            [n267._tabulated_radius(str(symbol)) for symbol in work.get_chemical_symbols()],
            dtype=float,
        )
        cells, contacts = n279.periodic_radical_cells_and_contacts(work, radii=radii)
        if len(cells) != len(work) or any(cell.empty for cell in cells):
            raise ValueError("PCGR graph requires every labelled power cell to be nonempty")
        if not contacts:
            raise ValueError("PCGR graph has no active-facet contact incidence")
        if not n279.contacts_are_reciprocal(contacts):
            raise ValueError("PCGR active-facet contacts are not reciprocal")
        result = charge_green_resistance_features(
            charges=charges,
            directed_contacts=np.asarray(
                [(contact.center, contact.neighbor) for contact in contacts], dtype=int
            ),
        )
        if not result.supported:
            return result
        return replace(result, valence_policy=str(assignment.policy))
    except Exception as exc:
        return _failure(exc)


def compute_pcgr_row(atoms: Atoms) -> dict[str, object]:
    result = compute_pcgr_features(atoms)
    row: dict[str, object] = {name: math.nan for name in FEATURE_NAMES}
    row.update(
        {
            "pcgr_supported": bool(result.supported),
            "pcgr_failure": result.failure_reason,
            "pcgr_site_count": result.site_count,
            "pcgr_directed_contact_count": result.directed_contact_count,
            "pcgr_intersite_contact_count": result.intersite_contact_count,
            "pcgr_positive_mode_count": result.positive_mode_count,
            "pcgr_nullity": result.nullity,
            "pcgr_minimum_positive_eigenvalue": result.minimum_positive_eigenvalue,
            "pcgr_maximum_eigenvalue": result.maximum_eigenvalue,
            "pcgr_valence_policy": result.valence_policy,
        }
    )
    if result.supported:
        row.update({name: float(result.features[name]) for name in FEATURE_NAMES})
    return row


def _error_row(exc: Exception) -> dict[str, object]:
    return {
        **{name: math.nan for name in FEATURE_NAMES},
        "pcgr_supported": False,
        "pcgr_failure": f"{type(exc).__name__}: {exc}",
        "pcgr_site_count": 0,
        "pcgr_directed_contact_count": 0,
        "pcgr_intersite_contact_count": 0,
        "pcgr_positive_mode_count": 0,
        "pcgr_nullity": 0,
        "pcgr_minimum_positive_eigenvalue": math.nan,
        "pcgr_maximum_eigenvalue": math.nan,
        "pcgr_valence_policy": None,
    }


def _compute_scigen_payload(item: tuple[str, bytes]) -> tuple[str, dict[str, object]]:
    material_id, payload = item
    try:
        atoms = n267.n85._parse_frame(payload, strict_output=True).atoms
        return material_id, compute_pcgr_row(atoms)
    except Exception as exc:
        return material_id, _error_row(exc)


def _compute_wyformer_payload(item: tuple[str, str]) -> tuple[str, dict[str, object]]:
    material_id, payload = item
    try:
        structure = n267.Structure.from_dict(json.loads(payload))
        atoms = n267.AseAtomsAdaptor.get_atoms(structure)
        return material_id, compute_pcgr_row(atoms)
    except Exception as exc:
        return material_id, _error_row(exc)


def _compute_many(payloads, *, source: str, workers: int):
    worker = _compute_scigen_payload if source == "scigen" else _compute_wyformer_payload
    if workers == 1:
        return [worker(item) for item in payloads]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(worker, payloads, chunksize=8))


def _label_free_statistics(table: pd.DataFrame) -> dict[str, object]:
    supported = table["pcgr_supported"].fillna(False).astype(bool)
    statistics: dict[str, object] = {}
    for name in FEATURE_NAMES:
        values = pd.to_numeric(table.loc[supported, name], errors="coerce").to_numpy(float)
        statistics[name] = {
            "unique_rounded_10": int(len(np.unique(np.round(values, 10)))),
            "exact_zero_fraction": float(np.mean(values < 1.0e-10)),
            "minimum": float(values.min()),
            "q10": _inverted_cdf(values, 0.10),
            "median": _inverted_cdf(values, 0.50),
            "q90": _inverted_cdf(values, 0.90),
            "maximum": float(values.max()),
        }
    return statistics


def build_cross_source_pcgr_features(
    *,
    scigen_cohort_dir: Path,
    wyformer_cohort_dir: Path,
    design_path: Path,
    output_dir: Path,
    workers: int = 16,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    """Build NEXT315 from physically isolated discovery geometry only."""

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
        "next19_source": Path(n19.__file__).resolve(),
        "next267_source": Path(n267.__file__).resolve(),
        "next279_source": Path(n279.__file__).resolve(),
        "next295_source": Path(n295.__file__).resolve(),
    }
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if type(workers) is not int or workers <= 0:
        raise ValueError("NEXT315 workers must be a positive exact integer")
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT315 input is missing")
    input_hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if require_formal_inputs and input_hashes != EXPECTED_INPUT_SHA256:
        differing = sorted(
            name
            for name in set(input_hashes) | set(EXPECTED_INPUT_SHA256)
            if input_hashes.get(name) != EXPECTED_INPUT_SHA256.get(name)
        )
        raise ValueError(f"NEXT315 formal input identity differs: {differing}")

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
        raise ValueError("NEXT315 discovery geometry provenance differs")

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
            raise ValueError(f"NEXT315 {source} metadata differs")
        selected = frame.loc[frame["partition_role"].eq("discovery")].copy()
        selected = selected.sort_values("material_id", kind="mergesort").reset_index(drop=True)
        if len(selected) != EXPECTED_ROWS[source]:
            raise ValueError(f"NEXT315 {source} discovery identity differs")
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
    test_path = source_path.parent.parent / "tests/test_next315_periodic_charge_green_resistance.py"
    executed_hashes = {
        "src/next315_periodic_charge_green_resistance.py": _sha256_file(source_path),
        "tests/test_next315_periodic_charge_green_resistance.py": _sha256_file(test_path),
    }
    started = time.perf_counter()
    try:
        computed = {
            source: _compute_many(payloads[source], source=source, workers=workers)
            for source in ("scigen", "wyformer")
        }
        counts: dict[str, object] = {}
        outputs: list[Path] = []
        statistics: dict[str, object] = {}
        for source in ("scigen", "wyformer"):
            computed_frame = pd.DataFrame(
                [{"material_id": material_id, **row} for material_id, row in computed[source]]
            )
            if (
                computed_frame["material_id"].astype(str).duplicated().any()
                or set(computed_frame["material_id"].astype(str))
                != set(discovery[source]["material_id"].astype(str))
            ):
                raise RuntimeError(f"NEXT315 {source} material identity differs")
            table = discovery[source].merge(
                computed_frame, on="material_id", how="left", validate="one_to_one"
            )
            supported = table["pcgr_supported"].fillna(False).astype(bool)
            finite = np.column_stack(
                [
                    np.isfinite(pd.to_numeric(table[name], errors="coerce").to_numpy(float))
                    for name in FEATURE_NAMES
                ]
            )
            if not finite[supported].all() or finite[~supported].any():
                raise RuntimeError(f"NEXT315 {source} finite support semantics differ")
            coverage = float(supported.mean())
            if coverage < MINIMUM_FORMAL_COVERAGE:
                raise RuntimeError(f"NEXT315 {source} coverage below frozen minimum")
            source_statistics = _label_free_statistics(table)
            if any(value["unique_rounded_10"] < 2 for value in source_statistics.values()):
                raise RuntimeError(f"NEXT315 {source} feature is degenerate")
            statistics[source] = source_statistics
            failures = Counter(table.loc[~supported, "pcgr_failure"].fillna("unknown"))
            counts[source] = {
                "rows": int(len(table)),
                "supported": int(supported.sum()),
                "failures": int((~supported).sum()),
                "coverage": coverage,
                "site_count": int(pd.to_numeric(table.loc[supported, "pcgr_site_count"]).sum()),
                "directed_contact_count": int(
                    pd.to_numeric(table.loc[supported, "pcgr_directed_contact_count"]).sum()
                ),
                "intersite_contact_count": int(
                    pd.to_numeric(table.loc[supported, "pcgr_intersite_contact_count"]).sum()
                ),
                "positive_mode_count": int(
                    pd.to_numeric(table.loc[supported, "pcgr_positive_mode_count"]).sum()
                ),
                "minimum_positive_eigenvalue": float(
                    pd.to_numeric(
                        table.loc[supported, "pcgr_minimum_positive_eigenvalue"]
                    ).min()
                ),
                "maximum_eigenvalue": float(
                    pd.to_numeric(table.loc[supported, "pcgr_maximum_eigenvalue"]).max()
                ),
                "finite_feature_counts": {
                    name: int(finite[:, index].sum())
                    for index, name in enumerate(FEATURE_NAMES)
                },
                "failure_counts": {str(key): int(value) for key, value in failures.items()},
            }
            output_path = staging / FEATURE_FILES[source]
            table.to_parquet(output_path, index=False)
            outputs.append(output_path)

        catalogue = {
            "protocol": PROTOCOL,
            "feature_count": len(FEATURE_NAMES),
            "feature_names": list(FEATURE_NAMES),
            "feature_directions": FEATURE_DIRECTIONS,
            "directions_frozen_before_outcome": True,
            "graph": "NEXT279 full reciprocal periodic radical active-facet quotient multigraph",
            "charge_policy": "NEXT19 neutral formal-valence assignment",
            "conductance": "one unit per directed reciprocal active-facet incidence",
            "laplacian": "L = diag(W 1) - W",
            "solver": "symmetric eigensystem Moore-Penrose pseudoinverse",
            "positive_eigenvalue_tolerance": "eps * max(L.shape) * max(1, max(abs(eigenvalues)))",
            "null_projection_limit": "1e-8 * max(1, ||q||_2)",
            "charge_resistance": "(q^T L^+ q) / (q^T q)",
            "voltage_normalization": "RMS(q)",
            "quantile": 0.90,
            "quantile_method": "inverted_cdf",
            "minimum_formal_coverage": MINIMUM_FORMAL_COVERAGE,
            "label_free_statistics": statistics,
            "next316_audit_authorized": True,
            **BOUNDARY_FLAGS,
        }
        catalogue_path = staging / CATALOGUE_NAME
        _write_json(catalogue_path, catalogue)
        outputs.append(catalogue_path)
        manifest = {
            "protocol": PROTOCOL,
            "mode": "physically_isolated_discovery_x0_periodic_charge_green_resistance_freeze",
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
            "next316_audit_authorized": True,
            **BOUNDARY_FLAGS,
            "inputs_sha256": input_hashes,
            "executed_source_sha256": executed_hashes,
            "outputs_sha256": {path.name: _sha256_file(path) for path in outputs},
        }
        manifest_path = staging / MANIFEST_NAME
        _write_json(manifest_path, manifest)
        if (
            _sha256_file(source_path)
            != executed_hashes["src/next315_periodic_charge_green_resistance.py"]
            or _sha256_file(test_path)
            != executed_hashes["tests/test_next315_periodic_charge_green_resistance.py"]
        ):
            raise RuntimeError("NEXT315 executed artifact changed before publication")
        os.replace(staging, target)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scigen-cohort-dir", type=Path, required=True)
    parser.add_argument("--wyformer-cohort-dir", type=Path, required=True)
    parser.add_argument("--design-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--allow-nonformal-inputs", action="store_true")
    arguments = parser.parse_args(argv)
    manifest = build_cross_source_pcgr_features(
        scigen_cohort_dir=arguments.scigen_cohort_dir,
        wyformer_cohort_dir=arguments.wyformer_cohort_dir,
        design_path=arguments.design_path,
        output_dir=arguments.output_dir,
        workers=arguments.workers,
        require_formal_inputs=not arguments.allow_nonformal_inputs,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CATALOGUE_NAME",
    "EXPECTED_DESIGN_SHA256",
    "FEATURE_DIRECTIONS",
    "FEATURE_FILES",
    "FEATURE_NAMES",
    "MANIFEST_NAME",
    "MINIMUM_FORMAL_COVERAGE",
    "PCGRFeatureResult",
    "PROTOCOL",
    "build_cross_source_pcgr_features",
    "charge_green_resistance_features",
    "compute_pcgr_features",
]
