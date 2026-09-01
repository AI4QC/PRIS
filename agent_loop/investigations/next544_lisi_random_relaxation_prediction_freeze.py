#!/usr/bin/env python3
"""Freeze MUPR and controls on the NEXT543 geometry-only Li--Si x0 cohort."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import time
import zipfile

from ase import Atoms
import numpy as np
import pandas as pd
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from src.next16_elementa_basin_hull import pauling_control
from src.next19_feature_build import _publish_directory_no_replace, _sha256
from src.next32_inorganic_response_features import compute_periodic_contact_features
from src.next411_same_sign_shell_purity import (
    FEATURE_NAMES as SSSP_FEATURE_NAMES,
    PROTOCOL as SSSP_PROTOCOL,
    compute_sssp_features,
)
from src.next537_periodic_bond_angle_affine_accommodation import (
    FEATURE_NAMES as PBAAA_FEATURE_NAMES,
    PROTOCOL as PBAAA_PROTOCOL,
    compute_periodic_bond_angle_affine_accommodation,
)
import src.next541_jarvis_cdvae_initial_prediction_freeze as n541
import src.next543_lisi_random_relaxation_initial_cohort as n543


PROTOCOL = "2026-08-13-next544-lisi-random-relaxation-prediction-freeze-v1"
SSSP_FEATURE = SSSP_FEATURE_NAMES[0]
PBAAA_FEATURE = PBAAA_FEATURE_NAMES[0]
MINIMUM_COVERAGE = 0.95
MINIMUM_UNIQUE = 50
MAXIMUM_POINT_MASS = 0.10
TABLE_NAME = "next544_lisi_rr_x0_predictions.parquet"
FORMULA_NAME = "NEXT544_MUPR_FORMULA.json"
MANIFEST_NAME = "MANIFEST.json"
EXPECTED_INPUT_SHA256 = {
    "design": n543.DESIGN_SHA256,
    "next543_manifest": "5521ee2ad498f9529db0f88d82242efa46f930c5351fd3bf1fc0e069bbb51cf1",
    "next543_inventory": "7de250de1dba4b22670949e08049a5483c1f00f12a564c5790bf6f7702aa3eaf",
    "next543_cohort": "9022a3a0a81e3cca43f237f3b217391b892ba66781a83c09dc5055c1a02964bc",
    "next543_geometry": "1c249fbc66267c964eb840b06886acd5f21a4884b05533469567866753dd843e",
    "next543_source": "79f2d7e1b70677856f5596f1b28908cff53b3c037d27e36545abb64115d55cca",
    "next541_source": "16f612791b256db05e782d4f65dabf8b95320fba48db9f624a7a765ff07cb80b",
    "next32_source": n541.EXPECTED_INPUT_SHA256["next32_source"],
    "next411_source": n541.EXPECTED_INPUT_SHA256["next411_source"],
    "next537_source": n541.EXPECTED_INPUT_SHA256["next537_source"],
    "pauling_source": n541.EXPECTED_INPUT_SHA256["pauling_source"],
}


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _compute_structure_payload(item: tuple[str, bytes]) -> dict[str, object]:
    trajectory_id, payload = item
    started = time.perf_counter()
    base: dict[str, object] = {"trajectory_id": trajectory_id}
    try:
        value = json.loads(payload)
        if not isinstance(value, dict):
            raise ValueError("NEXT544 x0 member schema differs")
        structure = Structure.from_dict(value)
        atoms: Atoms = AseAtomsAdaptor.get_atoms(structure)
        atoms.info.clear()
        base.update(
            {
                "formula": structure.composition.reduced_formula,
                "n_sites": len(structure),
                "x0_member_sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        return {
            **base,
            "formula": "",
            "n_sites": 0,
            "x0_member_sha256": hashlib.sha256(payload).hexdigest(),
            "contact_supported": False,
            "contact_failure": reason,
            "cov_q05": math.nan,
            "sssp_supported": False,
            "sssp_failure": reason,
            SSSP_FEATURE: math.nan,
            "pbaaa_supported": False,
            "pbaaa_failure": reason,
            PBAAA_FEATURE: math.nan,
            "pauling_feature_error": reason,
            "pauling_violation_fraction": math.nan,
            "pauling_p2_p5_decision": "ABSTAIN",
            "feature_runtime_seconds": time.perf_counter() - started,
        }

    contact = compute_periodic_contact_features(atoms)
    base.update(
        {
            "contact_supported": bool(contact.supported),
            "contact_failure": contact.failure_reason,
            "cov_q05": float(contact.features.get("cov_q05", math.nan)),
        }
    )
    sssp = compute_sssp_features(atoms)
    base.update(
        {
            "sssp_supported": bool(sssp.supported),
            "sssp_failure": sssp.failure_reason,
            SSSP_FEATURE: float(sssp.features.get(SSSP_FEATURE, math.nan)),
        }
    )
    pbaaa = compute_periodic_bond_angle_affine_accommodation(atoms)
    base.update(
        {
            "pbaaa_supported": bool(pbaaa.supported),
            "pbaaa_failure": pbaaa.failure_reason,
            PBAAA_FEATURE: float(pbaaa.features.get(PBAAA_FEATURE, math.nan)),
        }
    )
    pauling = pauling_control(atoms)
    decisions = [str(pauling[f"pauling_p{number}_decision"]) for number in range(2, 6)]
    supported = [value for value in decisions if value != "ABSTAIN"]
    base.update(pauling)
    base["pauling_violation_fraction"] = (
        float(np.mean([value == "REJECT" for value in supported])) if supported else math.nan
    )
    base["feature_runtime_seconds"] = time.perf_counter() - started
    return base


def _compute_many(payloads: list[tuple[str, bytes]], workers: int) -> list[dict[str, object]]:
    if workers == 1:
        iterator = map(_compute_structure_payload, payloads)
        executor = None
    else:
        executor = ProcessPoolExecutor(max_workers=workers)
        iterator = executor.map(_compute_structure_payload, payloads, chunksize=1)
    rows: list[dict[str, object]] = []
    try:
        for offset, row in enumerate(iterator, start=1):
            rows.append(row)
            if offset % 25 == 0 or offset == len(payloads):
                print(f"NEXT544 x0-only scores: {offset}/{len(payloads)}", flush=True)
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    return rows


def _gate_statistics(table: pd.DataFrame) -> dict[str, object]:
    contact = table["contact_supported"].fillna(False).to_numpy(bool)
    mupr = pd.to_numeric(table["mupr_risk"], errors="coerce").to_numpy(float)
    supported = np.isfinite(mupr)
    finite = np.round(mupr[supported], 12)
    counts = Counter(finite.tolist())
    unique = len(counts)
    point_mass = max(counts.values(), default=len(table)) / len(table)
    bounded = bool(len(finite) and np.all((finite >= 0.0) & (finite <= 1.0)))
    result = {
        "rows": len(table),
        "contact_supported": int(contact.sum()),
        "contact_coverage": float(contact.mean()),
        "mupr_supported": int(supported.sum()),
        "mupr_coverage": float(supported.mean()),
        "mupr_unique_rounded_12": unique,
        "mupr_maximum_point_mass": float(point_mass),
        "mupr_bounded": bounded,
        "mechanism_count_distribution": {
            str(key): int(value)
            for key, value in sorted(Counter(table["mechanism_count"].astype(int)).items())
        },
    }
    result["passes"] = bool(
        len(table) == n543.EXPECTED_ROWS
        and result["contact_coverage"] >= MINIMUM_COVERAGE
        and result["mupr_coverage"] >= MINIMUM_COVERAGE
        and unique >= MINIMUM_UNIQUE
        and point_mass <= MAXIMUM_POINT_MASS
        and bounded
    )
    return result


def build_prediction_freeze(
    *, next543_dir: Path, design_path: Path, output_dir: Path, workers: int = 8,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    upstream = Path(next543_dir).resolve()
    target = Path(output_dir).resolve()
    paths = {
        "design": Path(design_path).resolve(),
        "next543_manifest": upstream / n543.MANIFEST_NAME,
        "next543_inventory": upstream / n543.INVENTORY_NAME,
        "next543_cohort": upstream / n543.COHORT_NAME,
        "next543_geometry": upstream / n543.GEOMETRY_NAME,
        "next543_source": Path(n543.__file__).resolve(),
        "next541_source": Path(n541.__file__).resolve(),
        "next32_source": Path(compute_periodic_contact_features.__code__.co_filename).resolve(),
        "next411_source": Path(compute_sssp_features.__code__.co_filename).resolve(),
        "next537_source": Path(
            compute_periodic_bond_angle_affine_accommodation.__code__.co_filename
        ).resolve(),
        "pauling_source": Path(pauling_control.__code__.co_filename).resolve(),
    }
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if type(workers) is not int or not 1 <= workers <= 32:
        raise ValueError("NEXT544 workers differ")
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT544 input is missing")
    hashes = {name: _sha256(path) for name, path in paths.items()}
    if require_formal_inputs and hashes != EXPECTED_INPUT_SHA256:
        differing = sorted(
            name for name in set(hashes) | set(EXPECTED_INPUT_SHA256)
            if hashes.get(name) != EXPECTED_INPUT_SHA256.get(name)
        )
        raise ValueError(f"NEXT544 formal input identity differs: {differing}")
    manifest = json.loads(paths["next543_manifest"].read_text())
    outputs = manifest.get("outputs_sha256")
    source_hashes = manifest.get("executed_source_sha256")
    if (
        manifest.get("protocol") != n543.PROTOCOL
        or manifest.get("endpoint_values_opened") is not False
        or manifest.get("dft_force_stress_energy_values_decoded_or_inspected") is not False
        or manifest.get("later_structure_objects_decoded_or_inspected") is not False
        or manifest.get("next544_prediction_freeze_authorized") is not True
        or not isinstance(outputs, dict)
        or outputs.get(n543.INVENTORY_NAME) != hashes["next543_inventory"]
        or outputs.get(n543.COHORT_NAME) != hashes["next543_cohort"]
        or outputs.get(n543.GEOMETRY_NAME) != hashes["next543_geometry"]
        or not isinstance(source_hashes, dict)
        or source_hashes.get("src/next543_lisi_random_relaxation_initial_cohort.py")
        != hashes["next543_source"]
    ):
        raise ValueError("NEXT544 upstream x0 firewall differs")
    cohort = json.loads(paths["next543_cohort"].read_text())
    if not isinstance(cohort, list) or len(cohort) != n543.EXPECTED_ROWS:
        raise ValueError("NEXT544 cohort schema differs")
    by_id = {str(row["trajectory_id"]): row for row in cohort}
    with zipfile.ZipFile(paths["next543_geometry"]) as archive:
        names = archive.namelist()
        if len(names) != n543.EXPECTED_ROWS:
            raise ValueError("NEXT544 x0 geometry count differs")
        payloads = []
        for name in names:
            trajectory_id = Path(name).stem
            payload = archive.read(name)
            if (
                trajectory_id not in by_id
                or hashlib.sha256(payload).hexdigest()
                != str(by_id[trajectory_id]["x0_structure_sha256"])
            ):
                raise ValueError("NEXT544 x0 member identity differs")
            payloads.append((trajectory_id, payload))
    payloads.sort(key=lambda item: item[0])
    rows = pd.DataFrame(_compute_many(payloads, workers))
    metadata = pd.DataFrame(cohort)[["trajectory_id", "prefix", "object_name"]]
    table = metadata.merge(rows, on="trajectory_id", validate="one_to_one")
    ranks = n541.mechanism_union_percentile_risk(
        -pd.to_numeric(table["cov_q05"], errors="coerce").to_numpy(float),
        -pd.to_numeric(table[SSSP_FEATURE], errors="coerce").to_numpy(float),
        pd.to_numeric(table[PBAAA_FEATURE], errors="coerce").to_numpy(float),
    )
    for name, values in ranks.items():
        table[name] = values
    table = table.sort_values(["prefix", "trajectory_id"], kind="mergesort").reset_index(drop=True)
    statistics = _gate_statistics(table)
    if statistics["passes"] is not True:
        raise RuntimeError(f"NEXT544 label-blind gates failed: {statistics}")

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    source_path = Path(__file__).resolve()
    source_hash = _sha256(source_path)
    try:
        table_path = staging / TABLE_NAME
        formula_path = staging / FORMULA_NAME
        table.to_parquet(table_path, index=False)
        formula = {
            "protocol": PROTOCOL,
            "name": "mechanism_union_percentile_risk",
            "short_name": "MUPR",
            "formula": "R=1-product_j(1-u_j)",
            "percentile": "u_j=(midrank(x_j)-0.5)/n_j",
            "risk_inputs": {
                "contact": "-cov_q05",
                "same_sign_shell": f"-{SSSP_FEATURE}",
                "affine_accommodation": PBAAA_FEATURE,
            },
            "missing_policy": "contact required; omit unsupported SSSP or PBAAA",
            "operating_rule": "reject top 15 percent; prefix then trajectory_id break ties",
            "identical_to_next541_formula": True,
            "coefficients_fitted_to_endpoint": False,
        }
        formula_path.write_bytes(_json_bytes(formula))
        outputs = {table_path.name: _sha256(table_path), formula_path.name: _sha256(formula_path)}
        manifest_out = {
            "protocol": PROTOCOL,
            "gates": statistics,
            "inputs_sha256": {
                name: {"path": str(path), "sha256": hashes[name]}
                for name, path in paths.items()
            },
            "outputs_sha256": outputs,
            "executed_source_sha256": {
                "src/next544_lisi_random_relaxation_prediction_freeze.py": source_hash
            },
            "feature_protocols": {
                "sssp": SSSP_PROTOCOL,
                "pbaaa": PBAAA_PROTOCOL,
                "contact": "2026-08-03-next32-omat24-inorganic-response-features-v1",
                "pauling": "2026-08-02-next12-prospective-pauling2-5-controls-v1",
                "mupr": n541.PROTOCOL,
            },
            "endpoint_summary_or_values_opened": False,
            "dft_force_stress_energy_values_used": False,
            "later_structure_objects_opened": False,
            "predictions_and_gates_frozen_before_endpoint_access": True,
            "next545_endpoint_access_authorized": True,
            "scientific_improvement_claim": False,
        }
        (staging / MANIFEST_NAME).write_bytes(_json_bytes(manifest_out))
        if _sha256(source_path) != source_hash:
            raise RuntimeError("NEXT544 source changed before publication")
        if any(_sha256(path) != hashes[name] for name, path in paths.items()):
            raise RuntimeError("NEXT544 input changed before publication")
        _publish_directory_no_replace(staging, target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest_out


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--next543-dir", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = build_prediction_freeze(
        next543_dir=args.next543_dir,
        design_path=args.design,
        output_dir=args.output_dir,
        workers=args.workers,
    )
    print(json.dumps(manifest["gates"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
