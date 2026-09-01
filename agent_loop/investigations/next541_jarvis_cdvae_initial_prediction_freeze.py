#!/usr/bin/env python3
"""Freeze x0-only scores before opening JARVIS-CDVAE DFT endpoint members."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
import time
import zipfile

from ase import Atoms
import numpy as np
import pandas as pd
from pymatgen.core import Composition
from scipy.stats import rankdata

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


PROTOCOL = "2026-08-13-next541-jarvis-cdvae-initial-prediction-freeze-v1"
DESIGN_SHA256 = "9fb267b4e9dce69c330b279006634c729391f3b08a4fdfbb2ed5da9939588c6e"
EXPECTED_INITIAL_ROWS = 2_895
EXPECTED_ENDPOINT_FILES = 61
EXPECTED_COMPOSITIONS = 48
EXPECTED_CANDIDATES = 149
MINIMUM_CONTACT_COVERAGE = 0.95
MINIMUM_MUPR_COVERAGE = 0.95
MINIMUM_MUPR_UNIQUE = 30
MAXIMUM_MUPR_POINT_MASS = 0.25
FORBIDDEN_MODEL_FIELDS = ("bg", "fenp", "pred")
SSSP_FEATURE = SSSP_FEATURE_NAMES[0]
PBAAA_FEATURE = PBAAA_FEATURE_NAMES[0]
TABLE_NAME = "next541_jarvis_cdvae_initial_predictions.parquet"
INVENTORY_NAME = "NEXT541_ENDPOINT_NAME_INVENTORY.json"
FORMULA_NAME = "NEXT541_MUPR_FORMULA.json"
MANIFEST_NAME = "MANIFEST.json"
EXPECTED_INPUT_SHA256 = {
    "design": DESIGN_SHA256,
    "initial_archive": "8f879acec0e6b6f8a201d9c4ff59de55e2bfd26b790c7aceace05511f753a55b",
    "endpoint_archive": "1308bb32ec05bfcad1e9858dfaa25927f30c1fec61886fea47bd3b298e5db4cd",
    "next32_source": "6ecdd040f631e2196830706e66e92d4b64a461a5368e32a700d7c70a82f7931b",
    "next411_source": "172543534328a387b7d2b12ffd6cad919793ace56ec1124dd6e228f96d8cc9a4",
    "next537_source": "ccaedb99e9a62589a184ff40c3b9f66c6c5928c386b262e2c9cbf945a2a1a918",
    "pauling_source": "4f05c616640a523116d0ee1782913d5e4f06345fffd5ee9539b81cfe3084594c",
}
_JSON_NUMBER = rb"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?"
_MODEL_VALUE_PATTERN = re.compile(
    rb'"(?P<key>bg|fenp|pred)"(?P<separator>\s*:\s*)' + _JSON_NUMBER
)


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _composition_key(formula: str) -> str:
    return Composition(str(formula)).reduced_composition.alphabetical_formula


def _mask_forbidden_model_values(raw: bytes) -> tuple[bytes, dict[str, int]]:
    """Replace forbidden numeric tokens before the JSON decoder sees their values."""

    counts: Counter[str] = Counter()

    def replace(match: re.Match[bytes]) -> bytes:
        key = match.group("key").decode("ascii")
        counts[key] += 1
        return b'"' + match.group("key") + b'"' + match.group("separator") + b"null"

    masked = _MODEL_VALUE_PATTERN.sub(replace, raw)
    return masked, {name: int(counts[name]) for name in FORBIDDEN_MODEL_FIELDS}


def _read_initial_records(path: Path) -> tuple[list[dict[str, object]], dict[str, int]]:
    with zipfile.ZipFile(path) as archive:
        members = [name for name in archive.namelist() if Path(name).name == "all_pred_data.json"]
        if len(members) != 1:
            raise ValueError("NEXT541 initial JSON member inventory differs")
        raw = archive.read(members[0])
    masked, counts = _mask_forbidden_model_values(raw)
    value = json.loads(masked)
    if not isinstance(value, list) or len(value) != EXPECTED_INITIAL_ROWS:
        raise ValueError("NEXT541 initial record count differs")
    records: list[dict[str, object]] = []
    for raw_record in value:
        if not isinstance(raw_record, dict):
            raise ValueError("NEXT541 initial record schema differs")
        if any(raw_record.get(name) is not None for name in FORBIDDEN_MODEL_FIELDS):
            raise ValueError("NEXT541 forbidden model value was not masked")
        if set(raw_record) != {"formula", "atoms", *FORBIDDEN_MODEL_FIELDS}:
            raise ValueError("NEXT541 initial record fields differ")
        records.append(raw_record)
    if counts != {name: EXPECTED_INITIAL_ROWS for name in FORBIDDEN_MODEL_FIELDS}:
        raise ValueError("NEXT541 forbidden model field counts differ")
    return records, counts


def _endpoint_formula_token(filename: str) -> str:
    match = re.fullmatch(r"POSCAR-(?P<formula>.+?)(?:-v[0-9]+)?\.vasp", filename)
    if match is None:
        raise ValueError(f"NEXT541 endpoint filename differs: {filename}")
    return str(match.group("formula"))


def endpoint_inventory(path: Path) -> pd.DataFrame:
    """Read only ZIP central-directory names; never open a member payload."""

    with zipfile.ZipFile(path) as archive:
        members = [
            info.filename
            for info in archive.infolist()
            if not info.is_dir()
            and "__MACOSX" not in Path(info.filename).parts
            and Path(info.filename).name.startswith("POSCAR-")
            and Path(info.filename).suffix == ".vasp"
        ]
    rows = []
    for member in sorted(members, key=lambda value: (Path(value).name, value)):
        filename = Path(member).name
        token = _endpoint_formula_token(filename)
        rows.append(
            {
                "endpoint_member": member,
                "endpoint_filename": filename,
                "endpoint_formula_token": token,
                "composition_key": _composition_key(token),
            }
        )
    return pd.DataFrame(rows)


def _atoms_from_initial_record(record: dict[str, object]) -> Atoms:
    atoms_value = record.get("atoms")
    if not isinstance(atoms_value, dict):
        raise ValueError("NEXT541 atoms payload differs")
    required = {"lattice_mat", "coords", "elements", "cartesian"}
    if required - set(atoms_value):
        raise ValueError("NEXT541 atoms geometry fields differ")
    elements = atoms_value["elements"]
    coords = np.asarray(atoms_value["coords"], dtype=float)
    cell = np.asarray(atoms_value["lattice_mat"], dtype=float)
    if (
        not isinstance(elements, list)
        or coords.shape != (len(elements), 3)
        or cell.shape != (3, 3)
        or not np.isfinite(coords).all()
        or not np.isfinite(cell).all()
        or abs(float(np.linalg.det(cell))) <= 1.0e-10
        or type(atoms_value["cartesian"]) is not bool
    ):
        raise ValueError("NEXT541 raw periodic geometry differs")
    kwargs: dict[str, object] = {
        "symbols": [str(value) for value in elements],
        "cell": cell,
        "pbc": True,
    }
    if atoms_value["cartesian"]:
        kwargs["positions"] = coords
    else:
        kwargs["scaled_positions"] = coords
    atoms = Atoms(**kwargs)
    atoms.info.clear()
    return atoms


def _midrank_percentile(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    result = np.full(values.shape, np.nan, dtype=float)
    finite = np.isfinite(values)
    count = int(finite.sum())
    if count:
        result[finite] = (rankdata(values[finite], method="average") - 0.5) / count
    return result


def mechanism_union_percentile_risk(
    contact_risk: np.ndarray, sssp_risk: np.ndarray, pbaaa_risk: np.ndarray
) -> dict[str, np.ndarray]:
    arrays = [np.asarray(value, dtype=float) for value in (contact_risk, sssp_risk, pbaaa_risk)]
    if len({value.shape for value in arrays}) != 1 or arrays[0].ndim != 1:
        raise ValueError("NEXT541 mechanism arrays differ")
    percentiles = [_midrank_percentile(value) for value in arrays]
    contact_supported = np.isfinite(percentiles[0])
    product = np.ones(arrays[0].shape, dtype=float)
    count = np.zeros(arrays[0].shape, dtype=int)
    for values in percentiles:
        supported = np.isfinite(values)
        product[supported] *= 1.0 - values[supported]
        count[supported] += 1
    risk = 1.0 - product
    risk[~contact_supported] = np.nan
    count[~contact_supported] = 0
    return {
        "contact_percentile": percentiles[0],
        "sssp_percentile": percentiles[1],
        "pbaaa_percentile": percentiles[2],
        "mupr_risk": risk,
        "mechanism_count": count,
    }


def _failure_text(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def _compute_candidate(item: tuple[int, dict[str, object]]) -> dict[str, object]:
    index, record = item
    started = time.perf_counter()
    base: dict[str, object] = {
        "initial_index": int(index),
        "candidate_id": f"jarvis_cdvae_initial_{index:04d}",
        "formula": str(record["formula"]),
        "composition_key": _composition_key(str(record["formula"])),
    }
    try:
        atoms = _atoms_from_initial_record(record)
        geometry_bytes = np.asarray(atoms.cell.array, dtype="<f8").tobytes()
        geometry_bytes += np.asarray(atoms.get_scaled_positions(wrap=True), dtype="<f8").tobytes()
        geometry_bytes += "\0".join(atoms.get_chemical_symbols()).encode()
        base.update(
            {
                "n_sites": len(atoms),
                "initial_geometry_sha256": hashlib.sha256(geometry_bytes).hexdigest(),
            }
        )
    except Exception as exc:
        base.update(
            {
                "n_sites": 0,
                "initial_geometry_sha256": "",
                "contact_supported": False,
                "contact_failure": _failure_text(exc),
                "cov_q05": math.nan,
                "sssp_supported": False,
                "sssp_failure": _failure_text(exc),
                SSSP_FEATURE: math.nan,
                "pbaaa_supported": False,
                "pbaaa_failure": _failure_text(exc),
                PBAAA_FEATURE: math.nan,
                "pauling_feature_error": _failure_text(exc),
                "pauling_violation_fraction": math.nan,
                "pauling_p2_p5_decision": "ABSTAIN",
                "feature_runtime_seconds": time.perf_counter() - started,
            }
        )
        return base

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
    supported_decisions = [value for value in decisions if value != "ABSTAIN"]
    base.update(pauling)
    base["pauling_violation_fraction"] = (
        float(np.mean([value == "REJECT" for value in supported_decisions]))
        if supported_decisions
        else math.nan
    )
    base["feature_runtime_seconds"] = time.perf_counter() - started
    return base


def _compute_many(items: list[tuple[int, dict[str, object]]], workers: int) -> list[dict[str, object]]:
    if workers == 1:
        iterator = map(_compute_candidate, items)
        executor = None
    else:
        executor = ProcessPoolExecutor(max_workers=workers)
        iterator = executor.map(_compute_candidate, items, chunksize=1)
    rows: list[dict[str, object]] = []
    try:
        for offset, row in enumerate(iterator, start=1):
            rows.append(row)
            if offset % 25 == 0 or offset == len(items):
                print(f"NEXT541 initial-only features: {offset}/{len(items)}", flush=True)
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    return rows


def _gate_statistics(table: pd.DataFrame) -> dict[str, object]:
    rows = int(len(table))
    contact = table["contact_supported"].fillna(False).to_numpy(bool)
    mupr = pd.to_numeric(table["mupr_risk"], errors="coerce").to_numpy(float)
    supported = np.isfinite(mupr)
    finite = mupr[supported]
    rounded = np.round(finite, 12)
    frequencies = Counter(rounded.tolist())
    unique = len(frequencies)
    maximum_point_mass = max(frequencies.values(), default=rows) / rows if rows else 1.0
    bounded = bool(len(finite) and np.all((finite >= 0.0) & (finite <= 1.0)))
    statistics = {
        "rows": rows,
        "contact_supported": int(contact.sum()),
        "contact_coverage": float(contact.mean()) if rows else 0.0,
        "mupr_supported": int(supported.sum()),
        "mupr_coverage": float(supported.mean()) if rows else 0.0,
        "mupr_unique_rounded_12": int(unique),
        "mupr_maximum_point_mass": float(maximum_point_mass),
        "mupr_bounded": bounded,
        "mechanism_count_distribution": {
            str(key): int(value)
            for key, value in sorted(Counter(table["mechanism_count"].astype(int)).items())
        },
    }
    statistics["passes"] = bool(
        rows == EXPECTED_CANDIDATES
        and statistics["contact_coverage"] >= MINIMUM_CONTACT_COVERAGE
        and statistics["mupr_coverage"] >= MINIMUM_MUPR_COVERAGE
        and unique >= MINIMUM_MUPR_UNIQUE
        and maximum_point_mass <= MAXIMUM_MUPR_POINT_MASS
        and bounded
    )
    return statistics


def build_initial_prediction_freeze(
    *,
    initial_archive: Path,
    endpoint_archive: Path,
    design_path: Path,
    output_dir: Path,
    workers: int = 8,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    target = Path(output_dir).resolve()
    paths = {
        "design": Path(design_path).resolve(),
        "initial_archive": Path(initial_archive).resolve(),
        "endpoint_archive": Path(endpoint_archive).resolve(),
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
        raise ValueError("NEXT541 workers differ")
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT541 input is missing")
    hashes = {name: _sha256(path) for name, path in paths.items()}
    if require_formal_inputs and hashes != EXPECTED_INPUT_SHA256:
        differing = sorted(
            name
            for name in set(hashes) | set(EXPECTED_INPUT_SHA256)
            if hashes.get(name) != EXPECTED_INPUT_SHA256.get(name)
        )
        raise ValueError(f"NEXT541 formal input identity differs: {differing}")

    inventory = endpoint_inventory(paths["endpoint_archive"])
    if (
        len(inventory) != EXPECTED_ENDPOINT_FILES
        or inventory["endpoint_member"].duplicated().any()
        or inventory["composition_key"].nunique() != EXPECTED_COMPOSITIONS
    ):
        raise ValueError("NEXT541 endpoint name inventory differs")
    records, masked_counts = _read_initial_records(paths["initial_archive"])
    endpoint_compositions = set(inventory["composition_key"].astype(str))
    items = [
        (index, record)
        for index, record in enumerate(records)
        if _composition_key(str(record["formula"])) in endpoint_compositions
    ]
    if len(items) != EXPECTED_CANDIDATES:
        raise ValueError("NEXT541 possible initial candidate count differs")

    started = time.perf_counter()
    table = pd.DataFrame(_compute_many(items, workers)).sort_values(
        "initial_index", kind="mergesort"
    )
    ranks = mechanism_union_percentile_risk(
        -pd.to_numeric(table["cov_q05"], errors="coerce").to_numpy(float),
        -pd.to_numeric(table[SSSP_FEATURE], errors="coerce").to_numpy(float),
        pd.to_numeric(table[PBAAA_FEATURE], errors="coerce").to_numpy(float),
    )
    for name, values in ranks.items():
        table[name] = values
    forbidden_columns = [name for name in table if name in FORBIDDEN_MODEL_FIELDS]
    if forbidden_columns:
        raise RuntimeError(f"NEXT541 forbidden model columns escaped: {forbidden_columns}")
    statistics = _gate_statistics(table)
    if statistics["passes"] is not True:
        raise RuntimeError(f"NEXT541 label-blind gates failed: {statistics}")
    failure_counts = {
        family: dict(
            sorted(
                Counter(
                    table.loc[~table[f"{family}_supported"], f"{family}_failure"]
                    .fillna("")
                    .astype(str)
                ).items()
            )
        )
        for family in ("contact", "sssp", "pbaaa")
    }

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    source_path = Path(__file__).resolve()
    source_hash = _sha256(source_path)
    try:
        table_path = staging / TABLE_NAME
        inventory_path = staging / INVENTORY_NAME
        formula_path = staging / FORMULA_NAME
        table.to_parquet(table_path, index=False)
        inventory_path.write_bytes(_json_bytes(inventory.to_dict("records")))
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
            "missing_policy": "contact_required; omit unsupported SSSP or PBAAA",
            "operating_rule": "reject top 15 percent among mapped batch by R; initial_index breaks ties",
            "coefficients_fitted_to_endpoint": False,
        }
        formula_path.write_bytes(_json_bytes(formula))
        outputs = {
            table_path.name: _sha256(table_path),
            inventory_path.name: _sha256(inventory_path),
            formula_path.name: _sha256(formula_path),
        }
        table_digest = hashlib.sha256(
            pd.util.hash_pandas_object(table, index=False).to_numpy(dtype="<u8").tobytes()
        ).hexdigest()
        manifest = {
            "protocol": PROTOCOL,
            "mode": "all_possible_initial_candidates_scored_before_endpoint_content_access",
            "counts": {
                "initial_records": len(records),
                "endpoint_filenames": len(inventory),
                "endpoint_compositions": inventory["composition_key"].nunique(),
                "possible_initial_candidates": len(table),
            },
            "gates": statistics,
            "failure_counts": failure_counts,
            "forbidden_model_fields": list(FORBIDDEN_MODEL_FIELDS),
            "forbidden_model_value_tokens_masked_before_json_decode": masked_counts,
            "forbidden_model_values_used": False,
            "endpoint_archive_central_directory_read": True,
            "endpoint_member_payloads_read": False,
            "endpoint_final_lattices_or_coordinates_read": False,
            "endpoint_values_summarized_or_inspected": False,
            "scores_frozen_before_endpoint_member_payloads_read": True,
            "next542_endpoint_evaluation_authorized": True,
            "inputs_sha256": {
                name: {"path": str(path), "sha256": hashes[name]}
                for name, path in paths.items()
            },
            "outputs_sha256": outputs,
            "deterministic_table_digest": table_digest,
            "executed_source_sha256": {
                "src/next541_jarvis_cdvae_initial_prediction_freeze.py": source_hash
            },
            "feature_protocols": {
                "sssp": SSSP_PROTOCOL,
                "pbaaa": PBAAA_PROTOCOL,
                "contact": "2026-08-03-next32-omat24-inorganic-response-features-v1",
                "pauling": "2026-08-02-next12-prospective-pauling2-5-controls-v1",
            },
            "workers": workers,
            "runtime_seconds": time.perf_counter() - started,
            "dft_or_relaxed_input_used_by_screen": False,
            "ml_energy_force_stress_or_property_proxy_used_by_screen": False,
            "coordinate_relaxation_or_coordinate_generation_performed": False,
            "scientific_improvement_claim": False,
        }
        (staging / MANIFEST_NAME).write_bytes(_json_bytes(manifest))
        if _sha256(source_path) != source_hash:
            raise RuntimeError("NEXT541 source changed before publication")
        if any(_sha256(path) != hashes[name] for name, path in paths.items()):
            raise RuntimeError("NEXT541 input changed before publication")
        _publish_directory_no_replace(staging, target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-archive", type=Path, required=True)
    parser.add_argument("--endpoint-archive", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = build_initial_prediction_freeze(
        initial_archive=args.initial_archive,
        endpoint_archive=args.endpoint_archive,
        design_path=args.design,
        output_dir=args.output_dir,
        workers=args.workers,
    )
    print(json.dumps(manifest["gates"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
