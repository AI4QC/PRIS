#!/usr/bin/env python3
"""Re-run the frozen HEA catalogue on the development-calibrated extreme endpoint."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd

from src.next19_feature_build import _publish_directory_no_replace, _sha256
import src.next552_hea_analytic_feature_freeze as n552
import src.next553_hea_development_search as n553


PROTOCOL = "2026-08-13-next555-hea-extreme-waste-search-v1"
DESIGN_SHA256 = "2186cbaaad2a80b6142948af2381a32ced0aace01d1ae727d04560fa37ccc01f"
ENERGY_HULL_THRESHOLD = 0.40
PROTECTED_ENERGY_MAX = 0.10
PROTECTED_DISPLACEMENT_MAX = 0.10
PROTECTED_CELL_LOGSTRAIN_MAX = 0.04
PROTECTED_VOLUME_LOGCHANGE_MAX = 0.05
MINIMUM_PROTECTED = 50
ENDPOINT_TABLE_NAME = "next555_hea_extreme_development_endpoints.parquet"
SEARCH_NAME = "NEXT555_HEA_EXTREME_SEARCH.json"
FORMULA_NAME = "NEXT555_FROZEN_FORMULA.json"
MANIFEST_NAME = "MANIFEST.json"
EXPECTED_INPUT_SHA256 = {
    "design": DESIGN_SHA256,
    "next552_manifest": "925d0678a5a7347c277f5a2a5040adf0f80f2b95c2b18fced2f432a7819dc947",
    "next552_table": "7e8fa3bbe92ffaff186c641fb487972e691c1d583eb01e9b3cd714611334a46d",
    "next552_catalogue": "8a4ef3882c640ca08f9d3ffdfca0589ee04de81e24f836bf542876501c156ada",
    "next553_manifest": "63096e2a7771b06b81506511845f5ef3c331f58f2f0123fe9171c6baff0bbf1e",
    "next553_endpoints": "6c4438e7f775d3e000556b78ccad89e3666e9105a48dfe158c977c65ecfbd029",
    "next553_search": "edd74b60212f0cc5de38c1b1815051dac80d9abd321e1a8956165c2b3306f198",
    "next552_source": "20c74208c9072aadfdaeb310ad391b491fe05f5dcb52e69f5f1c0aaf2ff0e25f",
    "next553_source": "9370c8db471f784c7f4314ea3c6f132e52d17963797580f59e47059d8501e927",
}


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def apply_extreme_waste_endpoint(table: pd.DataFrame) -> pd.DataFrame:
    required = {
        "fid", "e_above_hull", "disp_p90", "cell_logstrain_max", "volume_logchange"
    }
    if required - set(table):
        raise ValueError("NEXT555 endpoint table differs")
    result = table.copy()
    energy = pd.to_numeric(result["e_above_hull"], errors="coerce").to_numpy(float)
    displacement = pd.to_numeric(result["disp_p90"], errors="coerce").to_numpy(float)
    strain = pd.to_numeric(result["cell_logstrain_max"], errors="coerce").to_numpy(float)
    volume = pd.to_numeric(result["volume_logchange"], errors="coerce").to_numpy(float)
    if not np.isfinite(np.column_stack([energy, displacement, strain, volume])).all():
        raise ValueError("NEXT555 endpoint contains nonfinite values")
    result["energetically_unstable"] = energy >= ENERGY_HULL_THRESHOLD
    result["large_geometric_response"] = (
        (displacement >= n553.DISPLACEMENT_P90_THRESHOLD)
        | (strain >= n553.CELL_LOGSTRAIN_THRESHOLD)
        | (volume >= n553.VOLUME_LOGCHANGE_THRESHOLD)
    )
    result["dft_waste"] = (
        result["energetically_unstable"].to_numpy(bool)
        | result["large_geometric_response"].to_numpy(bool)
    )
    result["waste_severity"] = np.max(
        np.column_stack(
            [
                energy / ENERGY_HULL_THRESHOLD,
                displacement / n553.DISPLACEMENT_P90_THRESHOLD,
                strain / n553.CELL_LOGSTRAIN_THRESHOLD,
                volume / n553.VOLUME_LOGCHANGE_THRESHOLD,
            ]
        ),
        axis=1,
    )
    result["protected"] = (
        (energy <= PROTECTED_ENERGY_MAX)
        & (displacement <= PROTECTED_DISPLACEMENT_MAX)
        & (strain <= PROTECTED_CELL_LOGSTRAIN_MAX)
        & (volume <= PROTECTED_VOLUME_LOGCHANGE_MAX)
    )
    return result


def _class_gates(table: pd.DataFrame) -> dict[str, object]:
    labels = table["dft_waste"].to_numpy(bool)
    families: dict[str, object] = {}
    for family in ("ordered", "sqs"):
        subset = table.loc[table["size_family"].astype(str).eq(family)]
        families[family] = {
            "rows": len(subset),
            "positive": int(subset["dft_waste"].sum()),
            "negative": int((~subset["dft_waste"]).sum()),
            "protected": int(subset["protected"].sum()),
        }
    result = {
        "rows": len(table),
        "positive": int(labels.sum()),
        "negative": int((~labels).sum()),
        "protected": int(table["protected"].sum()),
        "families": families,
    }
    result["passes"] = bool(
        len(table) == 1_200
        and result["positive"] >= n553.MINIMUM_CLASS_COUNT
        and result["negative"] >= n553.MINIMUM_CLASS_COUNT
        and result["protected"] >= MINIMUM_PROTECTED
        and all(
            families[family]["positive"] >= n553.MINIMUM_FAMILY_CLASS_COUNT
            and families[family]["negative"] >= n553.MINIMUM_FAMILY_CLASS_COUNT
            for family in ("ordered", "sqs")
        )
    )
    return result


def build_extreme_search(
    *, next552_dir: Path, next553_dir: Path, design_path: Path, output_dir: Path,
    bootstrap_draws: int = n553.BOOTSTRAP_DRAWS,
) -> dict[str, object]:
    root552 = Path(next552_dir).resolve()
    root553 = Path(next553_dir).resolve()
    design_path = Path(design_path).resolve()
    target = Path(output_dir).resolve()
    paths = {
        "design": design_path,
        "next552_manifest": root552 / n552.MANIFEST_NAME,
        "next552_table": root552 / n552.TABLE_NAME,
        "next552_catalogue": root552 / n552.CATALOGUE_NAME,
        "next553_manifest": root553 / n553.MANIFEST_NAME,
        "next553_endpoints": root553 / n553.ENDPOINT_TABLE_NAME,
        "next553_search": root553 / n553.SEARCH_NAME,
        "next552_source": Path(n552.__file__).resolve(),
        "next553_source": Path(n553.__file__).resolve(),
    }
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT555 input is missing")
    hashes = {name: _sha256(path) for name, path in paths.items()}
    if hashes != EXPECTED_INPUT_SHA256:
        differing = sorted(
            name for name in set(hashes) | set(EXPECTED_INPUT_SHA256)
            if hashes.get(name) != EXPECTED_INPUT_SHA256.get(name)
        )
        raise ValueError(f"NEXT555 formal input identity differs: {differing}")
    manifest553 = json.loads(paths["next553_manifest"].read_text())
    outputs553 = manifest553.get("outputs_sha256")
    if (
        manifest553.get("protocol") != n553.PROTOCOL
        or manifest553.get("development_endpoint_values_opened") is not True
        or manifest553.get("validation_endpoint_values_opened") is not False
        or manifest553.get("validation_final_structures_opened") is not False
        or manifest553.get("validation_endpoint_fields_copied_or_decoded") is not False
        or manifest553.get("search_executed") is not False
        or not isinstance(outputs553, dict)
        or outputs553.get(n553.ENDPOINT_TABLE_NAME) != hashes["next553_endpoints"]
        or outputs553.get(n553.SEARCH_NAME) != hashes["next553_search"]
    ):
        raise ValueError("NEXT555 NEXT553 stopped-state identity differs")
    endpoints = pd.read_parquet(paths["next553_endpoints"])
    endpoints = apply_extreme_waste_endpoint(endpoints)
    class_gates = _class_gates(endpoints)
    if class_gates["passes"] is not True:
        raise RuntimeError(f"NEXT555 calibrated endpoint gates failed: {class_gates}")
    features = pd.read_parquet(paths["next552_table"])
    features = features.loc[features["partition"].astype(str).eq("development")].copy()
    table = features.merge(
        endpoints[
            ["fid", "e_above_hull", "disp_p90", "cell_logstrain_max", "volume_logchange",
             "dft_waste", "waste_severity", "protected"]
        ],
        on="fid", validate="one_to_one",
    )
    if len(table) != 1_200:
        raise ValueError("NEXT555 development join differs")
    catalogue = json.loads(paths["next552_catalogue"].read_text())
    search = n553.run_bounded_search(table, catalogue, bootstrap_draws=bootstrap_draws)
    search["protocol"] = PROTOCOL
    search["endpoint"] = {
        "development_calibrated": True,
        "energy_hull_threshold": ENERGY_HULL_THRESHOLD,
        "displacement_p90_threshold": n553.DISPLACEMENT_P90_THRESHOLD,
        "cell_logstrain_threshold": n553.CELL_LOGSTRAIN_THRESHOLD,
        "volume_logchange_threshold": n553.VOLUME_LOGCHANGE_THRESHOLD,
    }
    search["class_gates"] = class_gates

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    source_path = Path(__file__).resolve()
    source_hash = _sha256(source_path)
    try:
        endpoint_path = staging / ENDPOINT_TABLE_NAME
        search_path = staging / SEARCH_NAME
        endpoints.to_parquet(endpoint_path, index=False)
        search_path.write_bytes(_json_bytes(search))
        outputs_out = {ENDPOINT_TABLE_NAME: _sha256(endpoint_path), SEARCH_NAME: _sha256(search_path)}
        winner = search.get("winner")
        if isinstance(winner, dict):
            formula = {
                "protocol": PROTOCOL,
                "formula_id": winner["formula_id"],
                "first": winner["first"],
                "second": winner["second"],
                "formula": winner["formula"],
                "feature_ranks_frozen_before_development_endpoints": True,
                "endpoint_fitted_coefficients": False,
                "validation_endpoints_opened": False,
                "endpoint_is_development_calibrated": True,
            }
            formula_path = staging / FORMULA_NAME
            formula_path.write_bytes(_json_bytes(formula))
            outputs_out[FORMULA_NAME] = _sha256(formula_path)
        manifest = {
            "protocol": PROTOCOL,
            "class_gates": class_gates,
            "eligible_pair_count": int(search["eligible_pair_count"]),
            "winner_frozen": isinstance(winner, dict),
            "next556_validation_opening_authorized": isinstance(winner, dict),
            "inputs_sha256": {
                name: {"path": str(path), "sha256": hashes[name]}
                for name, path in paths.items()
            },
            "outputs_sha256": outputs_out,
            "executed_source_sha256": {
                "src/next555_hea_extreme_waste_search.py": source_hash
            },
            "development_endpoint_reused": True,
            "endpoint_is_development_calibrated": True,
            "validation_endpoint_values_opened": False,
            "validation_final_structures_opened": False,
            "dft_values_used_by_executable_formula": False,
            "model_or_proxy_potential_used": False,
            "coordinates_or_cell_modified_by_formula": False,
            "scientific_improvement_claim": False,
        }
        (staging / MANIFEST_NAME).write_bytes(_json_bytes(manifest))
        if _sha256(source_path) != source_hash:
            raise RuntimeError("NEXT555 source changed before publication")
        _publish_directory_no_replace(staging, target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--next552-dir", required=True, type=Path)
    parser.add_argument("--next553-dir", required=True, type=Path)
    parser.add_argument("--design-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bootstrap-draws", type=int, default=n553.BOOTSTRAP_DRAWS)
    args = parser.parse_args(argv)
    manifest = build_extreme_search(
        next552_dir=args.next552_dir,
        next553_dir=args.next553_dir,
        design_path=args.design_path,
        output_dir=args.output_dir,
        bootstrap_draws=args.bootstrap_draws,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["apply_extreme_waste_endpoint", "build_extreme_search"]
