#!/usr/bin/env python3
"""Freeze the development-derived one-term HEA packing-deficit law."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile

from src.next19_feature_build import _publish_directory_no_replace, _sha256
import src.next555_hea_extreme_waste_search as n555


PROTOCOL = "2026-08-14-next557-hea-packing-deficit-freeze-v1"
DESIGN_SHA256 = "bebb4b157c95bc0aa769e54809d54515f812978f49f8c5dfacea88c8b4cc45f6"
FEATURE = "primitive_covalent_packing_fraction__risk_low"
FORMULA_NAME = "NEXT557_FROZEN_PACKING_DEFICIT.json"
MANIFEST_NAME = "MANIFEST.json"


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def freeze_law(
    *, next555_dir: Path, design_path: Path, output_dir: Path
) -> dict[str, object]:
    upstream = Path(next555_dir).resolve()
    design_path = Path(design_path).resolve()
    target = Path(output_dir).resolve()
    paths = {
        "design": design_path,
        "manifest": upstream / n555.MANIFEST_NAME,
        "search": upstream / n555.SEARCH_NAME,
        "source": Path(n555.__file__).resolve(),
    }
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT557 input is missing")
    if _sha256(design_path) != DESIGN_SHA256:
        raise ValueError("NEXT557 design identity differs")
    manifest555 = json.loads(paths["manifest"].read_text())
    search = json.loads(paths["search"].read_text())
    outputs = manifest555.get("outputs_sha256")
    if (
        manifest555.get("protocol") != n555.PROTOCOL
        or manifest555.get("eligible_pair_count") != 0
        or manifest555.get("winner_frozen") is not False
        or manifest555.get("validation_endpoint_values_opened") is not False
        or manifest555.get("validation_final_structures_opened") is not False
        or not isinstance(outputs, dict)
        or outputs.get(n555.SEARCH_NAME) != _sha256(paths["search"])
    ):
        raise ValueError("NEXT557 NEXT555 stopped-state differs")
    rows = [
        row for row in search.get("retained_directions", []) if row.get("key") == FEATURE
    ]
    if len(rows) != 1:
        raise ValueError("NEXT557 development direction identity differs")
    metrics = rows[0]["metrics"]
    overall, ordered, sqs = metrics["overall"], metrics["ordered"], metrics["sqs"]
    top = overall["top_15_percent"]
    if not (
        overall["coverage"] >= 0.95
        and overall["roc_auc"] >= 0.70
        and ordered["roc_auc"] >= 0.67
        and sqs["roc_auc"] >= 0.67
        and overall["spearman_severity"] >= 0.34
        and top["lift"] >= 1.60
        and top["protected"] / top["rows"] <= 0.02
    ):
        raise ValueError("NEXT557 development evidence differs")
    formula = {
        "protocol": PROTOCOL,
        "name": "HEA covalent packing-capacity deficit",
        "feature": FEATURE,
        "raw_feature": "sum_i(4*pi*r_cov_i^3/3)/V_cell",
        "risk_direction": "lower packing fraction is higher risk",
        "score": "cohort-wide midrank percentile of negative raw feature",
        "operating_rule": "reject highest-risk 15 percent; FID breaks ties",
        "development_metrics": metrics,
        "one_term": True,
        "endpoint_fitted_coefficients": False,
        "validation_endpoints_opened": False,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    source_path = Path(__file__).resolve()
    source_hash = _sha256(source_path)
    try:
        formula_path = staging / FORMULA_NAME
        formula_path.write_bytes(_json_bytes(formula))
        manifest = {
            "protocol": PROTOCOL,
            "inputs_sha256": {
                name: {"path": str(path), "sha256": _sha256(path)}
                for name, path in paths.items()
            },
            "outputs_sha256": {FORMULA_NAME: _sha256(formula_path)},
            "executed_source_sha256": {
                "src/next557_hea_packing_deficit_freeze.py": source_hash
            },
            "development_derived_hypothesis": True,
            "validation_endpoint_values_opened": False,
            "validation_final_structures_opened": False,
            "dft_values_used_by_executable_formula": False,
            "model_or_proxy_potential_used": False,
            "coordinates_or_cell_modified": False,
            "next558_validation_opening_authorized": True,
            "scientific_improvement_claim": False,
        }
        (staging / MANIFEST_NAME).write_bytes(_json_bytes(manifest))
        if _sha256(source_path) != source_hash:
            raise RuntimeError("NEXT557 source changed before publication")
        _publish_directory_no_replace(staging, target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--next555-dir", required=True, type=Path)
    parser.add_argument("--design-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    manifest = freeze_law(
        next555_dir=args.next555_dir,
        design_path=args.design_path,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
