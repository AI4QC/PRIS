"""Freeze the development-promoted NEXT17 rule before external evaluation."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import shutil
import tempfile
from typing import Mapping, Sequence

from src.next13d_acsc_dft_pairs import _json_bytes, _sha256_file, _strict_json
from src.next14_wbm_holdout import _publish_directory_no_replace
from src.next17_frozen_rule import (
    FROZEN_FORMULA,
    FROZEN_RELAXATION,
    FROZEN_THRESHOLD_EV_PER_ATOM,
)
from src.next17_strict_relax_gap import PROTOCOL as FEATURE_PROTOCOL
from src.next17_strict_relax_gap_evaluate import PROTOCOL as DEVELOPMENT_PROTOCOL


PROTOCOL = "2026-08-02-next17-strict-relax-gap-freeze-v1"
FROZEN_NAME = "FROZEN_PROTOCOL.json"
MANIFEST_NAME = "MANIFEST.json"
FROZEN_FORMAL_SHA256: Mapping[str, str] = {
    "strict_features": "afb86ff81b5bee3159dc96355b9402ee3fef7ac07479b2110ef6d510a306af2a",
    "strict_manifest": "674fc360fdb2f9fbb2c98c1853683d6a390999ac3c84f47776036ced42b80102",
    "development_result": "a5f09124c539316235cd86f07c377a56130b04358ac7f6f6185e3e956ebec3fd",
    "development_manifest": "1fd8816593fc33f22a2fea3e3d3706701365cd367bc67594c5c3a9d6dd39f1b6",
    "checkpoint": "e3df9fa708725e3d453140646c7d1838324b347a3d1214cf1440522146f872b5",
}


def _bound_output(
    manifest: Mapping[str, object], *, path: Path, data: bytes, role: str
) -> None:
    outputs = manifest.get("outputs_sha256")
    if not isinstance(outputs, Mapping) or outputs.get(path.name) != hashlib.sha256(data).hexdigest():
        raise ValueError(f"{role} output hash differs from manifest")


def freeze_next17(
    *,
    strict_features_path: Path,
    strict_manifest_path: Path,
    development_result_path: Path,
    development_manifest_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
    require_formal_inputs: bool = True,
) -> dict[str, object]:
    """Publish one immutable candidate only after verifying its development promotion."""

    target = Path(output_dir).resolve()
    if os.path.lexists(target):
        raise FileExistsError(f"refusing existing output: {target}")
    paths = {
        "strict_features": Path(strict_features_path).resolve(),
        "strict_manifest": Path(strict_manifest_path).resolve(),
        "development_result": Path(development_result_path).resolve(),
        "development_manifest": Path(development_manifest_path).resolve(),
        "checkpoint": Path(checkpoint_path).resolve(),
    }
    for role, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{role} is not a file: {path}")
    input_hashes = {role: _sha256_file(path) for role, path in paths.items()}
    if require_formal_inputs and input_hashes != dict(FROZEN_FORMAL_SHA256):
        raise ValueError("formal NEXT17 freeze inputs differ")

    feature_data = paths["strict_features"].read_bytes()
    feature_manifest = _strict_json(
        paths["strict_manifest"].read_bytes(), role="NEXT17 strict feature manifest"
    )
    if feature_manifest.get("protocol") != FEATURE_PROTOCOL:
        raise ValueError("NEXT17 feature protocol differs")
    _bound_output(
        feature_manifest,
        path=paths["strict_features"],
        data=feature_data,
        role="NEXT17 strict feature",
    )
    if (
        feature_manifest.get("formula") != FROZEN_FORMULA
        or feature_manifest.get("relaxation") != FROZEN_RELAXATION
        or feature_manifest.get("elementa_endpoint_bytes_read_by_execution") is not False
        or feature_manifest.get("mp_hull_bytes_read_by_execution") is not False
        or feature_manifest.get("threshold_selected") is not False
    ):
        raise ValueError("NEXT17 feature formula, relaxation, or leakage contract differs")
    feature_inputs = feature_manifest.get("inputs_sha256")
    checkpoint_binding = (
        feature_inputs.get("checkpoint") if isinstance(feature_inputs, Mapping) else None
    )
    if (
        not isinstance(checkpoint_binding, Mapping)
        or checkpoint_binding.get("sha256") != input_hashes["checkpoint"]
    ):
        raise ValueError("NEXT17 feature checkpoint binding differs")

    development_data = paths["development_result"].read_bytes()
    development_manifest = _strict_json(
        paths["development_manifest"].read_bytes(), role="NEXT17 development manifest"
    )
    if (
        development_manifest.get("protocol") != DEVELOPMENT_PROTOCOL
        or development_manifest.get("identifier_bearing") is not False
    ):
        raise ValueError("NEXT17 development manifest contract differs")
    _bound_output(
        development_manifest,
        path=paths["development_result"],
        data=development_data,
        role="NEXT17 development",
    )
    development = _strict_json(development_data, role="NEXT17 development result")
    strict_result = development.get("strict_relax")
    comparison = development.get("selected_comparison_strict_minus_x0")
    catalog = strict_result.get("catalog_scan") if isinstance(strict_result, Mapping) else None
    selected_entry = catalog.get(str(FROZEN_THRESHOLD_EV_PER_ATOM)) if isinstance(catalog, Mapping) else None
    if (
        development.get("protocol") != DEVELOPMENT_PROTOCOL
        or development.get("development_promotion") is not True
        or development.get("scientific_improvement_claim") is not False
        or not isinstance(strict_result, Mapping)
        or strict_result.get("selected_threshold_ev_per_atom") != FROZEN_THRESHOLD_EV_PER_ATOM
        or not isinstance(selected_entry, Mapping)
        or selected_entry.get("eligible") is not True
        or not isinstance(comparison, Mapping)
        or comparison.get("passes_development_promotion") is not True
    ):
        raise ValueError("NEXT17 development promotion or selected threshold differs")

    repo_root = Path(__file__).resolve().parents[1]
    source_paths = {
        "src/next17_frozen_rule.py": repo_root / "src/next17_frozen_rule.py",
        "src/next17_strict_relax_gap.py": repo_root / "src/next17_strict_relax_gap.py",
        "src/next17_strict_relax_gap_evaluate.py": repo_root / "src/next17_strict_relax_gap_evaluate.py",
        "src/next17_freeze.py": Path(__file__).resolve(),
    }
    source_hashes = {name: _sha256_file(path) for name, path in source_paths.items()}
    frozen: dict[str, object] = {
        "protocol": PROTOCOL,
        "status": "development-promoted candidate frozen for external falsification",
        "evidence_role": "immutable pre-external-evaluation candidate",
        "development_promotion_verified": True,
        "fresh_lockbox": False,
        "rule": {
            "name": "NEXT17_R64s",
            "formula": FROZEN_FORMULA,
            "comparison": ">=",
            "threshold_ev_per_atom": FROZEN_THRESHOLD_EV_PER_ATOM,
            "failure_policy": "ABSTAIN entire incomplete group",
            "application_scope": "complete same-composition candidate groups",
        },
        "relaxation": dict(FROZEN_RELAXATION),
        "model": {
            "package": "mattersim",
            "version": "1.2.3",
            "checkpoint_sha256": input_hashes["checkpoint"],
        },
        "inputs_sha256": input_hashes,
        "executed_source_sha256": source_hashes,
        "external_endpoint_bytes_read_by_freeze": False,
        "thresholds_refit": False,
        "scientific_improvement_claim": False,
        "interpretation_guard": (
            "This freeze follows a historically exposed development result. "
            "Only a later physically isolated x0-to-DFT cohort can provide external support."
        ),
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        frozen_path = staging / FROZEN_NAME
        frozen_path.write_bytes(_json_bytes(frozen))
        manifest = {
            "protocol": PROTOCOL,
            "identifier_bearing": False,
            "inputs_sha256": input_hashes,
            "executed_source_sha256": source_hashes,
            "outputs_sha256": {FROZEN_NAME: _sha256_file(frozen_path)},
            "scientific_improvement_claim": False,
        }
        (staging / MANIFEST_NAME).write_bytes(_json_bytes(manifest))
        _publish_directory_no_replace(staging, target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return frozen


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict-features", required=True, type=Path)
    parser.add_argument("--strict-manifest", required=True, type=Path)
    parser.add_argument("--development-result", required=True, type=Path)
    parser.add_argument("--development-manifest", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    freeze_next17(
        strict_features_path=args.strict_features,
        strict_manifest_path=args.strict_manifest,
        development_result_path=args.development_result,
        development_manifest_path=args.development_manifest,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
