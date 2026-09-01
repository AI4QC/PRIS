"""Freeze classical Pauling 2--5 controls on the prospective x0 cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from ase import Atoms

from src.next12_prospective_cohort import PROTOCOL as COHORT_PROTOCOL
from src.next12_prospective_gates import _load_cohort, _sha256_file
from src.next8_mattersim_committee_features import (
    _atomic_publish_directory_no_replace,
)


PROTOCOL = "2026-08-02-next12-prospective-pauling2-5-controls-v1"
OUTPUT_NAME = "prospective_pauling_controls.parquet"
MANIFEST_NAME = "MANIFEST.json"
FROZEN_FORMAL_INPUT_SHA256 = {
    "cohort": "fc08be4f1b28dc82f4a26aeb49819b914ad8df7c7c7ee3887dea7a6c61095215",
    "geometry_only_frames": "3b392bdd38120dae579dc22b1b51e7c30bcbbed0e72c9b462c1bce16eda96959",
    "cohort_manifest": "8649853dcbb40a081183b671101fdf2933f30358ad1e5cb5b8694e8e451a846a",
}
RULES: Mapping[str, Mapping[str, object]] = {
    "pauling_p2": {
        "feature": "p2_mean_dev",
        "operator": "<=",
        "threshold": 0.01,
        "description": "Pauling 2 bond-strength mean deviation",
    },
    "pauling_p3": {
        "feature": "p3_frac_edge_face",
        "operator": "<=",
        "threshold": 0.10,
        "description": "Pauling 3 edge/face-sharing fraction",
    },
    "pauling_p4": {
        "feature": "p4_violate",
        "operator": "<=",
        "threshold": 0.5,
        "description": "Pauling 4 high-valence contact violation",
    },
    "pauling_p5": {
        "feature": "p5_ok",
        "operator": ">=",
        "threshold": 0.5,
        "description": "Pauling 5 parsimony condition",
    },
}
DECISIONS = ("KEEP", "REJECT", "ABSTAIN")
FeatureCalculator = Callable[[Atoms], tuple[Mapping[str, object] | None, str | None]]


def _rule_decision(value: object, *, operator: str, threshold: float) -> str:
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        return "ABSTAIN"
    if not np.isfinite(scalar):
        return "ABSTAIN"
    if operator == "<=":
        passes = scalar <= threshold
    elif operator == ">=":
        passes = scalar >= threshold
    else:
        raise ValueError(f"unsupported Pauling operator: {operator}")
    return "KEEP" if passes else "REJECT"


def _combined_decision(decisions: Sequence[str]) -> str:
    if len(decisions) != len(RULES) or any(value not in DECISIONS for value in decisions):
        raise ValueError("combined Pauling decision requires four valid decisions")
    if "REJECT" in decisions:
        return "REJECT"
    if "ABSTAIN" in decisions:
        return "ABSTAIN"
    return "KEEP"


def _classical_features(atoms: Atoms) -> tuple[Mapping[str, object] | None, str | None]:
    from pymatgen.io.ase import AseAtomsAdaptor

    from src.apply_rules import frac_oxi
    from src.discriminate import criteria, guess_oxi

    structure = AseAtomsAdaptor.get_structure(atoms)
    valences, ok = guess_oxi(structure)
    if not ok:
        valences = frac_oxi(structure)
        if valences is None:
            return None, "integer and fractional oxidation-state inference failed"
    try:
        values = criteria(structure, valences)
    except Exception as exc:
        return None, f"criteria failed: {type(exc).__name__}: {exc}"
    if not isinstance(values, Mapping):
        return None, "criteria returned no feature mapping"
    return values, None


def _decision_counts(table: pd.DataFrame, column: str) -> dict[str, int]:
    return {decision: int(table[column].eq(decision).sum()) for decision in DECISIONS}


def run_pauling_controls(
    *,
    cohort_path: Path,
    frames_zip_path: Path,
    cohort_manifest_path: Path,
    output_dir: Path,
    feature_calculator: FeatureCalculator | None = None,
) -> dict[str, object]:
    """Apply the repository's frozen Pauling 2--5 operational definitions."""

    target = Path(output_dir).resolve()
    if os.path.lexists(target):
        raise FileExistsError(f"refusing to overwrite existing output: {target}")
    paths = {
        "cohort": Path(cohort_path).resolve(),
        "geometry_only_frames": Path(frames_zip_path).resolve(),
        "cohort_manifest": Path(cohort_manifest_path).resolve(),
    }
    for role, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{role} is not a file: {path}")
    input_hashes = {role: _sha256_file(path) for role, path in paths.items()}
    table, generated_sids, structures, upstream_manifest = _load_cohort(
        cohort_data=paths["cohort"].read_bytes(),
        archive_data=paths["geometry_only_frames"].read_bytes(),
        manifest_data=paths["cohort_manifest"].read_bytes(),
    )
    if upstream_manifest.get("protocol") != COHORT_PROTOCOL:
        raise ValueError("upstream cohort protocol differs")
    production = feature_calculator is None
    if production:
        if input_hashes != FROZEN_FORMAL_INPUT_SHA256:
            raise ValueError("formal Pauling-control inputs differ from frozen identities")
        if upstream_manifest.get("production_protocol_eligible") is not True:
            raise ValueError("formal upstream cohort is not production eligible")
        calculator = _classical_features
    else:
        calculator = feature_calculator

    generated_index = {sid: index for index, sid in enumerate(generated_sids)}
    started = time.perf_counter()
    rows: list[dict[str, object]] = []
    for upstream in table.to_dict("records"):
        sid = str(upstream["sid"])
        index = generated_index.get(sid)
        if index is None:
            features = None
            error = "generation failed before geometry freeze"
        else:
            try:
                features, error = calculator(structures[index])
            except Exception as exc:
                features = None
                error = f"calculator failed: {type(exc).__name__}: {exc}"
        feature_values = dict(features) if isinstance(features, Mapping) else {}
        row: dict[str, object] = {
            "attempt_index": int(upstream["attempt_index"]),
            "sid": sid,
            "formula": str(upstream["formula"]),
            "natoms": int(upstream["natoms"]),
            "generation_status": str(upstream["generation_status"]),
            "pauling_feature_error": error,
        }
        decisions: list[str] = []
        for name, rule in RULES.items():
            value = feature_values.get(str(rule["feature"]), np.nan)
            decision = _rule_decision(
                value,
                operator=str(rule["operator"]),
                threshold=float(rule["threshold"]),
            )
            row[f"{name}_value"] = value
            row[f"{name}_decision"] = decision
            decisions.append(decision)
        row["pauling_p2_p5_decision"] = _combined_decision(decisions)
        rows.append(row)
    elapsed = time.perf_counter() - started
    output_table = pd.DataFrame(rows).sort_values("sid", kind="stable", ignore_index=True)
    if len(output_table) != len(table) or output_table["sid"].duplicated().any():
        raise RuntimeError("Pauling controls did not retain every prospective attempt")

    repository_root = Path(__file__).resolve().parents[1]
    source_relatives = (
        "src/next12_pauling_controls.py",
        "src/next12_prospective_gates.py",
        "src/next12_prospective_cohort.py",
        "src/apply_rules.py",
        "src/discriminate.py",
    )
    source_paths = {relative: repository_root / relative for relative in source_relatives}
    source_hashes = {relative: _sha256_file(path) for relative, path in source_paths.items()}
    counts: dict[str, object] = {
        "rows": len(output_table),
        "generated_rows": len(generated_sids),
        "feature_error_rows": int(output_table["pauling_feature_error"].notna().sum()),
        "combined": _decision_counts(output_table, "pauling_p2_p5_decision"),
    }
    for name in RULES:
        counts[name] = _decision_counts(output_table, f"{name}_decision")
    manifest: dict[str, object] = {
        "protocol": PROTOCOL,
        "mode": "prospective_x0_pauling_controls",
        "labels_opened": False,
        "endpoint_artifacts_opened": False,
        "thresholds_refit": False,
        "rules": {name: dict(rule) for name, rule in RULES.items()},
        "counts": counts,
        "execution": {"wall_time_seconds": elapsed},
        "inputs_sha256": {
            role: {"path": str(path), "sha256": input_hashes[role]}
            for role, path in paths.items()
        },
        "executed_source_sha256": source_hashes,
        "production_protocol_eligible": bool(production),
        "scientific_improvement_claim": False,
        "known_limitations": [
            "Pauling rule 1 is excluded because its Shannon-radius/CN use is circular in this repository's audit.",
            "Oxidation-state or topology failures are abstentions, not passes.",
            "No DFT endpoint was opened in this run.",
        ],
    }

    def verify_unchanged() -> None:
        for role, path in paths.items():
            if _sha256_file(path) != input_hashes[role]:
                raise RuntimeError(f"input {role} changed before publication")
        for relative, path in source_paths.items():
            if _sha256_file(path) != source_hashes[relative]:
                raise RuntimeError(f"source {relative} changed before publication")

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        output_path = staging / OUTPUT_NAME
        output_table.to_parquet(output_path, index=False)
        manifest["outputs_sha256"] = {OUTPUT_NAME: _sha256_file(output_path)}
        manifest_path = staging / MANIFEST_NAME
        payload = json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True) + "\n"
        with manifest_path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        verify_unchanged()
        _atomic_publish_directory_no_replace(staging, target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", required=True, type=Path)
    parser.add_argument("--frames-zip", required=True, type=Path)
    parser.add_argument("--cohort-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    arguments = parser.parse_args(argv)
    run_pauling_controls(
        cohort_path=arguments.cohort,
        frames_zip_path=arguments.frames_zip,
        cohort_manifest_path=arguments.cohort_manifest,
        output_dir=arguments.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
