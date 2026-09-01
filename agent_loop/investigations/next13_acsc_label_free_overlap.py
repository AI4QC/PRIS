"""Compose sealed ACSC-v0 evidence with the frozen label-free gate decisions."""

from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Sequence

import pandas as pd

from src.next10_lrrc_mattersim_features import (
    _sha256_file,
    _snapshot,
    _strict_json_document,
)
from src.next8_mattersim_committee_features import (
    _atomic_publish_directory_no_replace,
)


PROTOCOL = "2026-08-02-next13-acsc-label-free-overlap-v1"
UPSTREAM_OVERLAP_PROTOCOL = "2026-08-02-next12-chsc-label-free-overlap-v1"
UPSTREAM_ACSC_PROTOCOL = "2026-08-02-next13-acsc-old-cohort-v1"
OUTPUT_NAME = "acsc_label_free_transitions.parquet"
MANIFEST_NAME = "MANIFEST.json"
ALLOWED_DECISIONS = frozenset({"KEEP", "REJECT", "ABSTAIN"})


def compose_decision(old_decision: str, coupling_only_negative: bool) -> str:
    """Let resolved coupling-negative evidence override KEEP or ABSTAIN."""

    if type(old_decision) is not str or old_decision not in ALLOWED_DECISIONS:
        raise ValueError("old decision must be KEEP, REJECT, or ABSTAIN")
    if type(coupling_only_negative) is not bool:
        raise ValueError("coupling_only_negative must be an exact bool")
    return "REJECT" if coupling_only_negative else old_decision


def _validated_manifest(
    data: bytes,
    *,
    protocol: str,
    output_name: str,
    output_sha256: str,
    require_label_flags: bool,
) -> dict[str, object]:
    manifest = dict(_strict_json_document(data, role=f"{protocol} manifest"))
    if manifest.get("protocol") != protocol:
        raise ValueError("upstream manifest protocol mismatch")
    outputs = manifest.get("outputs_sha256")
    if not isinstance(outputs, dict) or outputs.get(output_name) != output_sha256:
        raise ValueError("upstream output hash differs from its manifest")
    if require_label_flags and (
        manifest.get("labels_opened") is not False
        or manifest.get("endpoint_artifacts_opened") is not False
    ):
        raise ValueError("ACSC manifest does not prove label/endpoint isolation")
    return manifest


def run_label_free_overlap(
    *,
    old_transitions_path: Path,
    old_manifest_path: Path,
    acsc_features_path: Path,
    acsc_manifest_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Publish complete old-to-ACSC transitions without opening labels."""

    target = Path(output_dir)
    if os.path.lexists(os.fspath(target)):
        raise FileExistsError(target)
    paths = {
        "old_transitions": Path(old_transitions_path),
        "old_manifest": Path(old_manifest_path),
        "acsc_features": Path(acsc_features_path),
        "acsc_manifest": Path(acsc_manifest_path),
    }
    snapshots = {
        role: _snapshot(path, include_data=True) for role, path in paths.items()
    }
    _validated_manifest(
        snapshots["old_manifest"].data or b"",
        protocol=UPSTREAM_OVERLAP_PROTOCOL,
        output_name=paths["old_transitions"].name,
        output_sha256=snapshots["old_transitions"].sha256,
        require_label_flags=False,
    )
    _validated_manifest(
        snapshots["acsc_manifest"].data or b"",
        protocol=UPSTREAM_ACSC_PROTOCOL,
        output_name=paths["acsc_features"].name,
        output_sha256=snapshots["acsc_features"].sha256,
        require_label_flags=True,
    )
    old = pd.read_parquet(io.BytesIO(snapshots["old_transitions"].data or b""))
    acsc = pd.read_parquet(io.BytesIO(snapshots["acsc_features"].data or b""))
    old_required = {
        "sid",
        "rk",
        "m5_gap_ev_per_atom",
        "m5_decision",
        "phsc_status",
        "chsc_status",
        "m5_phsc_decision",
        "m5_phsc_chsc_decision",
    }
    acsc_required = {
        "sid",
        "rk",
        "acsc_status",
        "coupling_only_negative",
        "pure_status_drift",
        "lambda_r_ev_per_atom",
        "e_num_ev_per_atom",
        "u_num_ev_per_atom",
    }
    if not old_required.issubset(old.columns) or not acsc_required.issubset(acsc.columns):
        raise ValueError("upstream transition/ACSC table lacks required columns")
    if old["sid"].astype(str).duplicated().any() or acsc["sid"].astype(str).duplicated().any():
        raise ValueError("upstream sid values must be unique")
    if not set(acsc["sid"].astype(str)).issubset(set(old["sid"].astype(str))):
        raise ValueError("ACSC sid set is not a subset of the old transition cohort")
    merge_columns = [
        "sid",
        "rk",
        "acsc_status",
        "coupling_only_negative",
        "pure_status_drift",
        "lambda_r_ev_per_atom",
        "e_num_ev_per_atom",
        "u_num_ev_per_atom",
    ]
    combined = old.merge(
        acsc.loc[:, merge_columns],
        on=["sid", "rk"],
        how="left",
        validate="one_to_one",
        sort=False,
    )
    combined["acsc_evaluated"] = combined["acsc_status"].notna()
    combined["acsc_status"] = combined["acsc_status"].fillna("not_evaluated_pure_gate")
    combined["coupling_only_negative"] = combined["coupling_only_negative"].fillna(False).astype(bool)
    combined["pure_status_drift"] = combined["pure_status_drift"].fillna(False).astype(bool)
    combined["m5_phsc_chsc_acsc_decision"] = [
        compose_decision(str(old_decision), bool(coupling))
        for old_decision, coupling in zip(
            combined["m5_phsc_chsc_decision"],
            combined["coupling_only_negative"],
            strict=True,
        )
    ]
    combined["decision_transition"] = (
        combined["m5_phsc_chsc_decision"].astype(str)
        + "->"
        + combined["m5_phsc_chsc_acsc_decision"].astype(str)
    )
    combined = combined.sort_values("sid", kind="stable", ignore_index=True)

    old_counts = combined["m5_phsc_chsc_decision"].value_counts().to_dict()
    new_counts = combined["m5_phsc_chsc_acsc_decision"].value_counts().to_dict()
    transitions = combined["decision_transition"].value_counts().to_dict()
    manifest: dict[str, object] = {
        "protocol": PROTOCOL,
        "mode": "development_gate_label_free_composition",
        "labels_opened": False,
        "endpoint_artifacts_opened": False,
        "composition_rule": (
            "ACSC coupling_only_negative overrides KEEP or ABSTAIN to REJECT; "
            "otherwise preserve frozen M5+PHSC+CHSC decision"
        ),
        "counts": {
            "rows": len(combined),
            "acsc_evaluated_rows": int(combined["acsc_evaluated"].sum()),
            "acsc_coupling_only_negative_rows": int(combined["coupling_only_negative"].sum()),
            "old_decisions": {key: int(old_counts.get(key, 0)) for key in sorted(ALLOWED_DECISIONS)},
            "new_decisions": {key: int(new_counts.get(key, 0)) for key in sorted(ALLOWED_DECISIONS)},
            "transitions": {str(key): int(value) for key, value in sorted(transitions.items())},
        },
        "inputs_sha256": {
            role: {"path": str(snapshot.path.resolve()), "sha256": snapshot.sha256}
            for role, snapshot in snapshots.items()
        },
        "executed_source_sha256": {
            "src/next13_acsc_label_free_overlap.py": _sha256_file(Path(__file__).resolve())
        },
        "scientific_improvement_claim": False,
        "known_limitations": [
            "This artifact contains no DFT labels and measures decision coverage only.",
            "ACSC evidence is MatterSim-based and requires independent DFT validation.",
        ],
    }

    def verify_unchanged() -> None:
        for role, snapshot in snapshots.items():
            if _sha256_file(snapshot.path) != snapshot.sha256:
                raise RuntimeError(f"input {role} changed after initial hash")
        source = Path(__file__).resolve()
        if _sha256_file(source) != manifest["executed_source_sha256"][
            "src/next13_acsc_label_free_overlap.py"
        ]:
            raise RuntimeError("executed overlap source changed after initial hash")

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        table_path = staging / OUTPUT_NAME
        combined.to_parquet(table_path, index=False)
        manifest["outputs_sha256"] = {OUTPUT_NAME: _sha256_file(table_path)}
        payload = json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True) + "\n"
        manifest_path = staging / MANIFEST_NAME
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
    parser.add_argument("--old-transitions", required=True, type=Path)
    parser.add_argument("--old-manifest", required=True, type=Path)
    parser.add_argument("--acsc-features", required=True, type=Path)
    parser.add_argument("--acsc-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    arguments = parser.parse_args(argv)
    run_label_free_overlap(
        old_transitions_path=arguments.old_transitions,
        old_manifest_path=arguments.old_manifest,
        acsc_features_path=arguments.acsc_features,
        acsc_manifest_path=arguments.acsc_manifest,
        output_dir=arguments.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
