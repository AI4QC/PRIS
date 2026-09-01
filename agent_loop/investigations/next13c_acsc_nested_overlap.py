"""Compose nested three-scale ACSC confirmation with the frozen old gate."""

from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Mapping, Sequence

import pandas as pd

from src.next10_lrrc_mattersim_features import (
    _sha256_file,
    _snapshot,
    _strict_json_document,
)
from src.next13_acsc_label_free_overlap import compose_decision
from src.next8_mattersim_committee_features import (
    _atomic_publish_directory_no_replace,
)


PROTOCOL = "2026-08-02-next13c-acsc-nested-overlap-v1"
UPSTREAM_OLD_PROTOCOL = "2026-08-02-next12-chsc-label-free-overlap-v1"
UPSTREAM_LADDER_PROTOCOL = "2026-08-02-next13c-acsc-direct-ladder-v1"
OUTPUT_NAME = "acsc_nested_transitions.parquet"
MANIFEST_NAME = "MANIFEST.json"
DECISIONS = ("ABSTAIN", "KEEP", "REJECT")


def nested_decision(old_decision: str, confirmed: bool) -> str:
    """Override the old decision only for nested three-scale confirmation."""

    return compose_decision(old_decision, confirmed)


def _manifest(
    data: bytes,
    *,
    protocol: str,
    output_name: str,
    output_sha256: str,
    require_isolation: bool,
) -> dict[str, object]:
    manifest = dict(_strict_json_document(data, role=f"{protocol} manifest"))
    if manifest.get("protocol") != protocol:
        raise ValueError("upstream protocol mismatch")
    outputs = manifest.get("outputs_sha256")
    if not isinstance(outputs, Mapping) or outputs.get(output_name) != output_sha256:
        raise ValueError("upstream output hash differs from manifest")
    if require_isolation and (
        manifest.get("labels_opened") is not False
        or manifest.get("endpoint_artifacts_opened") is not False
    ):
        raise ValueError("ladder manifest does not prove label/endpoint isolation")
    return manifest


def run_overlap(
    *,
    old_transitions_path: Path,
    old_manifest_path: Path,
    ladder_features_path: Path,
    ladder_manifest_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Publish full-cohort transitions for the conservative nested rule."""

    target = Path(output_dir)
    if os.path.lexists(os.fspath(target)):
        raise FileExistsError(target)
    paths = {
        "old_transitions": Path(old_transitions_path),
        "old_manifest": Path(old_manifest_path),
        "ladder_features": Path(ladder_features_path),
        "ladder_manifest": Path(ladder_manifest_path),
    }
    snapshots = {
        role: _snapshot(path, include_data=True) for role, path in paths.items()
    }
    _manifest(
        snapshots["old_manifest"].data or b"",
        protocol=UPSTREAM_OLD_PROTOCOL,
        output_name=paths["old_transitions"].name,
        output_sha256=snapshots["old_transitions"].sha256,
        require_isolation=False,
    )
    _manifest(
        snapshots["ladder_manifest"].data or b"",
        protocol=UPSTREAM_LADDER_PROTOCOL,
        output_name=paths["ladder_features"].name,
        output_sha256=snapshots["ladder_features"].sha256,
        require_isolation=True,
    )
    old = pd.read_parquet(io.BytesIO(snapshots["old_transitions"].data or b""))
    ladder = pd.read_parquet(io.BytesIO(snapshots["ladder_features"].data or b""))
    old_required = {"sid", "rk", "m5_phsc_chsc_decision"}
    ladder_required = {
        "sid",
        "rk",
        "small_direct_confirmed",
        "large_direct_status",
        "nested_three_scale_confirmed",
        "small_u_num_ev_per_atom",
        "large_u_num_ev_per_atom",
    }
    if not old_required.issubset(old.columns) or not ladder_required.issubset(ladder.columns):
        raise ValueError("old/ladder table lacks required columns")
    if old["sid"].astype(str).duplicated().any() or ladder["sid"].astype(str).duplicated().any():
        raise ValueError("old/ladder sid values must be unique")
    if not set(ladder["sid"].astype(str)).issubset(set(old["sid"].astype(str))):
        raise ValueError("ladder sid set is not a subset of old cohort")
    merge_columns = [
        "sid",
        "rk",
        "small_direct_confirmed",
        "large_direct_status",
        "nested_three_scale_confirmed",
        "small_u_num_ev_per_atom",
        "large_u_num_ev_per_atom",
    ]
    combined = old.merge(
        ladder.loc[:, merge_columns],
        on=["sid", "rk"],
        how="left",
        validate="one_to_one",
        sort=False,
    )
    combined["nested_three_scale_confirmed"] = (
        combined["nested_three_scale_confirmed"].eq(True).astype(bool)
    )
    combined["nested_evaluated"] = combined["large_direct_status"].notna()
    combined["large_direct_status"] = combined["large_direct_status"].fillna(
        "not_evaluated_pure_gate"
    )
    combined["m5_phsc_chsc_acsc_nested_decision"] = [
        nested_decision(str(old_decision), bool(confirmed))
        for old_decision, confirmed in zip(
            combined["m5_phsc_chsc_decision"],
            combined["nested_three_scale_confirmed"],
            strict=True,
        )
    ]
    combined["nested_decision_transition"] = (
        combined["m5_phsc_chsc_decision"].astype(str)
        + "->"
        + combined["m5_phsc_chsc_acsc_nested_decision"].astype(str)
    )
    combined = combined.sort_values("sid", kind="stable", ignore_index=True)

    old_counts = combined["m5_phsc_chsc_decision"].value_counts()
    new_counts = combined["m5_phsc_chsc_acsc_nested_decision"].value_counts()
    transition_counts = combined["nested_decision_transition"].value_counts()
    manifest: dict[str, object] = {
        "protocol": PROTOCOL,
        "mode": "full_cohort_label_free_nested_composition",
        "labels_opened": False,
        "endpoint_artifacts_opened": False,
        "composition_rule": (
            "nested three-scale confirmation overrides old KEEP/ABSTAIN to REJECT; "
            "otherwise preserve M5+PHSC+CHSC"
        ),
        "counts": {
            "rows": len(combined),
            "nested_evaluated_rows": int(combined["nested_evaluated"].sum()),
            "nested_confirmed_rows": int(combined["nested_three_scale_confirmed"].sum()),
            "old_decisions": {key: int(old_counts.get(key, 0)) for key in DECISIONS},
            "new_decisions": {key: int(new_counts.get(key, 0)) for key in DECISIONS},
            "transitions": {
                str(key): int(value) for key, value in sorted(transition_counts.items())
            },
        },
        "inputs_sha256": {
            role: {"path": str(snapshot.path.resolve()), "sha256": snapshot.sha256}
            for role, snapshot in snapshots.items()
        },
        "executed_source_sha256": {
            "src/next13c_acsc_nested_overlap.py": _sha256_file(Path(__file__).resolve()),
            "src/next13_acsc_label_free_overlap.py": _sha256_file(
                Path(__file__).with_name("next13_acsc_label_free_overlap.py")
            ),
        },
        "scientific_improvement_claim": False,
        "known_limitations": [
            "This artifact measures label-free decision coverage, not correctness.",
            "Nested confirmation still uses MatterSim and requires independent DFT.",
        ],
    }

    def verify_unchanged() -> None:
        for role, snapshot in snapshots.items():
            if _sha256_file(snapshot.path) != snapshot.sha256:
                raise RuntimeError(f"input {role} changed after initial hash")
        for logical, expected in manifest["executed_source_sha256"].items():
            path = Path(__file__).resolve() if logical.endswith(
                "next13c_acsc_nested_overlap.py"
            ) else Path(__file__).with_name("next13_acsc_label_free_overlap.py")
            if _sha256_file(path) != expected:
                raise RuntimeError(f"executed source {logical} changed after initial hash")

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
    parser.add_argument("--ladder-features", required=True, type=Path)
    parser.add_argument("--ladder-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    arguments = parser.parse_args(argv)
    run_overlap(
        old_transitions_path=arguments.old_transitions,
        old_manifest_path=arguments.old_manifest,
        ladder_features_path=arguments.ladder_features,
        ladder_manifest_path=arguments.ladder_manifest,
        output_dir=arguments.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
