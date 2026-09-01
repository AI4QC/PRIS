#!/usr/bin/env python3
"""Search a finite B+E/contact conjunction catalogue on two exposed sources."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd

from src.next19_feature_build import _publish_directory_no_replace, _sha256
from src.next23_evaluate import _decision_metrics
from src.next23_relaxation_rule import ENDPOINT_COLUMN, PRIMARY_GATES


PROTOCOL = "2026-08-03-next41-source-balanced-contact-guard-search-v1"
SCAN_NAME = "next41_source_balanced_guard_scan.parquet"
RESULT_NAME = "NEXT41_SOURCE_BALANCED_GUARD_RESULT.json"
RULE_NAME = "NEXT41_FROZEN_CONTACT_GUARD_RULE.json"
MANIFEST_NAME = "MANIFEST.json"
FROZEN_NEXT23_THRESHOLD = 2.0327814658380157
GUARDS: Mapping[str, tuple[str, int]] = {
    "neg_cov_q01": ("cov_q01", -1),
    "neg_cov_q05": ("cov_q05", -1),
    "cov_contact085_pa": ("cov_contact085_pa", 1),
    "cov_overlap2_pa": ("cov_overlap2_pa", 1),
    "cov_site_overlap_q95": ("cov_site_overlap_q95", 1),
    "cov_site_overlap_max": ("cov_site_overlap_max", 1),
}


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _validate_source(frame: pd.DataFrame, *, role: str) -> pd.DataFrame:
    required = {
        "material_id",
        "score",
        "score_supported",
        "contact_supported",
        ENDPOINT_COLUMN,
        *(column for column, _direction in GUARDS.values()),
    }
    if not required.issubset(frame.columns):
        raise ValueError(f"{role} lacks columns: {sorted(required - set(frame))}")
    out = frame.copy()
    out["material_id"] = out.material_id.astype(str)
    if out.material_id.duplicated().any() or not len(out):
        raise ValueError(f"{role} identities are invalid")
    endpoint = pd.to_numeric(out[ENDPOINT_COLUMN], errors="coerce").to_numpy(float)
    if not np.isfinite(endpoint).all():
        raise ValueError(f"{role} endpoint is invalid")
    return out.sort_values("material_id", kind="stable", ignore_index=True)


def scan_source_balanced_guards(
    sources: Mapping[str, pd.DataFrame], *, quantile_count: int = 41
) -> tuple[dict[str, object], pd.DataFrame]:
    """Evaluate every finite rectangle and require every source to pass."""

    if len(sources) < 2 or type(quantile_count) is not int or quantile_count < 3:
        raise ValueError("NEXT41 requires at least two sources and three quantiles")
    validated = {
        str(name): _validate_source(frame, role=str(name))
        for name, frame in sorted(sources.items())
    }
    quantiles = np.linspace(0.0, 1.0, quantile_count)
    score_pool = np.concatenate(
        [
            frame.loc[frame.score_supported.astype(bool), "score"]
            .dropna()
            .to_numpy(float)
            for frame in validated.values()
        ]
    )
    score_thresholds = np.unique(
        np.r_[np.quantile(score_pool, quantiles), FROZEN_NEXT23_THRESHOLD]
    )
    rows: list[dict[str, object]] = []
    eligible_records: list[dict[str, object]] = []
    for guard_name, (column, direction) in GUARDS.items():
        guard_pool = np.concatenate(
            [
                direction
                * frame.loc[frame.contact_supported.astype(bool), column]
                .dropna()
                .to_numpy(float)
                for frame in validated.values()
            ]
        )
        guard_thresholds = np.unique(np.quantile(guard_pool, quantiles))
        for score_index, score_threshold in enumerate(score_thresholds):
            for guard_index, guard_threshold in enumerate(guard_thresholds):
                candidate_id = f"{guard_name}|s{score_index:03d}|g{guard_index:03d}"
                record: dict[str, object] = {
                    "candidate_id": candidate_id,
                    "guard": guard_name,
                    "guard_column": column,
                    "guard_direction": direction,
                    "score_threshold": float(score_threshold),
                    "guard_threshold": float(guard_threshold),
                }
                source_passes: list[bool] = []
                total_rejected = 0
                for source_name, frame in validated.items():
                    score = pd.to_numeric(frame.score, errors="coerce").to_numpy(float)
                    guard = direction * pd.to_numeric(
                        frame[column], errors="coerce"
                    ).to_numpy(float)
                    supported = (
                        frame.score_supported.to_numpy(bool)
                        & frame.contact_supported.to_numpy(bool)
                        & np.isfinite(score)
                        & np.isfinite(guard)
                    )
                    reject = supported & (score >= score_threshold) & (
                        guard >= guard_threshold
                    )
                    metrics = _decision_metrics(
                        supported=supported,
                        reject=reject,
                        endpoint=frame[ENDPOINT_COLUMN].to_numpy(float),
                    )
                    total_rejected += int(metrics["rejected"])
                    source_passes.append(bool(metrics["passes_primary_gates"]))
                    for name, value in metrics.items():
                        record[f"{source_name}_{name}"] = value
                record["all_sources_pass"] = all(source_passes)
                record["total_rejected"] = total_rejected
                rows.append(record)
                if record["all_sources_pass"]:
                    eligible_records.append(record)

    table = pd.DataFrame(rows).sort_values(
        ["guard", "score_threshold", "guard_threshold"],
        kind="stable",
        ignore_index=True,
    )
    selected: dict[str, object] | None = None
    if eligible_records:
        winner = max(
            eligible_records,
            key=lambda row: (
                int(row["total_rejected"]),
                -float(row["score_threshold"]),
                -float(row["guard_threshold"]),
                str(row["candidate_id"]),
            ),
        )
        selected = {
            name: winner[name]
            for name in (
                "candidate_id",
                "guard",
                "guard_column",
                "guard_direction",
                "score_threshold",
                "guard_threshold",
                "total_rejected",
            )
        }
    result: dict[str, object] = {
        "protocol": PROTOCOL,
        "development_sources": list(validated),
        "development_labels_opened": True,
        "confirmation_source_opened": False,
        "quantile_count": quantile_count,
        "score_threshold_count": len(score_thresholds),
        "guard_count": len(GUARDS),
        "candidate_count": len(table),
        "eligible_count": len(eligible_records),
        "primary_gates_required_per_source": dict(PRIMARY_GATES),
        "eligible": selected is not None,
        "selected_candidate": selected["candidate_id"] if selected else None,
        "selected_rule": selected,
    }
    return result, table


def _assemble_source(joined_path: Path, contact_path: Path, *, role: str) -> pd.DataFrame:
    joined = pd.read_parquet(joined_path)
    contact = pd.read_parquet(contact_path)
    if role == "wbm":
        joined = joined.rename(
            columns={"analytic_supported": "score_supported", "next23_risk_score": "score"}
        )
    elif role == "omat24_short":
        joined = joined.rename(
            columns={"next23_supported": "score_supported", "next23_score": "score"}
        )
    else:
        raise ValueError("unknown NEXT41 development source")
    return joined.merge(
        contact.drop(columns=["natoms"], errors="ignore"),
        on="material_id",
        how="left",
        validate="one_to_one",
    )


def run_source_balanced_guard_search(
    *,
    wbm_joined_path: Path,
    wbm_contact_path: Path,
    omat_joined_path: Path,
    omat_contact_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    target = Path(output_dir).resolve()
    if os.path.lexists(target):
        raise FileExistsError(str(target))
    paths = {
        "wbm_joined": Path(wbm_joined_path).resolve(),
        "wbm_contact": Path(wbm_contact_path).resolve(),
        "omat_joined": Path(omat_joined_path).resolve(),
        "omat_contact": Path(omat_contact_path).resolve(),
    }
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("NEXT41 search input is missing")
    input_hashes = {name: _sha256(path) for name, path in paths.items()}
    sources = {
        "wbm": _assemble_source(paths["wbm_joined"], paths["wbm_contact"], role="wbm"),
        "omat24_short": _assemble_source(
            paths["omat_joined"], paths["omat_contact"], role="omat24_short"
        ),
    }
    result, table = scan_source_balanced_guards(sources, quantile_count=41)

    repository = Path(__file__).resolve().parents[1]
    source_names = (
        "src/next23_evaluate.py",
        "src/next41_source_balanced_guard_search.py",
    )
    source_hashes = {name: _sha256(repository / name) for name in source_names}
    manifest: dict[str, object] = {
        "protocol": PROTOCOL,
        "development_labels_opened": True,
        "confirmation_source_opened": False,
        "thresholds_fit_on_exposed_development_sources": True,
        "candidate_count": result["candidate_count"],
        "eligible_count": result["eligible_count"],
        "promoted": result["eligible"],
        "inputs_sha256": input_hashes,
        "executed_source_sha256": source_hashes,
        "scientific_improvement_claim": False,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        scan_path = staging / SCAN_NAME
        result_path = staging / RESULT_NAME
        table.to_parquet(scan_path, index=False)
        result_path.write_bytes(_json_bytes(result))
        output_names = [SCAN_NAME, RESULT_NAME]
        if result["eligible"]:
            rule_path = staging / RULE_NAME
            rule_path.write_bytes(_json_bytes(result["selected_rule"]))
            output_names.append(RULE_NAME)
        manifest["outputs_sha256"] = {
            name: _sha256(staging / name) for name in output_names
        }
        (staging / MANIFEST_NAME).write_bytes(_json_bytes(manifest))
        if any(_sha256(path) != input_hashes[name] for name, path in paths.items()):
            raise RuntimeError("NEXT41 search input changed during publication")
        if any(_sha256(repository / name) != digest for name, digest in source_hashes.items()):
            raise RuntimeError("NEXT41 search source changed during publication")
        _publish_directory_no_replace(staging, target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wbm-joined", required=True, type=Path)
    parser.add_argument("--wbm-contact", required=True, type=Path)
    parser.add_argument("--omat-joined", required=True, type=Path)
    parser.add_argument("--omat-contact", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    run_source_balanced_guard_search(
        wbm_joined_path=args.wbm_joined,
        wbm_contact_path=args.wbm_contact,
        omat_joined_path=args.omat_joined,
        omat_contact_path=args.omat_contact,
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "GUARDS",
    "RESULT_NAME",
    "SCAN_NAME",
    "scan_source_balanced_guards",
    "run_source_balanced_guard_search",
]
