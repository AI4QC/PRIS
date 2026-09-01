"""Seal CHSC's incremental label-free transition over frozen M5+PHSC."""

from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from src.next11_phsc_mattersim_features import PROTOCOL as PHSC_PROTOCOL
from src.next12_chsc_mattersim_features import PROTOCOL as CHSC_PROTOCOL
from src.next12_prospective_gates import (
    FROZEN_PRIMARY_M5_THRESHOLD,
    _compose_decision,
    _compose_phsc_decision,
    _extract_frozen_primary_m5_rule,
    _sha256_file,
    _strict_json,
)
from src.next8_mattersim_committee_features import (
    _atomic_publish_directory_no_replace,
)


PROTOCOL = "2026-08-02-next12-chsc-label-free-overlap-v1"
RESULT_NAME = "CHSC_LABEL_FREE_OVERLAP.json"
TABLE_NAME = "chsc_label_free_transitions.parquet"
MANIFEST_NAME = "MANIFEST.json"
FROZEN_INPUT_SHA256 = {
    "committee_features": "65f0234010f17f43a96789bde7858bae038ffaa4aaa2130eaee163fd3245bc8c",
    "frozen_protocol": "b8049ad2f627ad91973ae86178c704871086097462f287b21c5330e3d4916fd4",
    "phsc_features": "f3492a81cef37b9887ee86b784c172c5cf1667ab0a7206379f787cb54aec6875",
    "phsc_manifest": "2b1ffd28995747352fe5a2ec4263e5822bf5f104846a0624c78de466d15ef9f5",
    "chsc_features": "cfe357605e09755ed128d56909e9fae27d8ad9003fb919c8881073e4000b7403",
    "chsc_manifest": "e2cfca1ed6fb3795a733d466d02bfa2f8acfb34708914ef01393e0dc197a8b71",
}
DECISIONS = ("KEEP", "REJECT", "ABSTAIN")


def _m5_baseline(features: pd.DataFrame, *, threshold: float) -> pd.DataFrame:
    required = {"sid", "rk", "m5_prediction_ok", "m5_energy_ev_per_atom"}
    if not required.issubset(features.columns):
        raise ValueError("committee features lack M5 baseline columns")
    table = features.loc[:, ["sid", "rk", "m5_prediction_ok", "m5_energy_ev_per_atom"]].copy()
    if table["sid"].isna().any() or table["sid"].astype(str).duplicated().any():
        raise ValueError("M5 baseline sid values must be unique")
    table["sid"] = table["sid"].astype(str)
    ready = table["m5_prediction_ok"].astype(bool)
    complete = ready.groupby(table["rk"], sort=False).transform("all")
    energy = pd.to_numeric(table["m5_energy_ev_per_atom"], errors="coerce").where(complete)
    minimum = energy.groupby(table["rk"], sort=False).transform("min")
    gap = energy - minimum
    decision = np.full(len(table), "ABSTAIN", dtype=object)
    supported = complete.to_numpy(bool) & np.isfinite(gap.to_numpy(float))
    decision[supported] = np.where(
        gap.to_numpy(float)[supported] > threshold, "REJECT", "KEEP"
    )
    table["m5_gap_ev_per_atom"] = gap
    table["m5_decision"] = decision
    return table.loc[:, ["sid", "rk", "m5_gap_ev_per_atom", "m5_decision"]]


def _counts(table: pd.DataFrame, column: str) -> dict[str, int]:
    return {decision: int(table[column].eq(decision).sum()) for decision in DECISIONS}


def _transition_table(
    baseline: pd.DataFrame, phsc: pd.DataFrame, chsc: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, object]]:
    joined = baseline.merge(
        phsc.loc[:, ["sid", "rk", "phsc_status"]],
        on=["sid", "rk"],
        validate="one_to_one",
    ).merge(
        chsc.loc[:, ["sid", "rk", "chsc_status"]],
        on=["sid", "rk"],
        validate="one_to_one",
    )
    if len(joined) != len(baseline):
        raise ValueError("M5, PHSC, and CHSC rows are not exactly aligned")
    joined["m5_phsc_decision"] = [
        _compose_phsc_decision(str(base), str(status))
        for base, status in zip(joined["m5_decision"], joined["phsc_status"], strict=True)
    ]
    joined["m5_phsc_chsc_decision"] = [
        _compose_decision(str(base), str(phsc_status), str(chsc_status))
        for base, phsc_status, chsc_status in zip(
            joined["m5_decision"],
            joined["phsc_status"],
            joined["chsc_status"],
            strict=True,
        )
    ]
    new_mask = joined["m5_phsc_decision"].ne("REJECT") & joined[
        "m5_phsc_chsc_decision"
    ].eq("REJECT")
    prior = _counts(joined, "m5_phsc_decision")
    composed = _counts(joined, "m5_phsc_chsc_decision")
    cross = (
        joined.groupby(["phsc_status", "chsc_status"], dropna=False)
        .size()
        .rename("count")
        .reset_index()
        .to_dict("records")
    )
    summary: dict[str, object] = {
        "rows": len(joined),
        "m5_counts": _counts(joined, "m5_decision"),
        "m5_phsc_counts": prior,
        "m5_phsc_chsc_counts": composed,
        "chsc_net_reject_delta_over_m5_phsc": composed["REJECT"] - prior["REJECT"],
        "new_chsc_reject_sids": sorted(joined.loc[new_mask, "sid"].astype(str)),
        "phsc_chsc_status_cross_tab": cross,
    }
    return joined.sort_values("sid", kind="stable", ignore_index=True), summary


def _validate_feature_manifest(
    payload: bytes, *, protocol: str, output_name: str, output_sha256: str, role: str
) -> dict[str, object]:
    manifest = _strict_json(payload, role=role)
    if manifest.get("protocol") != protocol or manifest.get("labels_opened") is not False:
        raise ValueError(f"{role} is not the frozen label-free protocol")
    outputs = manifest.get("outputs_sha256")
    if not isinstance(outputs, Mapping) or outputs.get(output_name) != output_sha256:
        raise ValueError(f"{role} output hash differs")
    return manifest


def run_label_free_overlap(
    *,
    committee_features_path: Path,
    frozen_protocol_path: Path,
    phsc_features_path: Path,
    phsc_manifest_path: Path,
    chsc_features_path: Path,
    chsc_manifest_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    target = Path(output_dir).resolve()
    if os.path.lexists(target):
        raise FileExistsError(f"refusing to overwrite existing output: {target}")
    paths = {
        "committee_features": Path(committee_features_path).resolve(),
        "frozen_protocol": Path(frozen_protocol_path).resolve(),
        "phsc_features": Path(phsc_features_path).resolve(),
        "phsc_manifest": Path(phsc_manifest_path).resolve(),
        "chsc_features": Path(chsc_features_path).resolve(),
        "chsc_manifest": Path(chsc_manifest_path).resolve(),
    }
    for role, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{role} is not a file: {path}")
    hashes = {role: _sha256_file(path) for role, path in paths.items()}
    if hashes != FROZEN_INPUT_SHA256:
        raise ValueError("formal CHSC overlap inputs differ from frozen identities")
    _validate_feature_manifest(
        paths["phsc_manifest"].read_bytes(),
        protocol=PHSC_PROTOCOL,
        output_name=paths["phsc_features"].name,
        output_sha256=hashes["phsc_features"],
        role="PHSC manifest",
    )
    _validate_feature_manifest(
        paths["chsc_manifest"].read_bytes(),
        protocol=CHSC_PROTOCOL,
        output_name=paths["chsc_features"].name,
        output_sha256=hashes["chsc_features"],
        role="CHSC manifest",
    )
    frozen = _strict_json(paths["frozen_protocol"].read_bytes(), role="frozen protocol")
    rule = _extract_frozen_primary_m5_rule(frozen)
    phsc = pd.read_parquet(paths["phsc_features"])
    chsc = pd.read_parquet(paths["chsc_features"])
    target_sids = set(phsc["sid"].astype(str))
    committee = pd.read_parquet(
        paths["committee_features"],
        columns=["sid", "rk", "m5_prediction_ok", "m5_energy_ev_per_atom"],
    )
    committee = committee.loc[committee["sid"].astype(str).isin(target_sids)].copy()
    baseline = _m5_baseline(committee, threshold=float(rule["threshold"]))
    transitions, summary = _transition_table(baseline, phsc, chsc)
    expected = {
        "m5_counts": {"KEEP": 2052, "REJECT": 93, "ABSTAIN": 26},
        "m5_phsc_counts": {"KEEP": 1903, "REJECT": 242, "ABSTAIN": 26},
        "m5_phsc_chsc_counts": {"KEEP": 1898, "REJECT": 247, "ABSTAIN": 26},
        "chsc_net_reject_delta_over_m5_phsc": 5,
        "new_chsc_reject_sids": [
            "elem-1254981",
            "elem-1739511",
            "elem-1861288",
            "elem-1970884",
            "elem-360622",
        ],
    }
    for key, value in expected.items():
        if summary.get(key) != value:
            raise RuntimeError(f"formal CHSC overlap expectation differs: {key}")
    result: dict[str, object] = {
        "protocol": PROTOCOL,
        "labels_opened": False,
        "endpoint_metrics_computed": False,
        "thresholds_refit": False,
        "frozen_m5_rule": rule,
        "summary": summary,
        "scientific_improvement_claim": False,
        "interpretation": "label-free incremental rejection capacity only; DFT safety remains unmeasured",
    }
    source = Path(__file__).resolve()
    source_hash = _sha256_file(source)

    def verify_unchanged() -> None:
        for role, path in paths.items():
            if _sha256_file(path) != hashes[role]:
                raise RuntimeError(f"input {role} changed before publication")
        if _sha256_file(source) != source_hash:
            raise RuntimeError("overlap source changed before publication")

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        table_path = staging / TABLE_NAME
        transitions.to_parquet(table_path, index=False)
        result_path = staging / RESULT_NAME
        result_path.write_text(
            json.dumps(result, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest = {
            "protocol": PROTOCOL,
            "inputs_sha256": {
                role: {"path": str(path), "sha256": hashes[role]}
                for role, path in paths.items()
            },
            "executed_source_sha256": {
                "src/next12_chsc_label_free_overlap.py": source_hash
            },
            "outputs_sha256": {
                RESULT_NAME: _sha256_file(result_path),
                TABLE_NAME: _sha256_file(table_path),
            },
        }
        (staging / MANIFEST_NAME).write_text(
            json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        verify_unchanged()
        _atomic_publish_directory_no_replace(staging, target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--committee-features", required=True, type=Path)
    parser.add_argument("--frozen-protocol", required=True, type=Path)
    parser.add_argument("--phsc-features", required=True, type=Path)
    parser.add_argument("--phsc-manifest", required=True, type=Path)
    parser.add_argument("--chsc-features", required=True, type=Path)
    parser.add_argument("--chsc-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    arguments = parser.parse_args(argv)
    run_label_free_overlap(
        committee_features_path=arguments.committee_features,
        frozen_protocol_path=arguments.frozen_protocol,
        phsc_features_path=arguments.phsc_features,
        phsc_manifest_path=arguments.phsc_manifest,
        chsc_features_path=arguments.chsc_features,
        chsc_manifest_path=arguments.chsc_manifest,
        output_dir=arguments.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
