"""Open fully accounted DFT endpoints only after the NEXT12 queue is frozen."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from src.next12_dft_queue import (
    PROTOCOL as QUEUE_PROTOCOL,
)
from src.next12_prospective_gates import _sha256_file, _strict_json
from src.next8_mattersim_committee_features import (
    _atomic_publish_directory_no_replace,
)


PROTOCOL = "2026-08-02-next12-isolated-dft-endpoint-evaluation-v1"
ENDPOINT_PROTOCOL = "2026-08-02-next12-vasp-pbe-endpoints-v1"
RESULT_NAME = "DFT_ENDPOINT_EVALUATION.json"
JOINED_NAME = "dft_joined_evaluation.parquet"
MANIFEST_NAME = "MANIFEST.json"
ENDPOINT_COLUMNS = (
    "sid",
    "status",
    "static_energy_ev",
    "static_fmax_ev_per_a",
    "static_max_stress_gpa",
    "relaxed_energy_ev",
    "relaxed_fmax_ev_per_a",
    "ionic_steps",
    "initial_volume_angstrom3",
    "relaxed_volume_angstrom3",
    "max_displacement_angstrom",
    "wall_time_seconds",
    "error",
)
METHOD_COLUMNS: Mapping[str, str] = {
    "pauling_p2": "pauling_p2_decision",
    "pauling_p3": "pauling_p3_decision",
    "pauling_p4": "pauling_p4_decision",
    "pauling_p5": "pauling_p5_decision",
    "pauling_p2_p5": "pauling_p2_p5_decision",
    "m5": "m5_decision",
    "m5_phsc": "m5_phsc_decision",
    "m5_phsc_chsc": "m5_phsc_chsc_decision",
}
DECISIONS = ("KEEP", "REJECT", "ABSTAIN")
STATUS_VALUES = ("converged", "failed", "timeout")
NEAR_MIN_EV_PER_ATOM = 0.001
VALUABLE_EV_PER_ATOM = 0.05
HIGH_ENERGY_EV_PER_ATOM = 0.20
SEVERE_ENERGY_DROP_EV_PER_ATOM = 0.10
SEVERE_INITIAL_FMAX_EV_PER_A = 1.0
SEVERE_MAX_DISPLACEMENT_ANGSTROM = 0.5
SEVERE_ABS_LOG_VOLUME_RATIO = 0.10


def _json_bytes(value: Mapping[str, object]) -> bytes:
    return (json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def _proportion(successes: int, total: int) -> dict[str, object]:
    if type(successes) is not int or type(total) is not int or not 0 <= successes <= total:
        raise ValueError("invalid proportion counts")
    if total == 0:
        return {
            "numerator": successes,
            "denominator": total,
            "estimate": None,
            "wilson_ci95": [None, None],
        }
    estimate = successes / total
    z = 1.959963984540054
    denominator = 1.0 + z * z / total
    center = (estimate + z * z / (2.0 * total)) / denominator
    half = (
        z
        * math.sqrt(
            estimate * (1.0 - estimate) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return {
        "numerator": successes,
        "denominator": total,
        "estimate": float(estimate),
        "wilson_ci95": [float(max(0.0, center - half)), float(min(1.0, center + half))],
    }


def _validated_queue(
    queue_data: bytes, manifest_data: bytes, *, queue_name: str
) -> tuple[pd.DataFrame, dict[str, object]]:
    manifest = _strict_json(manifest_data, role="DFT queue manifest")
    if manifest.get("protocol") != QUEUE_PROTOCOL:
        raise ValueError("DFT queue protocol differs")
    if manifest.get("all_attempts_queued") is not True or manifest.get(
        "selection_by_gate"
    ) is not False:
        raise ValueError("DFT queue was not a full gate-independent cohort")
    if manifest.get("licensed_potcar_contents_included") is not False:
        raise ValueError("DFT queue manifest license flag differs")
    outputs = manifest.get("outputs_sha256")
    digest = hashlib.sha256(queue_data).hexdigest()
    if not isinstance(outputs, Mapping) or outputs.get(queue_name) != digest:
        raise ValueError("DFT queue hash differs from its manifest")
    definitions = manifest.get("run_protocol")
    if not isinstance(definitions, Mapping):
        raise ValueError("DFT queue run protocol is missing")
    thresholds = definitions.get("endpoint_definitions_frozen_before_DFT")
    expected = {
        "near_min_eV_per_atom": NEAR_MIN_EV_PER_ATOM,
        "valuable_eV_per_atom": VALUABLE_EV_PER_ATOM,
        "high_energy_eV_per_atom": HIGH_ENERGY_EV_PER_ATOM,
        "complete_composition_groups_only": True,
    }
    if not isinstance(thresholds, Mapping) or dict(thresholds) != expected:
        raise ValueError("DFT endpoint thresholds differ from the frozen queue")
    table = pd.read_parquet(io.BytesIO(queue_data))
    required = {"sid", "formula", "natoms", "task_available", *METHOD_COLUMNS.values()}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"DFT queue lacks columns: {sorted(missing)}")
    table = table.loc[:, ["sid", "formula", "natoms", "task_available", *METHOD_COLUMNS.values()]].copy()
    if table["sid"].isna().any() or table["sid"].astype(str).duplicated().any():
        raise ValueError("DFT queue sid values must be nonmissing and unique")
    table["sid"] = table["sid"].astype(str)
    if not table["task_available"].map(lambda value: isinstance(value, (bool, np.bool_))).all():
        raise ValueError("DFT task_available must be boolean")
    if not table["task_available"].astype(bool).all():
        raise ValueError("endpoint evaluation requires a runnable task for every attempt")
    if (pd.to_numeric(table["natoms"], errors="coerce") <= 0).any():
        raise ValueError("DFT queue atom counts must be positive")
    for column in METHOD_COLUMNS.values():
        if not set(table[column].astype(str)).issubset(DECISIONS):
            raise ValueError(f"DFT queue contains invalid decisions in {column}")
    return table.sort_values("sid", kind="stable", ignore_index=True), manifest


def _validated_endpoints(
    endpoint_data: bytes,
    manifest_data: bytes,
    *,
    endpoint_name: str,
    queue_manifest_sha256: str,
) -> pd.DataFrame:
    manifest = _strict_json(manifest_data, role="DFT endpoint manifest")
    if manifest.get("protocol") != ENDPOINT_PROTOCOL:
        raise ValueError("DFT endpoint protocol differs")
    if manifest.get("queue_manifest_sha256") != queue_manifest_sha256:
        raise ValueError("DFT endpoints are not bound to this queue manifest")
    if manifest.get("all_attempts_accounted") is not True or manifest.get(
        "vasp_execution_complete"
    ) is not True:
        raise ValueError("DFT endpoint accounting is incomplete")
    outputs = manifest.get("outputs_sha256")
    if not isinstance(outputs, Mapping) or outputs.get(endpoint_name) != hashlib.sha256(
        endpoint_data
    ).hexdigest():
        raise ValueError("DFT endpoint hash differs from its manifest")
    table = pd.read_parquet(io.BytesIO(endpoint_data))
    if list(table.columns) != list(ENDPOINT_COLUMNS):
        raise ValueError("DFT endpoint columns differ from the frozen schema")
    if table["sid"].isna().any() or table["sid"].astype(str).duplicated().any():
        raise ValueError("DFT endpoint sid values must be nonmissing and unique")
    table["sid"] = table["sid"].astype(str)
    if not set(table["status"].astype(str)).issubset(STATUS_VALUES):
        raise ValueError("DFT endpoint status is invalid")
    wall = pd.to_numeric(table["wall_time_seconds"], errors="coerce").to_numpy(float)
    if (~np.isfinite(wall) | (wall < 0.0)).any():
        raise ValueError("DFT endpoint wall times must be finite and nonnegative")
    converged = table["status"].eq("converged").to_numpy(bool)
    required_numeric = [
        "static_energy_ev",
        "static_fmax_ev_per_a",
        "static_max_stress_gpa",
        "relaxed_energy_ev",
        "relaxed_fmax_ev_per_a",
        "initial_volume_angstrom3",
        "relaxed_volume_angstrom3",
        "max_displacement_angstrom",
    ]
    for column in required_numeric:
        values = pd.to_numeric(table[column], errors="coerce").to_numpy(float)
        if (~np.isfinite(values[converged])).any():
            raise ValueError(f"converged DFT endpoints require finite {column}")
    for column in ("static_fmax_ev_per_a", "relaxed_fmax_ev_per_a", "initial_volume_angstrom3", "relaxed_volume_angstrom3", "max_displacement_angstrom"):
        values = pd.to_numeric(table[column], errors="coerce").to_numpy(float)
        if (values[converged] < 0.0).any():
            raise ValueError(f"converged DFT endpoints require nonnegative {column}")
    return table.sort_values("sid", kind="stable", ignore_index=True)


def _method_metrics(joined: pd.DataFrame, decision_column: str) -> dict[str, object]:
    decisions = joined[decision_column].astype(str)
    reject = decisions.eq("REJECT").to_numpy(bool)
    coverage = decisions.ne("ABSTAIN").to_numpy(bool)
    evaluable = joined["endpoint_evaluable"].to_numpy(bool)
    valuable = joined["valuable"].to_numpy(bool)
    exact = joined["exact_min"].to_numpy(bool)
    high = joined["high_energy"].to_numpy(bool)
    severe = joined["severe_relaxation"].to_numpy(bool)
    valuable_mask = evaluable & valuable
    exact_mask = evaluable & exact
    high_mask = evaluable & high
    severe_mask = evaluable & severe
    group_all_reject = 0
    for _, group in joined.loc[evaluable].groupby("formula", sort=False):
        if group[decision_column].eq("REJECT").all():
            group_all_reject += 1
    wall = joined["wall_time_seconds"].to_numpy(float)
    wall_total = float(wall.sum())
    return {
        "decision_counts": {
            decision: int(decisions.eq(decision).sum()) for decision in DECISIONS
        },
        "coverage": _proportion(int(coverage.sum()), len(joined)),
        "dft_savings_row_fraction": float(reject.sum() / len(joined)) if len(joined) else None,
        "dft_savings_wall_fraction": float(wall[reject].sum() / wall_total) if wall_total > 0.0 else None,
        "valuable_recall": _proportion(
            int((valuable_mask & ~reject).sum()), int(valuable_mask.sum())
        ),
        "exact_min_retention": _proportion(
            int((exact_mask & ~reject).sum()), int(exact_mask.sum())
        ),
        "high_energy_rejection_recall": _proportion(
            int((high_mask & reject).sum()), int(high_mask.sum())
        ),
        "severe_relaxation_rejection_recall": _proportion(
            int((severe_mask & reject).sum()), int(severe_mask.sum())
        ),
        "all_rejected_complete_groups": int(group_all_reject),
    }


def _estimate(metric: Mapping[str, object]) -> float | None:
    value = metric.get("estimate")
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def evaluate_dft_endpoints(
    *,
    queue_path: Path,
    queue_manifest_path: Path,
    endpoints_path: Path,
    endpoint_manifest_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Join endpoints only after verifying full queue accounting and isolation."""

    target = Path(output_dir).resolve()
    if os.path.lexists(target):
        raise FileExistsError(f"refusing to overwrite existing output: {target}")
    paths = {
        "queue": Path(queue_path).resolve(),
        "queue_manifest": Path(queue_manifest_path).resolve(),
        "endpoints": Path(endpoints_path).resolve(),
        "endpoint_manifest": Path(endpoint_manifest_path).resolve(),
    }
    for role, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{role} is not a file: {path}")
    if paths["queue"].parent == paths["endpoints"].parent:
        raise ValueError("DFT endpoints must be physically separated from the queue")
    input_hashes = {role: _sha256_file(path) for role, path in paths.items()}
    queue, _queue_manifest = _validated_queue(
        paths["queue"].read_bytes(),
        paths["queue_manifest"].read_bytes(),
        queue_name=paths["queue"].name,
    )
    endpoints = _validated_endpoints(
        paths["endpoints"].read_bytes(),
        paths["endpoint_manifest"].read_bytes(),
        endpoint_name=paths["endpoints"].name,
        queue_manifest_sha256=input_hashes["queue_manifest"],
    )
    if set(queue.sid) != set(endpoints.sid):
        raise ValueError("DFT queue and endpoint sid sets differ")
    joined = queue.merge(endpoints, on="sid", validate="one_to_one", sort=False)
    joined = joined.sort_values("sid", kind="stable", ignore_index=True)
    converged = joined["status"].eq("converged")
    joined["group_complete"] = converged.groupby(joined["formula"], sort=False).transform("all")
    joined["endpoint_evaluable"] = converged & joined["group_complete"]
    joined["relaxed_energy_ev_per_atom"] = joined["relaxed_energy_ev"] / joined["natoms"]
    eligible_energy = joined["relaxed_energy_ev_per_atom"].where(joined["endpoint_evaluable"])
    group_min = eligible_energy.groupby(joined["formula"], sort=False).transform("min")
    joined["delta_e_ev_per_atom"] = eligible_energy - group_min
    tolerance = 1e-12
    joined["exact_min"] = joined["endpoint_evaluable"] & joined["delta_e_ev_per_atom"].le(tolerance)
    joined["near_min"] = joined["endpoint_evaluable"] & joined["delta_e_ev_per_atom"].le(NEAR_MIN_EV_PER_ATOM + tolerance)
    joined["valuable"] = joined["endpoint_evaluable"] & joined["delta_e_ev_per_atom"].le(VALUABLE_EV_PER_ATOM + tolerance)
    joined["high_energy"] = joined["endpoint_evaluable"] & joined["delta_e_ev_per_atom"].ge(HIGH_ENERGY_EV_PER_ATOM - tolerance)
    energy_drop = (joined["static_energy_ev"] - joined["relaxed_energy_ev"]) / joined["natoms"]
    with np.errstate(divide="ignore", invalid="ignore"):
        volume_change = np.abs(
            np.log(joined["relaxed_volume_angstrom3"] / joined["initial_volume_angstrom3"])
        )
    joined["severe_relaxation"] = joined["endpoint_evaluable"] & (
        energy_drop.ge(SEVERE_ENERGY_DROP_EV_PER_ATOM)
        | joined["static_fmax_ev_per_a"].ge(SEVERE_INITIAL_FMAX_EV_PER_A)
        | joined["max_displacement_angstrom"].ge(SEVERE_MAX_DISPLACEMENT_ANGSTROM)
        | volume_change.ge(SEVERE_ABS_LOG_VOLUME_RATIO)
    )

    methods = {
        name: _method_metrics(joined, column) for name, column in METHOD_COLUMNS.items()
    }
    evaluable = joined["endpoint_evaluable"].to_numpy(bool)
    valuable = joined["valuable"].to_numpy(bool)
    high = joined["high_energy"].to_numpy(bool)
    m5_reject = joined[METHOD_COLUMNS["m5"]].eq("REJECT").to_numpy(bool)
    paired: dict[str, object] = {}
    for name, column in METHOD_COLUMNS.items():
        if name == "m5":
            continue
        candidate = joined[column].eq("REJECT").to_numpy(bool)
        paired[name] = {
            "net_reject_delta": int(candidate.sum() - m5_reject.sum()),
            "valuable_added_false_rejects": int(
                (evaluable & valuable & candidate & ~m5_reject).sum()
            ),
            "valuable_recovered": int(
                (evaluable & valuable & ~candidate & m5_reject).sum()
            ),
            "high_energy_added_true_rejects": int(
                (evaluable & high & candidate & ~m5_reject).sum()
            ),
            "high_energy_lost": int(
                (evaluable & high & ~candidate & m5_reject).sum()
            ),
        }

    candidate = methods["m5_phsc_chsc"]
    valuable_ci = candidate["valuable_recall"]["wilson_ci95"]
    exact_ci = candidate["exact_min_retention"]["wilson_ci95"]
    safety = {
        "valuable_recall_wilson_lower_at_least_0.95": bool(
            valuable_ci[0] is not None and float(valuable_ci[0]) >= 0.95
        ),
        "exact_min_retention_wilson_lower_at_least_0.95": bool(
            exact_ci[0] is not None and float(exact_ci[0]) >= 0.95
        ),
        "no_all_rejected_complete_group": candidate["all_rejected_complete_groups"] == 0,
    }
    superiority: dict[str, object] = {}
    candidate_high = _estimate(candidate["high_energy_rejection_recall"])
    candidate_savings = candidate["dft_savings_row_fraction"]
    for comparator in ("m5", "pauling_p2_p5"):
        other = methods[comparator]
        other_high = _estimate(other["high_energy_rejection_recall"])
        other_savings = other["dft_savings_row_fraction"]
        superiority[comparator] = {
            "higher_high_energy_recall": bool(
                candidate_high is not None
                and other_high is not None
                and candidate_high > other_high
            ),
            "higher_dft_savings": bool(
                candidate_savings is not None
                and other_savings is not None
                and float(candidate_savings) > float(other_savings)
            ),
        }
    all_checks = [*safety.values()]
    for values in superiority.values():
        all_checks.extend(values.values())
    gate = {"safety": safety, "superiority": superiority, "passes": bool(all(all_checks))}

    counts = {
        "attempts": len(joined),
        "converged": int(joined["status"].eq("converged").sum()),
        "failed": int(joined["status"].eq("failed").sum()),
        "timeout": int(joined["status"].eq("timeout").sum()),
        "complete_groups": int(joined.loc[joined["group_complete"], "formula"].nunique()),
        "evaluable_rows": int(joined["endpoint_evaluable"].sum()),
        "valuable_rows": int(joined["valuable"].sum()),
        "high_energy_rows": int(joined["high_energy"].sum()),
    }
    result: dict[str, object] = {
        "protocol": PROTOCOL,
        "all_attempts_accounted": True,
        "endpoint_opening_was_isolated": True,
        "thresholds_refit_after_endpoint_opening": False,
        "endpoint_definitions": {
            "near_min_eV_per_atom": NEAR_MIN_EV_PER_ATOM,
            "valuable_eV_per_atom": VALUABLE_EV_PER_ATOM,
            "high_energy_eV_per_atom": HIGH_ENERGY_EV_PER_ATOM,
            "complete_composition_groups_only": True,
            "severe_relaxation": {
                "energy_drop_eV_per_atom_at_least": SEVERE_ENERGY_DROP_EV_PER_ATOM,
                "initial_fmax_eV_per_A_at_least": SEVERE_INITIAL_FMAX_EV_PER_A,
                "max_displacement_A_at_least": SEVERE_MAX_DISPLACEMENT_ANGSTROM,
                "abs_log_volume_ratio_at_least": SEVERE_ABS_LOG_VOLUME_RATIO,
            },
        },
        "counts": counts,
        "methods": methods,
        "paired_vs_m5": paired,
        "superiority_gate": gate,
    }

    repository_source = Path(__file__).resolve()
    source_hash = _sha256_file(repository_source)

    def verify_unchanged() -> None:
        for role, path in paths.items():
            if _sha256_file(path) != input_hashes[role]:
                raise RuntimeError(f"input {role} changed before publication")
        if _sha256_file(repository_source) != source_hash:
            raise RuntimeError("endpoint evaluator source changed before publication")

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        joined_path = staging / JOINED_NAME
        joined.to_parquet(joined_path, index=False)
        result_path = staging / RESULT_NAME
        result_path.write_bytes(_json_bytes(result))
        manifest = {
            "protocol": PROTOCOL,
            "inputs_sha256": {
                role: {"path": str(path), "sha256": input_hashes[role]}
                for role, path in paths.items()
            },
            "executed_source_sha256": {
                "src/next12_dft_endpoint_evaluate.py": source_hash
            },
            "outputs_sha256": {
                RESULT_NAME: _sha256_file(result_path),
                JOINED_NAME: _sha256_file(joined_path),
            },
            "integrity": {"prepublish_rehash": "passed"},
        }
        (staging / MANIFEST_NAME).write_bytes(_json_bytes(manifest))
        verify_unchanged()
        _atomic_publish_directory_no_replace(staging, target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", required=True, type=Path)
    parser.add_argument("--queue-manifest", required=True, type=Path)
    parser.add_argument("--endpoints", required=True, type=Path)
    parser.add_argument("--endpoint-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    arguments = parser.parse_args(argv)
    evaluate_dft_endpoints(
        queue_path=arguments.queue,
        queue_manifest_path=arguments.queue_manifest,
        endpoints_path=arguments.endpoints,
        endpoint_manifest_path=arguments.endpoint_manifest,
        output_dir=arguments.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
