"""Open fully accounted blinded NEXT13d VASP endpoints, then unblind pairs."""

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
from scipy.stats import binomtest, wilcoxon

from src.next13d_acsc_dft_pairs import (
    BLINDED_QUEUE_NAME,
    PRIVATE_PAIRS_NAME,
    PROTOCOL as QUEUE_PROTOCOL,
    _json_bytes,
    _sha256_file,
    _strict_json,
)
from src.next8_mattersim_committee_features import (
    _atomic_publish_directory_no_replace,
)


PROTOCOL = "2026-08-02-next13d-acsc-isolated-paired-dft-evaluation-v1"
ENDPOINT_PROTOCOL = "2026-08-02-next13d-acsc-blinded-vasp-pbe-endpoints-v1"
RESULT_NAME = "PAIRED_DFT_EVALUATION.json"
JOINED_NAME = "paired_dft_joined.parquet"
MANIFEST_NAME = "MANIFEST.json"
ENDPOINT_COLUMNS = (
    "task_id",
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
STATUS_VALUES = {"converged", "failed", "timeout"}
FROZEN_ENDPOINT_DEFINITIONS: Mapping[str, object] = {
    "severe_energy_drop_eV_per_atom": 0.10,
    "severe_initial_fmax_eV_per_A": 1.0,
    "severe_max_displacement_angstrom": 0.5,
    "severe_abs_log_volume_ratio": 0.10,
    "nonconvergence_is_severe": True,
    "same_rk_relaxed_energy_primary": True,
    "direction": "treatment_minus_control; positive severity/energy supports ACSC rejection",
}


def _validated_queue(
    *, blinded_data: bytes, private_data: bytes, manifest_data: bytes
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    manifest = _strict_json(manifest_data, role="paired queue manifest")
    if manifest.get("protocol") != QUEUE_PROTOCOL:
        raise ValueError("paired queue protocol differs")
    if manifest.get("all_tasks_queued") is not True:
        raise ValueError("paired queue is not fully accounted")
    if manifest.get("executor_blinded_to_sid_and_role") is not True or manifest.get(
        "private_mapping_must_be_withheld_from_executor"
    ) is not True:
        raise ValueError("paired queue blinding contract differs")
    if manifest.get("licensed_potcar_contents_included") is not False:
        raise ValueError("paired queue license flag differs")
    outputs = manifest.get("outputs_sha256")
    expected = {
        BLINDED_QUEUE_NAME: hashlib.sha256(blinded_data).hexdigest(),
        PRIVATE_PAIRS_NAME: hashlib.sha256(private_data).hexdigest(),
    }
    if not isinstance(outputs, Mapping) or any(outputs.get(name) != digest for name, digest in expected.items()):
        raise ValueError("paired queue output hash differs from its manifest")
    run_protocol = manifest.get("run_protocol")
    if not isinstance(run_protocol, Mapping) or run_protocol.get(
        "endpoint_definitions_frozen_before_DFT"
    ) != dict(FROZEN_ENDPOINT_DEFINITIONS):
        raise ValueError("paired DFT endpoint definitions differ")

    blinded = pd.read_parquet(io.BytesIO(blinded_data))
    blind_required = {
        "task_index", "task_id", "formula", "natoms", "task_available",
        "task_prefix", "encut_ev",
    }
    missing = blind_required - set(blinded.columns)
    if missing:
        raise ValueError(f"blinded queue lacks columns: {sorted(missing)}")
    if {"sid", "role", "pair_id"} & set(blinded.columns):
        raise ValueError("blinded queue leaks private mapping fields")
    blinded = blinded.loc[:, sorted(blind_required)].copy()
    if blinded["task_id"].isna().any() or blinded["task_id"].astype(str).duplicated().any():
        raise ValueError("blinded task IDs must be unique")
    blinded["task_id"] = blinded["task_id"].astype(str)
    if not blinded["task_available"].map(lambda value: isinstance(value, (bool, np.bool_))).all() or not blinded["task_available"].astype(bool).all():
        raise ValueError("every blinded task must be runnable")
    natoms = pd.to_numeric(blinded["natoms"], errors="coerce").to_numpy(float)
    if (~np.isfinite(natoms) | (natoms <= 0) | (natoms != np.floor(natoms))).any():
        raise ValueError("blinded atom counts must be positive integers")
    blinded["natoms"] = natoms.astype(int)

    private = pd.read_parquet(io.BytesIO(private_data))
    private_required = {
        "pair_id", "task_id", "role", "sid", "rk", "formula", "natoms",
        "m5_gap_ev_per_atom", "match_tier", "same_rk",
    }
    missing = private_required - set(private.columns)
    if missing:
        raise ValueError(f"private pair mapping lacks columns: {sorted(missing)}")
    private = private.loc[:, sorted(private_required)].copy()
    for column in ("pair_id", "task_id", "role", "sid", "rk"):
        if private[column].isna().any():
            raise ValueError(f"private mapping {column} values must be present")
        private[column] = private[column].astype(str)
    if private["task_id"].duplicated().any() or set(private["task_id"]) != set(blinded["task_id"]):
        raise ValueError("private and blinded task coverage differs")
    if not set(private["role"]).issubset({"treatment", "control"}):
        raise ValueError("private mapping role differs")
    for pair_id, group in private.groupby("pair_id", sort=False):
        if len(group) != 2 or set(group["role"]) != {"treatment", "control"}:
            raise ValueError(f"pair {pair_id} does not contain one treatment and one control")
        if group["sid"].duplicated().any():
            raise ValueError(f"pair {pair_id} reuses a sid")
        if group["match_tier"].nunique() != 1 or group["same_rk"].nunique() != 1:
            raise ValueError(f"pair {pair_id} metadata differs by role")
    if private["sid"].duplicated().any():
        raise ValueError("private mapping reuses a structure across pairs")
    return (
        blinded.sort_values("task_id", kind="stable", ignore_index=True),
        private.sort_values(["pair_id", "role"], kind="stable", ignore_index=True),
        manifest,
    )


def _validated_endpoints_after_manifest(
    *, endpoint_data: bytes, manifest: Mapping[str, object], endpoint_name: str,
    expected_task_ids: set[str]
) -> pd.DataFrame:
    outputs = manifest.get("outputs_sha256")
    if not isinstance(outputs, Mapping) or outputs.get(endpoint_name) != hashlib.sha256(endpoint_data).hexdigest():
        raise ValueError("DFT endpoint hash differs from its manifest")
    table = pd.read_parquet(io.BytesIO(endpoint_data))
    if list(table.columns) != list(ENDPOINT_COLUMNS):
        raise ValueError("DFT endpoint columns differ from the frozen schema")
    if table["task_id"].isna().any() or table["task_id"].astype(str).duplicated().any():
        raise ValueError("DFT endpoint task IDs must be unique")
    table["task_id"] = table["task_id"].astype(str)
    if set(table["task_id"]) != expected_task_ids:
        raise ValueError("DFT endpoint task coverage differs from the blinded queue")
    if not set(table["status"].astype(str)).issubset(STATUS_VALUES):
        raise ValueError("DFT endpoint status is invalid")
    wall = pd.to_numeric(table["wall_time_seconds"], errors="coerce").to_numpy(float)
    if (~np.isfinite(wall) | (wall < 0.0)).any():
        raise ValueError("DFT endpoint wall times must be finite and nonnegative")
    converged = table["status"].eq("converged").to_numpy(bool)
    required_numeric = (
        "static_energy_ev", "static_fmax_ev_per_a", "static_max_stress_gpa",
        "relaxed_energy_ev", "relaxed_fmax_ev_per_a", "ionic_steps",
        "initial_volume_angstrom3", "relaxed_volume_angstrom3",
        "max_displacement_angstrom",
    )
    numeric: dict[str, np.ndarray] = {}
    for column in required_numeric:
        numeric[column] = pd.to_numeric(table[column], errors="coerce").to_numpy(float)
        if (~np.isfinite(numeric[column][converged])).any():
            raise ValueError(f"converged endpoints require finite {column}")
    for column in ("static_fmax_ev_per_a", "relaxed_fmax_ev_per_a", "max_displacement_angstrom"):
        if (numeric[column][converged] < 0.0).any():
            raise ValueError(f"converged endpoints require nonnegative {column}")
    for column in ("initial_volume_angstrom3", "relaxed_volume_angstrom3"):
        if (numeric[column][converged] <= 0.0).any():
            raise ValueError(f"converged endpoints require positive {column}")
    steps = numeric["ionic_steps"][converged]
    if (steps < 0.0).any() or (steps != np.floor(steps)).any():
        raise ValueError("converged endpoint ionic steps must be nonnegative integers")
    return table.sort_values("task_id", kind="stable", ignore_index=True)


def _wilson(successes: int, total: int) -> list[float | None]:
    if total == 0:
        return [None, None]
    z = 1.959963984540054
    estimate = successes / total
    denominator = 1.0 + z * z / total
    center = (estimate + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(estimate * (1.0 - estimate) / total + z * z / (4.0 * total * total)) / denominator
    return [float(max(0.0, center - half)), float(min(1.0, center + half))]


def _paired_difference_summary(values: Sequence[float]) -> dict[str, object]:
    data = np.asarray(values, dtype=float)
    if data.ndim != 1 or (~np.isfinite(data)).any():
        raise ValueError("paired differences must be a finite vector")
    positive = int((data > 0.0).sum())
    negative = int((data < 0.0).sum())
    zero = int((data == 0.0).sum())
    discordant = positive + negative
    exact_p = float(binomtest(positive, discordant, p=0.5, alternative="greater").pvalue) if discordant else None
    if discordant:
        try:
            wilcoxon_p = float(wilcoxon(data, zero_method="wilcox", alternative="greater").pvalue)
        except ValueError:
            wilcoxon_p = None
    else:
        wilcoxon_p = None
    return {
        "paired_differences_treatment_minus_control": [float(value) for value in data],
        "pairs": int(len(data)),
        "mean_difference": float(data.mean()) if len(data) else None,
        "median_difference": float(np.median(data)) if len(data) else None,
        "positive": positive,
        "negative": negative,
        "zero": zero,
        "positive_fraction_nonzero": float(positive / discordant) if discordant else None,
        "positive_fraction_wilson_ci95": _wilson(positive, discordant),
        "one_sided_exact_sign_p": exact_p,
        "one_sided_wilcoxon_p": wilcoxon_p,
    }


def evaluate_paired_dft_endpoints(
    *,
    blinded_queue_path: Path,
    private_pairs_path: Path,
    queue_manifest_path: Path,
    endpoints_path: Path,
    endpoint_manifest_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Require sealed full accounting before endpoint bytes are parsed and roles unblinded."""

    target = Path(output_dir).resolve()
    if os.path.lexists(target):
        raise FileExistsError(f"refusing to overwrite existing output: {target}")
    paths = {
        "blinded_queue": Path(blinded_queue_path).resolve(),
        "private_pairs": Path(private_pairs_path).resolve(),
        "queue_manifest": Path(queue_manifest_path).resolve(),
        "endpoints": Path(endpoints_path).resolve(),
        "endpoint_manifest": Path(endpoint_manifest_path).resolve(),
    }
    for role, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{role} is not a file: {path}")
    queue_directory = paths["queue_manifest"].parent
    if paths["endpoints"].parent == queue_directory or paths["endpoint_manifest"].parent == queue_directory:
        raise ValueError("DFT endpoints must be physically separated from the queue")
    input_hashes = {role: _sha256_file(path) for role, path in paths.items()}
    blinded, private, _queue_manifest = _validated_queue(
        blinded_data=paths["blinded_queue"].read_bytes(),
        private_data=paths["private_pairs"].read_bytes(),
        manifest_data=paths["queue_manifest"].read_bytes(),
    )

    # Check the external seal before opening the endpoint table itself.
    endpoint_manifest = _strict_json(
        paths["endpoint_manifest"].read_bytes(), role="DFT endpoint manifest"
    )
    if endpoint_manifest.get("protocol") != ENDPOINT_PROTOCOL:
        raise ValueError("DFT endpoint protocol differs")
    if endpoint_manifest.get("queue_manifest_sha256") != input_hashes["queue_manifest"]:
        raise ValueError("DFT endpoints are not bound to this paired queue")
    if endpoint_manifest.get("all_tasks_accounted") is not True or endpoint_manifest.get(
        "vasp_execution_complete"
    ) is not True:
        raise ValueError("DFT endpoint accounting is incomplete")
    if endpoint_manifest.get("executor_received_private_mapping") is not False:
        raise ValueError("DFT executor was not blinded from the private mapping")
    endpoints = _validated_endpoints_after_manifest(
        endpoint_data=paths["endpoints"].read_bytes(),
        manifest=endpoint_manifest,
        endpoint_name=paths["endpoints"].name,
        expected_task_ids=set(blinded["task_id"]),
    )

    joined = private.merge(
        blinded.loc[:, ["task_id", "natoms", "formula"]].rename(
            columns={"natoms": "blinded_natoms", "formula": "blinded_formula"}
        ),
        on="task_id", validate="one_to_one",
    ).merge(endpoints, on="task_id", validate="one_to_one")
    if not (joined["natoms"].astype(int) == joined["blinded_natoms"].astype(int)).all() or not (
        joined["formula"].astype(str) == joined["blinded_formula"].astype(str)
    ).all():
        raise ValueError("private and blinded task metadata differs")
    converged = joined["status"].eq("converged")
    joined["energy_drop_ev_per_atom"] = np.where(
        converged,
        (pd.to_numeric(joined["static_energy_ev"], errors="coerce") - pd.to_numeric(joined["relaxed_energy_ev"], errors="coerce")) / joined["natoms"].astype(float),
        np.nan,
    )
    joined["relaxed_energy_ev_per_atom"] = np.where(
        converged,
        pd.to_numeric(joined["relaxed_energy_ev"], errors="coerce") / joined["natoms"].astype(float),
        np.nan,
    )
    joined["abs_log_volume_ratio"] = np.where(
        converged,
        np.abs(np.log(pd.to_numeric(joined["relaxed_volume_angstrom3"], errors="coerce") / pd.to_numeric(joined["initial_volume_angstrom3"], errors="coerce"))),
        np.nan,
    )
    joined["severe_relaxation"] = (
        ~converged
        | (joined["energy_drop_ev_per_atom"] >= 0.10)
        | (pd.to_numeric(joined["static_fmax_ev_per_a"], errors="coerce") >= 1.0)
        | (pd.to_numeric(joined["max_displacement_angstrom"], errors="coerce") >= 0.5)
        | (joined["abs_log_volume_ratio"] >= 0.10)
    )
    joined = joined.sort_values(["pair_id", "role"], kind="stable", ignore_index=True)

    paired = joined.pivot(index="pair_id", columns="role")
    treatment_severe = paired["severe_relaxation"]["treatment"].astype(bool)
    control_severe = paired["severe_relaxation"]["control"].astype(bool)
    treatment_only = int((treatment_severe & ~control_severe).sum())
    control_only = int((~treatment_severe & control_severe).sum())
    discordant = treatment_only + control_only
    severe_exact_p = float(binomtest(treatment_only, discordant, p=0.5, alternative="greater").pvalue) if discordant else None
    primary = {
        "treatment_severe": int(treatment_severe.sum()),
        "control_severe": int(control_severe.sum()),
        "both_severe": int((treatment_severe & control_severe).sum()),
        "neither_severe": int((~treatment_severe & ~control_severe).sum()),
        "treatment_severe_control_not": treatment_only,
        "control_severe_treatment_not": control_only,
        "paired_risk_difference": float(treatment_severe.mean() - control_severe.mean()),
        "treatment_share_of_discordant": float(treatment_only / discordant) if discordant else None,
        "treatment_share_wilson_ci95": _wilson(treatment_only, discordant),
        "one_sided_exact_p": severe_exact_p,
        "frozen_success_rule": "treatment-only discordances exceed control-only with one-sided exact p < 0.05",
        "frozen_success": bool(treatment_only > control_only and severe_exact_p is not None and severe_exact_p < 0.05),
    }

    both_converged_ids = paired.index[
        paired["status"]["treatment"].eq("converged")
        & paired["status"]["control"].eq("converged")
    ]
    continuous: dict[str, object] = {}
    for output_name, column in (
        ("energy_drop_ev_per_atom", "energy_drop_ev_per_atom"),
        ("initial_fmax_ev_per_a", "static_fmax_ev_per_a"),
        ("max_displacement_angstrom", "max_displacement_angstrom"),
        ("abs_log_volume_ratio", "abs_log_volume_ratio"),
    ):
        differences = (
            paired.loc[both_converged_ids, (column, "treatment")].to_numpy(float)
            - paired.loc[both_converged_ids, (column, "control")].to_numpy(float)
        )
        continuous[output_name] = _paired_difference_summary(differences)

    same_rk = paired["same_rk"]["treatment"].astype(bool)
    same_rk_converged_ids = paired.index.intersection(both_converged_ids)[
        same_rk.loc[paired.index.intersection(both_converged_ids)].to_numpy(bool)
    ]
    relaxed_differences = (
        paired.loc[same_rk_converged_ids, ("relaxed_energy_ev_per_atom", "treatment")].to_numpy(float)
        - paired.loc[same_rk_converged_ids, ("relaxed_energy_ev_per_atom", "control")].to_numpy(float)
    )
    same_rk_energy = _paired_difference_summary(relaxed_differences)

    counts = {
        "tasks": len(joined),
        "pairs": int(joined["pair_id"].nunique()),
        "converged_tasks": int(joined["status"].eq("converged").sum()),
        "failed_tasks": int(joined["status"].eq("failed").sum()),
        "timeout_tasks": int(joined["status"].eq("timeout").sum()),
        "both_converged_pairs": int(len(both_converged_ids)),
        "same_rk_both_converged_pairs": int(len(same_rk_converged_ids)),
    }
    result: dict[str, object] = {
        "protocol": PROTOCOL,
        "queue_protocol": QUEUE_PROTOCOL,
        "endpoint_protocol": ENDPOINT_PROTOCOL,
        "all_endpoints_opened_only_after_full_accounting": True,
        "thresholds_refit": False,
        "counts": counts,
        "primary_severe_relaxation": primary,
        "continuous_paired_endpoints": continuous,
        "same_rk_relaxed_energy_ev_per_atom": same_rk_energy,
        "scientific_improvement_claim": False,
        "interpretation_guard": "A positive result is independent DFT support for this frozen subset, not proof of universal superiority over DFT.",
    }

    repository_root = Path(__file__).resolve().parents[1]
    source_paths = {
        "src/next13d_acsc_dft_endpoint_evaluate.py": Path(__file__).resolve(),
        "src/next13d_acsc_dft_pairs.py": repository_root / "src/next13d_acsc_dft_pairs.py",
    }
    source_hashes = {relative: _sha256_file(path) for relative, path in source_paths.items()}
    manifest: dict[str, object] = {
        "protocol": PROTOCOL,
        "all_tasks_accounted_before_unblinding": True,
        "executor_received_private_mapping": False,
        "thresholds_refit": False,
        "inputs_sha256": {role: {"path": str(paths[role]), "sha256": digest} for role, digest in input_hashes.items()},
        "executed_source_sha256": source_hashes,
        "scientific_improvement_claim": False,
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
        result_path = staging / RESULT_NAME
        joined_path = staging / JOINED_NAME
        result_path.write_bytes(_json_bytes(result))
        joined.to_parquet(joined_path, index=False)
        manifest["outputs_sha256"] = {
            RESULT_NAME: _sha256_file(result_path), JOINED_NAME: _sha256_file(joined_path)
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
    parser.add_argument("--blinded-queue", required=True, type=Path)
    parser.add_argument("--private-pairs", required=True, type=Path)
    parser.add_argument("--queue-manifest", required=True, type=Path)
    parser.add_argument("--endpoints", required=True, type=Path)
    parser.add_argument("--endpoint-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    arguments = parser.parse_args(argv)
    evaluate_paired_dft_endpoints(
        blinded_queue_path=arguments.blinded_queue,
        private_pairs_path=arguments.private_pairs,
        queue_manifest_path=arguments.queue_manifest,
        endpoints_path=arguments.endpoints,
        endpoint_manifest_path=arguments.endpoint_manifest,
        output_dir=arguments.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
