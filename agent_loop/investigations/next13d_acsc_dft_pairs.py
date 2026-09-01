"""Freeze a label-free, blinded paired VASP queue for conservative ACSC."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
from typing import Callable, Mapping, Sequence
import zipfile

from ase import Atoms
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from src.next11_geometry_only_frames import load_geometry_only_archive
from src.next12_dft_queue import (
    _incar,
    _json_bytes,
    _poscar_bytes,
    _potcar_record,
    _zip_info,
)
from src.next13_acsc_old_cohort import PROTOCOL as ACSC_PROTOCOL
from src.next13c_acsc_nested_overlap import PROTOCOL as NESTED_PROTOCOL
from src.next8_mattersim_committee_features import (
    _atomic_publish_directory_no_replace,
)


PROTOCOL = "2026-08-02-next13d-acsc-blinded-paired-vasp-pbe-queue-v1"
BLINDED_QUEUE_NAME = "blinded_execution_queue.parquet"
PRIVATE_PAIRS_NAME = "private_pair_mapping.parquet"
TASKS_NAME = "blinded_vasp_tasks.zip"
RUN_PROTOCOL_NAME = "RUN_PROTOCOL.json"
MANIFEST_NAME = "MANIFEST.json"
FORMAL_EXPECTED_PAIR_COUNT = 58
FORMAL_EXPECTED_TIER_COUNTS: Mapping[str, int] = {
    "same_rk_same_natoms": 37,
    "same_rk_diff_natoms": 9,
    "fallback_same_natoms_chemistry": 12,
}
FROZEN_FORMAL_INPUT_SHA256: Mapping[str, str] = {
    "nested_features": "640f8a92b731d1c4c67826fd886d518ea52856803ce63139ef3713a8ab277072",
    "nested_manifest": "23ad7493ebc1676e909f0a28b89de8c26d470d1c6e7887d5c288ef251c861192",
    "acsc_features": "323bd5a25f9e746a972024486e789717157e2a91252dce8066f295d0c5d20e22",
    "acsc_manifest": "ca492c2a3466439335035a6b4ca169194176c80fac5db7511b888fb35658222c",
    "geometry_only_frames": "9b99226a7dc5497fca2aaadbf6ac554c657cb5475705072bcd56b92db9515de9",
    "geometry_manifest": "2e5559595fa1dbc3f16470b005e1dc4f9dbe4a65de81a39a52f53c0af9b14901",
}
FROZEN_POTENTIAL_MAP: Mapping[str, str] = {
    "Ag": "Ag", "Al": "Al", "As": "As", "Au": "Au", "B": "B",
    "Be": "Be_sv", "Bi": "Bi_d", "Br": "Br", "Cd": "Cd", "Ce": "Ce_3",
    "Cl": "Cl", "Cr": "Cr_pv", "Cs": "Cs_sv", "Cu": "Cu_pv", "Fe": "Fe_pv",
    "Ga": "Ga_d", "Gd": "Gd_3", "Ge": "Ge_d", "Hg": "Hg", "Ho": "Ho_3",
    "I": "I", "In": "In_d", "K": "K_sv", "Li": "Li_sv", "Lu": "Lu_3",
    "Mg": "Mg_pv", "Mn": "Mn_pv", "N": "N", "Na": "Na_pv", "Nb": "Nb_pv",
    "Ni": "Ni", "O": "O", "P": "P", "Pb": "Pb_d", "Pd": "Pd",
    "Pr": "Pr_3", "Pt": "Pt", "Rb": "Rb_sv", "Re": "Re_pv", "Rh": "Rh_pv",
    "Ru": "Ru_pv", "S": "S", "Sb": "Sb", "Se": "Se", "Si": "Si",
    "Sr": "Sr_sv", "Ta": "Ta_pv", "Te": "Te", "Tl": "Tl_d", "Yb": "Yb_2",
    "Zn": "Zn",
}

_RK_TOKEN = re.compile(r"([A-Z][a-z]?)([1-9][0-9]*)")
_ARCHIVE_SID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def _sha256_file(path: Path) -> str:
    with Path(path).open("rb") as handle:
        digest = hashlib.sha256()
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _paired_potcar_record(
    root: Path, element: str, potential: str
) -> dict[str, object]:
    """Read the local PAW identity, including the legacy standard-B filename."""

    try:
        return _potcar_record(root, element, potential)
    except FileNotFoundError:
        source = Path(root) / potential / f"POTCAR.{element}"
        if not source.is_file():
            raise
    content = source.read_bytes()
    text = content.decode("latin-1", errors="replace")
    title_match = re.search(r"^\s*TITEL\s*=\s*(.+?)\s*$", text, flags=re.MULTILINE)
    enmax_match = re.search(r"\bENMAX\s*=\s*([0-9]+(?:\.[0-9]*)?)", text)
    if title_match is None or enmax_match is None:
        raise ValueError(f"POTCAR metadata missing for {potential}")
    enmax = float(enmax_match.group(1))
    if not math.isfinite(enmax) or enmax <= 0.0:
        raise ValueError(f"POTCAR ENMAX invalid for {potential}")
    digest = hashlib.sha256(content).hexdigest()
    return {
        "element": element,
        "potential": potential,
        "source_relative": source.relative_to(root).as_posix(),
        "source_sha256": digest,
        "content_sha256": digest,
        "title": title_match.group(1).strip(),
        "enmax_ev": enmax,
    }


def _strict_json(data: bytes, *, role: str) -> dict[str, object]:
    try:
        value = json.loads(
            data.decode("utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"nonfinite JSON token {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid {role}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{role} must be a JSON object")
    return value


def _validated_features(
    *, path: Path, manifest_path: Path, protocol: str, required: Sequence[str], role: str
) -> pd.DataFrame:
    data = Path(path).read_bytes()
    manifest = _strict_json(Path(manifest_path).read_bytes(), role=f"{role} manifest")
    if manifest.get("protocol") != protocol:
        raise ValueError(f"{role} protocol differs")
    if manifest.get("endpoint_artifacts_opened") is not False:
        raise ValueError(f"{role} is not endpoint-isolated")
    if role == "nested" and manifest.get("labels_opened") is not False:
        raise ValueError("nested selection is not label-free")
    outputs = manifest.get("outputs_sha256")
    if not isinstance(outputs, Mapping) or outputs.get(Path(path).name) != hashlib.sha256(data).hexdigest():
        raise ValueError(f"{role} feature hash differs from its manifest")
    table = pd.read_parquet(io.BytesIO(data))
    missing = set(required) - set(table.columns)
    if missing:
        raise ValueError(f"{role} features lack columns: {sorted(missing)}")
    table = table.loc[:, list(required)].copy()
    if table["sid"].isna().any() or table["sid"].astype(str).duplicated().any():
        raise ValueError(f"{role} sid values must be unique")
    table["sid"] = table["sid"].astype(str)
    return table.sort_values("sid", kind="stable", ignore_index=True)


def _parse_rk(rk: str) -> dict[str, int]:
    tokens = str(rk).split("|")
    parsed: dict[str, int] = {}
    for token in tokens:
        match = _RK_TOKEN.fullmatch(token)
        if match is None or match.group(1) in parsed:
            raise ValueError(f"invalid reduced composition: {rk}")
        parsed[match.group(1)] = int(match.group(2))
    if not parsed:
        raise ValueError(f"invalid reduced composition: {rk}")
    return parsed


def _assignment(
    left: pd.DataFrame,
    right: pd.DataFrame,
    cost: Callable[[pd.Series, pd.Series, int], float],
) -> list[tuple[int, int]]:
    if left.empty or right.empty:
        return []
    left = left.sort_values("sid", kind="stable")
    right = right.sort_values("sid", kind="stable")
    matrix = np.empty((len(left), len(right)), dtype=float)
    for i, (_, a) in enumerate(left.iterrows()):
        for j, (_, b) in enumerate(right.iterrows()):
            matrix[i, j] = cost(a, b, j)
    rows, columns = linear_sum_assignment(matrix)
    return [(int(left.index[i]), int(right.index[j])) for i, j in zip(rows, columns, strict=True)]


def _chemistry_cost(a: pd.Series, b: pd.Series, control_rank: int) -> float:
    ca, cb = _parse_rk(str(a["rk"])), _parse_rk(str(b["rk"]))
    ea, eb = set(ca), set(cb)
    symmetric_difference = len(ea ^ eb)
    element_count_difference = abs(len(ea) - len(eb))
    ta, tb = float(sum(ca.values())), float(sum(cb.values()))
    stoichiometric_l1 = sum(
        abs(ca.get(element, 0) / ta - cb.get(element, 0) / tb)
        for element in ea | eb
    )
    gap_difference = abs(float(a["m5_gap_ev_per_atom"]) - float(b["m5_gap_ev_per_atom"]))
    return (
        symmetric_difference * 1.0e12
        + element_count_difference * 1.0e10
        + round(stoichiometric_l1 * 1.0e6) * 1.0e3
        + gap_difference * 10.0
        + control_rank * 1.0e-6
    )


def select_blinded_pairs(nested: pd.DataFrame, acsc: pd.DataFrame) -> pd.DataFrame:
    """Select controls without DFT fields, outcomes, labels, or replacement."""

    nested_required = {
        "sid", "rk", "m5_gap_ev_per_atom", "m5_phsc_chsc_decision",
        "nested_three_scale_confirmed",
    }
    acsc_required = {"sid", "natoms", "acsc_status"}
    if not nested_required.issubset(nested.columns) or not acsc_required.issubset(acsc.columns):
        raise ValueError("pairing inputs lack required label-free columns")
    if nested["sid"].astype(str).duplicated().any() or acsc["sid"].astype(str).duplicated().any():
        raise ValueError("pairing sid values must be unique")
    joined = nested.loc[:, sorted(nested_required)].copy()
    joined["sid"] = joined["sid"].astype(str)
    acsc_small = acsc.loc[:, sorted(acsc_required)].copy()
    acsc_small["sid"] = acsc_small["sid"].astype(str)
    joined = joined.merge(acsc_small, on="sid", how="left", validate="one_to_one")
    joined["natoms"] = pd.to_numeric(joined["natoms"], errors="coerce")
    joined["m5_gap_ev_per_atom"] = pd.to_numeric(joined["m5_gap_ev_per_atom"], errors="coerce")

    treatments = joined.loc[
        joined["nested_three_scale_confirmed"].eq(True)
        & joined["m5_phsc_chsc_decision"].eq("KEEP")
    ].copy()
    controls = joined.loc[
        joined["m5_phsc_chsc_decision"].eq("KEEP")
        & joined["acsc_status"].eq("resolved_nonnegative")
    ].copy()
    if treatments.empty or len(controls) < len(treatments):
        raise ValueError("insufficient treatment or eligible control rows")
    for name, table in (("treatment", treatments), ("control", controls)):
        if (~np.isfinite(table["natoms"].to_numpy(float)) | (table["natoms"].to_numpy(float) <= 0)).any():
            raise ValueError(f"{name} atom counts must be finite and positive")
        if (~np.isfinite(table["m5_gap_ev_per_atom"].to_numpy(float))).any():
            raise ValueError(f"{name} M5 gaps must be finite")
    treatments["natoms"] = treatments["natoms"].astype(int)
    controls["natoms"] = controls["natoms"].astype(int)
    treatments = treatments.sort_values("sid", kind="stable")
    controls = controls.sort_values("sid", kind="stable")

    chosen: list[tuple[int, int, str]] = []
    used_treatments: set[int] = set()
    used_controls: set[int] = set()

    for (rk, natoms), group in treatments.groupby(["rk", "natoms"], sort=True):
        candidates = controls.loc[(controls["rk"] == rk) & (controls["natoms"] == natoms)]
        pairs = _assignment(
            group, candidates,
            lambda a, b, j: abs(float(a["m5_gap_ev_per_atom"]) - float(b["m5_gap_ev_per_atom"])) * 1.0e6 + j * 1.0e-3,
        )
        chosen.extend((i, j, "same_rk_same_natoms") for i, j in pairs)
        used_treatments.update(i for i, _ in pairs)
        used_controls.update(j for _, j in pairs)

    remaining_treatments = treatments.loc[~treatments.index.isin(used_treatments)]
    for rk, group in remaining_treatments.groupby("rk", sort=True):
        candidates = controls.loc[(controls["rk"] == rk) & ~controls.index.isin(used_controls)]
        pairs = _assignment(
            group, candidates,
            lambda a, b, j: abs(int(a["natoms"]) - int(b["natoms"])) * 1.0e9
            + abs(float(a["m5_gap_ev_per_atom"]) - float(b["m5_gap_ev_per_atom"])) * 1.0e6
            + j * 1.0e-3,
        )
        chosen.extend((i, j, "same_rk_diff_natoms") for i, j in pairs)
        used_treatments.update(i for i, _ in pairs)
        used_controls.update(j for _, j in pairs)

    remaining_treatments = treatments.loc[~treatments.index.isin(used_treatments)]
    for natoms, group in remaining_treatments.groupby("natoms", sort=True):
        candidates = controls.loc[(controls["natoms"] == natoms) & ~controls.index.isin(used_controls)]
        pairs = _assignment(group, candidates, _chemistry_cost)
        chosen.extend((i, j, "fallback_same_natoms_chemistry") for i, j in pairs)
        used_treatments.update(i for i, _ in pairs)
        used_controls.update(j for _, j in pairs)

    remaining_treatments = treatments.loc[~treatments.index.isin(used_treatments)]
    remaining_controls = controls.loc[~controls.index.isin(used_controls)]
    pairs = _assignment(
        remaining_treatments,
        remaining_controls,
        lambda a, b, j: abs(int(a["natoms"]) - int(b["natoms"])) * 1.0e15
        + _chemistry_cost(a, b, j),
    )
    chosen.extend((i, j, "fallback_diff_natoms_chemistry") for i, j in pairs)
    used_treatments.update(i for i, _ in pairs)
    used_controls.update(j for _, j in pairs)

    if len(chosen) != len(treatments) or len(used_controls) != len(treatments):
        raise RuntimeError("deterministic matching did not cover treatments one-to-one")
    rows: list[dict[str, object]] = []
    for treatment_index, control_index, tier in chosen:
        treatment = treatments.loc[treatment_index]
        control = controls.loc[control_index]
        rows.append(
            {
                "treatment_sid": str(treatment["sid"]),
                "control_sid": str(control["sid"]),
                "treatment_rk": str(treatment["rk"]),
                "control_rk": str(control["rk"]),
                "treatment_natoms": int(treatment["natoms"]),
                "control_natoms": int(control["natoms"]),
                "treatment_m5_gap_ev_per_atom": float(treatment["m5_gap_ev_per_atom"]),
                "control_m5_gap_ev_per_atom": float(control["m5_gap_ev_per_atom"]),
                "match_tier": tier,
                "same_rk": bool(treatment["rk"] == control["rk"]),
            }
        )
    result = pd.DataFrame(rows).sort_values("treatment_sid", kind="stable", ignore_index=True)
    result.insert(0, "pair_id", [f"pair-{index:04d}" for index in range(len(result))])
    return result


def opaque_task_id(*, pair_index: int, role: str, sid: str) -> str:
    if role not in {"treatment", "control"} or pair_index < 0 or not sid:
        raise ValueError("invalid opaque task identity input")
    payload = f"{PROTOCOL}|{pair_index}|{role}|{sid}".encode("utf-8")
    return "task-" + hashlib.sha256(payload).hexdigest()[:16]


def _archive_sids(path: Path) -> list[str]:
    try:
        context = zipfile.ZipFile(path)
    except (OSError, zipfile.BadZipFile) as exc:
        raise ValueError("invalid geometry-only archive") from exc
    with context as archive:
        sids: list[str] = []
        for info in archive.infolist():
            member = PurePosixPath(info.filename)
            if info.is_dir() or member.parent != PurePosixPath(".") or member.suffix != ".extxyz":
                raise ValueError("geometry-only archive contains an unexpected member")
            if _ARCHIVE_SID.fullmatch(member.stem) is None:
                raise ValueError("geometry-only archive contains an invalid sid")
            sids.append(member.stem)
    if not sids or len(sids) != len(set(sids)):
        raise ValueError("geometry-only archive sid coverage is invalid")
    return sorted(sids)


def build_paired_dft_queue(
    *,
    nested_features_path: Path,
    nested_manifest_path: Path,
    acsc_features_path: Path,
    acsc_manifest_path: Path,
    frames_zip_path: Path,
    geometry_manifest_path: Path,
    potcar_root: Path,
    output_dir: Path,
    potential_map: Mapping[str, str] | None = None,
    geometry_loader: Callable[..., tuple[list[str], list[Atoms]]] = load_geometry_only_archive,
    expected_pair_count: int = FORMAL_EXPECTED_PAIR_COUNT,
    expected_tier_counts: Mapping[str, int] | None = FORMAL_EXPECTED_TIER_COUNTS,
) -> dict[str, object]:
    """Build the sealed queue before any DFT endpoint is opened."""

    target = Path(output_dir).resolve()
    if os.path.lexists(target):
        raise FileExistsError(f"refusing to overwrite existing output: {target}")
    paths = {
        "nested_features": Path(nested_features_path).resolve(),
        "nested_manifest": Path(nested_manifest_path).resolve(),
        "acsc_features": Path(acsc_features_path).resolve(),
        "acsc_manifest": Path(acsc_manifest_path).resolve(),
        "geometry_only_frames": Path(frames_zip_path).resolve(),
        "geometry_manifest": Path(geometry_manifest_path).resolve(),
    }
    for role, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{role} is not a file: {path}")
    root = Path(potcar_root).resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    input_hashes = {role: _sha256_file(path) for role, path in paths.items()}
    production = potential_map is None
    if production and input_hashes != dict(FROZEN_FORMAL_INPUT_SHA256):
        raise ValueError("formal paired DFT queue inputs differ from frozen identities")

    nested = _validated_features(
        path=paths["nested_features"], manifest_path=paths["nested_manifest"],
        protocol=NESTED_PROTOCOL,
        required=("sid", "rk", "m5_gap_ev_per_atom", "m5_phsc_chsc_decision", "nested_three_scale_confirmed"),
        role="nested",
    )
    acsc = _validated_features(
        path=paths["acsc_features"], manifest_path=paths["acsc_manifest"],
        protocol=ACSC_PROTOCOL, required=("sid", "natoms", "acsc_status"), role="acsc",
    )
    pairs = select_blinded_pairs(nested, acsc)
    tier_counts = {str(key): int(value) for key, value in pairs["match_tier"].value_counts().sort_index().items()}
    if len(pairs) != expected_pair_count:
        raise ValueError(f"paired queue expected {expected_pair_count} pairs, observed {len(pairs)}")
    if expected_tier_counts is not None and tier_counts != dict(expected_tier_counts):
        raise ValueError(f"paired queue tier counts differ: {tier_counts}")

    selected_sids = sorted(set(pairs["treatment_sid"]) | set(pairs["control_sid"]))
    loader_sids = _archive_sids(paths["geometry_only_frames"]) if geometry_loader is load_geometry_only_archive else selected_sids
    loaded_sids, structures = geometry_loader(
        archive_path=paths["geometry_only_frames"],
        manifest_path=paths["geometry_manifest"],
        expected_sids=loader_sids,
    )
    structure_by_sid = dict(zip(loaded_sids, structures, strict=True))
    missing = set(selected_sids) - set(structure_by_sid)
    if missing:
        raise ValueError(f"paired sids lack sanitized geometries: {sorted(missing)}")
    selected_structures = {sid: structure_by_sid[sid] for sid in selected_sids}
    for sid, atoms in selected_structures.items():
        expected_natoms = pairs.loc[
            pairs["treatment_sid"].eq(sid), "treatment_natoms"
        ].tolist() + pairs.loc[pairs["control_sid"].eq(sid), "control_natoms"].tolist()
        if expected_natoms != [len(atoms)]:
            raise ValueError(f"paired atom count differs from geometry for {sid}")

    all_elements = sorted({element for atoms in selected_structures.values() for element in atoms.get_chemical_symbols()})
    selected_map = dict(FROZEN_POTENTIAL_MAP if production else potential_map)
    if set(selected_map) != set(all_elements):
        raise ValueError(f"potential map must exactly cover paired elements: {all_elements}")
    potcars = {
        element: _paired_potcar_record(root, element, selected_map[element])
        for element in all_elements
    }
    global_encut = int(math.ceil(1.3 * max(float(record["enmax_ev"]) for record in potcars.values())))

    run_protocol: dict[str, object] = {
        "protocol": PROTOCOL,
        "functional": "PBE",
        "paw_family": "local GGA/PBE PAW set identified by hashes",
        "encut_rule": "ceil(1.3 * global maximum POTCAR ENMAX)",
        "encut_ev": global_encut,
        "kpoints": {"KSPACING_inverse_angstrom": 0.22, "KGAMMA": True, "KPOINTS_file": False},
        "spin": {"ISPIN": 2, "MAGMOM": "1.0 per atom for every task"},
        "static_stage": {"NSW": 0, "IBRION": -1, "ISIF": 2, "EDIFF": 1.0e-6},
        "relax_stage": {"NSW": 200, "IBRION": 2, "ISIF": 3, "EDIFF": 1.0e-6, "EDIFFG_eV_per_A": -0.03, "POTIM": 0.30},
        "attempt_policy": {
            "stages": ["static_x0", "full_cell_relax"],
            "maximum_attempts_per_stage": 1,
            "timeout_hours_per_stage": 24,
            "failures_and_timeouts_retained": True,
        },
        "endpoint_definitions_frozen_before_DFT": {
            "severe_energy_drop_eV_per_atom": 0.10,
            "severe_initial_fmax_eV_per_A": 1.0,
            "severe_max_displacement_angstrom": 0.5,
            "severe_abs_log_volume_ratio": 0.10,
            "nonconvergence_is_severe": True,
            "same_rk_relaxed_energy_primary": True,
            "direction": "treatment_minus_control; positive severity/energy supports ACSC rejection",
        },
        "potentials": potcars,
        "licensed_potcar_contents_included": False,
    }

    private_rows: list[dict[str, object]] = []
    blind_rows: list[dict[str, object]] = []
    payloads: list[tuple[str, bytes]] = []
    for pair_index, pair in pairs.iterrows():
        for role in ("treatment", "control"):
            sid = str(pair[f"{role}_sid"])
            atoms = selected_structures[sid]
            task_id = opaque_task_id(pair_index=int(pair_index), role=role, sid=sid)
            formula = atoms.get_chemical_formula(mode="hill")
            prefix = f"tasks/{task_id}"
            private_rows.append(
                {
                    "pair_id": str(pair["pair_id"]), "task_id": task_id, "role": role,
                    "sid": sid, "rk": str(pair[f"{role}_rk"]), "formula": formula,
                    "natoms": len(atoms), "m5_gap_ev_per_atom": float(pair[f"{role}_m5_gap_ev_per_atom"]),
                    "match_tier": str(pair["match_tier"]), "same_rk": bool(pair["same_rk"]),
                }
            )
            blind_rows.append(
                {
                    "task_id": task_id, "formula": formula, "natoms": len(atoms),
                    "task_available": True, "task_prefix": prefix, "encut_ev": global_encut,
                }
            )
            species_order = sorted(set(atoms.get_chemical_symbols()))
            spec = {
                "task_id": task_id,
                "sequence": [potcars[element] for element in species_order],
                "materialization": "read or decompress each source_relative and concatenate in sequence order",
                "licensed_contents_included": False,
            }
            task = {
                "protocol": PROTOCOL, "task_id": task_id, "formula": formula,
                "natoms": len(atoms), "task_available": True,
                "stages": ["static_x0", "full_cell_relax"],
                "expected_endpoint_statuses": ["converged", "failed", "timeout"],
                "blinded_role_and_sid": True,
            }
            payloads.extend(
                [
                    (f"{prefix}/TASK.json", _json_bytes(task)),
                    (f"{prefix}/POSCAR.x0", _poscar_bytes(atoms)),
                    (f"{prefix}/INCAR.static", _incar(sid=task_id, natoms=len(atoms), encut=global_encut, relax=False)),
                    (f"{prefix}/INCAR.relax", _incar(sid=task_id, natoms=len(atoms), encut=global_encut, relax=True)),
                    (f"{prefix}/POTCAR.spec.json", _json_bytes(spec)),
                ]
            )
    private = pd.DataFrame(private_rows).sort_values(["pair_id", "role"], kind="stable", ignore_index=True)
    blind = pd.DataFrame(blind_rows).sort_values("task_id", kind="stable", ignore_index=True)
    blind.insert(0, "task_index", np.arange(len(blind), dtype=int))
    if len(private) != 2 * len(pairs) or len(blind) != 2 * len(pairs) or not blind["task_id"].is_unique:
        raise RuntimeError("paired task accounting differs")
    if {"sid", "role", "pair_id"} & set(blind.columns):
        raise RuntimeError("blinded queue leaks private mapping fields")
    payloads.sort(key=lambda item: item[0])

    repository_root = Path(__file__).resolve().parents[1]
    source_paths = {
        "src/next13d_acsc_dft_pairs.py": Path(__file__).resolve(),
        "src/next12_dft_queue.py": repository_root / "src/next12_dft_queue.py",
        "src/next11_geometry_only_frames.py": repository_root / "src/next11_geometry_only_frames.py",
    }
    source_hashes = {relative: _sha256_file(path) for relative, path in source_paths.items()}
    manifest: dict[str, object] = {
        "protocol": PROTOCOL,
        "labels_opened": False,
        "dft_endpoints_opened": False,
        "thresholds_refit": False,
        "selection_uses_only_label_free_fields": True,
        "control_replacement": False,
        "executor_blinded_to_sid_and_role": True,
        "private_mapping_must_be_withheld_from_executor": True,
        "all_tasks_queued": True,
        "licensed_potcar_contents_included": False,
        "counts": {
            "pairs": len(pairs), "tasks": len(blind), "treatments": len(pairs),
            "controls": len(pairs), "same_rk_pairs": int(pairs["same_rk"].sum()),
            "total_atoms": int(blind["natoms"].sum()), "match_tiers": tier_counts,
        },
        "selection": {
            "treatment": "nested_three_scale_confirmed AND old M5+PHSC+CHSC KEEP",
            "control": "old M5+PHSC+CHSC KEEP AND ACSC resolved_nonnegative",
            "hierarchy": list(FORMAL_EXPECTED_TIER_COUNTS) + ["fallback_diff_natoms_chemistry"],
            "DFT_fields_available_to_matcher": [],
        },
        "run_protocol": run_protocol,
        "inputs_sha256": {role: {"path": str(paths[role]), "sha256": digest} for role, digest in input_hashes.items()},
        "executed_source_sha256": source_hashes,
        "production_protocol_eligible": bool(production),
        "scientific_improvement_claim": False,
        "vasp_execution_completed": False,
        "known_limitations": [
            "The paired queue contains no DFT results.",
            "POTCAR contents are licensed and deliberately absent.",
            "The private mapping must be withheld from the DFT executor until all endpoints are sealed.",
            "Independent DFT is required before any superiority claim.",
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
        blind_path = staging / BLINDED_QUEUE_NAME
        private_path = staging / PRIVATE_PAIRS_NAME
        tasks_path = staging / TASKS_NAME
        protocol_path = staging / RUN_PROTOCOL_NAME
        blind.to_parquet(blind_path, index=False)
        private.to_parquet(private_path, index=False)
        with zipfile.ZipFile(tasks_path, "x") as archive:
            for member, payload in payloads:
                archive.writestr(_zip_info(member), payload)
        protocol_path.write_bytes(_json_bytes(run_protocol))
        manifest["outputs_sha256"] = {
            BLINDED_QUEUE_NAME: _sha256_file(blind_path),
            PRIVATE_PAIRS_NAME: _sha256_file(private_path),
            TASKS_NAME: _sha256_file(tasks_path),
            RUN_PROTOCOL_NAME: _sha256_file(protocol_path),
        }
        (staging / MANIFEST_NAME).write_bytes(_json_bytes(manifest))
        verify_unchanged()
        _atomic_publish_directory_no_replace(staging, target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nested-features", required=True, type=Path)
    parser.add_argument("--nested-manifest", required=True, type=Path)
    parser.add_argument("--acsc-features", required=True, type=Path)
    parser.add_argument("--acsc-manifest", required=True, type=Path)
    parser.add_argument("--frames-zip", required=True, type=Path)
    parser.add_argument("--geometry-manifest", required=True, type=Path)
    parser.add_argument("--potcar-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    arguments = parser.parse_args(argv)
    build_paired_dft_queue(
        nested_features_path=arguments.nested_features,
        nested_manifest_path=arguments.nested_manifest,
        acsc_features_path=arguments.acsc_features,
        acsc_manifest_path=arguments.acsc_manifest,
        frames_zip_path=arguments.frames_zip,
        geometry_manifest_path=arguments.geometry_manifest,
        potcar_root=arguments.potcar_root,
        output_dir=arguments.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
