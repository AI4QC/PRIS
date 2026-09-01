"""Build a full-cohort, license-safe VASP PBE relaxation queue for NEXT12."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Mapping, Sequence
import zipfile

from ase import Atoms
from ase.io import write
import pandas as pd

from src.next12_pauling_controls import PROTOCOL as PAULING_PROTOCOL
from src.next12_prospective_gates import (
    PROTOCOL as GATE_PROTOCOL,
    _load_cohort,
    _sha256_file,
    _strict_json,
)
from src.next8_mattersim_committee_features import (
    _atomic_publish_directory_no_replace,
)


PROTOCOL = "2026-08-02-next12-full-cohort-vasp-pbe-queue-v1"
QUEUE_NAME = "dft_queue.parquet"
TASKS_NAME = "vasp_tasks.zip"
RUN_PROTOCOL_NAME = "RUN_PROTOCOL.json"
MANIFEST_NAME = "MANIFEST.json"
FROZEN_FORMAL_INPUT_SHA256 = {
    "cohort": "fc08be4f1b28dc82f4a26aeb49819b914ad8df7c7c7ee3887dea7a6c61095215",
    "geometry_only_frames": "3b392bdd38120dae579dc22b1b51e7c30bcbbed0e72c9b462c1bce16eda96959",
    "cohort_manifest": "8649853dcbb40a081183b671101fdf2933f30358ad1e5cb5b8694e8e451a846a",
    "gate_features": "43c23c56e0de017f291060faed0f249f6013bacba9bffcbceea030839d0bda43",
    "gate_manifest": "da37c43ba69e40a5c8d1e364ce32211c84091e94b33f46f7eedf2d96d140c98f",
    "pauling_features": "72014c90806ed74a0a50390a6c3698c903f432cf92edd658de5a62d45b8de756",
    "pauling_manifest": "e32494a5a1f4375841239d6a20ee3979aa3c7a2f8a37aa77d56d65c9bb674db9",
}
FROZEN_POTENTIAL_MAP: Mapping[str, str] = {
    "Ag": "Ag",
    "Cu": "Cu_pv",
    "K": "K_sv",
    "O": "O",
    "S": "S",
    "Se": "Se",
    "Sm": "Sm_3",
    "Sn": "Sn_d",
    "Sr": "Sr_sv",
    "Tb": "Tb_3",
    "Tl": "Tl_d",
    "Y": "Y_sv",
}
GATE_COLUMNS = (
    "sid",
    "m5_decision",
    "m5_phsc_decision",
    "composed_decision",
)
PAULING_COLUMNS = (
    "sid",
    "pauling_p2_decision",
    "pauling_p3_decision",
    "pauling_p4_decision",
    "pauling_p5_decision",
    "pauling_p2_p5_decision",
)
DECISIONS = {"KEEP", "REJECT", "ABSTAIN"}


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _zip_info(member: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(member, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    return info


def _json_bytes(value: Mapping[str, object]) -> bytes:
    return (json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def _validated_decision_table(
    *, data: bytes, manifest_data: bytes, protocol: str, name: str, columns: Sequence[str]
) -> pd.DataFrame:
    manifest = _strict_json(manifest_data, role=f"{name} manifest")
    if manifest.get("protocol") != protocol:
        raise ValueError(f"{name} protocol differs")
    if manifest.get("labels_opened") is not False or manifest.get(
        "endpoint_artifacts_opened"
    ) is not False:
        raise ValueError(f"{name} is not endpoint-isolated")
    if manifest.get("thresholds_refit") is not False:
        raise ValueError(f"{name} thresholds were refit")
    outputs = manifest.get("outputs_sha256")
    if not isinstance(outputs, Mapping) or outputs.get(name) != _sha256_bytes(data):
        raise ValueError(f"{name} hash differs from its manifest")
    table = pd.read_parquet(io.BytesIO(data))
    missing = set(columns) - set(table.columns)
    if missing:
        raise ValueError(f"{name} lacks columns: {sorted(missing)}")
    table = table.loc[:, list(columns)].copy()
    if table["sid"].isna().any() or table["sid"].astype(str).duplicated().any():
        raise ValueError(f"{name} sid values must be unique")
    table["sid"] = table["sid"].astype(str)
    for column in columns[1:]:
        if not set(table[column].astype(str)).issubset(DECISIONS):
            raise ValueError(f"{name} contains invalid decisions in {column}")
    return table.sort_values("sid", kind="stable", ignore_index=True)


def _potcar_source(root: Path, potential: str) -> Path:
    directory = root / potential
    for filename in ("POTCAR", "POTCAR.gz", "POTCAR.Z"):
        path = directory / filename
        if path.is_file():
            return path
    raise FileNotFoundError(f"POTCAR source missing for {potential}: {directory}")


def _potcar_content(path: Path) -> bytes:
    if path.name == "POTCAR":
        return path.read_bytes()
    if path.name == "POTCAR.gz":
        with gzip.open(path, "rb") as handle:
            return handle.read()
    if path.name == "POTCAR.Z":
        completed = subprocess.run(
            ["gzip", "-cd", str(path)],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"could not decompress {path}: {completed.stderr.decode(errors='replace')}"
            )
        return completed.stdout
    raise ValueError(f"unsupported POTCAR source: {path}")


def _potcar_record(root: Path, element: str, potential: str) -> dict[str, object]:
    source = _potcar_source(root, potential)
    compressed = source.read_bytes()
    content = _potcar_content(source)
    text = content.decode("latin-1", errors="replace")
    title_match = re.search(r"^\s*TITEL\s*=\s*(.+?)\s*$", text, flags=re.MULTILINE)
    enmax_match = re.search(r"\bENMAX\s*=\s*([0-9]+(?:\.[0-9]*)?)", text)
    if title_match is None or enmax_match is None:
        raise ValueError(f"POTCAR metadata missing for {potential}")
    enmax = float(enmax_match.group(1))
    if not math.isfinite(enmax) or enmax <= 0.0:
        raise ValueError(f"POTCAR ENMAX invalid for {potential}")
    return {
        "element": element,
        "potential": potential,
        "source_relative": source.relative_to(root).as_posix(),
        "source_sha256": _sha256_bytes(compressed),
        "content_sha256": _sha256_bytes(content),
        "title": title_match.group(1).strip(),
        "enmax_ev": enmax,
    }


def _poscar_bytes(atoms: Atoms) -> bytes:
    stream = io.StringIO()
    write(stream, atoms, format="vasp", direct=True, sort=True, vasp5=True)
    return stream.getvalue().encode("utf-8")


def _incar(*, sid: str, natoms: int, encut: int, relax: bool) -> bytes:
    common = [
        f"SYSTEM = {sid} {'relax' if relax else 'static_x0'}",
        "GGA = PE",
        f"ENCUT = {encut}",
        "PREC = Accurate",
        "EDIFF = 1E-6",
        "NELM = 200",
        "ALGO = Normal",
        "ISMEAR = 0",
        "SIGMA = 0.05",
        "ISPIN = 2",
        f"MAGMOM = {natoms}*1.0",
        "LASPH = .TRUE.",
        "LREAL = .FALSE.",
        "ADDGRID = .TRUE.",
        "KSPACING = 0.22",
        "KGAMMA = .TRUE.",
        "ISYM = 0",
        "LWAVE = .FALSE.",
        "LCHARG = .FALSE.",
    ]
    if relax:
        common.extend(
            [
                "NSW = 200",
                "IBRION = 2",
                "ISIF = 3",
                "EDIFFG = -0.03",
                "POTIM = 0.30",
            ]
        )
    else:
        common.extend(["NSW = 0", "IBRION = -1", "ISIF = 2"])
    return ("\n".join(common) + "\n").encode("utf-8")


def build_dft_queue(
    *,
    cohort_path: Path,
    frames_zip_path: Path,
    cohort_manifest_path: Path,
    gate_features_path: Path,
    gate_manifest_path: Path,
    pauling_features_path: Path,
    pauling_manifest_path: Path,
    potcar_root: Path,
    output_dir: Path,
    potential_map: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Freeze all prospective attempts; decisions never select queue membership."""

    target = Path(output_dir).resolve()
    if os.path.lexists(target):
        raise FileExistsError(f"refusing to overwrite existing output: {target}")
    paths = {
        "cohort": Path(cohort_path).resolve(),
        "geometry_only_frames": Path(frames_zip_path).resolve(),
        "cohort_manifest": Path(cohort_manifest_path).resolve(),
        "gate_features": Path(gate_features_path).resolve(),
        "gate_manifest": Path(gate_manifest_path).resolve(),
        "pauling_features": Path(pauling_features_path).resolve(),
        "pauling_manifest": Path(pauling_manifest_path).resolve(),
    }
    for role, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{role} is not a file: {path}")
    root = Path(potcar_root).resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    input_hashes = {role: _sha256_file(path) for role, path in paths.items()}
    cohort_data = paths["cohort"].read_bytes()
    archive_data = paths["geometry_only_frames"].read_bytes()
    cohort_table, generated_sids, structures, cohort_manifest = _load_cohort(
        cohort_data=cohort_data,
        archive_data=archive_data,
        manifest_data=paths["cohort_manifest"].read_bytes(),
    )
    gates = _validated_decision_table(
        data=paths["gate_features"].read_bytes(),
        manifest_data=paths["gate_manifest"].read_bytes(),
        protocol=GATE_PROTOCOL,
        name=paths["gate_features"].name,
        columns=GATE_COLUMNS,
    )
    pauling = _validated_decision_table(
        data=paths["pauling_features"].read_bytes(),
        manifest_data=paths["pauling_manifest"].read_bytes(),
        protocol=PAULING_PROTOCOL,
        name=paths["pauling_features"].name,
        columns=PAULING_COLUMNS,
    )
    cohort_sids = set(cohort_table["sid"].astype(str))
    if set(gates.sid) != cohort_sids or set(pauling.sid) != cohort_sids:
        raise ValueError("cohort, gate, and Pauling sid sets differ")
    production = potential_map is None
    selected_map = dict(FROZEN_POTENTIAL_MAP if production else potential_map)
    if production:
        if input_hashes != FROZEN_FORMAL_INPUT_SHA256:
            raise ValueError("formal DFT queue inputs differ from frozen identities")
        if cohort_manifest.get("production_protocol_eligible") is not True:
            raise ValueError("formal cohort is not production eligible")
    if not selected_map:
        raise ValueError("potential map must be nonempty")

    all_elements = sorted({symbol for atoms in structures for symbol in atoms.get_chemical_symbols()})
    if set(all_elements) != set(selected_map):
        raise ValueError(
            f"potential map must exactly cover cohort elements: {all_elements}"
        )
    potcars = {
        element: _potcar_record(root, element, selected_map[element])
        for element in all_elements
    }
    global_encut = int(math.ceil(1.3 * max(float(row["enmax_ev"]) for row in potcars.values())))
    structure_by_sid = dict(zip(generated_sids, structures, strict=True))
    decisions = gates.merge(pauling, on="sid", validate="one_to_one")
    decision_by_sid = {str(row["sid"]): row for row in decisions.to_dict("records")}

    run_protocol: dict[str, object] = {
        "protocol": PROTOCOL,
        "functional": "PBE",
        "paw_family": "local GGA/PBE PAW set identified by hashes",
        "encut_rule": "ceil(1.3 * global maximum POTCAR ENMAX)",
        "encut_ev": global_encut,
        "kpoints": {"KSPACING_inverse_angstrom": 0.22, "KGAMMA": True, "KPOINTS_file": False},
        "spin": {"ISPIN": 2, "MAGMOM": "1.0 per atom for every task"},
        "static_stage": {"NSW": 0, "IBRION": -1, "ISIF": 2, "EDIFF": 1e-6},
        "relax_stage": {
            "NSW": 200,
            "IBRION": 2,
            "ISIF": 3,
            "EDIFF": 1e-6,
            "EDIFFG_eV_per_A": -0.03,
            "POTIM": 0.30,
        },
        "attempt_policy": {
            "stages": ["static_x0", "full_cell_relax"],
            "maximum_attempts_per_stage": 1,
            "timeout_hours_per_stage": 24,
            "failures_and_timeouts_retained": True,
            "gate_based_task_selection": False,
        },
        "endpoint_definitions_frozen_before_DFT": {
            "near_min_eV_per_atom": 0.001,
            "valuable_eV_per_atom": 0.05,
            "high_energy_eV_per_atom": 0.20,
            "complete_composition_groups_only": True,
        },
        "potentials": potcars,
        "licensed_potcar_contents_included": False,
    }

    queue_rows: list[dict[str, object]] = []
    task_payloads: list[tuple[str, bytes]] = []
    for upstream in cohort_table.to_dict("records"):
        sid = str(upstream["sid"])
        prefix = f"tasks/{sid}"
        decision = decision_by_sid[sid]
        atoms = structure_by_sid.get(sid)
        task_available = atoms is not None
        queue_rows.append(
            {
                "attempt_index": int(upstream["attempt_index"]),
                "sid": sid,
                "formula": str(upstream["formula"]),
                "natoms": int(upstream["natoms"]),
                "generation_status": str(upstream["generation_status"]),
                "task_available": task_available,
                "task_prefix": prefix,
                "encut_ev": global_encut,
                "m5_decision": str(decision["m5_decision"]),
                "m5_phsc_decision": str(decision["m5_phsc_decision"]),
                "m5_phsc_chsc_decision": str(decision["composed_decision"]),
                **{
                    column: str(decision[column]) for column in PAULING_COLUMNS[1:]
                },
            }
        )
        task = {
            "protocol": PROTOCOL,
            "sid": sid,
            "formula": str(upstream["formula"]),
            "natoms": int(upstream["natoms"]),
            "task_available": task_available,
            "stages": ["static_x0", "full_cell_relax"] if task_available else [],
            "expected_endpoint_statuses": ["converged", "failed", "timeout"],
            "decisions_are_metadata_not_selection": True,
        }
        task_payloads.append((f"{prefix}/TASK.json", _json_bytes(task)))
        if atoms is None:
            continue
        species_order = sorted(set(atoms.get_chemical_symbols()))
        spec = {
            "sid": sid,
            "sequence": [potcars[element] for element in species_order],
            "materialization": "decompress each source_relative and concatenate in sequence order",
            "licensed_contents_included": False,
        }
        task_payloads.extend(
            (
                (f"{prefix}/POSCAR.x0", _poscar_bytes(atoms)),
                (f"{prefix}/INCAR.static", _incar(sid=sid, natoms=len(atoms), encut=global_encut, relax=False)),
                (f"{prefix}/INCAR.relax", _incar(sid=sid, natoms=len(atoms), encut=global_encut, relax=True)),
                (f"{prefix}/POTCAR.spec.json", _json_bytes(spec)),
            )
        )
    queue = pd.DataFrame(queue_rows).sort_values("sid", kind="stable", ignore_index=True)
    if len(queue) != len(cohort_table) or queue["sid"].duplicated().any():
        raise RuntimeError("DFT queue did not retain every unique attempt")
    task_payloads.sort(key=lambda item: item[0])

    repository_root = Path(__file__).resolve().parents[1]
    source_paths = {
        "src/next12_dft_queue.py": Path(__file__).resolve(),
        "src/next12_prospective_gates.py": repository_root / "src/next12_prospective_gates.py",
    }
    source_hashes = {relative: _sha256_file(path) for relative, path in source_paths.items()}
    manifest: dict[str, object] = {
        "protocol": PROTOCOL,
        "all_attempts_queued": True,
        "selection_by_gate": False,
        "licensed_potcar_contents_included": False,
        "run_protocol": run_protocol,
        "counts": {
            "attempts": len(queue),
            "tasks": len(queue),
            "total_atoms": int(queue["natoms"].sum()),
        },
        "inputs_sha256": {
            role: {"path": str(paths[role]), "sha256": digest}
            for role, digest in input_hashes.items()
        },
        "executed_source_sha256": source_hashes,
        "production_protocol_eligible": bool(production),
        "scientific_improvement_claim": False,
        "vasp_execution_completed": False,
        "known_limitations": [
            "No VASP executable or scheduler was available on the build host.",
            "POTCAR contents are licensed and deliberately absent from this artifact.",
            "DFT superiority cannot be claimed until every queued endpoint is accounted for.",
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
        queue_path = staging / QUEUE_NAME
        queue.to_parquet(queue_path, index=False)
        tasks_path = staging / TASKS_NAME
        with zipfile.ZipFile(tasks_path, "x") as archive:
            for member, payload in task_payloads:
                archive.writestr(_zip_info(member), payload)
        protocol_path = staging / RUN_PROTOCOL_NAME
        protocol_path.write_bytes(_json_bytes(run_protocol))
        manifest["outputs_sha256"] = {
            QUEUE_NAME: _sha256_file(queue_path),
            TASKS_NAME: _sha256_file(tasks_path),
            RUN_PROTOCOL_NAME: _sha256_file(protocol_path),
        }
        manifest_path = staging / MANIFEST_NAME
        manifest_path.write_bytes(_json_bytes(manifest))
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
    parser.add_argument("--gate-features", required=True, type=Path)
    parser.add_argument("--gate-manifest", required=True, type=Path)
    parser.add_argument("--pauling-features", required=True, type=Path)
    parser.add_argument("--pauling-manifest", required=True, type=Path)
    parser.add_argument("--potcar-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    arguments = parser.parse_args(argv)
    build_dft_queue(
        cohort_path=arguments.cohort,
        frames_zip_path=arguments.frames_zip,
        cohort_manifest_path=arguments.cohort_manifest,
        gate_features_path=arguments.gate_features,
        gate_manifest_path=arguments.gate_manifest,
        pauling_features_path=arguments.pauling_features,
        pauling_manifest_path=arguments.pauling_manifest,
        potcar_root=arguments.potcar_root,
        output_dir=arguments.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
