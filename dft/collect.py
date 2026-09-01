#!/usr/bin/env python3
"""Extract raw results from finished VASP runs into one JSON per package.

Runs on the cluster: standard library only, no numpy, no pymatgen. It reads what VASP
actually wrote and makes no scientific decision — fitting, symmetry analysis and the
pre-registered comparisons all live in `analyze.py`, which runs afterwards.

    python3 collect.py                       # every package found beside this file
    python3 collect.py --package E4_design   # one
    python3 collect.py --package E4_design --stage-b

Writes `<package>/collected.json` and `<package>/collected.csv`, plus a short status line
per package. A task that did not finish is recorded with its reason rather than dropped,
so a missing number is always traceable.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACKAGES = ("E1_rho_curve", "E1b_paw_control", "E2_ordering", "E3_crosscheck",
            "E4_design")

RE_TOTEN = re.compile(r"free\s+energy\s+TOTEN\s*=\s*(-?\d+\.\d+)")
RE_SIGMA0 = re.compile(r"energy\(sigma->0\)\s*=\s*(-?\d+\.\d+)")
RE_VOLUME = re.compile(r"volume of cell\s*:\s*(\d+\.\d+)")
RE_TIME = re.compile(r"Elapsed time \(sec\):\s*(\d+\.\d+)")
RE_CORES = re.compile(r"running\s+(?:on\s+)?(\d+)\s+(?:total\s+)?(?:mpi-ranks|cores|nodes)")
RE_NBANDS = re.compile(r"NBANDS=\s*(\d+)")
RE_IRRK = re.compile(r"irreducible k-points:\s*(\d+)")
# `energy(sigma->0)` is printed for every electronic iteration, so its first value is the
# opening SCF guess, hundreds of eV above the ground state, and is not the energy of the
# initial geometry. The converged energy of an ionic step is the one that follows this
# header; the F=/E0= summary line lives in OSZICAR, not here.
IONIC_HEADER = "FREE ENERGIE OF THE ION-ELECTRON SYSTEM"
CONVERGED_MARK = "reached required accuracy"


def read_outcar(path: Path) -> dict:
    """Pull the per-ionic-step energies and the run's own convergence verdict."""
    toten: list[float] = []
    sigma0: list[float] = []
    ionic: list[float] = []
    in_ionic_block = False
    volumes: list[float] = []
    forces: list[float] = []
    info: dict = {"converged_flag": False, "elapsed_s": None, "cores": None,
                  "nbands": None, "irreducible_kpoints": None,
                  "warnings": [], "close_ions_warning": False}
    with open(path, "r", errors="replace") as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        m = RE_TOTEN.search(line)
        if m:
            toten.append(float(m.group(1)))
        if IONIC_HEADER in line:
            in_ionic_block = True
        m = RE_SIGMA0.search(line)
        if m:
            sigma0.append(float(m.group(1)))
            if in_ionic_block:
                ionic.append(float(m.group(1)))
                in_ionic_block = False
        m = RE_VOLUME.search(line)
        if m:
            volumes.append(float(m.group(1)))
        if CONVERGED_MARK in line:
            info["converged_flag"] = True
        if "WARNING" in line or "W A R N I N G" in line:
            w = line.strip()
            if w not in info["warnings"] and len(info["warnings"]) < 20:
                info["warnings"].append(w)
        # VASP's own verdict on overlapping augmentation spheres, worth keeping because
        # E1 compresses on purpose and the compressed end must stay auditable
        if "distance between some ions is very small" in line.lower():
            info["close_ions_warning"] = True
        m = RE_TIME.search(line)
        if m:
            info["elapsed_s"] = float(m.group(1))
        m = RE_CORES.search(line)
        if m and info["cores"] is None:
            info["cores"] = int(m.group(1))
        m = RE_NBANDS.search(line)
        if m and info["nbands"] is None:
            info["nbands"] = int(m.group(1))
        m = RE_IRRK.search(line)
        if m and info["irreducible_kpoints"] is None:
            info["irreducible_kpoints"] = int(m.group(1))
        if "TOTAL-FORCE" in line:
            j = i + 2
            worst = 0.0
            while j < len(lines) and not lines[j].strip().startswith("---"):
                parts = lines[j].split()
                if len(parts) >= 6:
                    try:
                        fx, fy, fz = (float(x) for x in parts[3:6])
                        worst = max(worst, (fx * fx + fy * fy + fz * fz) ** 0.5)
                    except ValueError:
                        pass
                j += 1
            forces.append(worst)
            i = j
            continue
        i += 1
    info.update({
        "n_electronic_steps": len(sigma0),
        "n_ionic_steps": len(ionic),
        # the converged energy of the first and last ionic step; the relaxation energy is
        # the difference between them, with electronic convergence excluded from both
        "energy_first_ev": ionic[0] if ionic else None,
        "energy_last_ev": ionic[-1] if ionic else (sigma0[-1] if sigma0 else None),
        "energy_scf_last_ev": sigma0[-1] if sigma0 else None,
        "ionic_energies_ev": ionic,
        "toten_last_ev": toten[-1] if toten else None,
        "volume_first_a3": volumes[0] if volumes else None,
        "volume_last_a3": volumes[-1] if volumes else None,
        "fmax_last_ev_per_a": forces[-1] if forces else None,
    })
    return info


def read_poscar_head(path: Path) -> dict:
    """Cell vectors, species counts and volume, without numpy."""
    lines = path.read_text().splitlines()
    scale = float(lines[1].split()[0])
    vec = [[float(x) for x in lines[i].split()[:3]] for i in (2, 3, 4)]
    if scale < 0:
        det = abs(_det(vec))
        scale = (abs(scale) / det) ** (1 / 3)
    vec = [[c * scale for c in row] for row in vec]
    symbols = lines[5].split()
    counts = [int(x) for x in lines[6].split()]
    return {"lattice": vec, "volume_a3": abs(_det(vec)),
            "symbols": symbols, "counts": counts, "n_atoms": sum(counts)}


def _det(m) -> float:
    return (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))


def collect_task(task: Path) -> dict:
    meta = json.loads((task / "TASK.json").read_text())
    stages = json.loads((task / "stages.json").read_text())
    record = {k: v for k, v in meta.items() if k != "points"}
    record["task_dir"] = str(task)
    record["stage_results"] = {}
    record["status"] = "complete"
    record["status_reason"] = ""

    for s in stages:
        name = s["stage"]
        d = task / name
        entry = {"ran": d.exists()}
        code_file = d / "exit_code"
        if code_file.exists():
            entry["exit_code"] = code_file.read_text().strip()
        outcar = d / "OUTCAR"
        if outcar.exists() and outcar.stat().st_size:
            try:
                entry.update(read_outcar(outcar))
            except Exception as exc:
                entry["parse_error"] = str(exc)
        contcar = d / "CONTCAR"
        if contcar.exists() and contcar.stat().st_size:
            try:
                entry["final_cell"] = read_poscar_head(contcar)
                entry["contcar"] = contcar.read_text()
            except Exception as exc:
                entry["contcar_error"] = str(exc)
        record["stage_results"][name] = entry

        if not entry.get("ran"):
            record["status"] = "incomplete"
            record["status_reason"] = f"stage {name} never ran"
            break
        if entry.get("exit_code") not in (None, "0"):
            record["status"] = "failed"
            record["status_reason"] = f"stage {name} exited {entry.get('exit_code')}"
            break
        if entry.get("energy_last_ev") is None:
            record["status"] = "failed"
            record["status_reason"] = f"stage {name} produced no energy"
            break
        if name.startswith("relax") and not entry.get("converged_flag"):
            # a relaxation that hit NSW without meeting EDIFFG is usable but flagged
            record["status"] = "unconverged"
            record["status_reason"] = f"stage {name} hit the ionic step limit"

    # derived quantities that need no scientific judgement
    n = meta.get("n_atoms")
    for name, entry in record["stage_results"].items():
        if entry.get("energy_first_ev") is not None and n:
            entry["energy_first_ev_per_atom"] = entry["energy_first_ev"] / n
            entry["energy_last_ev_per_atom"] = entry["energy_last_ev"] / n
            entry["relax_release_ev_per_atom"] = (
                (entry["energy_first_ev"] - entry["energy_last_ev"]) / n)
    return record


def collect_package(pkg: Path, stage_b: bool = False) -> dict:
    root = pkg / "stage_b" if stage_b else pkg
    tasks_dir = root / "tasks"
    if not tasks_dir.exists():
        return {"package": pkg.name, "stage_b": stage_b, "error": "no tasks directory"}
    records = [collect_task(t) for t in sorted(tasks_dir.iterdir()) if t.is_dir()]
    counts: dict[str, int] = {}
    for r in records:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    n_close = sum(1 for r in records
                  for e in r["stage_results"].values() if e.get("close_ions_warning"))
    out = {
        "package": pkg.name,
        "stage": "B" if stage_b else "A",
        "n_stages_with_close_ion_warning": n_close,
        "protocol": records[0]["protocol"] if records else None,
        "n_tasks": len(records),
        "status_counts": counts,
        "records": records,
    }
    (root / "collected.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")

    # a flat table for quick inspection; the JSON keeps the structures
    flat_keys = ["task", "experiment", "status", "status_reason", "formula", "n_atoms",
                 "encut_ev", "n_kpoints"]
    extra = {"E1": ["source_id", "anion", "rho_c_parent", "radius_sum_a", "potentials"],
             "E1b": ["source_id", "anion", "rho_c_parent", "radius_sum_a", "potentials"],
             "E2": ["entry", "kind", "merge_class", "ordering_index",
                    "is_released_ordering"],
             "E3": ["parent", "kind", "variant"],
             "E4": ["candidate_id", "role", "pss", "uma_bulk_modulus_gpa",
                    "volume_factor"]}
    exp = records[0].get("experiment") if records else None
    flat_keys += [k for k in extra.get(exp, []) if any(k in r for r in records)]
    stage_names = []
    for r in records:
        for s in r["stage_results"]:
            if s not in stage_names:
                stage_names.append(s)
    with open(root / "collected.csv", "w", newline="") as f:
        w = csv.writer(f)
        header = list(flat_keys)
        for s in stage_names:
            header += [f"{s}.energy_ev", f"{s}.energy_ev_per_atom",
                       f"{s}.volume_a3", f"{s}.release_ev_per_atom",
                       f"{s}.n_ionic_steps", f"{s}.converged", f"{s}.elapsed_s",
                       f"{s}.close_ions_warning"]
        w.writerow(header)
        for r in records:
            row = [r.get(k, "") for k in flat_keys]
            for s in stage_names:
                e = r["stage_results"].get(s, {})
                cell = e.get("final_cell") or {}
                row += [e.get("energy_last_ev", ""), e.get("energy_last_ev_per_atom", ""),
                        cell.get("volume_a3", e.get("volume_last_a3", "")),
                        e.get("relax_release_ev_per_atom", ""),
                        e.get("n_ionic_steps", ""), e.get("converged_flag", ""),
                        e.get("elapsed_s", ""), e.get("close_ions_warning", "")]
            w.writerow(row)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(HERE))
    ap.add_argument("--package", action="append", default=None)
    ap.add_argument("--stage-b", action="store_true",
                    help="collect E4_design/stage_b instead of the stage-A tasks")
    a = ap.parse_args()
    root = Path(a.root)
    names = a.package or list(PACKAGES)

    total_cpu_s = 0.0
    for name in names:
        pkg = root / name
        if not pkg.exists():
            print(f"{name}: not present, skipped")
            continue
        out = collect_package(pkg, stage_b=a.stage_b)
        if "error" in out:
            print(f"{name}: {out['error']}")
            continue
        for r in out["records"]:
            for e in r["stage_results"].values():
                if e.get("elapsed_s") and e.get("cores"):
                    total_cpu_s += e["elapsed_s"] * e["cores"]
        note = (f", {out['n_stages_with_close_ion_warning']} stages where VASP flagged "
                f"very close ions") if out["n_stages_with_close_ion_warning"] else ""
        print(f"{name} stage {out['stage']}: {out['n_tasks']} tasks, "
              f"{out['status_counts']}{note}")
    if total_cpu_s:
        print(f"measured cost so far: {total_cpu_s / 3600:.0f} core-hours")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
