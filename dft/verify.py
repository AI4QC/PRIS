#!/usr/bin/env python3
"""Verify the E1-E4 VASP task packages before anything is submitted.

Everything here is checked against the files on disk rather than against the builder's
intentions, so a package that passes can be shipped to a cluster and run unattended.

    python dft/verify.py [--out dft] [--report dft/VERIFY_REPORT.md]

Checks, per package:
  structure   POSCAR parses, atom count and species order agree with TASK.json and with
              the POTCAR spec, and no two atoms sit implausibly close
  inputs      every INCAR a stage names exists, KPOINTS matches the KSPACING rule, ENCUT
              satisfies the frozen rule for the species actually present
  potcar      each spec entry resolves in the local library and its SHA256 still matches;
              no licensed POTCAR content is shipped inside the package
  manifest    every recorded file hash still matches, and no unrecorded file has appeared
  runnable    tasklist entries exist, run.sh and stages.json are consistent, and the
              submit script's array size equals the number of tasks
  cost        a k-point-weighted estimate of the campaign's size, reported not enforced
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path("<repo>/")
# The library the packages were built against. On a cluster, point --potcar-lib (or
# VASP_PP_PATH) at the local one: a hash mismatch then means the potentials genuinely
# differ, which changes the physics and is worth knowing before submitting.
POTCAR_LIB = Path(os.environ.get("VASP_PP_PATH") or "<path>")
KSPACING = 0.22
ENCUT_FLOOR = 520.0
ENCUT_ENMAX_FACTOR = 1.3
MIN_CONTACT_A = 0.5          # below this a cell is unusable even as deliberate damage
# PAW spheres overlap in ordinary dense solids (Ir-Ir sits at 0.91 of the radius sum), so
# overlap alone means nothing. These bound where the frozen core stops being trustworthy.
PAW_RATIO_ERROR = 0.40
PAW_RATIO_WARN = 0.60
PAW_RATIO_REPORT = 0.75
PACKAGES = ("E1_rho_curve", "E1b_paw_control", "E2_ordering", "E3_crosscheck",
            "E4_design")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Report:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.notes: list[str] = []

    def error(self, msg: str) -> None:
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def note(self, msg: str) -> None:
        self.notes.append(msg)


def parse_poscar(path: Path):
    """Return (lattice 3x3, species list per site, frac coords) without pymatgen."""
    lines = path.read_text().splitlines()
    scale = float(lines[1].split()[0])
    lat = np.array([[float(x) for x in lines[i].split()[:3]] for i in (2, 3, 4)])
    if scale < 0:                      # negative scale means "target volume"
        vol = abs(np.linalg.det(lat))
        scale = (abs(scale) / vol) ** (1 / 3)
    lat = lat * scale
    symbols = lines[5].split()
    counts = [int(x) for x in lines[6].split()]
    idx = 7
    mode = lines[idx].strip().lower()
    if mode.startswith("s"):           # selective dynamics
        idx += 1
        mode = lines[idx].strip().lower()
    idx += 1
    n = sum(counts)
    coords = np.array([[float(x) for x in lines[idx + i].split()[:3]] for i in range(n)])
    if mode.startswith("c") or mode.startswith("k"):
        coords = coords @ np.linalg.inv(lat)
    species = [s for s, c in zip(symbols, counts) for _ in range(c)]
    return lat, symbols, counts, species, coords


def min_distance(lat: np.ndarray, frac: np.ndarray, species: list[str] | None = None):
    """Shortest minimum-image distance, and the pair it belongs to when species are given."""
    n = len(frac)
    if n < 2:
        return (float("inf"), None) if species else float("inf")
    rng = (-2, -1, 0, 1, 2)
    shifts = np.array([[i, j, k] for i in rng for j in rng for k in rng])
    best = float("inf")
    pair = None
    cart = frac @ lat
    for s in shifts:
        off = s @ lat
        d = cart[:, None, :] - cart[None, :, :] + off
        dist = np.sqrt((d ** 2).sum(-1))
        if not s.any():
            np.fill_diagonal(dist, np.inf)
        idx = np.unravel_index(int(np.argmin(dist)), dist.shape)
        if float(dist[idx]) < best:
            best = float(dist[idx])
            pair = (int(idx[0]), int(idx[1]))
    if species is None:
        return best
    return best, (species[pair[0]], species[pair[1]]) if pair else None


def parse_kpoints(path: Path) -> tuple[int, int, int]:
    lines = path.read_text().splitlines()
    return tuple(int(x) for x in lines[3].split()[:3])


def parse_incar(path: Path) -> dict[str, str]:
    out = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip().upper()] = v.strip()
    return out


def expected_mesh(lat: np.ndarray) -> tuple[int, int, int]:
    recip = 2 * math.pi * np.linalg.inv(lat).T
    lengths = np.sqrt((recip ** 2).sum(axis=1))
    return tuple(max(1, int(math.ceil(x / KSPACING))) for x in lengths)


def verify_task(task: Path, rep: Report, stats: dict) -> None:
    tag = f"{task.parent.parent.name}/{task.name}"
    for required in ("TASK.json", "stages.json", "KPOINTS", "POTCAR.spec.json", "run.sh"):
        if not (task / required).exists():
            rep.error(f"{tag}: missing {required}")
            return
    meta = json.loads((task / "TASK.json").read_text())
    stages = json.loads((task / "stages.json").read_text())
    spec = json.loads((task / "POTCAR.spec.json").read_text())

    # --- inputs each stage names must exist ------------------------------------
    for s in stages:
        if not (task / f"INCAR.{s['incar']}").exists():
            rep.error(f"{tag}: stage {s['stage']} names INCAR.{s['incar']}, which is absent")
        src = s.get("from_contcar") or ""
        if not src and not (task / s["poscar"]).exists():
            rep.error(f"{tag}: stage {s['stage']} needs {s['poscar']}, which is absent")
        if src and src not in [t["stage"] for t in stages]:
            rep.error(f"{tag}: stage {s['stage']} chains from unknown stage {src}")
    if [s["stage"] for s in stages] != meta["stages"]:
        rep.error(f"{tag}: stages.json and TASK.json disagree on the stage list")

    # --- POTCAR spec ------------------------------------------------------------
    enmax = []
    rcore = {}
    for entry in spec["order"]:
        p = POTCAR_LIB / entry["library_relative"]
        if not p.exists():
            rep.error(f"{tag}: POTCAR {entry['library_relative']} not in the local library")
            continue
        if sha256_file(p) != entry["sha256"]:
            rep.error(f"{tag}: POTCAR {entry['library_relative']} hash differs from the spec")
        enmax.append(entry["enmax_ev"])
        if entry.get("rcore_a"):
            rcore[entry["element"]] = entry["rcore_a"]
    listed = [ln for ln in (task / "POTCAR.list").read_text().splitlines() if ln.strip()] \
        if (task / "POTCAR.list").exists() else None
    if listed is None:
        rep.error(f"{tag}: POTCAR.list is missing, so run.sh cannot build a POTCAR")
    elif listed != [e["library_relative"] for e in spec["order"]]:
        rep.error(f"{tag}: POTCAR.list disagrees with POTCAR.spec.json")
    tsv = (task / "stages.tsv")
    if not tsv.exists():
        rep.error(f"{tag}: stages.tsv is missing, so run.sh has no plan")
    else:
        parsed = [ln.split("\t") for ln in tsv.read_text().splitlines() if ln.strip()]
        want = [[s["stage"], s["incar"], s["poscar"], s.get("from_contcar") or "-"]
                for s in stages]
        if parsed != want:
            rep.error(f"{tag}: stages.tsv disagrees with stages.json")
    for stray in task.rglob("POTCAR"):
        rep.error(f"{tag}: licensed POTCAR content present at {stray.relative_to(task)}")

    # --- every POSCAR in the task ----------------------------------------------
    poscars = sorted(task.glob("POSCAR.*"))
    if not poscars:
        rep.error(f"{tag}: no POSCAR")
        return
    mesh = parse_kpoints(task / "KPOINTS")
    if list(mesh) != list(meta["kmesh"]):
        rep.error(f"{tag}: KPOINTS {mesh} disagrees with TASK.json {meta['kmesh']}")

    spec_symbols = [e["element"] for e in spec["order"]]
    for pos in poscars:
        try:
            lat, symbols, counts, species, frac = parse_poscar(pos)
        except Exception as exc:
            rep.error(f"{tag}: {pos.name} does not parse ({exc})")
            continue
        if symbols != spec_symbols:
            rep.error(f"{tag}: {pos.name} species order {symbols} "
                      f"does not match the POTCAR spec {spec_symbols}")
        if sum(counts) != meta["n_atoms"]:
            rep.error(f"{tag}: {pos.name} has {sum(counts)} atoms, TASK.json says "
                      f"{meta['n_atoms']}")
        d, closest = min_distance(lat, frac, species)
        if d < MIN_CONTACT_A:
            rep.error(f"{tag}: {pos.name} has atoms {d:.3f} A apart")
        elif d < 1.0:
            rep.warn(f"{tag}: {pos.name} shortest contact {d:.3f} A")
        stats["min_contact"].append(d)
        # two atoms closer than the sum of their PAW cutoff radii have overlapping
        # augmentation spheres; VASP still runs but the energy is not to be trusted
        if closest and all(rcore.get(e) for e in closest):
            limit = rcore[closest[0]] + rcore[closest[1]]
            q = d / limit
            if q < PAW_RATIO_REPORT:
                stats["paw_overlap"].append(
                    {"task": tag, "poscar": pos.name, "distance_a": round(d, 3),
                     "pair": list(closest), "rcore_sum_a": round(limit, 3),
                     "ratio": round(q, 3)})
            if q < PAW_RATIO_ERROR:
                rep.error(f"{tag}: {pos.name} has {closest[0]}-{closest[1]} at "
                          f"{q:.2f} of the PAW radius sum, beyond any frozen-core validity")
            elif q < PAW_RATIO_WARN:
                rep.warn(f"{tag}: {pos.name} {closest[0]}-{closest[1]} at {q:.2f} of the "
                         f"PAW radius sum")

    # the mesh is taken from the first POSCAR, which must be the densest requirement
    lat0 = parse_poscar(poscars[0])[0]
    exp = expected_mesh(lat0)
    if list(exp) != list(mesh):
        rep.error(f"{tag}: KPOINTS {mesh} does not follow KSPACING={KSPACING} "
                  f"for the first POSCAR (expected {exp})")
    for pos in poscars[1:]:
        e = expected_mesh(parse_poscar(pos)[0])
        if any(a > b for a, b in zip(e, mesh)):
            rep.warn(f"{tag}: {pos.name} would want a denser mesh {e} than the fixed {mesh}")

    # --- INCAR ------------------------------------------------------------------
    want_encut = max(ENCUT_FLOOR, math.ceil(ENCUT_ENMAX_FACTOR * max(enmax))) if enmax else None
    for incar_path in sorted(task.glob("INCAR.*")):
        inc = parse_incar(incar_path)
        if want_encut is not None and float(inc.get("ENCUT", 0)) < want_encut:
            rep.error(f"{tag}: {incar_path.name} ENCUT {inc.get('ENCUT')} below the "
                      f"required {want_encut:.0f}")
        stage_name = incar_path.name.split(".", 1)[1]
        isym = inc.get("ISYM")
        if stage_name.startswith("relax") and isym != "0":
            rep.error(f"{tag}: {incar_path.name} relaxes with ISYM={isym}, which would "
                      f"freeze the symmetry the experiment measures")
        if not stage_name.startswith("relax") and isym != "2":
            rep.warn(f"{tag}: {incar_path.name} static stage has ISYM={isym}")
        if inc.get("ISPIN") == "2" and "MAGMOM" not in inc:
            rep.error(f"{tag}: {incar_path.name} sets ISPIN=2 without MAGMOM")
        if stage_name == "static" and inc.get("ISMEAR") == "-5" and np.prod(mesh) < 4:
            rep.error(f"{tag}: {incar_path.name} uses tetrahedron smearing with "
                      f"{int(np.prod(mesh))} k-points")

    stats["tasks"] += 1
    stats["vasp_runs"] += len(stages)
    stats["atoms"] += meta["n_atoms"]
    # crude but monotone cost proxy: irreducible-free k-points times atoms cubed
    stats["cost_proxy"] += float(np.prod(mesh)) * meta["n_atoms"] ** 3 * len(stages)


def verify_package(pkg: Path, rep: Report) -> dict:
    stats = {"tasks": 0, "vasp_runs": 0, "atoms": 0, "cost_proxy": 0.0,
             "min_contact": [], "paw_overlap": []}
    if not pkg.exists():
        rep.error(f"{pkg.name}: package directory missing")
        return stats

    manifest = json.loads((pkg / "MANIFEST.json").read_text())
    selection = json.loads((pkg / "selection.json").read_text())

    # --- manifest integrity ------------------------------------------------------
    on_disk = {str(p.relative_to(pkg)): p for p in pkg.rglob("*")
               if p.is_file() and p.name != "MANIFEST.json"}
    recorded = manifest["files_sha256"]
    for rel, digest in recorded.items():
        p = pkg / rel
        if not p.exists():
            rep.error(f"{pkg.name}: manifest lists {rel}, which is missing")
        elif sha256_file(p) != digest:
            rep.error(f"{pkg.name}: {rel} changed since the manifest was written")
    RESULT_FILES = {"collected.json", "collected.csv", "curves.json",
                    "ordering_energies.json", "bulk_moduli.json", "paw_shift.json",
                    "mattersim_reference.json"}
    extra = sorted(r for r in set(on_disk) - set(recorded)
                   if Path(r).name not in RESULT_FILES
                   and not r.startswith(("logs/", ".farm", "stage_b/")))
    for rel in extra:
        rep.warn(f"{pkg.name}: {rel} is present but not recorded in the manifest")

    # --- tasklist and submit script ---------------------------------------------
    tasklist = [ln for ln in (pkg / "tasklist.txt").read_text().splitlines() if ln.strip()]
    for rel in tasklist:
        if not (pkg / rel).is_dir():
            rep.error(f"{pkg.name}: tasklist entry {rel} is not a directory")
    submit = (pkg / "submit.slurm").read_text()
    m = re.search(r"--array=1-(\d+)%", submit)
    if not m:
        rep.error(f"{pkg.name}: submit.slurm has no array range")
    elif int(m.group(1)) != len(tasklist):
        rep.error(f"{pkg.name}: submit.slurm array is 1-{m.group(1)} but tasklist has "
                  f"{len(tasklist)} entries")
    if "VASP_PP_PATH" not in submit or "VASP_BIN" not in submit:
        rep.error(f"{pkg.name}: submit.slurm does not set VASP_BIN and VASP_PP_PATH")

    dirs = sorted(p for p in (pkg / "tasks").iterdir() if p.is_dir())
    if len(dirs) != len(tasklist):
        rep.error(f"{pkg.name}: {len(dirs)} task directories but {len(tasklist)} "
                  f"tasklist entries")
    for task in dirs:
        verify_task(task, rep, stats)

    if stats["tasks"] != manifest["n_tasks"]:
        rep.error(f"{pkg.name}: manifest says {manifest['n_tasks']} tasks, found "
                  f"{stats['tasks']}")
    if stats["vasp_runs"] != manifest["n_vasp_runs"]:
        rep.error(f"{pkg.name}: manifest says {manifest['n_vasp_runs']} VASP runs, found "
                  f"{stats['vasp_runs']}")
    rep.note(f"{pkg.name}: {stats['tasks']} tasks, {stats['vasp_runs']} VASP runs, "
             f"{stats['atoms']} atoms, rule: {selection['rule'][:80]}...")
    return stats


def _check_e1_grid(pkg: Path, sel: dict, rep: Report) -> None:
    """Every point must sit exactly where the frozen grid says, in every task."""
    grid = sel["rho_grid"]
    for task in sorted((pkg / "tasks").iterdir()):
        meta = json.loads((task / "TASK.json").read_text())
        lat0 = parse_poscar(task / "POSCAR.v00")[0]
        v0 = abs(np.linalg.det(lat0))
        for point in meta["points"]:
            lat = parse_poscar(task / f"POSCAR.v{point['index']:02d}")[0]
            ratio = (abs(np.linalg.det(lat)) / v0) ** (1 / 3)
            want = point["rho_target"] / meta["points"][0]["rho_target"]
            if abs(ratio - want) > 1e-5:
                rep.error(f"{pkg.name}/{task.name}: point {point['index']} scales by "
                          f"{ratio:.6f}, expected {want:.6f}")
        # the radius sum must reproduce the parent contact the law divides by
        if meta.get("radius_sum_a") and meta.get("rho_c_parent"):
            got = meta["radius_sum_a"] * meta["rho_c_parent"]
            if abs(got - meta["parent_min_contact_a"]) > 0.01:
                rep.error(f"{pkg.name}/{task.name}: radius sum times rho_c gives {got:.3f} A, "
                          f"but the parent contact is {meta['parent_min_contact_a']:.3f} A")


def check_experiment_specific(out: Path, rep: Report) -> None:
    """Checks that only make sense for one experiment."""
    # E1: the grid must actually sweep rho_c, and the scaling must be exactly uniform
    for pkg_name in ("E1_rho_curve", "E1b_paw_control"):
        sel = json.loads((out / pkg_name / "selection.json").read_text())
        _check_e1_grid(out / pkg_name, sel, rep)
    sel = json.loads((out / "E1_rho_curve" / "selection.json").read_text())
    grid = sel["rho_grid"]
    if len(set(grid)) != len(grid) or grid != sorted(grid):
        rep.error("E1: rho grid is not a strictly increasing set")
    anions = {s["anion"] for s in sel["structures"]}
    if len(anions) < 5:
        rep.warn(f"E1: only {len(anions)} anion classes represented")
    # E1b must replicate E1 cells exactly, differing only in the potentials
    b_sel = json.loads((out / "E1b_paw_control" / "selection.json").read_text())
    e1_ids = {s["source_id"] for s in sel["structures"]}
    for s_b in b_sel["structures"]:
        if s_b["source_id"] not in e1_ids:
            rep.error(f"E1b: {s_b['source_id']} is not one of the E1 compounds")
        base = next(x for x in sel["structures"] if x["source_id"] == s_b["source_id"])
        if abs(base["rho_c_parent"] - s_b["rho_c_parent"]) > 1e-9:
            rep.error(f"E1b: {s_b['source_id']} has a different parent rho_c than E1")
        if s_b["encut_ev"] <= base["encut_ev"]:
            rep.error(f"E1b: {s_b['source_id']} cutoff {s_b['encut_ev']} does not exceed "
                      f"E1's {base['encut_ev']}, so the potentials cannot be harder")
    for task in sorted((out / "E1b_paw_control" / "tasks").iterdir()):
        sid = json.loads((task / "TASK.json").read_text())["source_id"]
        twin = out / "E1_rho_curve" / "tasks" / f"E1-{sid}"
        for k in range(len(grid)):
            a = (task / f"POSCAR.v{k:02d}").read_text()
            b = (twin / f"POSCAR.v{k:02d}").read_text()
            if a != b:
                rep.error(f"E1b/{task.name}: POSCAR.v{k:02d} differs from the E1 twin")
                break

    # E2: every entry must include its released ordering exactly once, and all orderings
    # of an entry must share one composition
    sel = json.loads((out / "E2_ordering" / "selection.json").read_text())
    for entry in sel["entries"]:
        native = [o for o in entry["orderings"] if o["is_released_ordering"]]
        if len(native) != 1:
            rep.error(f"E2/{entry['id']}: {len(native)} orderings marked as released")
        if entry["n_orderings"] != len(entry["orderings"]):
            rep.error(f"E2/{entry['id']}: ordering count disagrees with the listed orderings")
    formulas: dict[str, set] = {}
    for task in sorted((out / "E2_ordering" / "tasks").iterdir()):
        meta = json.loads((task / "TASK.json").read_text())
        formulas.setdefault(meta["entry"], set()).add(meta["formula"])
    for entry, fset in formulas.items():
        if len(fset) != 1:
            rep.error(f"E2/{entry}: orderings do not share one composition: {sorted(fset)}")
    kinds = {}
    for entry in sel["entries"]:
        kinds[entry["kind"]] = kinds.get(entry["kind"], 0) + 1
    rep.note(f"E2: {kinds}, control classes "
             f"{sorted({e['merge_class'] for e in sel['entries'] if e['kind'] == 'experimental'})}")

    # E3: each parent must carry the undamaged cell and all five damages, and each damage
    # must preserve composition
    sel = json.loads((out / "E3_crosscheck" / "selection.json").read_text())
    by_parent: dict[str, dict] = {}
    for task in sorted((out / "E3_crosscheck" / "tasks").iterdir()):
        meta = json.loads((task / "TASK.json").read_text())
        if meta["kind"] != "experimental":
            continue
        by_parent.setdefault(meta["parent"], {})[meta["variant"]] = meta
    for parent, variants in by_parent.items():
        missing = {"P0", "S1", "S2", "S3", "S4", "S5"} - set(variants)
        if missing:
            rep.error(f"E3/{parent}: missing variants {sorted(missing)}")
        formulas = {v["formula"] for v in variants.values()}
        if len(formulas) != 1:
            rep.error(f"E3/{parent}: damage changed the composition: {sorted(formulas)}")
        atoms = {v["n_atoms"] for v in variants.values()}
        if len(atoms) != 1:
            rep.error(f"E3/{parent}: damage changed the atom count: {sorted(atoms)}")
    # an exchange that leaves the crystal unchanged is not a damaged structure
    from pymatgen.core import Structure
    try:
        from pymatgen.core.structure_matcher import StructureMatcher
    except ImportError:
        from pymatgen.analysis.structure_matcher import StructureMatcher
    matcher = StructureMatcher(primitive_cell=False, attempt_supercell=False, scale=False,
                               ltol=0.01, stol=0.02, angle_tol=0.5)
    for parent, variants in by_parent.items():
        if "P0" not in variants:
            continue
        p0 = Structure.from_file(str(Path(variants["P0"]["task_dir"]) / "POSCAR.init"))
        for kind in ("S2", "S5"):
            if kind not in variants:
                continue
            st = Structure.from_file(str(Path(variants[kind]["task_dir"]) / "POSCAR.init"))
            if matcher.fit(st, p0):
                rep.error(f"E3/{parent}: the {kind} exchange left the crystal unchanged")
    if len(by_parent) != sel["n_experimental_parents"]:
        rep.error(f"E3: {len(by_parent)} parents on disk, selection.json says "
                  f"{sel['n_experimental_parents']}")

    # E4: roles must reproduce the frozen counts and the CIF hashes must still match
    sel = json.loads((out / "E4_design" / "selection.json").read_text())
    counts: dict[str, int] = {}
    pss_screened, pss_retained = [], []
    for task in sorted((out / "E4_design" / "tasks").iterdir()):
        meta = json.loads((task / "TASK.json").read_text())
        counts[meta["role"]] = counts.get(meta["role"], 0) + 1
        (pss_screened if meta["role"] == "screened" else pss_retained).append(meta["pss"])
        if not (task / "INCAR.relax_ions").exists():
            rep.error(f"E4/{task.name}: stage B needs INCAR.relax_ions, which is absent")
    if counts.get("screened") != 61 or counts.get("priority") != 140:
        rep.error(f"E4: role counts {counts} do not reproduce the frozen 61 screened "
                  f"and 140 priority")
    if pss_screened and max(pss_screened) >= sel["pss_cutoff"]:
        rep.error("E4: a task labelled screened has a PSS at or above the cutoff")
    if pss_retained and min(pss_retained) < sel["pss_cutoff"]:
        rep.error("E4: a task labelled retained has a PSS below the cutoff")
    rep.note(f"E4: roles {counts}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "dft"))
    ap.add_argument("--report", default=str(ROOT / "dft" / "VERIFY_REPORT.md"))
    ap.add_argument("--potcar-lib", default=None,
                    help="PAW library to check the specs against; defaults to "
                         "$VASP_PP_PATH, then the path the packages were built with")
    a = ap.parse_args()
    global POTCAR_LIB
    if a.potcar_lib:
        POTCAR_LIB = Path(a.potcar_lib)
    out = Path(a.out)
    rep = Report()

    totals = {"tasks": 0, "vasp_runs": 0, "atoms": 0, "cost_proxy": 0.0}
    overlaps: dict[str, list] = {}
    per_pkg = {}
    for name in PACKAGES:
        stats = verify_package(out / name, rep)
        per_pkg[name] = stats
        overlaps[name] = stats["paw_overlap"]
        for k in totals:
            totals[k] += stats[k]

    check_experiment_specific(out, rep)

    lines = ["# DFT task package verification", ""]
    lines.append(f"- packages checked: {', '.join(PACKAGES)}")
    lines.append(f"- tasks: {totals['tasks']}, VASP runs: {totals['vasp_runs']}, "
                 f"atoms across all cells: {totals['atoms']}")
    lines.append(f"- errors: {len(rep.errors)}, warnings: {len(rep.warnings)}")
    lines.append("")
    lines.append("| package | tasks | VASP runs | atoms | shortest contact (A) |")
    lines.append("|---|---|---|---|---|")
    for name, s in per_pkg.items():
        mc = min(s["min_contact"]) if s["min_contact"] else float("nan")
        lines.append(f"| {name} | {s['tasks']} | {s['vasp_runs']} | {s['atoms']} | {mc:.3f} |")
    n_ov = sum(len(v) for v in overlaps.values())
    if n_ov:
        lines += [f"## PAW core overlap (shortest contact below {PAW_RATIO_REPORT} of "
                  f"the radius sum)", "",
                  "Overlap by itself is normal in dense solids, so only tight cells are "
                  "listed. E1 compresses on purpose and is expected here, which is why it "
                  "carries the E1b hard-potential control; anywhere else a tight cell is "
                  "worth a second look.", ""]
        for name, items in overlaps.items():
            if not items:
                continue
            worst = min(items, key=lambda x: x["ratio"])
            lines.append(f"- {name}: {len(items)} cells below {PAW_RATIO_REPORT}, worst "
                         f"{worst['ratio']:.2f} ({'-'.join(worst['pair'])} at "
                         f"{worst['distance_a']} A in {worst['task']})")
        lines.append("")
    lines.append("")
    if rep.errors:
        lines += ["## Errors", ""] + [f"- {e}" for e in rep.errors] + [""]
    if rep.warnings:
        lines += ["## Warnings", ""] + [f"- {w}" for w in rep.warnings[:80]] + [""]
        if len(rep.warnings) > 80:
            lines += [f"- ... and {len(rep.warnings) - 80} more", ""]
    lines += ["## Notes", ""] + [f"- {n}" for n in rep.notes] + [""]
    Path(a.report).write_text("\n".join(lines))

    print("\n".join(lines[:14]))
    print(f"\nerrors={len(rep.errors)} warnings={len(rep.warnings)}  -> {a.report}")
    for e in rep.errors[:20]:
        print("  ERROR", e)
    return 1 if rep.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
