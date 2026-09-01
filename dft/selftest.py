#!/usr/bin/env python3
"""End-to-end test of collect.py, make_stage_b.py and analyze.py on synthetic results.

The real calculations are weeks away, so the extraction chain is exercised here against
VASP output written by hand with known answers. Real task directories are copied, so all
metadata, stage chaining and selection bookkeeping are the genuine ones; only the OUTCAR
and CONTCAR content is synthetic.

    python dft/selftest.py

Checks that the reduced-contact curve comes back in the right shape, that an ordering
energy and its order-disorder temperature are computed from static energies, that the
damage table and the DFT-MatterSim rank correlation are formed, and that the
Birch-Murnaghan fit recovers an injected bulk modulus.
"""
from __future__ import annotations

import importlib.util
import json
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
EV_A3_TO_GPA = 160.21766208


def load_module(name: str, path: Path, here: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    mod.HERE = here
    return mod


def write_outcar(stage_dir: Path, energies: list[float], volume: float,
                 n_atoms: int, converged: bool = True, fmax: float = 0.01) -> None:
    """Write the parts of an OUTCAR that collect.py reads."""
    stage_dir.mkdir(parents=True, exist_ok=True)
    lines = [" running   32 mpi-ranks, with    1 threads/rank",
             "   NBANDS=     48",
             "   irreducible k-points:    12"]
    for e in energies:
        lines += [f"  volume of cell :     {volume:.2f}",
                  "  TOTAL-FORCE (eV/Angst)",
                  "  -----------------------------------------------------------------"]
        for _ in range(n_atoms):
            lines.append(f"   0.000  0.000  0.000    {fmax:.6f}  0.000000  0.000000")
        lines += ["  -----------------------------------------------------------------",
                  f"  free  energy   TOTEN  =    {e:.8f} eV",
                  f"  energy  without entropy=    {e:.8f}  energy(sigma->0) =    {e:.8f}"]
    if converged:
        lines.append(" reached required accuracy - stopping structural energy minimisation")
    lines.append(" Elapsed time (sec):      120.0")
    (stage_dir / "OUTCAR").write_text("\n".join(lines) + "\n")
    (stage_dir / "exit_code").write_text("0\n")


def fake_contcar(stage_dir: Path, source_poscar: Path) -> None:
    shutil.copy2(source_poscar, stage_dir / "CONTCAR")


def bm_energy(v: float, e0: float, v0: float, b0_ev_a3: float, bp: float) -> float:
    eta = (v0 / v) ** (2.0 / 3.0)
    return e0 + 9.0 * v0 * b0_ev_a3 / 16.0 * (
        (eta - 1.0) ** 3 * bp + (eta - 1.0) ** 2 * (6.0 - 4.0 * eta))


def build_mock(tmp: Path) -> None:
    """Copy a small slice of the real packages and fill in synthetic VASP output."""
    # ---- E1: three parents, a V-shaped curve with a steep repulsive wall -------
    src = HERE / "E1_rho_curve"
    dst = tmp / "E1_rho_curve"
    (dst / "tasks").mkdir(parents=True)
    shutil.copy2(src / "selection.json", dst / "selection.json")
    grid = json.loads((src / "selection.json").read_text())["rho_grid"]
    stiffness = {}
    for idx, task in enumerate(sorted((src / "tasks").iterdir())[:3]):
        t = dst / "tasks" / task.name
        shutil.copytree(task, t)
        meta = json.loads((t / "TASK.json").read_text())
        n = meta["n_atoms"]
        # a different wall per compound, so the spread comparison has something to measure
        a = 8.0 + 2.0 * idx
        stiffness[meta["source_id"]] = a
        for k, rho in enumerate(grid):
            # steep below 1.0, shallow above: the asymmetry the experiment looks for
            excess = a * (1.0 - rho) ** 2 if rho < 1.0 else 0.3 * (rho - 1.0) ** 2
            write_outcar(t / f"v{k:02d}", [(-10.0 + excess) * n], 100.0, n)

    # ---- E1b: the same three compounds, a slightly softer wall under hard potentials
    srcb = HERE / "E1b_paw_control"
    dstb = tmp / "E1b_paw_control"
    (dstb / "tasks").mkdir(parents=True)
    shutil.copy2(srcb / "selection.json", dstb / "selection.json")
    # only the E1b compounds that were also mocked in E1, so the pairs actually match
    for task in sorted((srcb / "tasks").iterdir()):
        sid = json.loads((task / "TASK.json").read_text())["source_id"]
        if sid not in stiffness:
            continue
        t = dstb / "tasks" / task.name
        shutil.copytree(task, t)
        meta = json.loads((t / "TASK.json").read_text())
        n = meta["n_atoms"]
        a = 0.9 * stiffness[meta["source_id"]]   # 10% softer, inside the 25% tolerance
        for k, rho in enumerate(grid):
            excess = a * (1.0 - rho) ** 2 if rho < 1.0 else 0.3 * (rho - 1.0) ** 2
            write_outcar(t / f"v{k:02d}", [(-10.0 + excess) * n], 100.0, n)

    # ---- E2: one whole entry, the released ordering deliberately not the lowest --
    src = HERE / "E2_ordering"
    dst = tmp / "E2_ordering"
    (dst / "tasks").mkdir(parents=True)
    shutil.copy2(src / "selection.json", dst / "selection.json")
    sel = json.loads((src / "selection.json").read_text())
    entry = next(e for e in sel["entries"] if e["kind"] == "gnome")
    for task in sorted((src / "tasks").iterdir()):
        meta = json.loads((task / "TASK.json").read_text())
        if meta.get("entry") != entry["id"]:
            continue
        t = dst / "tasks" / task.name
        shutil.copytree(task, t)
        n = meta["n_atoms"]
        # 2 meV per atom above the best ordering: far below room-temperature entropy
        offset = 0.002 if meta["is_released_ordering"] else 0.0
        write_outcar(t / "relax_cell", [(-9.9 + offset) * n, (-10.0 + offset) * n],
                     100.0, n)
        fake_contcar(t / "relax_cell", t / "POSCAR.init")
        write_outcar(t / "static", [(-10.0 + offset) * n], 100.0, n)

    # ---- E3: two parents with all six cells, damage costing more than the parent -
    src = HERE / "E3_crosscheck"
    dst = tmp / "E3_crosscheck"
    (dst / "tasks").mkdir(parents=True)
    shutil.copy2(src / "selection.json", dst / "selection.json")
    release = {"P0": 0.001, "S1": 0.25, "S2": 0.10, "S3": 1.20, "S4": 0.40, "S5": 0.60}
    parents = sorted({json.loads((t / "TASK.json").read_text())["parent"]
                      for t in (src / "tasks").iterdir()
                      if json.loads((t / "TASK.json").read_text())["kind"] == "experimental"})[:2]
    ms_records = []
    for task in sorted((src / "tasks").iterdir()):
        meta = json.loads((task / "TASK.json").read_text())
        if meta.get("parent") not in parents:
            continue
        t = dst / "tasks" / task.name
        shutil.copytree(task, t)
        n = meta["n_atoms"]
        r = release[meta["variant"]]
        write_outcar(t / "relax_ions_fixed_cell", [(-10.0) * n, (-10.0 - r) * n], 100.0, n)
        fake_contcar(t / "relax_ions_fixed_cell", t / "POSCAR.init")
        write_outcar(t / "relax_cell", [(-10.0 - r) * n, (-10.05 - r) * n], 98.0, n)
        fake_contcar(t / "relax_cell", t / "POSCAR.init")
        write_outcar(t / "static", [(-10.05 - r) * n], 98.0, n)
        # a MatterSim reference that tracks DFT but understates the wrong-site exchange
        ms_records.append({"task": meta["task"], "variant": meta["variant"],
                           "release_ev_per_atom": r * (0.25 if meta["variant"] == "S5"
                                                       else 0.95)})
    (dst / "mattersim_reference.json").write_text(
        json.dumps({"records": ms_records}, indent=1) + "\n")

    # ---- E4: two candidates, each with an injected bulk modulus -----------------
    src = HERE / "E4_design"
    dst = tmp / "E4_design"
    (dst / "tasks").mkdir(parents=True)
    for f in ("selection.json", "submit.slurm", "make_stage_b.py"):
        shutil.copy2(src / f, dst / f)
    picked = []
    for task in sorted((src / "tasks").iterdir()):
        meta = json.loads((task / "TASK.json").read_text())
        if meta["role"] not in ("screened", "priority"):
            continue
        if sum(1 for p in picked if p[1] == meta["role"]) >= 1:
            continue
        picked.append((task, meta["role"]))
        if len(picked) == 2:
            break
    injected = {}
    for task, role in picked:
        t = dst / "tasks" / task.name
        shutil.copytree(task, t)
        meta = json.loads((t / "TASK.json").read_text())
        write_outcar(t / "relax_cell", [-10.0 * meta["n_atoms"]], 100.0, meta["n_atoms"])
        fake_contcar(t / "relax_cell", t / "POSCAR.init")
        injected[t.name] = 450.0 if role == "priority" else 250.0
    subprocess.run([sys.executable, str(dst / "make_stage_b.py")], check=True,
                   capture_output=True)
    for child in sorted((dst / "stage_b" / "tasks").iterdir()):
        meta = json.loads((child / "TASK.json").read_text())
        n = meta["n_atoms"]
        b_gpa = injected[meta["parent_task"]]
        # the equilibrium volume is the real relaxed cell, and the scaled POSCAR the
        # generator wrote carries the volume this point actually runs at
        v0 = _poscar_volume(dst / "tasks" / meta["parent_task"] / "POSCAR.init")
        v = _poscar_volume(child / "POSCAR.init")
        e = bm_energy(v, -10.0 * n, v0, b_gpa / EV_A3_TO_GPA, 4.0)
        for stage in ("relax_ions", "static"):
            write_outcar(child / stage, [e], v, n)
            fake_contcar(child / stage, child / "POSCAR.init")
    return injected, stiffness


def _poscar_volume(path: Path) -> float:
    lines = path.read_text().splitlines()
    s = float(lines[1].split()[0])
    m = [[float(x) for x in lines[i].split()[:3]] for i in (2, 3, 4)]
    det = (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
           - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
           + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))
    return abs(det) * s ** 3


def main() -> int:
    failures = []
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        injected, stiffness_used = build_mock(tmp)

        collect = load_module("dft_collect", HERE / "collect.py", tmp)
        for name in ("E1_rho_curve", "E1b_paw_control", "E2_ordering",
                     "E3_crosscheck", "E4_design"):
            out = collect.collect_package(tmp / name)
            print(f"collect {name}: {out['n_tasks']} tasks {out['status_counts']}")
            if out["status_counts"].get("failed"):
                failures.append(f"{name}: collect reported failed tasks")
        out = collect.collect_package(tmp / "E4_design", stage_b=True)
        print(f"collect E4 stage B: {out['n_tasks']} tasks {out['status_counts']}")

        analyze = load_module("dft_analyze", HERE / "analyze.py", tmp)
        lines: list[str] = []
        checks = []
        for fn in (analyze.analyse_e1, analyze.analyse_e2,
                   analyze.analyse_e3, analyze.analyse_e4):
            checks += fn(lines) or []
        print("\n".join(lines))

        # ---- assertions on known answers ---------------------------------------
        curves = json.loads((tmp / "E1_rho_curve" / "curves.json").read_text())
        # the mock curve is analytic, so the interpolated read-off must land on it
        a0 = stiffness_used[curves[0]["source_id"]] if curves else 8.0
        want = a0 * (1.0 - 0.735) ** 2
        if not curves or abs(curves[0]["excess_at_floor"] - want) > 0.01:
            failures.append(f"E1: excess at the floor is "
                            f"{curves[0]['excess_at_floor'] if curves else None}, "
                            f"expected {want:.4f} within 0.01")
        want_strict = a0 * (1.0 - 0.804) ** 2
        if curves and abs(curves[0]["excess_at_strict_floor"] - want_strict) > 0.01:
            failures.append(f"E1: excess at the strict floor is "
                            f"{curves[0]['excess_at_strict_floor']:.4f}, "
                            f"expected {want_strict:.4f} within 0.01")
        # a*(1-rho)^2 = 0.1 at rho = 1 - sqrt(0.1/a)
        want_star = 1.0 - (0.1 / a0) ** 0.5
        if curves and abs(curves[0]["rho_star"] - want_star) > 0.005:
            failures.append(f"E1: rho* is {curves[0]['rho_star']:.4f}, "
                            f"expected {want_star:.4f}")
        if curves and curves[0]["d_star_a"] is None:
            failures.append("E1: d* missing, so the transferability test cannot run")
        shift = json.loads((tmp / "E1b_paw_control" / "paw_shift.json").read_text())
        if not shift:
            failures.append("E1b: no matched pairs against E1")
        else:
            got = shift[0]["relative_shift"]
            if abs(got - 0.10) > 0.005:
                failures.append(f"E1b: relative shift {got:.3f}, expected 0.10")

        orders = json.loads((tmp / "E2_ordering" / "ordering_energies.json").read_text())
        if not orders:
            failures.append("E2: no ordering energies produced")
        else:
            row = orders[0]
            if abs(row["dE_order_ev_per_atom"] - 0.002) > 1e-6:
                failures.append(f"E2: ordering energy {row['dE_order_ev_per_atom']}, "
                                f"expected 0.002")
            want_t = 0.002 / (analyze.KB_EV * row["dS_per_atom_kB"])
            if abs(row["T_od_K"] - want_t) > 1.0:
                failures.append(f"E2: T_od {row['T_od_K']:.1f} K, expected {want_t:.1f} K")

        e3 = [c for c in checks if c["experiment"] == "E3"]
        if not e3:
            failures.append("E3: no checks produced")
        else:
            ratio = next((c for c in e3 if "ratio" in c["prediction"]), None)
            if ratio is None or not ratio["met"]:
                failures.append("E3: the S5 ratio check should pass on mock data where "
                                "MatterSim understates the wrong-site exchange")

        moduli = json.loads((tmp / "E4_design" / "bulk_moduli.json").read_text())
        if len(moduli) != len(injected):
            failures.append(f"E4: fitted {len(moduli)} candidates, expected {len(injected)}")
        for row in moduli:
            want_b = injected[row["parent_task"]]
            err = abs(row["dft_bulk_modulus_gpa"] - want_b) / want_b
            print(f"  E4 {row['parent_task']}: fitted {row['dft_bulk_modulus_gpa']:.1f} GPa, "
                  f"injected {want_b:.1f} GPa, error {err:.2%}")
            if err > 0.02:
                failures.append(f"E4: {row['parent_task']} fit off by {err:.1%}")

    print()
    if failures:
        for f in failures:
            print("FAIL:", f)
        return 1
    print("selftest passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
