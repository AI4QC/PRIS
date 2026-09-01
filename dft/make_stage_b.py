#!/usr/bin/env python3
"""Build the E4 stage-B energy-volume tasks from the stage-A relaxed cells.

`build_tasks.py` copies this file into `E4_design/`, so run it from there on the cluster
after `submit.slurm` (stage A) finishes:

    python3 make_stage_b.py && sbatch stage_b/submit.slurm

Stage A relaxes each generated candidate. Stage B holds the relaxed cell at five fixed
volumes and relaxes ions and cell shape inside each, giving the energy-volume curve that
a Birch-Murnaghan fit turns into a bulk modulus. Only the standard library is used, since
the cluster is not assumed to carry pymatgen: a uniform volume change is a scaling of the
three lattice vectors, and the fractional coordinates are untouched.

A candidate whose stage A did not converge is skipped and listed in stage_b/SKIPPED.json,
so a missing bulk modulus is always traceable to a named failure rather than a silent gap.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from pathlib import Path

KSPACING = 0.22   # must match the value build_tasks.py froze

HERE = Path(__file__).resolve().parent
TASKS = HERE / "tasks"
OUT = HERE / "stage_b"


def read_poscar(path: Path):
    lines = path.read_text().splitlines()
    scale = float(lines[1].split()[0])
    vectors = [[float(x) for x in lines[i].split()[:3]] for i in (2, 3, 4)]
    return lines, scale, vectors


def write_scaled(src: Path, dst: Path, volume_factor: float) -> None:
    """Uniform volume change: every lattice vector scales by the cube root of the factor."""
    lines, scale, vectors = read_poscar(src)
    f = volume_factor ** (1.0 / 3.0)
    out = list(lines)
    out[1] = f"{scale:.16f}"
    for k, vec in enumerate(vectors):
        out[2 + k] = "  " + "  ".join(f"{c * f:.16f}" for c in vec)
    dst.write_text("\n".join(out) + "\n")


def _det(m):
    return (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))


def _inverse(m):
    """True 3x3 inverse: the adjugate is the cofactor matrix transposed, so j indexes
    the rows of the minor and i its columns. Getting this backwards returns the
    transpose, which silently yields a different k-mesh."""
    d = _det(m)
    return [[(m[(j + 1) % 3][(i + 1) % 3] * m[(j + 2) % 3][(i + 2) % 3]
              - m[(j + 1) % 3][(i + 2) % 3] * m[(j + 2) % 3][(i + 1) % 3]) / d
             for j in range(3)] for i in range(3)]


def kmesh_for(vectors, scale: float = 1.0):
    """Gamma-centred mesh from KSPACING, on a lattice scaled by `scale`.

    Stage A's mesh was fixed on the generated cell. Relaxation usually shrinks a generated
    cell, so reusing that mesh would under-sample the relaxed one; the mesh is therefore
    re-derived here, from the smallest volume in the curve so that every point on the
    curve is at least as well sampled, and shared by all five points so the energies are
    comparable.
    """
    lat = [[c * scale for c in row] for row in vectors]
    inv = _inverse(lat)
    mesh = []
    for i in range(3):
        # rows of 2*pi*inv(A)^T are the reciprocal vectors
        b = math.sqrt(sum((2 * math.pi * inv[j][i]) ** 2 for j in range(3)))
        mesh.append(max(1, int(math.ceil(b / KSPACING))))
    return mesh


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "tasks").mkdir(parents=True)
    (OUT / "logs").mkdir(parents=True)

    built, skipped = [], []
    for task in sorted(TASKS.iterdir()):
        if not task.is_dir():
            continue
        meta = json.loads((task / "TASK.json").read_text())
        contcar = task / "relax_cell" / "CONTCAR"
        code = task / "relax_cell" / "exit_code"
        if not contcar.exists() or not contcar.stat().st_size:
            skipped.append({"task": task.name, "reason": "no CONTCAR from stage A"})
            continue
        if code.exists() and code.read_text().strip() != "0":
            skipped.append({"task": task.name, "reason": "stage A exit code not zero"})
            continue
        _, contcar_scale, contcar_vec = read_poscar(contcar)
        smallest = min(meta["ev_volume_factors"])
        mesh = kmesh_for([[c * contcar_scale for c in row] for row in contcar_vec],
                         smallest ** (1.0 / 3.0))
        kpoints_text = (f"Gamma-centred mesh from KSPACING={KSPACING} on the relaxed cell\n"
                        f"0\nGamma\n{mesh[0]} {mesh[1]} {mesh[2]}\n0 0 0\n")
        for i, factor in enumerate(meta["ev_volume_factors"]):
            name = f"{task.name}-v{i}"
            d = OUT / "tasks" / name
            d.mkdir(parents=True)
            write_scaled(contcar, d / "POSCAR.init", float(factor))
            for f in ("POTCAR.spec.json", "POTCAR.list", "run.sh"):
                shutil.copy2(task / f, d / f)
            (d / "KPOINTS").write_text(kpoints_text)
            os.chmod(d / "run.sh", 0o755)
            shutil.copy2(task / "INCAR.relax_ions", d / "INCAR.relax_ions")
            shutil.copy2(task / "INCAR.static", d / "INCAR.static")
            plan = [
                {"stage": "relax_ions", "incar": "relax_ions",
                 "poscar": "POSCAR.init", "from_contcar": ""},
                {"stage": "static", "incar": "static",
                 "poscar": "POSCAR.static", "from_contcar": "relax_ions"},
            ]
            (d / "stages.json").write_text(json.dumps(plan, indent=1) + "\n")
            (d / "stages.tsv").write_text("".join(
                f"{r['stage']}\t{r['incar']}\t{r['poscar']}\t{r['from_contcar'] or '-'}\n"
                for r in plan))
            child = dict(meta)
            child.update({"task": name, "stage": "B", "parent_task": task.name,
                          "volume_factor": float(factor),
                          "stage_a_contcar_sha256": sha256_file(contcar),
                          "kmesh": mesh, "n_kpoints": mesh[0] * mesh[1] * mesh[2],
                          "kmesh_source": "re-derived from the relaxed cell at the "
                                          "smallest volume of the curve",
                          "stages": ["relax_ions", "static"]})
            (d / "TASK.json").write_text(json.dumps(child, indent=1, sort_keys=True) + "\n")
            built.append(f"tasks/{name}")

    (OUT / "tasklist.txt").write_text("\n".join(built) + "\n")
    (OUT / "SKIPPED.json").write_text(json.dumps(skipped, indent=1) + "\n")

    submit = (HERE / "submit.slurm").read_text()
    submit = submit.replace("pris-e4a", "pris-e4b")
    for line in submit.splitlines():
        if line.startswith("#SBATCH --array="):
            submit = submit.replace(line, f"#SBATCH --array=1-{len(built)}%32")
            break
    (OUT / "submit.slurm").write_text(submit)
    os.chmod(OUT / "submit.slurm", 0o755)

    n_cells = len({Path(b).name.rsplit("-v", 1)[0] for b in built})
    print(f"stage B: {len(built)} tasks from {n_cells} relaxed cells, "
          f"{len(skipped)} candidates skipped")
    if skipped:
        print("skipped (see stage_b/SKIPPED.json):")
        for s in skipped[:10]:
            print("  ", s["task"], "-", s["reason"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
