#!/usr/bin/env python3
"""Build the E1-E4 VASP task packages for the PRIS first-principles campaign.

The builder is deterministic: given the same input stores it writes byte-identical
task directories, and every selection rule below is fixed here rather than chosen
after looking at results. `dft/PREREG-DFT.md` states the predictions these tasks test.

  E1  rho_c energy landscape        physical calibration of the D1 floors and the D2 ceiling
  E1b hard-potential control        bounds the frozen-core error at the compressed end
  E2  ordering degeneracy           whether GNoME's low-symmetry excess is thermodynamically real
  E3  damage severity cross-check   DFT against MatterSim on the five controlled damages
  E4  inverse-design verification   DFT bulk modulus of the screened and retained queues

Run:  python dft/build_tasks.py [--out dft] [--only E1,E2]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path("<repo>/")
sys.path.insert(0, str(ROOT / "src"))

FEATURES = Path("$PRIS_FEATURES/")
GNOME_DIR = Path("$PRIS_ARCHIVE/external_sources/gnome_d7")
MERGE_TEST = ROOT / "outputs/20260817_p1_gnome_attr/merge_test.parquet"
INVERSE_SCORES = ROOT / "outputs/20260822_property_design_synthesis_score/inverse_scores.parquet"
SHARD_ROOT = ROOT / "outputs/20260821_property_design"

POTCAR_LIB = Path("<path>")

# ---------------------------------------------------------------- frozen protocol

PROTOCOL_ID = "2026-08-25-pris-dft-e1e4-v1"
VASP_VERSION = "6.3.0"
KSPACING = 0.22          # inverse angstrom, reciprocal lattice includes 2*pi
# Parallel layout for one task on one 64-core node, chosen by measurement on this cluster
# rather than by default: at 64 ranks the two settings below ran a representative cell in
# 12.2 s against 39.5 s for the eight-rank default, and VASP itself warns that NCORE=1 is
# inefficient on modern hardware. Neither changes the physics; they only divide the work.
NCORE = 8
KPAR = 2
BOHR_A = 0.529177210903  # POTCAR radii are quoted in Bohr
ENCUT_FLOOR = 520.0      # eV
ENCUT_ENMAX_FACTOR = 1.3
PSS_CUTOFF = -0.6368790173149083   # frozen in outputs/20260823_property_design_pss_supportmatched_v2
PRIORITY_BULK_GPA = 400.0
SPLIT_SEED = "20260728"  # PREREG.md section 1

# elements that get a spin-polarised start; everything else runs ISPIN=1
MAGNETIC = set("Sc Ti V Cr Mn Fe Co Ni Cu Zn".split()) | set(
    "La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu".split()
) | set("Ac Th Pa U Np Pu".split())

# MPRelaxSet symbols, with the one local substitution recorded explicitly
POTCAR_OVERRIDES = {"W": "W_sv"}   # local library carries W_sv, not W_pv

# Hard potentials for the E1b control. Compressing a lattice pushes contacts inside the
# sum of the PAW cutoff radii, where the frozen core is least reliable and its error runs
# repulsive, in the same direction as E1's own prediction. These have much smaller cores,
# so agreement between the two sets bounds that error instead of assuming it away.
HARD_OVERRIDES = {"O": "O_h", "F": "F_h", "N": "N_h", "S": "S_h", "Cl": "Cl_h",
                  "B": "B_h", "C": "C_h", "P": "P_h", "H": "H_h", "Ga": "Ga_h",
                  "Ge": "Ge_h"}
E1B_N_COMPOUNDS = 8      # the 8 with the tightest PAW overlap at the D1 floor

LN = set("La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu".split())
RE_CLASS = LN | {"Sc", "Y"}

E1_RHO_GRID = tuple(round(0.60 + 0.05 * i, 2) for i in range(17))   # 0.60 .. 1.40
E1_N_PER_ANION = 4
E1_ANIONS = ("O", "S", "F", "Cl", "N")
E1_MAX_SITES = 12

E2_N_GNOME = 25
E2_N_CONTROL = 10
E2_MAX_SITES = 20
E2_MAX_GROUP_SITES = 10
E2_ORDERING_RANGE = (2, 12)      # enumerate all orderings only when the count lands here
E2_ENUM_CAP = 5000               # skip an entry rather than enumerate more decorations

E3_N_PARENTS = 30
E3_MAX_SITES = 16
E3_N_GNOME_PARENTS = 20
E3_DAMAGES = ("S1", "S2", "S3", "S4", "S5")

MIN_PARENT_CONTACT_A = 1.0    # source cells below this carry overlapping split sites
MIN_VARIANT_CONTACT_A = 0.9   # a damaged cell below this is numerically hopeless, not just bad

E4_N_CONTROL = 60
E4_EV_VOLUME_FACTORS = (0.94, 0.97, 1.00, 1.03, 1.06)   # volume factors about the relaxed cell


# ---------------------------------------------------------------- small helpers


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def split_of(sid: str) -> str:
    """PREREG.md section 1: seeded hash split, discovery 60 / calibration 25 / lockbox 15."""
    u = int(hashlib.sha256(f"{SPLIT_SEED}:{sid}".encode()).hexdigest()[:8], 16) / 2 ** 32
    return "discovery" if u < 0.60 else ("calibration" if u < 0.85 else "lockbox")


def stable_rng(*parts: str) -> np.random.Generator:
    seed = int(hashlib.sha256("|".join(parts).encode()).hexdigest()[:16], 16) % (2 ** 63)
    return np.random.default_rng(seed)


def merge_class(sym: str) -> str:
    from pymatgen.core.periodic_table import Element
    if sym in RE_CLASS:
        return "RE"
    try:
        return f"G{Element(sym).group}"
    except Exception:
        return sym


# ---------------------------------------------------------------- VASP input writers


_POTCAR_MAP: dict[str, str] | None = None
_POTCAR_META: dict[str, dict] = {}


def potcar_symbol(element: str, hard: bool = False) -> str:
    global _POTCAR_MAP
    if _POTCAR_MAP is None:
        from pymatgen.io.vasp.sets import MPRelaxSet
        cfg = getattr(MPRelaxSet, "CONFIG", None)
        if cfg is None:
            from pymatgen.io.vasp.sets import _load_yaml_config
            cfg = _load_yaml_config("MPRelaxSet")
        _POTCAR_MAP = dict(cfg["POTCAR"])
        _POTCAR_MAP.update(POTCAR_OVERRIDES)
    if hard and element in HARD_OVERRIDES:
        return HARD_OVERRIDES[element]
    return _POTCAR_MAP[element]


def potcar_meta(element: str, hard: bool = False) -> dict:
    key = (element, hard)
    if key not in _POTCAR_META:
        sym = potcar_symbol(element, hard)
        path = POTCAR_LIB / sym / "POTCAR"
        if not path.exists():
            raise FileNotFoundError(f"POTCAR missing for {element} ({sym}) at {path}")
        enmax = None
        title = None
        rcore_au = None
        with open(path, "r", errors="replace") as f:
            for line in f:
                if title is None and "PAW" in line:
                    title = line.strip()
                if enmax is None and "ENMAX" in line:
                    enmax = float(line.split("ENMAX")[1].split("=")[1].split(";")[0])
                if rcore_au is None and "RCORE" in line:
                    rcore_au = float(line.split("RCORE")[1].split("=")[1].split()[0])
                # RCORE precedes ENMAX in the PSCTR block, so stop only once both are read
                if enmax is not None and rcore_au is not None:
                    break
        _POTCAR_META[key] = {
            "element": element,
            "symbol": sym,
            "title": title,
            "enmax_ev": enmax,
            # outermost PAW cutoff radius, converted from Bohr: two atoms closer than the
            # sum of their radii have overlapping augmentation spheres and VASP says so
            "rcore_a": round(rcore_au * BOHR_A, 4) if rcore_au else None,
            "sha256": sha256_file(path),
            "library_relative": f"{sym}/POTCAR",
        }
    return _POTCAR_META[key]


def encut_for(species: list[str], hard: bool = False) -> float:
    mx = max(potcar_meta(e, hard)["enmax_ev"] for e in species)
    return float(max(ENCUT_FLOOR, math.ceil(ENCUT_ENMAX_FACTOR * mx)))


def kmesh_for(structure) -> tuple[int, int, int]:
    """VASP KSPACING convention; the reciprocal lattice in pymatgen already carries 2*pi."""
    b = structure.lattice.reciprocal_lattice.abc
    return tuple(max(1, int(math.ceil(x / KSPACING))) for x in b)


def write_kpoints(path: Path, mesh: tuple[int, int, int]) -> None:
    path.write_text(
        f"Gamma-centred mesh from KSPACING={KSPACING}\n0\nGamma\n"
        f"{mesh[0]} {mesh[1]} {mesh[2]}\n0 0 0\n"
    )


def incar_text(name: str, stage: str, structure, encut: float, mesh: tuple[int, int, int]) -> str:
    """stage in {static, static_soft, relax_cell, relax_ions, relax_ions_fixed_cell}."""
    species = [str(s.specie.symbol) for s in structure]
    ispin = 2 if any(s in MAGNETIC for s in species) else 1
    nk = mesh[0] * mesh[1] * mesh[2]
    # Relaxations run with ISYM=0 so the cell is free to change symmetry, which is the
    # quantity E2 measures. Static stages keep ISYM=2 purely to fold the k-point mesh.
    isym = 0 if stage.startswith("relax") else 2
    lines = [
        f"SYSTEM = {name} {stage}",
        "GGA = PE",
        f"ENCUT = {encut:.0f}",
        "PREC = Accurate",
        "EDIFF = 1E-6",
        "NELM = 200",
        "ALGO = Normal",
        "LASPH = .TRUE.",
        "LREAL = .FALSE.",
        "ADDGRID = .TRUE.",
        f"ISYM = {isym}",
        "LWAVE = .FALSE.",
        "LCHARG = .FALSE.",
        f"NCORE = {NCORE}",
        f"KPAR = {KPAR}",
        f"ISPIN = {ispin}",
    ]
    if ispin == 2:
        lines.append(f"MAGMOM = {len(structure)}*1.0")
    if stage == "static":
        # tetrahedron energies where the mesh allows it, gaussian otherwise
        if nk >= 4:
            lines += ["ISMEAR = -5", "NSW = 0", "IBRION = -1", "ISIF = 2"]
        else:
            lines += ["ISMEAR = 0", "SIGMA = 0.05", "NSW = 0", "IBRION = -1", "ISIF = 2"]
    elif stage == "static_soft":
        lines += ["ISMEAR = 0", "SIGMA = 0.05", "NSW = 0", "IBRION = -1", "ISIF = 2"]
    elif stage == "relax_cell":
        lines += ["ISMEAR = 0", "SIGMA = 0.10", "NSW = 200", "IBRION = 2",
                  "ISIF = 3", "EDIFFG = -0.02", "POTIM = 0.30"]
    elif stage == "relax_ions":
        lines += ["ISMEAR = 0", "SIGMA = 0.10", "NSW = 120", "IBRION = 2",
                  "ISIF = 4", "EDIFFG = -0.02", "POTIM = 0.30"]
    elif stage == "relax_ions_fixed_cell":
        # ions only, cell entirely frozen: the protocol the paper's MatterSim relaxation
        # uses, so that the two relaxation energies measure the same thing
        lines += ["ISMEAR = 0", "SIGMA = 0.10", "NSW = 200", "IBRION = 2",
                  "ISIF = 2", "EDIFFG = -0.02", "POTIM = 0.30"]
    else:
        raise ValueError(stage)
    return "\n".join(lines) + "\n"


RUN_SH = """#!/usr/bin/env bash
# Per-task driver. Stages run in order; each writes into its own subdirectory so a failed
# stage never overwrites a good one. Pure shell, so a compute node needs nothing but VASP.
#
# Two VASP failures are mechanical rather than physical, and both have a remedy that leaves
# the answer alone. VASP's line search reports `ZBRENT: fatal error in bracketing` when a
# relaxation has converged so tightly that it cannot take another step; VASP's own advice is
# to continue from CONTCAR, which is what happens here. Its symmetry analyser refuses a cell
# whose real and reciprocal lattices disagree on the Bravais type; since ISYM=2 is present
# only to fold the k-mesh, that stage is redone with ISYM=0, which computes the same
# quantity on the unfolded mesh. Every fallback writes FALLBACK_APPLIED into the stage
# directory and keeps the failed attempt beside it, so nothing is quietly repaired.
set -uo pipefail
cd "$(dirname "$0")"
: "${VASP_BIN:?VASP_BIN not set}"
: "${VASP_PP_PATH:?VASP_PP_PATH not set}"

build_potcar() {
  : > POTCAR
  while read -r rel; do
    [ -z "$rel" ] && continue
    cat "$VASP_PP_PATH/$rel" >> POTCAR || return 1
  done < POTCAR.list
  [ -s POTCAR ]
}

run_stage() {
  stage="$1"; incar_file="$2"; poscar="$3"
  mkdir -p "$stage"
  cp "$poscar" "$stage/POSCAR"
  cp "$incar_file" "$stage/INCAR"
  cp KPOINTS "$stage/KPOINTS"
  cp POTCAR.list POTCAR.spec.json "$stage/"
  # stdin is closed for VASP: the stage loop reads from a file descriptor, and a child that
  # touches stdin would swallow the rest of that file and silently end the loop early
  ( cd "$stage" && build_potcar && $VASP_BIN > vasp.out 2> vasp.err < /dev/null ; echo $? > exit_code )
  rm -f "$stage/POTCAR" "$stage/CHG" "$stage/CHGCAR" "$stage/WAVECAR" "$stage/PROCAR"
  return 0
}

why_failed() {
  grep -q "ZBRENT" "$1/vasp.out" 2>/dev/null && { echo zbrent; return; }
  # IBZKPT belongs here too: the same analyser refusing a cell whose crystal and
  # reciprocal point groups disagree, and it answers to the same tolerance
  grep -qE "Inconsistent Bravais|RHOSYG|INVGRP|IBZKPT|VERY BAD NEWS.*symmetr" "$1/vasp.out" 2>/dev/null \
    && { echo symmetry; return; }
  echo other
}

# the plan is read on descriptor 3, so nothing a stage runs can consume it
while IFS="$(printf '\\t')" read -r stage incar poscar from_contcar <&3; do
  [ -z "$stage" ] && continue
  if [ -f "$stage/exit_code" ] && [ "$(cat "$stage/exit_code")" = "0" ]; then
    echo "skip $stage (already converged)"; continue
  fi
  if [ "$from_contcar" != "-" ]; then
    if [ ! -s "$from_contcar/CONTCAR" ]; then
      echo "missing $from_contcar/CONTCAR, stopping" >&2; break
    fi
    cp "$from_contcar/CONTCAR" "$poscar"
  fi
  echo "==> $stage"
  incar_file="INCAR.$incar"
  attempt=0
  while : ; do
    run_stage "$stage" "$incar_file" "$poscar"
    code=$(cat "$stage/exit_code" 2>/dev/null)
    [ "$code" = "0" ] && break
    reason=$(why_failed "$stage")
    if [ "$reason" = "zbrent" ] && [ -s "$stage/CONTCAR" ] && [ "$attempt" -lt 2 ]; then
      attempt=$((attempt + 1))
      mv "$stage" "${stage}.failed${attempt}"
      cp "${stage}.failed${attempt}/CONTCAR" "$poscar"
      mkdir -p "$stage"; echo "continued from CONTCAR after ZBRENT, attempt $attempt" > "$stage/FALLBACK_APPLIED"
      echo "    ZBRENT: continuing from CONTCAR (attempt $attempt)"
      continue
    fi
    if [ "$reason" = "symmetry" ] && [ "$attempt" -lt 3 ]; then
      # VASP refuses a cell whose real and reciprocal lattices disagree on the Bravais
      # type. Turning symmetry off does not help: the lattice-type test runs regardless of
      # ISYM. What the error itself asks for is a looser SYMPREC, which was confirmed on
      # this cluster: ISYM=0 alone still failed, ISYM=0 with SYMPREC=1E-4 converged. The
      # k-mesh is written out explicitly, so a looser symmetry tolerance cannot change it.
      attempt=$((attempt + 1))
      mv "$stage" "${stage}.failed${attempt}"
      # A ladder, because one loosening is not always enough: on one E4 cell SYMPREC=1E-4
      # only moved the refusal from the Bravais test to IBZKPT. A cell that survives none
      # of these is reported as a failure rather than pushed further.
      case "$attempt" in
        1) sym="1E-4"; isym="keep" ;;
        2) sym="1E-3"; isym="0" ;;
        *) sym="1E-2"; isym="0" ;;
      esac
      incar_file="INCAR.${incar}.sym${attempt}"
      grep -viE "^SYMPREC" "INCAR.$incar" > "$incar_file"
      [ "$isym" = "0" ] && sed -i 's/^ISYM = 2$/ISYM = 0/' "$incar_file"
      echo "SYMPREC = $sym" >> "$incar_file"
      mkdir -p "$stage"
      echo "lattice-type check failed; retried with SYMPREC=$sym ISYM=$isym" > "$stage/FALLBACK_APPLIED"
      echo "    symmetry: retrying with SYMPREC=$sym ISYM=$isym"
      continue
    fi
    echo "stage $stage failed ($reason)" >&2
    break
  done
  [ "$code" = "0" ] || break
done 3< stages.tsv
"""


SUBMIT_SLURM = """#!/usr/bin/env bash
#SBATCH --job-name={job}
#SBATCH --array=1-{n}%{concurrent}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node={ntasks}
#SBATCH --time={time}
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
# Adjust --partition/--account for the cluster before submitting.

set -uo pipefail
HERE="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
cd "$HERE"
mkdir -p logs

# ---- cluster-specific settings -------------------------------------------------
module load vasp/{vasp} 2>/dev/null || true
export VASP_BIN="${{VASP_BIN:-srun -n $SLURM_NTASKS vasp_std}}"
export VASP_PP_PATH="${{VASP_PP_PATH:-$HOME/POTCAR/PBE}}"
# --------------------------------------------------------------------------------

TASK_DIR=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" tasklist.txt)
[ -z "$TASK_DIR" ] && exit 0
echo "task $SLURM_ARRAY_TASK_ID -> $TASK_DIR on $(hostname)"
bash "$TASK_DIR/run.sh"
"""


def write_task(
    task_dir: Path,
    name: str,
    structures: list[tuple[str, "object"]],   # (poscar filename, structure)
    stages: list[dict],
    meta: dict,
    extra_incars: tuple[str, ...] = (),
    hard: bool = False,
) -> dict:
    """Write one task directory. `stages` entries: {stage, poscar, incar_stage, from_contcar?}."""
    from pymatgen.io.vasp.inputs import Poscar
    task_dir.mkdir(parents=True, exist_ok=True)
    # sort so that identical species are contiguous in POSCAR; POTCAR order follows
    structures = [(fname, st.get_sorted_structure()) for fname, st in structures]
    ref = structures[0][1]
    species = sorted({str(s.specie.symbol) for s in ref})
    encut = encut_for(species, hard)
    mesh = kmesh_for(ref)

    for fname, st in structures:
        Poscar(st).write_file(str(task_dir / fname))

    write_kpoints(task_dir / "KPOINTS", mesh)

    incar_stages = sorted({s["incar_stage"] for s in stages} | set(extra_incars))
    for istage in incar_stages:
        (task_dir / f"INCAR.{istage}").write_text(incar_text(name, istage, ref, encut, mesh))

    # POTCAR blocks must follow the species order written into POSCAR, not alphabetical
    site_symbols = Poscar(ref).site_symbols
    order = [potcar_meta(sym, hard) for sym in site_symbols]
    (task_dir / "POTCAR.spec.json").write_text(
        json.dumps({"order": order, "library": str(POTCAR_LIB),
                    "site_symbols": site_symbols}, indent=1) + "\n"
    )
    # the same order as a bare list, so the job script can build POTCAR with cat alone
    (task_dir / "POTCAR.list").write_text(
        "\n".join(e["library_relative"] for e in order) + "\n")

    # a stage names its own output directory and the INCAR it runs, so several stages can
    # share one INCAR without duplicating it
    stage_records = [{"stage": s["stage"], "incar": s["incar_stage"], "poscar": s["poscar"],
                      "from_contcar": s.get("from_contcar", "")} for s in stages]
    (task_dir / "stages.json").write_text(json.dumps(stage_records, indent=1) + "\n")
    (task_dir / "stages.tsv").write_text("".join(
        f"{r['stage']}\t{r['incar']}\t{r['poscar']}\t{r['from_contcar'] or '-'}\n"
        for r in stage_records))

    (task_dir / "run.sh").write_text(RUN_SH)
    os.chmod(task_dir / "run.sh", 0o755)

    task_meta = {
        "protocol": PROTOCOL_ID,
        "task": name,
        "encut_ev": encut,
        "kmesh": list(mesh),
        "n_kpoints": int(np.prod(mesh)),
        "n_atoms": len(ref),
        "species": species,
        "formula": ref.composition.reduced_formula,
        "stages": [s["stage"] for s in stages],
        **meta,
    }
    (task_dir / "TASK.json").write_text(json.dumps(task_meta, indent=1, sort_keys=True) + "\n")
    return task_meta


def finish_package(pkg: Path, job: str, task_metas: list[dict], selection: dict,
                   ntasks: int, hours: int, concurrent: int) -> dict:
    rel = [str(Path(m["task_dir"]).relative_to(pkg)) for m in task_metas]
    (pkg / "tasklist.txt").write_text("\n".join(rel) + "\n")
    (pkg / "submit.slurm").write_text(
        SUBMIT_SLURM.format(job=job, n=len(rel), concurrent=concurrent, ntasks=ntasks,
                            time=f"{hours}:00:00", vasp=VASP_VERSION))
    os.chmod(pkg / "submit.slurm", 0o755)
    (pkg / "selection.json").write_text(json.dumps(selection, indent=1, sort_keys=True) + "\n")

    files = {}
    for p in sorted(pkg.rglob("*")):
        if p.is_file() and p.name != "MANIFEST.json":
            files[str(p.relative_to(pkg))] = sha256_file(p)
    manifest = {
        "protocol": PROTOCOL_ID,
        "package": pkg.name,
        "n_tasks": len(rel),
        "n_vasp_runs": int(sum(len(m["stages"]) for m in task_metas)),
        "total_atoms": int(sum(m["n_atoms"] for m in task_metas)),
        "builder_sha256": sha256_file(Path(__file__)),
        "vasp_version": VASP_VERSION,
        "kspacing": KSPACING,
        "encut_rule": f"max({ENCUT_FLOOR:.0f}, ceil({ENCUT_ENMAX_FACTOR} * max ENMAX))",
        "potcar_library": str(POTCAR_LIB),
        "potcar_contents_included": False,
        "files_sha256": files,
    }
    (pkg / "MANIFEST.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    return manifest


# ---------------------------------------------------------------- data loaders


def load_provenance() -> pd.DataFrame:
    p = pd.read_parquet(FEATURES / "provenance.parquet")
    p = p[p.in_analysis_set].copy()
    p["split"] = [split_of(s) for s in p.source_id]
    return p


def structure_from_blob(row) -> "object":
    from discriminate import read_blob_cif
    from pymatgen.core import Structure
    return Structure.from_str(read_blob_cif(int(row.blob_offset), int(row.blob_length)), fmt="cif")


def min_contact(structure) -> float:
    """Shortest interatomic distance under the minimum-image convention."""
    n = len(structure)
    if n < 2:
        return float(min(structure.lattice.abc))
    dm = structure.distance_matrix
    iu = np.triu_indices(n, 1)
    return float(min(dm[iu].min(), min(structure.lattice.abc)))


def usable_parent(structure) -> bool:
    """Reject cells that cannot be a physical parent, whatever the database says.

    Partial occupancies and split sites survive into some deposited CIFs as atoms sitting
    a few tenths of an angstrom apart. Those are a recording convention, not a structure,
    and DFT on them is meaningless.
    """
    return bool(structure.is_ordered) and min_contact(structure) >= MIN_PARENT_CONTACT_A


def rho_c(structure, val) -> float | None:
    """Reduced contact: min over cation-anion contacts of d / (r_cat + r_an), Shannon radii."""
    from pymatgen.core.periodic_table import Element
    cats = [i for i in range(len(structure)) if val[i] > 0]
    ans = [i for i in range(len(structure)) if val[i] < 0]
    if not cats or not ans:
        return None
    radii = {}
    for i in range(len(structure)):
        el = Element(structure[i].specie.symbol)
        try:
            r = el.average_ionic_radius
            if val[i] != 0:
                cn_r = el.ionic_radii.get(int(round(val[i])))
                if cn_r:
                    r = cn_r
        except Exception:
            r = None
        if not r:
            return None
        radii[i] = float(r)
    dm = structure.distance_matrix
    best = None
    for i in cats:
        for j in ans:
            d = dm[i][j]
            if d <= 0:
                continue
            v = d / (radii[i] + radii[j])
            if best is None or v < best:
                best = v
    return best


# ---------------------------------------------------------------- E1


def _e1_candidates():
    """The E1 selection, shared by the main package and its hard-potential control."""
    from discriminate import guess_oxi
    prov = load_provenance()
    pool = prov[(prov.split == "discovery") & (prov.n_sites <= E1_MAX_SITES)
                & (prov.n_elements >= 2)].sort_values("source_id")
    picked = []
    for anion in E1_ANIONS:
        got = 0
        for row in pool[pool.anion == anion].itertuples():
            if got >= E1_N_PER_ANION:
                break
            try:
                st = structure_from_blob(row)
            except Exception:
                continue
            if len(st) > E1_MAX_SITES or len(st) < 2 or not usable_parent(st):
                continue
            val, ok = guess_oxi(st)
            if not ok:
                continue
            r0 = rho_c(st, val)
            if r0 is None or not (0.75 <= r0 <= 1.30):
                continue
            picked.append((row.source_id, anion, st, float(r0)))
            got += 1
    return picked


def _e1_write(pkg: Path, picked, hard: bool):
    """Write one E1-style package; `hard` swaps in the small-core potentials."""
    metas, chosen = [], []
    for source_id, anion, st, r0 in picked:
        prefix = "E1b" if hard else "E1"
        name = f"{prefix}-{source_id}"
        tdir = pkg / "tasks" / name
        points, structures = [], []
        for k, target in enumerate(E1_RHO_GRID):
            s = st.copy()
            scale = target / r0
            s.lattice = type(s.lattice)(s.lattice.matrix * scale)
            structures.append((f"POSCAR.v{k:02d}", s))
            points.append({"index": k, "rho_target": target,
                           "scale": round(float(scale), 6)})
        stages = [{"stage": f"v{p['index']:02d}", "incar_stage": "static_soft",
                   "poscar": f"POSCAR.v{p['index']:02d}"} for p in points]
        # the radius sum the law divides by, recovered from the parent geometry, so the
        # analysis can move between the reduced coordinate and plain angstroms
        parent_contact = min_contact(st)
        meta = write_task(tdir, name, structures, stages,
                          {"experiment": "E1b" if hard else "E1", "source_id": source_id,
                           "anion": anion, "rho_c_parent": round(r0, 6),
                           "parent_min_contact_a": round(parent_contact, 4),
                           "radius_sum_a": round(parent_contact / r0, 4),
                           "potentials": "hard" if hard else "standard",
                           "points": points, "task_dir": str(tdir)},
                          hard=hard)
        meta["task_dir"] = str(tdir)
        metas.append(meta)
        chosen.append({"source_id": source_id, "anion": anion,
                       "formula": meta["formula"], "n_atoms": meta["n_atoms"],
                       "rho_c_parent": round(r0, 6),
                       "radius_sum_a": round(parent_contact / r0, 4),
                       "encut_ev": meta["encut_ev"]})
    return metas, chosen


def build_e1(out: Path) -> dict:
    pkg = out / "E1_rho_curve"
    pkg.mkdir(parents=True, exist_ok=True)
    metas, chosen = _e1_write(pkg, _e1_candidates(), hard=False)
    sel = {"experiment": "E1", "rule": (
        "discovery split, in_analysis_set, n_sites<=12, ordered with a shortest contact of "
        "at least 1.0 A, integer charge balance available, parent rho_c in [0.75,1.30]; "
        "first 4 per anion by source_id over O,S,F,Cl,N. Each parent is scaled rigidly and "
        "isotropically onto the frozen reduced-contact grid; static energies only."),
        "rho_grid": list(E1_RHO_GRID), "n_selected": len(chosen), "structures": chosen}
    return finish_package(pkg, "pris-e1", metas, sel, ntasks=16, hours=12, concurrent=20)


def build_e1b(out: Path) -> dict:
    """Hard-potential replicate of the most PAW-strained E1 compounds.

    Selection uses geometry alone, never a computed energy: the compounds are ranked by how
    far inside the sum of the PAW cutoff radii their shortest contact sits at the D1 floor,
    and the tightest eight are replicated.
    """
    pkg = out / "E1b_paw_control"
    pkg.mkdir(parents=True, exist_ok=True)
    picked = _e1_candidates()

    scored = []
    for source_id, anion, st, r0 in picked:
        s = st.copy()
        s.lattice = type(s.lattice)(s.lattice.matrix * (0.735 / r0))
        d = min_contact(s)
        radii = [potcar_meta(e.symbol)["rcore_a"] for e in s.composition.elements]
        radii = [r for r in radii if r]
        limit = sum(sorted(radii)[-2:]) if len(radii) >= 2 else (2 * radii[0] if radii else None)
        if not limit:
            continue
        scored.append((d / limit, source_id, anion, st, r0))
    scored.sort(key=lambda x: (x[0], x[1]))
    subset = [(sid, anion, st, r0) for _, sid, anion, st, r0 in scored[:E1B_N_COMPOUNDS]]

    metas, chosen = _e1_write(pkg, subset, hard=True)
    sel = {"experiment": "E1b", "rule": (
        f"The {E1B_N_COMPOUNDS} E1 compounds whose shortest contact at the D1 floor sits "
        "furthest inside the sum of the PAW cutoff radii, ranked on geometry alone and "
        "recomputed with the hard small-core potentials "
        f"({', '.join(f'{k}->{v}' for k, v in sorted(HARD_OVERRIDES.items()))}) at the "
        "cutoff those potentials require. Same grid, same cells, same k-mesh rule."),
        "rho_grid": list(E1_RHO_GRID),
        "paw_ratio_at_floor": [{"source_id": sid, "ratio": round(float(q), 4)}
                               for q, sid, _, _, _ in scored[:E1B_N_COMPOUNDS]],
        "n_selected": len(chosen), "structures": chosen}
    return finish_package(pkg, "pris-e1b", metas, sel, ntasks=16, hours=16, concurrent=20)


# ---------------------------------------------------------------- E2


def _multiset_permutations(items: list[str]):
    """Distinct permutations of a multiset, generated without materialising n! candidates."""
    items = sorted(items)
    n = len(items)
    used = [False] * n
    cur: list[str] = []

    def rec():
        if len(cur) == n:
            yield tuple(cur)
            return
        prev = None
        for i in range(n):
            if used[i] or items[i] == prev:
                continue
            prev = items[i]
            used[i] = True
            cur.append(items[i])
            yield from rec()
            cur.pop()
            used[i] = False

    return rec()


def _multiset_count(items: list[str]) -> int:
    from collections import Counter
    total = math.factorial(len(items))
    for c in Counter(items).values():
        total //= math.factorial(c)
    return total


def _site_permutations(structure, group_sites: list[int]):
    """Permutations of the group sites induced by the parent lattice's symmetry.

    The parent is the same structure with every group site carrying one dummy species, so
    its symmetry is exactly the symmetry the ordering breaks. Operations that map a group
    site outside the group are dropped rather than approximated.
    """
    from pymatgen.core import Structure
    from pymatgen.core.periodic_table import DummySpecies
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    base = structure.copy()
    for i in group_sites:
        base.replace(int(i), DummySpecies("X"))
    try:
        ds = SpacegroupAnalyzer(base, symprec=0.01).get_symmetry_dataset()
        rots = np.asarray(ds.rotations if hasattr(ds, "rotations") else ds["rotations"])
        trans = np.asarray(ds.translations if hasattr(ds, "translations") else ds["translations"])
    except Exception:
        return [tuple(range(len(group_sites)))]

    frac = np.asarray([structure[i].frac_coords for i in group_sites])
    perms = set()
    for R, t in zip(rots, trans):
        img = frac @ R.T + t
        perm = []
        ok = True
        for row in img:
            delta = np.abs(frac - row)
            delta -= np.round(delta)
            hit = np.flatnonzero((np.abs(delta) < 1e-3).all(axis=1))
            if len(hit) != 1:
                ok = False
                break
            perm.append(int(hit[0]))
        if ok and len(set(perm)) == len(perm):
            perms.add(tuple(perm))
    return sorted(perms) or [tuple(range(len(group_sites)))]


def orderings_for(structure, group_sites: list[int], group_species: list[str]):
    """All symmetry-distinct decorations of `group_sites` by the same species multiset.

    Returns None when the entry falls outside the frozen enumeration window, so that no
    ordering is ever sampled: an entry is either enumerated exhaustively or dropped.
    """
    if _multiset_count(group_species) > E2_ENUM_CAP:
        return None
    perms = _site_permutations(structure, group_sites)
    native = tuple(group_species)
    seen: set[tuple[str, ...]] = set()
    reps: list[tuple[str, ...]] = []
    for assign in _multiset_permutations(group_species):
        if assign in seen:
            continue
        orbit = {tuple(assign[p[k]] for k in range(len(assign))) for p in perms}
        orbit.add(assign)
        seen.update(orbit)
        # the released ordering represents its own class, so it is never canonicalised away
        reps.append(native if native in orbit else min(orbit))
        if len(reps) > E2_ORDERING_RANGE[1]:
            return None            # too many distinct orderings; excluded by the frozen rule
    reps.sort(key=lambda a: (a != native, a))
    out = []
    for assign in reps:
        s = structure.copy()
        for site_idx, sym in zip(group_sites, assign):
            s.replace(int(site_idx), sym)
        out.append((assign, s))
    return out


def pick_merge_group(structure, prefer: set[str] | None = None):
    """Largest set of sites carrying two or more elements of one merge class.

    `prefer` restricts the choice to named classes when any of them is present, so that a
    control can be matched to the mechanism the GNoME entries exercise instead of falling
    back to a chemically obvious pair elsewhere in the same structure.
    """
    by_class: dict[str, list[int]] = {}
    for i, site in enumerate(structure):
        by_class.setdefault(merge_class(str(site.specie.symbol)), []).append(i)
    candidates = []
    for cls, idx in by_class.items():
        elements = {str(structure[i].specie.symbol) for i in idx}
        if len(elements) < 2 or len(idx) > E2_MAX_GROUP_SITES:
            continue
        candidates.append((cls, idx))
    if prefer:
        preferred = [c for c in candidates if c[0] in prefer]
        if preferred:
            candidates = preferred
        else:
            return None
    if not candidates:
        return None
    return max(candidates, key=lambda c: (len(c[1]), c[0]))


def build_e2(out: Path) -> dict:
    from pymatgen.core import Structure
    from discriminate import guess_oxi
    pkg = out / "E2_ordering"
    pkg.mkdir(parents=True, exist_ok=True)

    metas, chosen, excluded = [], [], []

    def add_entry(kind: str, ident: str, st, prefer: set[str] | None = None) -> bool:
        grp = pick_merge_group(st, prefer)
        if grp is None:
            excluded.append({"id": ident, "kind": kind, "reason": "no mergeable pair"})
            return False
        cls, idx = grp
        species = [str(st[i].specie.symbol) for i in idx]
        res = orderings_for(st, idx, species)
        if res is None:
            excluded.append({"id": ident, "kind": kind, "reason": "more than 12 orderings"})
            return False
        if len(res) < E2_ORDERING_RANGE[0]:
            excluded.append({"id": ident, "kind": kind, "reason": "fewer than 2 orderings"})
            return False
        entry_orderings = []
        for k, (perm, s) in enumerate(res):
            name = f"E2-{ident}-o{k:02d}"
            tdir = pkg / "tasks" / name
            stages = [
                {"stage": "relax_cell", "incar_stage": "relax_cell", "poscar": "POSCAR.init"},
                {"stage": "static", "incar_stage": "static", "poscar": "POSCAR.static",
                 "from_contcar": "relax_cell"},
            ]
            is_native = list(perm) == species
            meta = write_task(tdir, name, [("POSCAR.init", s)], stages,
                              {"experiment": "E2", "kind": kind, "entry": ident,
                               "merge_class": cls, "group_sites": [int(i) for i in idx],
                               "ordering_index": k, "ordering": list(perm),
                               "is_released_ordering": bool(is_native),
                               "task_dir": str(tdir)})
            meta["task_dir"] = str(tdir)
            metas.append(meta)
            entry_orderings.append({"index": k, "ordering": list(perm),
                                    "is_released_ordering": bool(is_native)})
        chosen.append({"id": ident, "kind": kind, "merge_class": cls,
                       "n_group_sites": len(idx), "n_orderings": len(res),
                       "n_atoms": len(st), "formula": st.composition.reduced_formula,
                       "orderings": entry_orderings})
        return True

    # --- GNoME entries -----------------------------------------------------------
    mt = pd.read_parquet(MERGE_TEST)
    mt = mt[(mt.status == "ok") & mt.merged_any & (mt.econ_raw_01 > 2 / 3)]
    mt = mt.sort_values("material_id")
    zf = zipfile.ZipFile(GNOME_DIR / "by_id.zip")
    names = {Path(n).stem: n for n in zf.namelist() if n.endswith(".CIF")}
    got = 0
    for row in mt.itertuples():
        if got >= E2_N_GNOME:
            break
        member = names.get(row.material_id)
        if member is None:
            continue
        try:
            st = Structure.from_str(zf.read(member).decode(), fmt="cif").get_primitive_structure()
        except Exception:
            continue
        if len(st) > E2_MAX_SITES or len(st) < 2 or not usable_parent(st):
            continue
        if add_entry("gnome", row.material_id, st):
            got += 1

    # --- experimental controls, matched to the GNoME merge classes ----------------
    # An unmatched control would be too easy: merging Cs with Na costs a great deal of
    # energy for reasons that have nothing to do with artificial ordering. The controls
    # therefore carry the same classes the GNoME entries do, rare-earth pairs first.
    gnome_classes = {e["merge_class"] for e in chosen}
    prov = load_provenance()
    pool = prov[(prov.split == "discovery") & (prov.n_sites <= E2_MAX_SITES)
                & (prov.n_elements >= 2)].copy()
    pool["classes"] = [
        {merge_class(e) for e in els} for els in pool.elements
    ]
    got = 0
    control_formulas: set[str] = set()
    for prefer in ({"RE"}, gnome_classes):
        if got >= E2_N_CONTROL:
            break
        sub = pool[[bool(c & prefer) for c in pool.classes]].sort_values("source_id")
        for row in sub.itertuples():
            if got >= E2_N_CONTROL:
                break
            if any(c["id"] == row.source_id for c in chosen):
                continue
            try:
                st = structure_from_blob(row).get_primitive_structure()
            except Exception:
                continue
            if len(st) > E2_MAX_SITES or len(st) < 2 or not usable_parent(st):
                continue
            formula = st.composition.reduced_formula
            # one control per composition, so the control set is not one family repeated
            if formula in control_formulas:
                excluded.append({"id": row.source_id, "kind": "experimental",
                                 "reason": f"duplicate control composition {formula}"})
                continue
            _, ok = guess_oxi(st)
            if not ok:
                continue
            if add_entry("experimental", row.source_id, st, prefer):
                control_formulas.add(formula)
                got += 1

    sel = {"experiment": "E2", "rule": (
        "GNoME: merge_test status ok, merged_any, econ_raw_01>2/3, primitive cell <=20 atoms, "
        "largest merge class spanning <=10 sites, symmetry-distinct ordering count in [2,12]; "
        "taken in material_id order up to 25. Controls: discovery-split experimental "
        "structures under the same structural rules, restricted to merge class RE first and "
        "then to any class present in the selected GNoME set, taken in source_id order up to "
        "10. Every ordering of a selected entry is enumerated; none is sampled, so an entry "
        "is either exhaustive or excluded."),
        "gnome_target": E2_N_GNOME, "control_target": E2_N_CONTROL,
        "gnome_classes": sorted(gnome_classes),
        "n_entries": len(chosen), "entries": chosen, "excluded": excluded[:300],
        "n_excluded": len(excluded)}
    man = finish_package(pkg, "pris-e2", metas, sel, ntasks=32, hours=24, concurrent=24)
    return man


# ---------------------------------------------------------------- E3


def _has_noop_swap(variants: dict) -> bool:
    """True when an exchange damage left a crystal identical to its parent."""
    try:
        from pymatgen.core.structure_matcher import StructureMatcher
    except ImportError:
        from pymatgen.analysis.structure_matcher import StructureMatcher
    matcher = StructureMatcher(primitive_cell=False, attempt_supercell=False, scale=False,
                               ltol=0.01, stol=0.02, angle_tol=0.5)
    parent = variants.get("P0")
    for kind in ("S2", "S5"):          # the two damages that move no atom and no axis
        st = variants.get(kind)
        if st is not None and parent is not None and matcher.fit(st, parent):
            return True
    return False


def build_e3(out: Path) -> dict:
    from pymatgen.core import Structure
    from discriminate import guess_oxi
    from make_negatives import perturb
    pkg = out / "E3_crosscheck"
    pkg.mkdir(parents=True, exist_ok=True)
    prov = load_provenance()
    pool = prov[(prov.split == "discovery") & (prov.n_sites <= E3_MAX_SITES)
                & (prov.n_elements >= 2)].sort_values("source_id")

    metas, chosen = [], []
    # The ions-only stage mirrors the frozen-cell MatterSim protocol behind Fig. 5c,d, so
    # the two relaxation energies are measured on the same degrees of freedom. The cell is
    # then released, which is what a full first-principles relaxation would do anyway.
    stages = [
        {"stage": "relax_ions_fixed_cell", "incar_stage": "relax_ions_fixed_cell",
         "poscar": "POSCAR.init"},
        {"stage": "relax_cell", "incar_stage": "relax_cell", "poscar": "POSCAR.cell",
         "from_contcar": "relax_ions_fixed_cell"},
        {"stage": "static", "incar_stage": "static", "poscar": "POSCAR.static",
         "from_contcar": "relax_cell"},
    ]

    got = 0
    for row in pool.itertuples():
        if got >= E3_N_PARENTS:
            break
        try:
            st = structure_from_blob(row)
        except Exception:
            continue
        if len(st) > E3_MAX_SITES or len(st) < 2 or not usable_parent(st):
            continue
        val, ok = guess_oxi(st)
        if not ok:
            continue
        variants = {"P0": st}
        rng = stable_rng("E3", row.source_id)
        for kind in E3_DAMAGES:
            try:
                p = perturb(st, kind, rng, val)
            except Exception:
                p = None
            if p is not None:
                variants[kind] = p
        if len(variants) < 1 + len(E3_DAMAGES):
            continue                      # keep only parents where all five damages exist
        if _has_noop_swap(variants):
            # A cation swap does nothing when the two cation sublattices are related by a
            # lattice translation: the swapped crystal is the original, shifted. Such a
            # cell is not a damaged structure and would dilute the damage statistics.
            continue
        worst = min(min_contact(v) for v in variants.values())
        if worst < MIN_VARIANT_CONTACT_A:
            # the parent is dropped whole rather than in part, so every retained parent
            # contributes a complete and comparable six-cell set
            continue
        for label, s in variants.items():
            name = f"E3-{row.source_id}-{label}"
            tdir = pkg / "tasks" / name
            meta = write_task(tdir, name, [("POSCAR.init", s)], stages,
                              {"experiment": "E3", "kind": "experimental",
                               "parent": row.source_id, "variant": label,
                               "rng_seed_source": f"E3|{row.source_id}",
                               "task_dir": str(tdir)})
            meta["task_dir"] = str(tdir)
            metas.append(meta)
        chosen.append({"source_id": row.source_id, "formula": st.composition.reduced_formula,
                       "n_atoms": len(st), "variants": sorted(variants)})
        got += 1

    # GNoME parents, unmodified, for the MatterSim relaxation-energy comparison
    mt = pd.read_parquet(MERGE_TEST).sort_values("material_id")
    zf = zipfile.ZipFile(GNOME_DIR / "by_id.zip")
    names = {Path(n).stem: n for n in zf.namelist() if n.endswith(".CIF")}
    gnome_chosen = []
    got = 0
    for row in mt.itertuples():
        if got >= E3_N_GNOME_PARENTS:
            break
        member = names.get(row.material_id)
        if member is None:
            continue
        try:
            st = Structure.from_str(zf.read(member).decode(), fmt="cif").get_primitive_structure()
        except Exception:
            continue
        if len(st) > E3_MAX_SITES or len(st) < 2 or not usable_parent(st):
            continue
        name = f"E3-gnome-{row.material_id}"
        tdir = pkg / "tasks" / name
        meta = write_task(tdir, name, [("POSCAR.init", st)], stages,
                          {"experiment": "E3", "kind": "gnome", "parent": row.material_id,
                           "variant": "P0", "task_dir": str(tdir)})
        meta["task_dir"] = str(tdir)
        metas.append(meta)
        gnome_chosen.append({"material_id": row.material_id, "n_atoms": len(st),
                             "formula": st.composition.reduced_formula})
        got += 1

    sel = {"experiment": "E3", "rule": (
        "30 discovery-split experimental parents with <=16 sites for which all five damage "
        "operators S1-S5 of src/make_negatives.py apply, first by source_id, each with its "
        "five damaged variants; plus 20 unmodified GNoME parents <=16 atoms by material_id. "
        "Damage RNG seeded by sha256('E3|<source_id>'), not Python's salted hash(). "
        "A parent is also dropped when its S2 or S5 exchange leaves a crystal identical "
        "to the parent, which happens when the two cation sublattices are related by a "
        "lattice translation. "
        "Parents must be ordered with a shortest contact of at least 1.0 A, and a parent "
        "is dropped whole if any of its six cells falls below 0.9 A. Each cell relaxes "
        "ions at fixed cell first, matching the MatterSim protocol it is compared with, "
        "and only then releases the cell."),
        "n_experimental_parents": len(chosen), "experimental": chosen,
        "n_gnome_parents": len(gnome_chosen), "gnome": gnome_chosen,
        "damage_operators": list(E3_DAMAGES)}
    man = finish_package(pkg, "pris-e3", metas, sel, ntasks=32, hours=24, concurrent=24)
    return man


# ---------------------------------------------------------------- E4


def build_e4(out: Path) -> dict:
    from pymatgen.core import Structure
    pkg = out / "E4_design"
    pkg.mkdir(parents=True, exist_ok=True)
    d = pd.read_parquet(INVERSE_SCORES)
    score = d["synthesis_score"].to_numpy(float)
    bulk = d["clamped_bulk_modulus_proxy_gpa"].to_numpy(float)
    screened = score < PSS_CUTOFF
    priority = bulk >= PRIORITY_BULK_GPA
    assert int(screened.sum()) == 61 and int(priority.sum()) == 140, "frozen counts changed"

    rest = ~screened & ~priority
    rng = stable_rng("E4", "control")
    rest_idx = np.flatnonzero(rest)
    control_idx = np.sort(rng.choice(rest_idx, size=E4_N_CONTROL, replace=False))
    role = np.array(["none"] * len(d), dtype=object)
    role[rest] = "unused"
    role[control_idx] = "control"
    role[priority] = "priority"
    role[screened] = "screened"          # screened and priority are disjoint (verified above)

    shards: dict[str, zipfile.ZipFile] = {}
    metas, chosen = [], []
    stages = [
        {"stage": "relax_cell", "incar_stage": "relax_cell", "poscar": "POSCAR.init"},
        {"stage": "static", "incar_stage": "static", "poscar": "POSCAR.static",
         "from_contcar": "relax_cell"},
    ]
    for i in np.flatnonzero(role != "unused"):
        row = d.iloc[i]
        shard = str(row["source_shard"])
        if shard not in shards:
            zp = SHARD_ROOT / shard / "generated_crystals_cif.zip"
            shards[shard] = zipfile.ZipFile(zp)
        text = shards[shard].read(str(row["source_member"])).decode()
        if sha256_bytes(text.encode()) != row["cif_sha256"]:
            raise ValueError(f"CIF hash mismatch for {row['candidate_id']}")
        st = Structure.from_str(text, fmt="cif")
        name = f"E4-{row['candidate_id']}"
        tdir = pkg / "tasks" / name
        meta = write_task(tdir, name, [("POSCAR.init", st)], stages,
                          {"experiment": "E4", "candidate_id": str(row["candidate_id"]),
                           "role": str(role[i]),
                           "pss": float(row["synthesis_score"]),
                           "uma_bulk_modulus_gpa": float(row["clamped_bulk_modulus_proxy_gpa"]),
                           "rung_L4_verdict": str(row["rung_L4_verdict"]),
                           "cif_sha256": str(row["cif_sha256"]),
                           "ev_volume_factors": list(E4_EV_VOLUME_FACTORS),
                           "task_dir": str(tdir)},
                          extra_incars=("relax_ions",))
        meta["task_dir"] = str(tdir)
        metas.append(meta)
        chosen.append({"candidate_id": str(row["candidate_id"]), "role": str(role[i]),
                       "formula": meta["formula"], "n_atoms": meta["n_atoms"],
                       "pss": float(row["synthesis_score"]),
                       "uma_bulk_modulus_gpa": float(row["clamped_bulk_modulus_proxy_gpa"])})

    sel = {"experiment": "E4", "rule": (
        "All 61 PSS-screened candidates (synthesis_score < %.16f), all 140 UMA-priority "
        "candidates (proxy bulk modulus >= %.0f GPa), and 60 controls drawn without "
        "replacement from the remaining 880 with seed sha256('E4|control'). Stage A relaxes "
        "each cell; stage B builds the five-point energy-volume curve from the relaxed cell."
        % (PSS_CUTOFF, PRIORITY_BULK_GPA)),
        "pss_cutoff": PSS_CUTOFF, "priority_bulk_gpa": PRIORITY_BULK_GPA,
        "ev_volume_factors": list(E4_EV_VOLUME_FACTORS),
        "counts": {r: int((role == r).sum()) for r in ("screened", "priority", "control")},
        "n_selected": len(chosen), "structures": chosen}
    # stage B is generated on the cluster from stage A's relaxed cells, so its generator
    # travels inside the package and is covered by the manifest
    import shutil
    shutil.copy2(Path(__file__).resolve().parent / "make_stage_b.py",
                 pkg / "make_stage_b.py")
    os.chmod(pkg / "make_stage_b.py", 0o755)
    man = finish_package(pkg, "pris-e4a", metas, sel, ntasks=32, hours=24, concurrent=32)
    return man


# ---------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "dft"))
    ap.add_argument("--only", default="E1,E1b,E2,E3,E4")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    wanted = [w.strip() for w in a.only.split(",") if w.strip()]

    builders = {"E1": build_e1, "E1b": build_e1b, "E2": build_e2,
                "E3": build_e3, "E4": build_e4}
    summary = {}
    for key in wanted:
        print(f"==> building {key}", flush=True)
        man = builders[key](out)
        summary[key] = {k: man[k] for k in
                        ("package", "n_tasks", "n_vasp_runs", "total_atoms")}
        print(f"    {summary[key]}", flush=True)

    (out / "BUILD_SUMMARY.json").write_text(
        json.dumps({"protocol": PROTOCOL_ID,
                    "builder_sha256": sha256_file(Path(__file__)),
                    "packages": summary}, indent=1, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
