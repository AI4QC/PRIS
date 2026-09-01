#!/usr/bin/env python3
"""MatterSim relaxation energies for the E3 cells, on the protocol the manuscript used.

E3 asks whether MatterSim's ranking of damage severity survives first principles. That
comparison is only meaningful if both sides relax the same degrees of freedom, so this
script reproduces the protocol behind Fig. 5c,d exactly: MatterSim v1.0.0 5M, ions only
with the cell frozen, fmax < 0.05 eV/A, hard cap 200 steps
(`external_sources/mlip_groundtruth/protocol.md`). The matching DFT stage is
`relax_ions_fixed_cell`.

Run locally, in an environment that has mattersim (this repository's `newpauling` env):

    python dft/mattersim_reference.py

Writes `E3_crosscheck/mattersim_reference.json`, which `analyze.py` pairs with the DFT
results by task name.
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = "<path>"
FMAX = 0.05          # eV per A, as in the manuscript's relaxation protocol
MAX_STEPS = 200      # hard cap, as in the manuscript's relaxation protocol


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", default=str(HERE / "E3_crosscheck"))
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    pkg = Path(a.package)

    import numpy as np
    import torch
    from ase.io import read as ase_read
    from ase.optimize import FIRE
    from mattersim.forcefield import MatterSimCalculator

    device = a.device if torch.cuda.is_available() else "cpu"
    calc = MatterSimCalculator(load_path=a.checkpoint, device=device)

    records = []
    tasks = sorted(p for p in (pkg / "tasks").iterdir() if p.is_dir())
    for i, task in enumerate(tasks, 1):
        meta = json.loads((task / "TASK.json").read_text())
        atoms = ase_read(str(task / "POSCAR.init"), format="vasp")
        atoms.calc = calc
        n = len(atoms)
        try:
            e_initial = float(atoms.get_potential_energy())
            f0 = float(np.linalg.norm(atoms.get_forces(), axis=1).max())
            # the cell is frozen: FIRE moves ions only, exactly as the manuscript's run did
            opt = FIRE(atoms, logfile=None)
            opt.run(fmax=FMAX, steps=MAX_STEPS)
            e_final = float(atoms.get_potential_energy())
            f1 = float(np.linalg.norm(atoms.get_forces(), axis=1).max())
            steps = int(opt.get_number_of_steps())
            records.append({
                "task": meta["task"], "parent": meta.get("parent"),
                "kind": meta.get("kind"), "variant": meta.get("variant"),
                "formula": meta.get("formula"), "n_atoms": n,
                "energy_initial_ev": e_initial, "energy_final_ev": e_final,
                "release_ev_per_atom": (e_initial - e_final) / n,
                "fmax_init_ev_per_a": f0, "fmax_final_ev_per_a": f1,
                "n_steps": steps, "converged": bool(f1 < FMAX and steps < MAX_STEPS),
                "error": None,
            })
        except Exception as exc:
            records.append({"task": meta["task"], "variant": meta.get("variant"),
                            "release_ev_per_atom": None, "error": str(exc)})
        if i % 25 == 0:
            print(f"  {i}/{len(tasks)}", flush=True)

    ok = [r for r in records if r.get("release_ev_per_atom") is not None]
    out = {
        "protocol": "mattersim-v1.0.0-5M, ions only with the cell frozen, "
                    f"fmax<{FMAX} eV/A, cap {MAX_STEPS} steps",
        "checkpoint": a.checkpoint,
        "device": device,
        "n_tasks": len(records),
        "n_evaluated": len(ok),
        "records": records,
    }
    (pkg / "mattersim_reference.json").write_text(json.dumps(out, indent=1) + "\n")
    print(f"{len(ok)} of {len(records)} cells relaxed -> "
          f"{pkg / 'mattersim_reference.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
