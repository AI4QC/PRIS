# PRIS first-principles campaign — E1 to E4

VASP task packages that replace four of the manuscript's learned yardsticks with
first-principles calculations. What each experiment tests, what it predicts and how it may
fail is fixed in `PREREG-DFT.md`, written before any job was submitted.

Start with `GUIDE-zh.md` for the scientific background, what each experiment computes and
how the results come out; this file covers operations. `CONNECT-zh.md` covers getting onto
the cluster from this machine, where a local proxy has to be bypassed per connection.

```
dft/
  GUIDE-zh.md          background, contents and the data-extraction workflow
  CONNECT-zh.md        reaching the cluster past the local proxy; what is installed there
  PREREG-DFT.md        frozen predictions, protocol and decision rules
  build_tasks.py       deterministic builder for all four packages
  verify.py            pre-submission checks; writes VERIFY_REPORT.md
  collect.py           extracts finished runs into collected.json/csv (cluster, stdlib only)
  make_stage_b.py      builds E4's energy-volume tasks from stage A (copied into E4_design)
  mattersim_reference.py  the MatterSim side of E3, on the manuscript's own protocol
  analyze.py           evaluates the pre-registered predictions; writes RESULTS.md
  selftest.py          runs the whole extraction chain on synthetic VASP output
  VERIFY_REPORT.md     the current result: 0 errors, 6 warnings
  BUILD_SUMMARY.json   task counts and the builder hash
  E1_rho_curve/        20 tasks     340 VASP runs
  E1b_paw_control/      8 tasks     136 VASP runs   hard-potential control for E1
  E2_ordering/        128 tasks     256 VASP runs
  E3_crosscheck/      200 tasks     600 VASP runs
  E4_design/          261 tasks     522 VASP runs  (+ ~1,305 stage-B tasks on the cluster)
```

Each package holds `tasks/<name>/` with `POSCAR.*`, the `INCAR.*` its stages name,
`KPOINTS`, `POTCAR.spec.json`, `POTCAR.list`, `stages.json`, `stages.tsv`, `run.sh` and
`TASK.json`; plus
`selection.json` (which structures, chosen by which rule), `MANIFEST.json` (SHA256 of every
file), `tasklist.txt` and `submit.slurm`.

## POTCAR

**No POTCAR content is shipped.** VASP potentials are licensed, so each task carries only
`POTCAR.spec.json`: the element order, the `MPRelaxSet` symbol, the title line, ENMAX and
the SHA256 of the file it expects. `run.sh` concatenates the POTCAR at run time from
`$VASP_PP_PATH` and deletes it when the stage finishes. The driver is pure shell, so a
compute node needs nothing but VASP itself.

The build used the local library at `<path>` (329 potentials, PBE PAW).
Every element needed is present. If the cluster has its own PBE PAW library, point
`VASP_PP_PATH` at it — `verify.py` will confirm the hashes still match, and a mismatch is
an error rather than a silent difference in the physics.

If the cluster has no library, ship one:

```bash
python3 - <<'PY'
import json, tarfile, pathlib
need = set()
for spec in pathlib.Path("dft").rglob("POTCAR.spec.json"):
    for e in json.loads(spec.read_text())["order"]:
        need.add(e["library_relative"])
with tarfile.open("potcar_subset.tar.gz", "w:gz") as t:
    for rel in sorted(need):
        t.add(f"<path>", arcname=rel)
print(len(need), "potentials")
PY
```

## Running

```bash
rsync -a dft/ user@cluster:~/pris-dft/          # ~32 MB, no licensed content
ssh user@cluster
cd ~/pris-dft/E1_rho_curve
# set --partition/--account, and VASP_BIN / VASP_PP_PATH if the defaults are wrong
sbatch submit.slurm
```

`submit.slurm` is a Slurm array, one array element per task, throttled with `%N`. It loads
`vasp/6.3.0` if a module of that name exists and otherwise expects `vasp_std` on `PATH`.
Defaults it sets: `VASP_BIN="srun -n $SLURM_NTASKS vasp_std"`,
`VASP_PP_PATH="$HOME/POTCAR/PBE"`.

Before submitting anything, re-run the checks against the cluster's own PAW library, so a
different set of potentials shows up as a hash mismatch rather than as different physics:

```bash
python3 verify.py --potcar-lib $VASP_PP_PATH --report /tmp/verify_cluster.md
```

**Submit in three batches, not all at once** — the point is that a misconfigured module
name or POTCAR path ruins 28 tasks instead of 617:

1. `E1_rho_curve` + `E1b_paw_control` (28 tasks, hours). Static points only, no chaining, so
   it exercises the whole path cheaply and calibrates the per-point cost.
2. `E2_ordering` + `E3_crosscheck` (328 tasks, 1–4 days). Relaxation and stage chaining.
3. `E4_design` stage A, then `make_stage_b.py`, then stage B (261 + ~1,305 tasks).

`GUIDE-zh.md` section 4 lists what to check when each batch returns, and the failure rules
fixed in `PREREG-DFT.md`: only a wall-clock timeout may be re-run, once, with identical
inputs. Nothing else is retried with different settings.

Tasks are **resumable**: a stage that already exited zero is skipped, so re-submitting the
array after a timeout continues where it stopped. A stage that fails stops its task, and
later stages never run on a broken cell — verified by smoke test.

E4 runs in two stages:

```bash
cd ~/pris-dft/E4_design
sbatch submit.slurm                             # stage A: relax 261 candidates
python3 make_stage_b.py && sbatch stage_b/submit.slurm   # stage B: 5-point E-V curves
```

`make_stage_b.py` uses the standard library only — it does not need pymatgen on the
cluster. Candidates whose stage A did not converge are listed in `stage_b/SKIPPED.json`
rather than dropped silently.

## Size

Cells are small: at most 20 atoms, 6,861 atoms across all 617 stage-A tasks.

| package | tasks | VASP runs | median k-points | rough core-hours |
|---|---|---|---|---|
| E1_rho_curve | 20 | 340 | 667 | 100 – 500 |
| E1b_paw_control | 8 | 136 | 667 | 700 – 2,700 |
| E2_ordering | 128 | 256 | 140 | 2,500 – 10,000 |
| E3_crosscheck | 200 | 600 | 175 | 4,000 – 16,000 |
| E4_design stage A | 261 | 522 | 196 | 5,000 – 20,000 |
| E4_design stage B | ~1,305 | ~2,610 | 196 | 13,000 – 50,000 |
| **total** | | | | **25,000 – 100,000** |

E1b costs more per point than E1 despite being smaller: the hard potentials need 910–1005 eV
rather than 520.

The bands assume 15–60 core-hours for a full cell relaxation of a 10–20 atom cell at these
settings, which spans the difference between a simple oxide and a spin-polarised f-electron
metal. On 256 cores the campaign is 4–15 days; E1 alone finishes in hours. k-point counts
are the full mesh — `ISYM=2` folds them in the static stages, often by a large factor.

## Verifying

```bash
python dft/verify.py
```

Checks POSCAR parsing, species order against the POTCAR spec, atom counts against
`TASK.json`, shortest interatomic contacts, the k-mesh against the KSPACING rule, ENCUT
against the frozen rule for the species actually present, POTCAR hashes against the local
library, that no licensed content is inside the package, every manifest hash, and that the
Slurm array size equals the number of tasks. It also re-derives the E1 scaling grid, that
each E2 entry contains its released ordering exactly once and that all its orderings share
one composition, that every E3 parent carries all six cells at constant composition and
atom count, and that E4's roles reproduce the frozen 61 screened and 140 priority.

It also reports where PAW cutoff spheres overlap tightly. Overlap by itself is meaningless
(Ir–Ir in ordinary iridium sits at 0.91 of the radius sum), so only cells below 0.75 are
listed and only below 0.40 is an error.

Current state: **0 errors, 23 warnings** — the warnings are E1's two most compressed grid
points, which compress on purpose and are covered by the E1b control, and a handful of S3
random-displacement cells, which is the damage operator doing its job.

## Collecting results

```bash
python3 collect.py                                   # on the cluster, standard library only
python3 collect.py --package E4_design --stage-b
rsync -am --include='*/' --include='collected.json' --include='collected.csv' --exclude='*' \
      user@cluster:~/pris-dft/ dft/
python dft/analyze.py                                # locally; writes RESULTS.md
```

`GUIDE-zh.md` section 3 documents every field and the status codes. `selftest.py` exercises
the chain on synthetic VASP output with known answers and currently passes, including a
Birch-Murnaghan fit that recovers an injected bulk modulus to 0.00%.

## Rebuilding

```bash
python dft/build_tasks.py            # all four
python dft/build_tasks.py --only E2  # one
```

Selection is a pure function of the input stores and the rules in the builder, so a rebuild
reproduces the packages. Changing a selection rule after `PREREG-DFT.md` was tagged means
amending that file, not editing it.
