# PRIS first-principles verification (E1–E4)

This document covers three things: **why these four calculations are being run**, **what each one
actually computes**, and **how to get the data out once they finish**.

Operational detail (how to submit, how to resume, what to do about pseudopotentials) is in
`README.md`; reaching the cluster and getting past the local proxy is in `CONNECT-zh.md`;
predictions, criteria and failure handling are in `PREREG-DFT.md`, which was frozen and git-tagged
before the first job was submitted.

---

## 1. Background: why run these calculations

The central claim of the paper is that "structural plausibility is an independent layer of
judgement, and it belongs ahead of thermodynamic stability, kinetic stability and
synthesizability". PRIS itself depends on no model — the eight laws use only formal charges, ionic
radii and symmetry. But **every yardstick currently used to validate it or compare against it is a
learned one**:

| evidence in the paper | current source | what a referee will ask |
|---|---|---|
| energy released on relaxation (Fig. 5c,d: 0.001 / 0.22 / 6.05 eV per atom) | MatterSim | an ML potential is out of distribution on implausible structures; why trust it |
| E_hull comparison (72.0%) | MatterSim | same |
| the 140 high-property inverse-design candidates, 67.3% queue reduction | UMA surrogate | the bulk moduli were never actually computed |
| phonons (35.6%, 41.7%, 4,271) | published MP data | those are filtered database structures, not generated ones |
| the physical basis for the D1 threshold 0.735 (Fig. 3c) | Born–Mayer model potential | that is a toy model, not first principles |
| Counterexample 2: artificial ordering in GNoME | merge statistics on relabelled entries | that is a statistical argument, not a thermodynamic one |

A paper arguing that "first-principles verification should be ordered" has not itself run a single
first-principles calculation. At any journal above NCS this will be asked. Each of these four
calculations replaces one of those yardsticks.

Two more points worth stating up front:

**These calculations may overturn numbers in the paper.** E4 above all: if any of the 61 candidates
that PSS screens out turns out to be a DFT-confirmed high-modulus structure, the claim "all 140
retained" does not hold. Precisely for that reason the sampling rules, estimators and thresholds
must be fixed in writing before anything is computed. That is what `PREREG-DFT.md` is for.

**This is an entirely new chain of evidence.** The two sealed evaluations of 2026-08-14 have
already been spent; nothing here touches them, and nothing here consumes the remaining lockbox
openings. The experimental structures for E1/E2/E3 all come from the discovery split, so the
held-out set stays clean.

---

## 2. Contents: what each of the five packages computes

Four experiments, one of which (E1) carries a control package E1b. The cells are all small, at most
20 atoms; the 617 stage-A tasks come to 6,861 atoms in total.

### E1 — is the contact ratio ρ_c a transferable coordinate (20 tasks / 340 single points)

**The objection being answered**: D1 divides the contact distance by the sum of the ionic radii and
claims that one threshold then works across chemistries, which a cutoff fixed in ångström cannot do
(the paper's own wording, section 2.2). Do first principles support that?

**How it is computed**: 20 experimental parent structures from the discovery split (≤12 atoms, four
each of O/S/F/Cl/N), each **rigidly and isotropically scaled** over ρ_c ∈ [0.60, 1.40] in steps of
0.05, 17 points in all. Rigidity is the point — PRIS sees exactly the coordinates it is handed and
performs no internal relaxation. Static single points only, with the k-grid fixed at the density of
the most compressed point, so there is no grid jump anywhere along the curve.

**Estimator**: for each compound, `ρ*` is the contact ratio at which the energy cost of compression
first reaches 0.1 eV per atom, and `d* = ρ* × (r_cat + r_an)` is that same crossing converted to
ångström. Dispersion is the interquartile range over the median.

**Prediction**: across the 20 compounds the relative dispersion of `ρ*` is at least 1.5× smaller
than that of `d*`, and the median `ρ*` falls in [0.70, 1.00], bracketing the two D1 lower bounds. If
that fails, the contact ratio is no more transferable than the bare distance, and the
first-principles basis for "divide by the radius sum" in section 2.2 does not exist.

**Why not "the energy at the lower bound exceeds 0.5 eV per atom"**: that claim cannot fail —
ρ_c = 0.735 means compressing the volume to 40%, which costs any compound several eV per atom. And
at that degree of compression the shortest contact is only 0.66–0.94× the sum of the PAW cutoff
radii (median 0.77), where the frozen-core error is biased towards repulsion, in the same direction
as the prediction. With the crossing point instead, the main estimator lands near ρ_c ≈ 0.9, where
the PAW spheres are just touching, and the conclusion no longer rests on the most compressed end.
The excess energies at 0.735 / 0.804 / 1.05 are still reported, now with the overlap ratio at that
point alongside.

**This is the cheapest of the five**, a few hours to a result. Submit it first, and use it to
confirm that the VASP invocation, module name and pseudopotential paths on the cluster are all
correct.

### E1b — hard-pseudopotential control (8 tasks / 136 single points)

**The objection being answered**: compression pushes atoms inside each other's PAW core radii,
where the frozen core is least reliable, and the error is biased towards repulsion — exactly the
direction E1 is trying to demonstrate.

**How it is computed**: from E1's 20 compounds, take the 8 tightest by a geometric measure (the
shortest contact at the D1 lower bound relative to the sum of the PAW radii) and recompute them
with small-core hard pseudopotentials (O_h, F_h, N_h, S_h, Cl_h, B_h, C_h, P_h, H_h, Ga_h, Ge_h),
raising the cutoff to 910–1005 eV accordingly. The cells, grids and k-point rules are all
unchanged; only the pseudopotentials differ — `verify.py` compares the E1b and E1 POSCARs byte for
byte to confirm it.

**Prediction**: the hard pseudopotentials change the excess energy at ρ_c = 0.735 by less than 25%.
If the change is larger, the compressed end is a frozen-core artefact and all of E1 switches to the
hard-pseudopotential numbers.

### E2 — is GNoME's low-symmetry excess thermodynamically real (128 tasks / 256 runs)

**The objection being answered**: the paper attributes part of GNoME's low-symmetry excess to
"artificial ordering of chemically similar elements", on the evidence that symmetry rises once the
labels are merged. That is a statistical argument. DFT can turn it into a thermodynamic one: if the
ordering is artificial, other arrangements of the same composition should be near-degenerate, so at
any temperature at which the material could be synthesised it is a solid solution rather than a new
compound.

**How it is computed**: 24 GNoME entries that fail D7 and contain a mergeable pair of similar
elements (primitive cell ≤20 atoms, ≤10 sites in the merged class, 2–12 symmetry-inequivalent
arrangements). **Every arrangement of every entry is computed, with no sampling** — either
enumerate exhaustively or exclude the entry, so that the "lowest-energy arrangement" cannot be
overestimated by sampling. Symmetry inequivalence is determined by the orbits of the uncoloured
parent's symmetry operations acting on the site assignment. Ten experimental controls are selected
under the same structural rules, restricted to the RE class (matching the class that dominates the
GNoME set).

**Estimators**: ordering energy `ΔE_i = E(published arrangement) − min_j E(arrangement j)` (eV per
atom, after full-cell relaxation); disordering temperature `T_od = ΔE / (k_B · ΔS)`, with
`ΔS = −Σ x_a ln x_a` the configurational entropy per mixed site.

**Prediction**: the median `T_od` of the 24 GNoME entries is below 300 K and that of the 10 controls
is above 1000 K; at least 60% of the GNoME entries fall below 300 K and at most 20% of the controls
do.

**Declared limitation**: 9 of the 10 controls belong to the single structural family YbREX₃.
Genuinely ordered compounds containing two similar elements are rare to begin with, which is itself
part of the argument, but the control set is not diverse and this will be stated plainly.

### E3 — do the five controlled damage classes look like damage to DFT too (200 tasks / 600 runs)

**The objection being answered**: the severity of the five damage classes is currently calibrated
by MatterSim, and an ML potential is least reliable on exactly this kind of structure. S5
(cation–anion swap) is the sharpest case: it moves no coordinate at all, so a potential that takes
local geometry as input may simply not see it.

**How it is computed**: 30 experimental parent structures from the discovery split (≤16 sites, and
all five operators of `src/make_negatives.py` applicable), each with five damaged variants, plus 20
unmodified GNoME parents, for 200 cells. A parent must be ordered with a shortest contact ≥1.0 Å,
and the whole group is discarded if any of its six cells drops below 0.9 Å — DFT gives nothing
meaningful on such a cell. Each cell is first relaxed **with the cell fixed and only the ions
moving** (`ISIF=2`), then with the cell released (`ISIF=3`), and finally static.

The fixed-cell step is deliberate: the MatterSim relaxations behind Fig. 5c,d freeze the cell
(fmax < 0.05 eV/Å, capped at 200 steps), and the two sides only mean anything if they are compared
over the same degrees of freedom. This was noticed while writing the extraction script and has been
recorded as Amendment 1 under the pre-registration's revision procedure.

The random numbers used for damage are seeded by `sha256("E3|<source_id>")` rather than Python's
per-process salted `hash()`, so the variants are reproducible.

**One more thing this turned up**: when two cation sublattices differ by exactly one translation,
the crystal produced by an S2 swap is the original translated — nothing has been damaged. One of
the 30 parents (ZrCoF₆) is such a case; it has been filtered out in the generator, and `verify.py`
checks that none slip through. This is a property of the paper's own S2 operator. Such a cell would
be counted as "damage not detected" and can only make PRIS look worse, not better, so the published
numbers are not at risk.

**Estimator**: `ΔE_relax = E(first ionic step) − E(last ionic step)`, taken over the fixed-cell
stage.

**Prediction**: across the 200 cells the rank correlation (Spearman) between DFT and MatterSim is at
least 0.7; the median DFT `ΔE_relax` for S5 exceeds that of the undamaged parents by at least
0.3 eV per atom; and the S5/parent ratio is larger under DFT than under MatterSim.

### E4 — does the inverse-design screen survive first principles (261 tasks / 522 runs + about 1,305 stage-B tasks)

**The objection being answered**: the 67.3% queue reduction and "all 140 high-property candidates
retained" rest entirely on the UMA surrogate. Not one candidate has been relaxed and not one bulk
modulus has been computed from first principles.

**How it is computed**: all 61 candidates screened out by PSS (`synthesis_score` below the frozen
threshold −0.6368790173149083), all 140 that UMA puts at ≥400 GPa, and 60 controls drawn without
replacement from the remaining 880 by `sha256("E4|control")` — 261 in all.

- **Stage A**: relax each cell fully (this is precisely what "the DFT verification queue" means in
  the paper).
- **Stage B**: hold the relaxed cell at five volume factors 0.94 / 0.97 / 1.00 / 1.03 / 1.06,
  relaxing ions and cell shape at each (`ISIF=4`), to get an energy–volume curve; a third-order
  Birch–Murnaghan fit gives the bulk modulus.

The stage-B job package is generated on the cluster by `make_stage_b.py` from the stage-A results —
it uses only the standard library, so the cluster does not need pymatgen. **The k-grid is
re-derived from the relaxed cell** (taken at the smallest volume on the curve and shared by all five
points), because generated cells usually shrink on relaxation and reusing the stage-A grid would
undersample.

**Prediction**: at most 2 of the 61 screened-out candidates reach a DFT bulk modulus ≥400 GPa; at
least 70% of the 140 prioritised candidates are confirmed by DFT at ≥400 GPa; the Pearson r between
UMA and DFT bulk moduli over the 261 is at least 0.8.

---

## 3. How to get the data out afterwards

Three steps: **collect (run on the cluster, standard library only) → mattersim_reference (local, for
E3 only) → analyze (local, writes RESULTS.md)**.

### Step 1: extract the raw results on the cluster

```bash
cd ~/pris-dft
python3 collect.py                                  # all four packages together
python3 collect.py --package E4_design --stage-b    # E4's stage B separately
```

`collect.py` reads only what VASP actually wrote and **makes no scientific judgement** — fitting,
symmetry analysis and comparison against the pre-registration all live in `analyze.py`. It uses only
the standard library, no numpy and no pymatgen, so it is certain to run on the cluster.

Each package writes two files:

- `collected.json` — the full record: per-ionic-step energies for every stage, final CONTCAR text,
  convergence flags, core hours
- `collected.csv` — a flat table for reading directly

**Every task carries a `status`**, so nothing unfinished is dropped silently; the reason is recorded:

| status | meaning | what happens next |
|---|---|---|
| `complete` | every stage exited 0 and produced an energy | enters the analysis |
| `unconverged` | some relaxation hit the ionic-step cap but still has an energy | enters the analysis, flagged |
| `failed` | some stage exited non-zero, or wrote no energy | excluded, reason recorded in the report |
| `incomplete` | some stage never ran at all | excluded, reason recorded in the report |

The main fields of `collected.csv`:

| field | meaning |
|---|---|
| `<stage>.energy_ev` / `.energy_ev_per_atom` | `energy(sigma->0)` at that stage's last ionic step |
| `<stage>.release_ev_per_atom` | (first-step energy − last-step energy) / atoms for that stage, i.e. the energy released on relaxation |
| `<stage>.volume_a3` | cell volume at the end of that stage |
| `<stage>.n_ionic_steps` / `.converged` | number of ionic steps; whether VASP itself reported "reached required accuracy" |
| `<stage>.elapsed_s` | wall-clock time for that stage (multiply by the core count for core hours) |

At the end it prints a one-line summary of measured core hours, which can be used to calibrate the
queue parameters for the remaining batches.

### Step 2: the MatterSim reference for E3 (local, needs a GPU)

E3 has to be compared against MatterSim, so MatterSim must be run over the **same cells** under the
**same protocol**:

```bash
python dft/mattersim_reference.py
```

The protocol matches Fig. 5c,d of the paper: MatterSim v1.0.0 **5M**, fixed cell with only the ions
moving, fmax < 0.05 eV/Å, capped at 200 steps. It writes
`E3_crosscheck/mattersim_reference.json`, which `analyze.py` pairs with the DFT results by task
name. Skip this step and E3's other numbers still come out; only the rank-correlation entry will
read "MatterSim reference missing".

### Step 3: analyse locally and reach the conclusions

Sync `collected.json` back from the cluster (the whole tree is not needed — `collected.json` is
enough unless you want to look at the structures again):

```bash
rsync -am --include='*/' --include='collected.json' --include='collected.csv' --exclude='*' \
      user@cluster:~/pris-dft/ dft/
python dft/analyze.py                                      # writes dft/RESULTS.md
python dft/analyze.py --only E2                            # run one package alone
```

Every estimator and threshold in `analyze.py` is copied from `PREREG-DFT.md`; the script executes,
it does not decide. It writes `RESULTS.md`, which opens with a prediction check table:

```
| experiment | prediction | measured | met |
| E2 | GNoME median T_od below 300 K | 180 K | yes |
| E4 | at most 2 screened candidates reach 400 GPa | 5 | **no** |
```

It also writes intermediate results that can be plotted directly:

| file | contents |
|---|---|
| `E1_rho_curve/curves.json` | per-compound ρ_c–energy curve, `ρ*` and `d*`, the excess at 0.735 / 0.804 / 1.05, and the position of the minimum |
| `E1b_paw_control/paw_shift.json` | the excess for the 8 compounds under standard and hard pseudopotentials, and the relative difference |
| `E2_ordering/ordering_energies.json` | per-entry ordering energy, configurational entropy, `T_od`, relaxed space group and site economy |
| `E4_design/bulk_moduli.json` | per-candidate DFT bulk modulus, V0, B′, fit residual, and the paired UMA value and PSS |

E3's per-variant median table is written straight into `RESULTS.md`.

**What to do when a threshold is not a grid point**: D1's 0.735 and 0.804 are not on the 0.05 grid,
so the script reads them off the computed points with shape-preserving cubic interpolation (PCHIP),
which avoids the overshoot an ordinary spline gives on a steep repulsive wall. The upper bound 1.05
is a grid point and is read directly.

### The extraction chain has already been verified

Real data is still weeks away, so the extraction chain was verified first against **synthetic VASP
output**:

```bash
python dft/selftest.py
```

It copies the real task directories (metadata, stage chaining and sampling records are all genuine)
and swaps only OUTCAR and CONTCAR for synthetic content with known answers, then runs
`collect.py → make_stage_b.py → analyze.py` end to end and asserts: E1's curve readings and `ρ*`
land near the analytic values; the E1b-vs-E1 relative difference equals the injected 10%; E2's
ordering energies and `T_od` computed from static energies match the hand calculation; E3's damage
table and rank correlation take shape; and E4's Birch–Murnaghan fit recovers the injected bulk
modulus.

Current status: **passing**, with 0.00% recovery error on the bulk moduli of E4's two candidates.

Writing this self-test turned up a real bug in `analyze.py` — it used to index the energy at 0.735
straight off the grid key, and since 0.735 is not on the grid it raised `KeyError`. Finding that
only after a real run would have cost another round trip.

---

## 4. Submission plan: three batches, each with a checkpoint

Do not throw all 617 tasks up at once. The point of batching is not to save core hours, it is **to
let the first mistake destroy 28 tasks instead of 617** — the module name, pseudopotential path,
partition name and parallelisation on the cluster can each be wrong, and any one of them will make
the whole batch fail the same way.

### Before submitting, confirm five things on the cluster

| to confirm | how | where to fix it |
|---|---|---|
| partition and account | `sinfo`, `sacctmgr show user $USER` | add `#SBATCH --partition=...` / `--account=...` at the top of each `submit.slurm` |
| how VASP is invoked | `module avail vasp`; `which vasp_std` | `module load vasp/6.3.0` and `VASP_BIN` in `submit.slurm` |
| where the pseudopotential library is | `ls $VASP_PP_PATH/O/POTCAR` | `VASP_PP_PATH` in `submit.slurm`; if the library is absent, package and upload it as `README.md` describes |
| cores per node | `sinfo -o "%P %c"` | `#SBATCH --ntasks-per-node` (currently 16 or 32) |
| queue time limit | `sinfo -o "%P %l"` | `#SBATCH --time` (currently 12–24 hours) |

Once the pseudopotential library is confirmed, run the check again on the cluster so it hashes the
cluster's **own** library:

```bash
python3 verify.py --potcar-lib $VASP_PP_PATH --report /tmp/verify_cluster.md
```

A hash mismatch is not a minor thing — it means the cluster's pseudopotentials are a different
version from the local set, and the physics will differ. Either change the library or regenerate the
task packages on the cluster.

### Batch 1: E1 + E1b (28 tasks, a few hours)

```bash
cd ~/pris-dft/E1_rho_curve  && sbatch submit.slurm
cd ~/pris-dft/E1b_paw_control && sbatch submit.slurm
```

**Why these two first**: cheapest, fastest and structurally simplest (static single points only, no
relaxation, no stage chaining), and between them they exercise the whole chain — pseudopotential
assembly, k-points, cutoff, driver script, collection script. E1b additionally confirms that the
high cutoff (910–1005 eV) runs on the cluster at all.

**What to check when they come back**:

```bash
python3 collect.py --package E1_rho_curve --package E1b_paw_control
```

- The `status` distribution should be all `complete`. If anything is `failed`, start from that
  stage's `vasp.err`.
- The printed core hours divided by the task count give the calibrated cost per single point; use it
  to re-estimate the queue times and `--time` for E2/E3/E4.
- The `close_ions_warning` column in `collected.csv`: `True` at E1's most compressed v00/v01 is
  **expected**, and is exactly why E1b exists. If it also appears near v08 (ρ_c ≈ 1.0), some parent
  structure is wrong to begin with — come back and say so.
- Pick one task and check `ENCUT` and `NKPTS` at the top of `v00/OUTCAR` against `TASK.json`.

Once this batch passes, the cluster-side configuration is right and the rest can be submitted in
bulk with confidence.

### Batch 2: E2 + E3 (328 tasks, 1–3 days)

```bash
cd ~/pris-dft/E2_ordering   && sbatch submit.slurm
cd ~/pris-dft/E3_crosscheck && sbatch submit.slurm
```

These introduce relaxation and stage chaining, so adjust `--time` using the costs measured in
batch 1. E3 has three stages per task and is the slowest.

**When they come back**: a few `unconverged` entries are normal (a relaxation hit the ionic-step
cap); they still enter the analysis but are flagged. If `failed` exceeds 10%, do not press on —
first find out which damage class is failing.

E3 also needs one **local** MatterSim reference run (section 3, step 2), which does not involve the
cluster.

### Batch 3: E4 (261 + about 1,305 tasks, 3–10 days)

```bash
cd ~/pris-dft/E4_design
sbatch submit.slurm                                       # stage A
# wait for all of stage A to finish, then generate stage B
python3 make_stage_b.py && sbatch stage_b/submit.slurm
```

**Stage A must genuinely finish before stage B is generated** — every stage-B cell and k-grid is
derived from a stage-A `CONTCAR`. Candidates that did not converge go into `stage_b/SKIPPED.json`
rather than being dropped silently; check how long that file is before going further.

E4 is the one experiment that could overturn a number in the paper, and it is also the most
expensive, so it goes last: its result carries weight only once the first two batches have shown the
protocol is sound.

### What to do when something fails (pre-registration section 1.3; no improvising)

- A stage exits non-zero → that task stops there and later stages do not run. Record it as it is.
- **Only a wall-clock timeout** may be rerun, once, with **identical input** and a longer `--time`.
  Reruns go in the log.
- Nothing else may be retried with changed parameters. A number that only came out after switching
  `ALGO` or loosening `EDIFF` is not the same quantity as the others.
- Still not converged after a rerun → report it as unconverged, exclude it from the estimator, and
  state the reason and the count.
- Tasks are **resumable**: stages that already exited 0 are skipped, so after a timeout you can
  resubmit the same array directly.

### Core-hour budget

| batch | tasks | rough core hours | on 256 cores |
|---|---|---|---|
| 1: E1 + E1b | 28 | 800 – 3,200 | a few hours |
| 2: E2 + E3 | 328 | 6,500 – 26,000 | 1–4 days |
| 3: E4 A + B | 261 + ~1,305 | 18,000 – 70,000 | 3–11 days |
| total | | **25,000 – 100,000** | |

The intervals are wide because a simple oxide and a spin-polarised f-electron metal can differ by a
factor of three or four. The first batch will narrow them.

### Pitfalls already cleared out of the way

This package was audited twice before delivery, the second pass specifically to avoid wasting core
hours. Found and fixed: E1's original prediction could not fail and its evidence was
self-serving (estimator replaced, and the E1b hard-pseudopotential control added); one E3 parent's
S2 swap damaged nothing, and some damaged cells were as short as 0.63 Å; E4's stage B would have
reused the unrelaxed cell's k-grid, and the pure-Python matrix inverse had the transpose backwards,
which would have given all 1,305 jobs the wrong grid; and the driver script assumed Python was
available on the compute nodes. Details and the new hashes are in Amendment 2 of `PREREG-DFT.md`.

Before submitting, run these two locally once more; both must come back clean:

```bash
python dft/verify.py     # should report 0 errors
python dft/selftest.py   # should report selftest passed
```

---

## 5. What goes into the paper afterwards

The paper currently has 5 main figures and NMI/NCS allow at most 6 display items, so one more fits.
The suggestion is a combined **Fig. 6, "First-principles verification"**:

- **a** E1: ρ_c–energy curves for a few representative compounds, marking D1's two lower bounds and
  D2's upper bound (this can also replace the Born–Mayer curve in Fig. 3c); an inset comparing the
  dispersion of `ρ*` and `d*`, which is the direct evidence that transferability comes from dividing
  by the radius sum
- **b** E2: the `T_od` distributions for GNoME and the experimental controls, log x-axis, with a
  room-temperature line
- **c** E3: DFT against MatterSim relaxation energy released, coloured by damage class, with S5
  marked out
- **d** E4: DFT against UMA bulk modulus, coloured as screened / retained / prioritised, with a
  400 GPa line

In the text: E1 goes into section 2.3 (the five mechanisms) and the Fig. 3c caption; E2 goes into
the GNoME paragraph of section 2.4, upgrading "symmetry rises once labels are merged" to "the
ordering energy is below the configurational entropy, so it is a solid solution at any synthesis
temperature"; E3 goes into Methods and the SI, closing off the objection that all energetic evidence
comes from an ML potential; E4 goes into the inverse-design paragraph at the end of section 2.4,
replacing the UMA surrogate with DFT or reporting the two side by side.

**If a prediction is not met**: follow section 3 of `PREREG-DFT.md`, report it as it is, and change
the corresponding claim in the paper to the DFT conclusion — do not quietly drop the experiment. Any
change at the level of the numbers will be written up for you to decide on first.
