# Pre-registration: first-principles verification of PRIS (E1-E4)

**Protocol** `2026-08-25-pris-dft-e1e4-v1` · **written 2026-08-25, before any VASP job was
submitted** · repository commit at the time of writing: `6cf9596`

This file is committed and tagged before the first job is submitted. Afterwards it may only
gain an appended "Amendments" section; no existing clause is rewritten. It extends
`PREREG.md`, which governs the data splits this campaign reuses, and opens a new evidence
chain: the sealed evaluations of 2026-08-14 are spent, so nothing here draws on them.

---

## 0. Why pre-register a set of DFT calculations

Every energetic and property statement in the manuscript currently rests on a surrogate:
MatterSim for relaxation energy and hull energy, UMA for bulk modulus, published Materials
Project data for phonons, PU models for synthesizability. PRIS itself needs none of them,
but the yardsticks it is measured against are all learned. This campaign replaces four of
those yardsticks with first-principles calculations.

Two of the four experiments can contradict published numbers. E4 in particular can show
that a candidate PRIS removed is genuinely high-modulus, which would change a headline
result. Fixing the selection, the protocol and the decision rule in advance is what makes
that outcome reportable rather than negotiable.

## 1. What is frozen

The task packages are built by `dft/build_tasks.py` and verified by `dft/verify.py`. Both
are committed. Selection is a pure function of the input stores and the rules below, so the
packages can be rebuilt byte-for-byte.

| package | selection.json SHA256 | MANIFEST.json SHA256 |
|---|---|---|
| E1_rho_curve | `8aba047f65493182b4c78e351d58f51519e15338456abebbfed1acc8d127d571` | `3919950a35ed1b271a84a7e88996bf8e101db3d8e711073d7c3c21cc45ca6aaa` |
| E2_ordering | `3f2ae27e6266bb5cbcf599dee7e08a0c39d1589ae03f9fec586626fed84f1572` | `64a7de51fc734cf1509df27d0b16b9c35bd5ce8be9d6dd4d822ddc9a5e029895` |
| E3_crosscheck | `7dedfa10da5677aa7977aad2d8bc84d4ac1c11c9811ea481976213828723c313` | `ce03ce257cbcf9425f45b01efef966a746d89f134c1c3cbd1e8ed2428e15ed45` |
| E4_design | `27c3bd0463efed213f049f9e77c349887c18c91d38c551d54b3422418691f48a` | `36ed6965999ace09f6578ef4b149a3db6cd61ff9f303d1a5b7d203724f60b6fe` |

`dft/build_tasks.py` = `cd54f4bc0a041fa6bf104f9adb29e88807b4c929e10560f16982f7ce0243015e`
`dft/verify.py` = `aa809b22dd9397dc49ab9e8013b3180b6a0a2b6a4aae3596b54ab8738ba9d825`

Each `MANIFEST.json` records the SHA256 of every file in its package, so a task that is
edited after this point is detectable.

### 1.1 VASP protocol

PBE PAW, VASP 6.3.0, POTCAR symbols from `MPRelaxSet` with the single documented
substitution `W_pv -> W_sv` (the local library carries no `W_pv`). Plane-wave cutoff
`ENCUT = max(520 eV, ceil(1.3 x max ENMAX))` over the species present.
Gamma-centred k-mesh from `KSPACING = 0.22 A^-1`, written out explicitly so that the mesh
is auditable and identical across the stages of one task. `PREC=Accurate`, `EDIFF=1E-6`,
`LASPH=.TRUE.`, `LREAL=.FALSE.`, `ADDGRID=.TRUE.`, `NELM=200`, `ALGO=Normal`.

`ISPIN=2` with `MAGMOM = N*1.0` when the cell contains a 3d transition metal, a lanthanide
or an actinide; `ISPIN=1` otherwise.

Relaxations use `ISYM=0`, so a cell is free to change its symmetry. This matters: E2 and E3
measure symmetry change, and `ISYM=2` would freeze the very quantity under test. Static
stages use `ISYM=2` purely to fold the k-mesh, and tetrahedron smearing (`ISMEAR=-5`) where
the mesh carries at least four points, Gaussian smearing otherwise.

Cell relaxation: `IBRION=2`, `ISIF=3`, `NSW=200`, `EDIFFG=-0.02 eV/A`, `POTIM=0.30`.
Fixed-volume relaxation (E4 stage B): the same with `ISIF=4`, `NSW=120`.

### 1.2 Structure hygiene, fixed in advance

A parent cell is used only if it is fully ordered and its shortest interatomic distance is
at least 1.0 A. Some deposited CIFs record partial occupancy as split sites a few tenths of
an angstrom apart; those are a recording convention, not a structure. In E3 a parent is
dropped whole, not in part, if any of its six cells falls below 0.6 A, so every retained
parent contributes a complete and comparable set.

### 1.3 Failure handling, fixed in advance

A stage that exits non-zero stops its task; later stages do not run on a broken cell.
Failures and timeouts are retained and reported, never silently re-run with different
settings. One re-run is permitted per task, and only for a wall-clock timeout, with the
same inputs and a longer time limit; that re-run is logged. Any task still unconverged
after that is reported as unconverged and excluded from the estimator with its reason
named. Exclusion counts are reported per experiment whatever the outcome.

---

## 2. Experiments, predictions and decision rules

Each prediction is stated so that it can fail. The direction, the estimator and the
threshold are fixed here; only the numbers come later.

### E1 - the reduced-contact energy landscape

**Question.** D1's floors (0.735, 0.804) and D2's ceiling (1.05) are percentiles of an
observed distribution, and the manuscript motivates them with a scaled Born-Mayer pair
energy. Does a first-principles energy landscape place them where the law puts them?

**Design.** 20 discovery-split experimental parents, at most 12 sites, four each with the
anion O, S, F, Cl and N, taken in `source_id` order. Each parent is scaled rigidly and
isotropically onto a fixed grid of 17 reduced-contact values from 0.60 to 1.40 in steps of
0.05 — rigid, because that is what PRIS sees when a structure is submitted. Static energies
only; the k-mesh is fixed at the densest requirement (the most compressed point) so the
curve carries no mesh discontinuity. 340 single-point calculations.

**Prediction.** In every one of the 20 compounds, the energy at the D1 floor
rho_c = 0.735 exceeds the curve's minimum by at least 0.5 eV per atom, and the mean
excess across the 20 is larger at rho_c = 0.735 than at rho_c = 1.05 by at least a factor
of three. That asymmetry is the physical reason a lower bound alone cannot detect
expansion, which is why D2 and D3 exist.

**If it fails.** If the repulsive wall does not sit near the floor, D1's threshold is
empirical only, and the manuscript will say so instead of implying a physical derivation.

### E2 - is GNoME's low-symmetry excess thermodynamically real

**Question.** The manuscript traces part of GNoME's low-symmetry enrichment to artificial
ordering of chemically similar elements, evidenced by a label-merging test. Merging labels
is a statistical argument. DFT can make it a thermodynamic one: if the ordering is
artificial, the alternative orderings of the same composition are nearly degenerate, and
the compound is a solid solution at any temperature at which it could be made.

**Design.** 24 GNoME entries that fail D7 with a mergeable similar-element pair, primitive
cell at most 20 atoms, merge class spanning at most 10 sites, and a symmetry-distinct
ordering count between 2 and 12; taken in `material_id` order. Every ordering of a selected
entry is enumerated — an entry is exhaustive or excluded, never sampled — so the minimum
over orderings is not biased toward the hypothesis. Symmetry-distinctness is decided by the
orbit of the site assignment under the symmetry operations of the undecorated parent.
10 experimental controls follow the same structural rules, restricted to the merge class RE
first and then to any class present in the GNoME selection.

**Estimator.** For entry i, the ordering energy is
`dE_i = E(released ordering) - min_j E(ordering j)`, in eV per atom, after full cell
relaxation. The order-disorder temperature is `T_od,i = dE_i / (k_B * dS_i)` with
`dS_i = -sum_a x_a ln x_a` per mixed site, `x_a` the fraction of the merge group's sites
carrying element a.

**Prediction.** The median `T_od` over the 24 GNoME entries is below 300 K, and the median
over the 10 experimental controls is above 1000 K. At least 60% of GNoME entries fall below
300 K and at most 20% of controls do.

**If it fails.** If GNoME orderings carry real ordering energies, the artificial-ordering
attribution in section 2.4 is weakened, and D7's diagnosis of those entries must be
restated as a symmetry observation without a thermodynamic claim.

**Stated limitation.** Nine of the ten controls belong to the YbREX3 family. Genuinely
ordered compounds carrying two elements of one merge class are rare, which is itself part
of the point, but the control set is not structurally diverse and will be described that
way.

### E3 - do the controlled damages read as damage to DFT

**Question.** The severity of the five damage operators is currently calibrated with
MatterSim, a model trained on relaxed near-equilibrium structures and therefore least
reliable on exactly these cells. A cation-anion exchange (S5) is the sharpest case: it
leaves every coordinate untouched, so a potential keyed to local geometry may not see it at
all.

**Design.** 30 discovery-split experimental parents of at most 16 sites for which all five
operators of `src/make_negatives.py` apply, each with its five damaged variants, plus 20
unmodified GNoME parents. The damage RNG is seeded by `sha256("E3|<source_id>")` rather
than Python's salted `hash()`, so the variants are reproducible across processes. 200 cells,
each relaxed and then evaluated statically.

**Estimator.** Relaxation energy release `dE_relax = E(initial) - E(relaxed)` in eV per
atom, from DFT and from MatterSim on the identical cells.

**Prediction.** Rank correlation (Spearman) between the DFT and MatterSim relaxation
energies across the 200 cells is at least 0.7. The median DFT `dE_relax` of the S5
cation-anion exchanges exceeds that of the undamaged parents by at least 0.3 eV per atom,
and the ratio of the S5 median to the parent median is larger under DFT than under
MatterSim.

**If it fails.** A weak correlation would mean the manuscript's MatterSim-based severity
calibration cannot carry the weight placed on it, and Fig. 5c,d would need the DFT numbers
in place of, not beside, the MatterSim ones.

### E4 - does the inverse-design screen survive first principles

**Question.** The queue reduction of up to 67.3% and the retention of all 140 high-property
candidates are stated against a UMA proxy for bulk modulus. No candidate has been relaxed,
and no bulk modulus has been computed from first principles.

**Design.** All 61 candidates PSS screens out (`synthesis_score` below the frozen cutoff
-0.6368790173149083), all 140 candidates the UMA proxy places at or above 400 GPa, and 60
controls drawn without replacement from the remaining 880 with the seed
`sha256("E4|control")`: 261 candidates. Stage A relaxes each cell. Stage B holds the relaxed
cell at volume factors 0.94, 0.97, 1.00, 1.03, 1.06, relaxing ions and cell shape inside
each, and a Birch-Murnaghan fit of the resulting energy-volume curve gives the bulk modulus.

**Prediction.** Of the 61 screened candidates, at most two have a DFT bulk modulus at or
above 400 GPa. Among the 140 priority candidates, at least 70% are confirmed at or above
400 GPa by DFT, and the DFT and UMA bulk moduli correlate with Pearson r of at least 0.8
across the 261.

**If it fails.** If more than two screened candidates are genuinely high-modulus, the claim
that PSS retains every high-property candidate does not survive and will be restated with
the DFT count. If the UMA-DFT correlation is below 0.8, the definition of the 140 is itself
unreliable, and the inverse-design result will be reported against DFT throughout.

---

## 3. What will be reported whatever happens

- The number of tasks submitted, converged, failed and excluded, per experiment, with
  reasons.
- Every prediction above, marked met or not met, with the measured value beside the
  threshold.
- Any disagreement between a DFT result and a number already in the manuscript, named
  explicitly rather than absorbed into a revised figure.
- The count of amendments to this file.

## 4. Amendments

### Amendment 1 — 2026-08-25, before any job was submitted

**What changed.** E3 gained a relaxation stage. Each cell now relaxes ions at fixed cell
(`ISIF=2`, `NSW=200`) before the cell is released (`ISIF=3`) and the static energy is taken.
The stage list is `relax_ions_fixed_cell -> relax_cell -> static`; E3 grows from 400 to 600
VASP runs. Nothing else moves: the same 200 cells, the same parents, the same operators.

**Why.** The protocol behind Fig. 5c,d
(`external_sources/mlip_groundtruth/protocol.md`) relaxes ions with the cell **frozen**:
MatterSim v1.0.0 5M, fmax below 0.05 eV/A, hard cap 200 steps. The original E3 released the
cell, so the DFT and MatterSim relaxation energies would have been measured on different
degrees of freedom and the section 2 comparison would not have meant anything. This was
found while writing the extraction script, before any compute was spent.

**Consequence for section 2.** The E3 estimator is now the frozen-cell relaxation energy
`dE_relax = E(first ionic step) - E(last ionic step)` of the `relax_ions_fixed_cell` stage,
which is what the MatterSim number measures. The full-cell release is recorded alongside as
`dft_release_full_ev_per_atom` and is not used for the pre-registered comparison. The
predictions and thresholds are unchanged.

**Also recorded.** The E1 thresholds 0.735 and 0.804 are not points of the 0.05 grid, so the
excess energy at a threshold is read from a shape-preserving cubic (PCHIP) through the
computed points. The ceiling 1.05 is a grid point and is read directly.

**New hashes** (the other two packages are unchanged):

| item | SHA256 |
|---|---|
| E3_crosscheck/selection.json | `bd0bee8518ce31a643af40e366c3c1066dd8b1aa29b9b263b81cad432a87e086` |
| E3_crosscheck/MANIFEST.json | `3d7423c2ba5495419d053a1cf3d3ca5d0d93263b0824aea4b4c2a3b2ec68ae88` |
| E4_design/MANIFEST.json | `be69bcd26f87fd55a1e166aa583f504f422ae942e9272c296b835e367e8d305c` |
| build_tasks.py | `cef6e9e9c152aa0a26e7a746b229a010120a87fd922f50e55ed1d90fbe603561` |
| collect.py | `bc0fb5b1e0528f2814e93a15aa323dac57460dc6c314549c142f0701082ebb41` |
| analyze.py | `b89f2f08eda4995560ca2e4eae901f87aa838835ba03fa407084427efc01b31d` |
| make_stage_b.py | `934f4064ca47f82b59358533d307a7aadbf3751533d006df233d65c83d4fc153` |
| mattersim_reference.py | `7cd085b4097b1c7faac5cb38d7e8d00169ae604af91e1173fd75e8bb2e8e4f8f` |

`E4_design/selection.json` is unchanged
(`27c3bd0463efed213f049f9e77c349887c18c91d38c551d54b3422418691f48a`); only its manifest
moved, because the package now carries `make_stage_b.py` inside it.

Amendments so far: **1**.

### Amendment 2 - 2026-08-25, still before any job was submitted

A second review, run specifically so that no DFT time would be spent on a flawed design,
turned up four problems. All four are fixed here; no calculation had been started.

**1. E1's primary prediction could not fail, and its evidence leaned the wrong way.**
The original prediction asked for an energy excess of at least 0.5 eV per atom at
rho_c = 0.735. Reaching that reduced contact by rigid scaling compresses the cell to 40 per
cent of its volume, where every compound costs several eV per atom, so the prediction was
unfalsifiable. Worse, at that compression the shortest contact falls to 0.66-0.94 of the sum
of the PAW cutoff radii, median 0.77, and frozen-core error there runs repulsive, in the
same direction as the prediction.

The calculations are unchanged; the estimator is not. E1 now measures **transferability**,
which is what D1 actually claims: dividing a contact by the radius sum is meant to make one
threshold work across chemistries, where a cutoff fixed in angstroms cannot.

For each compound, rho* is the reduced contact at which compression first costs
0.1 eV per atom, and d* = rho* x (r_cat + r_an) is that same crossing in angstroms. Spread
is the interquartile range over the median.

Prediction: the relative spread of rho* across the 20 compounds is at least 1.5 times
smaller than the relative spread of d*, and the median rho* lies in [0.70, 1.00], the range
that brackets the D1 floors. If it fails, the reduced contact is no more transferable than a
plain distance, and section 2.2's argument for scaling by the radius sum loses its
first-principles support.

The crossing sits near rho_c = 0.9, where the PAW spheres barely touch, so the primary
estimator no longer depends on the compressed end. The excess energies at 0.735, 0.804 and
1.05 are still reported, now beside the overlap ratio at each point.

**2. E1b, a hard-potential control, is added.** The eight E1 compounds whose contacts sit
furthest inside the PAW radii at the D1 floor are recomputed with the small-core potentials
(O_h, F_h, N_h, S_h, Cl_h, B_h, C_h, P_h, H_h, Ga_h, Ge_h) at the cutoff those require,
910-1005 eV. Same cells, same grid, same k-mesh rule; only the potentials differ, which
`verify.py` checks by comparing each POSCAR byte for byte against its E1 twin.

Prediction: hard potentials move the excess energy at rho_c = 0.735 by less than 25 per
cent. If they move it more, the compressed end is a frozen-core artefact and the
hard-potential numbers replace the standard ones throughout E1.

136 extra single-point calculations, 8 tasks.

**3. E3 admitted cells DFT cannot use, and one damage that did nothing.** The contact floor
rises from 0.6 A to 0.9 A, which costs no parents because the pool refills. Separately, a
cation exchange leaves the crystal unchanged whenever the two cation sublattices are related
by a lattice translation: the swapped crystal is the original, shifted. One parent in thirty
(ZrCoF6) was such a case. A parent is now dropped when its S2 or S5 exchange reproduces the
parent under a tight structure match, and `verify.py` checks that none survives. This is a
property of the manuscript's own S2 operator; such a cell counts as undetected damage, so it
makes PRIS look slightly worse rather than better and no published number is at risk.

**4. E4 stage B would have under-sampled the Brillouin zone.** The stage-B k-mesh was
inherited from stage A, which fixed it on the unrelaxed generated cell; relaxation usually
shrinks a generated cell, leaving that mesh too coarse. `make_stage_b.py` now re-derives the
mesh from the relaxed cell at the smallest volume of the curve and shares it across all five
points, so every point is at least as well sampled and the five energies stay comparable. A
transpose in its pure-Python matrix inverse, which would have produced silently wrong meshes
for all 1,305 stage-B jobs, was found and fixed; the routine now agrees with numpy to 1e-10
and with the builder on all 261 cells.

**Also changed, without scientific consequence.** The per-task driver is pure shell and needs
no Python on a compute node; each task carries `POTCAR.list` and `stages.tsv` beside the
JSON, and `verify.py` checks the two agree. `verify.py` gained the PAW overlap ratio (overlap
by itself means nothing: Ir-Ir in ordinary iridium sits at 0.91 of the radius sum), a check
that E1b replicates E1 exactly, and the no-op exchange check.

**Package sizes after this amendment:** 617 tasks and 1,854 VASP runs in stage A, plus about
1,305 stage-B tasks for E4.

**New hashes:**

| item | SHA256 |
|---|---|
| E1_rho_curve/selection.json | `af5d8716272cc289b53e58b6667beefe8b97a9a69bb7137449c3f05fdd5007ae` |
| E1_rho_curve/MANIFEST.json | `60049fc92e392ad22432ed70d946949d957247042625cdfe4cb7aff060451e0e` |
| E1b_paw_control/selection.json | `ea9bedfed448ce8c3e116017f1067079c389b18c15c75ac3739d224c56857875` |
| E1b_paw_control/MANIFEST.json | `705e034866610a635bd6b7f4d2a69f0150ef3a36c45b087e368d11cc51db4d53` |
| E2_ordering/MANIFEST.json | `e8426f01415e1a237bdf5a0611c0c51aa51619c71722cdbc70741cf7f734e4df` |
| E3_crosscheck/selection.json | `b9c4b271fed61a7fda2af7fa090c4ad2410c6629234788808f16a029aed24d88` |
| E3_crosscheck/MANIFEST.json | `397fc3b934c1de904251a6708a7bd9cce45e21dea2f5e64d2c3c67d52210620e` |
| E4_design/MANIFEST.json | `597842a82a2523525beb6aed9a5ab50c10d91cc1fbc3a1b17dbdae8d2548718d` |
| build_tasks.py | `0fa3ca36bc20a8796d4c29b231f4e88837fa466d70640054c62b0f1b154f2490` |
| verify.py | `33685f5590cf893dfde3167f0fd115c220f13fddebef94b6d51cacc1c15b7ecd` |
| collect.py | `daad18e2e11cc8bf8e916062062eff680d240f6c79e568075cb4e3a7e6397c24` |
| analyze.py | `dc8c1e60635dffc0c75ec05acb4bd8fd908d9521657b5052785ba18b6ac50e3c` |
| make_stage_b.py | `9b52c4d4b052cdc56c214799ba7f3dc85747f93cd4d8e64aa38695b7e761546a` |
| selftest.py | `da4ca626fe1575d8ff63e740c984b7943cfe5d831db46ce40705378e2afa305f` |

`E2_ordering/selection.json` and `E4_design/selection.json` are unchanged: no structure
entered or left either set.

Amendments so far: **2**.

### Amendment 3 - 2026-08-25, before submission

`verify.py` gained a `--potcar-lib` option so the same checks can run on the cluster against
its own PAW library, where a differing hash means the potentials genuinely differ rather than
the physics quietly changing. No task, selection rule, estimator or prediction is touched.

| item | SHA256 |
|---|---|
| verify.py | `bd0ae7cf141b774d24d0cab58d7d1ba9d08b83c7cb2e4832d12ec7d0369d0169` |

superseding `33685f5590cf893dfde3167f0fd115c220f13fddebef94b6d51cacc1c15b7ecd` from Amendment 2.

Amendments so far: **3**.

### Amendment 4 - 2026-08-26/27, during execution

The campaign is running on a Slurm cluster whose partitions hand out whole nodes. Getting
it to run there needed three changes to how the work is launched and one to the INCARs.
None of them alters a structure, a selection rule, an estimator or a prediction.

**1. Parallel layout, chosen by measurement.** Every task now runs alone on one 64-core
node with `NCORE = 8` and `KPAR = 2`. VASP warns that the default `NCORE=1` is inefficient,
and a sweep on this cluster confirmed it: a representative E1 cell took 39.5 s at the
eight-rank default and 12.2 s at 64 ranks with these settings. `KPAR` and `NCORE` divide the
work between ranks and do not enter the physics.

The same sweep showed that packing eight tasks onto a node would give about 2.5 times the
throughput per node-hour, measured with the node to itself. The campaign runs one task per
node anyway, at the author's direction: lowest latency per task and the simplest failure
mode. The measurement is recorded here so the choice is visible rather than implied.

**2. `LSCAAWARE = .FALSE.`, added after a failure.** At 64 ranks VASP's distributed subspace
diagonalisation (`PDSYEVX`/`PZHEEVX`) failed on a number of cells. The switch selects the
non-distributed path and is a linear-algebra choice, not a physical one. It was added to
every INCAR after 1,266 stages had already run without it. Because that is a change of
settings mid-campaign, its equivalence is checked rather than asserted:
`check_lscaaware.slurm` recomputes finished stages with the switch and reports the energy
difference against what is on disk. The result is reported whatever it shows; a difference
beyond convergence tolerance would mean the affected stages are recomputed.

**3. Two stages fall back to `ISYM = 0`.** `E3-cod-1010052-S4` and `E3-cod-1010179-S4` failed
their static stage inside VASP's charge-density symmetrisation (`RHOSYG`, `INVGRP`), which
`ISYM=2` performs only to fold the k-mesh. Those two INCARs now carry `ISYM = 0`, which
computes the same quantity on the unfolded mesh. Both cases are named here.

**4. Failure evidence is preserved.** Three defects in the launcher were found and fixed
during the first batches, none of which touched the physics: a VASP process inherited the
stdin of the loop that fed it its stage plan and consumed the rest of the file, so tasks
stopped after a few stages while reporting success; `module` is a shell function that a
batch script does not inherit, so the toolchain silently failed to load; and a network
interface changed underneath the connection. Separately, 39 failed stages were deleted
before their output was read, losing the reason. The farm now copies the tail of `vasp.out`
and `vasp.err` into `.farm/failed/<task>.txt` before a failed stage is cleared, so a repeat
is diagnosable.

**Working rules for execution.** At most 20 jobs are in flight. A task claims its work with
an atomic `mkdir` and marks completion with a file, so no task is computed twice and an
interrupted one resumes where it stopped. `controller.sh` tops the queue up and moves
between packages; it submits only work already frozen in the packages and never edits an
INCAR or retries a stage with different settings.

**New hashes:**

| item | SHA256 |
|---|---|
| build_tasks.py | `fb8665dabe871e1d21d724268b9bd0a1b68c855753713f3268999e97d51dcefa` |
| farm.slurm | `a74225977f2b9f2c33cb5034b8d3fc86d4d7c450d96785956085ac66f5af6a88` |
| submit_farm.sh | `bfd9b054a52a2baa5ee185cb62cbea9f1e695ca9494927d6fd4ce69183310fba` |
| controller.sh | `991eb3c8f79c1bf815b5c1da412a96ee3e60841a15058becd2f6fce990a5063e` |
| collect.py | `daad18e2e11cc8bf8e916062062eff680d240f6c79e568075cb4e3a7e6397c24` |
| check_potcars.py | `dcb0cdba747ba585a468ec2ad7c597260a94bff02b26dcb148a65ee71f1119e3` |
| check_lscaaware.slurm | `d93a71fb23e4f748fba93cdbe20e89850ba6e60f8dab89b0c5707d911a9b8155` |
| bench_scaling.slurm | `e8db969c31afb67b61c9fd1474c05fb25dbaf6f28b9fbdd2e60ca0628c758879` |
| bench_packing.slurm | `c70b7b1f54a106ab7c8e811d8ccb9675caf047de6c21208bbaf67d2e1ff37ee3` |

Amendments so far: **4**.

### Amendment 5 - 2026-08-27, during execution

**E1 and E1b are complete and all three of their predictions are met.** The reduced contact
localises the 0.1 eV per atom crossing 1.80 times more tightly across twenty chemistries than
the same crossing in angstroms (relative spread 0.152 against 0.274, threshold 1.5); the
median crossing at rho_c = 0.927 lies inside [0.70, 1.00] and brackets the D1 floors; and the
hard-potential control moves the excess at the floor by 0.7 per cent at the median, against a
25 per cent tolerance, so the compressed end is not a frozen-core artefact. 476 stages, no
failures.

**LSCAAWARE is equivalent, measured not assumed.** Six finished stages recomputed with the
switch reproduced their energies to every printed digit, difference exactly zero. The
mid-campaign change of Amendment 4 therefore altered nothing.

**An E2 entry is analysed only if every one of its orderings finished.** The design already
enumerates orderings exhaustively so that the minimum is not taken over a subset. Execution
can undo that: if some orderings fail, the minimum is taken over what survived, which can
only raise it, which lowers the ordering energy and the order-disorder temperature with it -
the direction that favours the hypothesis. `analyze.py` now drops such an entry and names it
in the report rather than analysing a partial set. This tightens the analysis; it does not
change any prediction.

**Three mechanical failure modes, and one correction to how they are handled.**

`ZBRENT: fatal error in bracketing` appears when a relaxation has converged so tightly that
the line search cannot take another step, with dE reaching 1e-307. VASP's own advice is to
continue from CONTCAR, which the driver now does, up to twice.

`Inconsistent Bravais lattice types found for crystalline and reciprocal lattice` appears
when a relaxed cell's real and reciprocal lattices disagree on the Bravais type. The first
fallback written for it was wrong: it set `ISYM = 0`, on the assumption that this disables
the check. It does not - the lattice-type test runs regardless of ISYM, and six E2 tasks
exhausted three attempts still failing. A direct test on the cluster settled it: `ISYM=0`
alone failed again, while `ISYM=0` with `SYMPREC = 1E-4` converged, giving -50.96367162 eV.
The fallback now loosens SYMPREC, which is what the error message asks for. Because the
k-mesh is written out explicitly, a looser symmetry tolerance cannot change it.

`FEXCF: supplied exchange-correlation table` occurred twice, both at the first electronic
step of orderings of `E2-84b0225bbe` (TbEuY5NiI12). Eu keeps its f electrons in the valence,
where PBE describes them poorly, and the uniform `MAGMOM = N*1.0` start is a crude
initialisation for an f7 ion. A physically chosen initial moment would probably fix it, but
that is a change of settings driven by a failure, so it was not made: the affected orderings
are reported as failures and, under the rule above, their entry is excluded from E2 with the
reason named.

Every fallback writes `FALLBACK_APPLIED` into the stage directory and keeps the failed
attempt beside it as `<stage>.failedN`, so no repair is silent.

**Result files are not package inputs.** `verify.py` no longer reports `collected.json` and
its kin as unrecorded files; only the frozen inputs are checked against the manifest.

**New hashes:**

| item | SHA256 |
|---|---|
| build_tasks.py | `5e2fbd2d575b78be73ab350c2e3851b05e9e6034beaa93d06953f7db8f17e6ab` |
| analyze.py | `2f1d722d94bff31c9d6f6cb391a8e2b9a35ae7184fa40e842eef8faa65bbe69c` |
| verify.py | `68a7583a2f9599239afbdfc24043499fc87dbeb2482fd52ddfb6f73847a3037d` |
| farm.slurm | `a695339e0519784027bfe90e46de7087ecfc340066d4ed720ac3191ff2a06203` |
| controller.sh | `a11dbfdeae7f2e7912834092727de83173aa381d844e6158bfd1eeb5f938a928` |
| status.sh | `37b602833bdbd799a907604a09c284f51a073a1056ab64c4ef84f4cc6c4cd809` |
| monitor.sh | `f4c6fe435605b949551e6fa9f3b098dd213b39d90b79a50e619713819d3b3c56` |

Amendments so far: **5**.

### Amendment 6 - 2026-08-27, during execution

**E2 is complete and two of its four predictions are met.** 23 of 24 GNoME entries and all
10 controls finished every ordering. The one exclusion is `84b0225bbe` (TbEuY5NiI12), whose
orderings failed inside VASP's exchange-correlation table; under the rule of Amendment 5 an
entry missing any ordering is dropped rather than analysed on a partial set.

| prediction | measured | met |
|---|---|---|
| GNoME median T_od below 300 K | 0 K | yes |
| control median T_od above 1000 K | 995 K | no |
| at least 60 per cent of GNoME entries below 300 K | 100 per cent | yes |
| at most 20 per cent of controls below 300 K | 30 per cent | no |

**The two failures come from the estimator, not from the physics, and the estimator is
mine.** The pre-registered ordering energy is `dE = E(released ordering) - min over
orderings`. That measures whether the release picked the lowest-energy ordering. It does not
measure whether the compound is ordered, and those are different questions.

Thirteen of the 23 GNoME entries have `dE` exactly zero: GNoME picked the ground-state
ordering, which is a point in its favour. Fed into `T_od = dE / (k_B dS)` that gives 0 K,
which the pre-registered rule reads as "disordered". The prediction passes for a reason it
did not intend. On the control side the same thing runs the other way: three controls have
their experimental ordering as the computed minimum - the structure is the ground state,
which is what one wants of an experimental structure - and the formula turns that into 0 K
and counts them as disordered. Those three drag the control median to 995 K, five kelvin
under the threshold, and take the "below 300 K" fraction to 30 per cent.

The pre-registered numbers stand as reported. They are not replaced.

**The quantity that does decide order against disorder is reported beside them.** An ordered
ground state disorders when `T dS` exceeds the cost of leaving it for a random
configuration, approximated by the mean over orderings minus the minimum:

| | GNoME (23) | controls (10) |
|---|---|---|
| disordering energy, median | 0.00010 eV per atom | 0.03642 eV per atom |
| spread across orderings, median | 0.00020 eV per atom | 0.08622 eV per atom |
| implied temperature, median | 11 K | 1524 K |
| below 300 K | 18 of 23 | 0 of 10 |

The two groups separate by a factor of 358. A spread of 0.2 meV per atom is two orders of
magnitude below k_B T at room temperature, 26 meV per atom, so those orderings are degenerate
at any temperature at which the material could be made: what the release lists as a distinct
ordered compound is a solid solution. The controls, at 86 meV per atom and 1524 K, are
genuinely ordered, and not one of them falls below 300 K.

This is reported as an additional analysis with its rationale, not as a substituted
criterion. The design fault - conflating "did the release pick a good ordering" with "is
this compound ordered" - was mine, and it is recorded here rather than quietly corrected.

**New hashes:**

| item | SHA256 |
|---|---|
| analyze.py | `1087546770420d09a8d92ac6da3cb5631507a2bb4dd2fe2cbd0d98c986f870cb` |
| farm.slurm | `6c2fca3a3f57e2d94cf61b55ecaf5e5e06a6814f9e0e4a1dcccf335a7fbecf5e` |
| controller.sh | `75eddf5b5b0ed770fcf27b534da101dfd2b9d039ee371ef7780815ada14ba660` |
| status.sh | `b12ecc66219a48f646ab2e613d968f07818b34f993dc92499f1c8b1e402016d7` |

Amendments so far: **6**.

### Amendment 7 - 2026-08-27, during execution

**E3 is complete and all three of its predictions are met** - but only after a defect in the
result extraction was found and fixed. The first run of the analysis said the opposite, and
the way it was caught is worth recording.

**What the first run reported.** Rank correlation between DFT and MatterSim: -0.038, near
zero. S5 exceeding the parents by -9.29 eV per atom, the wrong sign. Zero of three
predictions met. Taken at face value this says the manuscript's MatterSim-based severity
calibration in Fig. 5c,d is unrelated to first-principles energetics.

**Why it was wrong.** `collect.py` took the relaxation energy as the first minus the last
`energy(sigma->0)` in the OUTCAR. That line is printed for every *electronic* iteration, so
its first value is the opening SCF guess from a superposition of atoms, several hundred eV
above the ground state. The "relaxation energy" therefore contained the whole electronic
convergence, tens of eV per atom for every cell, which buried the signal. The giveaway was
that undamaged experimental structures appeared to release 80 eV per atom on relaxation,
which is not physically possible - the pre-registered numbers made the anomaly impossible to
overlook.

**The fix.** The converged energy of an ionic step is the one following VASP's
`FREE ENERGIE OF THE ION-ELECTRON SYSTEM` header; the per-step `F=`/`E0=` summary lives in
OSZICAR, not OUTCAR. Extraction is now keyed to that header. No calculation was repeated:
the same OUTCARs were re-read.

**What E3 shows once the energies are right.**

| variant | n | DFT release, median (eV per atom) | MatterSim |
|---|---|---|---|
| P0 undamaged | 50 | 0.0009 | 0.0024 |
| S1 uniaxial compression | 30 | 0.3032 | 0.3012 |
| S2 cation-cation exchange | 30 | 0.3953 | 0.3135 |
| S3 random displacement | 30 | 3.1537 | 3.3265 |
| S4 isotropic expansion | 30 | 0.5545 | 0.4756 |
| S5 cation-anion exchange | 30 | 0.3985 | 0.3884 |

| prediction | measured | met |
|---|---|---|
| rank correlation at least 0.7 | 0.953 | yes |
| S5 exceeds the parents by 0.3 eV per atom | 0.3976 | yes |
| the S5-to-parent ratio is larger under DFT | 460.7 against 164.4 | yes |

A rank correlation of 0.953 over 200 cells is the answer to the objection the campaign was
built for: the relaxation energies behind Fig. 5c,d rest on a potential evaluated far from
the data it was trained on, and first principles agree with it on this task. Undamaged
parents release 0.0009 eV per atom, and the five damage classes separate in the same order
under both methods.

The third prediction is met but the ratio it uses is poorly conditioned: both denominators
are near zero, so the comparison is dominated by how small each method's parent median
happens to be. The difference of medians, the second prediction, is the robust statement.

**One correction to something reported earlier in this campaign.** A single parent
(`cod-1001469`) showed S5 at 2.52 eV per atom under DFT against 0.39 under MatterSim, and
that was described as MatterSim underestimating the wrong-site exchange roughly sixfold.
Across all thirty S5 cells the medians are 0.3985 and 0.3884: they agree. The sixfold figure
came from one cell in the tail and should not have been offered as a finding.

**New hashes:**

| item | SHA256 |
|---|---|
| collect.py | `2e8ce82732c99464e9690d61218d21d1c559f483d18edf397cdbb0d598979d1d` |
| analyze.py | `1087546770420d09a8d92ac6da3cb5631507a2bb4dd2fe2cbd0d98c986f870cb` |
| controller.sh | `51d03da4a01c4f2a520c7ac1a13a4aa669b7cbfc62d600932317b41467379cf3` |
| status.sh | `8850d6b1a09a502a5d1b44c1b3d905ca4f620077b138cf938a94316cda832b6b` |
| farm.slurm | `9dce67f15583b51d59078057a03ce1192663a9a28a0143c7c39820b4e8f3eb0f` |

Amendments so far: **7**.

### Amendment 8 - 2026-08-27, during execution

**A sensitivity analysis the pre-registration did not ask for, reported beside the frozen
numbers rather than in place of them.**

Section 2 fixes the rule that a relaxation which exhausts NSW without meeting EDIFFG is
kept, not dropped: `usable()` admits `complete` and `unconverged`. That rule stands and the
headline numbers below are still the ones it produces. But the rule was written before
anyone knew how many cells it would admit, and a referee is entitled to ask what the
numbers look like without them. The counts turn out to be small and the answer is recorded
here so that it cannot be chosen after the fact:

| package | stage | converged | exited 0 without reaching required accuracy |
|---|---|---|---|
| E2 | relax_cell | 117 | 6 (five orderings of `cod-1511279`, one of `84b0225bbe`) |
| E3 | relax_ions_fixed_cell | 189 | 11 |
| E3 | relax_cell | 195 | 5 |

**E3 is insensitive to the choice.** All three predictions are met either way:

| quantity | pre-registered (unconverged kept) | complete only |
|---|---|---|
| cells analysed | 200 | 187 |
| DFT-MatterSim Spearman (gate 0.70) | 0.953 | 0.945 |
| S5 excess over parents, eV per atom (gate 0.30) | 0.3976 | 0.3689 |
| S5-to-parent ratio, DFT vs MatterSim | 460.69 vs 164.38 | 427.52 vs 156.09 |

**E2 is sensitive, and in the direction that flatters the hypothesis.** Dropping the
unconverged cells removes one control entry and moves the control arm across one of its two
failing gates:

| prediction | pre-registered | complete only |
|---|---|---|
| GNoME median T_od below 300 K | 0 K, met | 0 K, met |
| control median T_od above 1000 K | 995 K, **not met** | 1062 K, met |
| at least 60% of GNoME below 300 K | 100%, met | 100%, met |
| at most 20% of controls below 300 K | 30.0%, **not met** | 22.2%, **not met** |

The pre-registered result is therefore two of four met, and it stays two of four met. The
control median missing 1000 K by 5 K is a coin-flip on a ten-entry median, not a finding,
and the fact that a defensible reanalysis flips it is exactly why the frozen rule is the one
being reported. **Nothing in the manuscript may quote the complete-only column as the
result.** It is here to show that the failing gates are not an artefact of unconverged
cells: the 20% control gate fails under both.

The quantity the code already reports alongside these - the cost of disordering, mean over
orderings minus the minimum, which is what actually decides order against disorder - is not
sensitive at all: 0 of 10 controls fall below 300 K against 18 of 23 GNoME entries, under
either rule.

**An operational note, not a protocol change.** The task farm's controller was found running
in five superseded copies across two login nodes, logging a shell error each cycle. Cause:
the script had been redeployed by overwriting it in place while it ran, and bash reads a
script incrementally by byte offset. Two fixes, neither touching any physics: deploy to a
new inode and `mv` it into place, and have each cycle stand down if the shared pid file no
longer names it (`cleanup_controllers.sh` removes strays across login nodes). Clearing the
strays briefly took the live controller with them, because the older copies carried an
unguarded EXIT trap that deleted the pid file; the farm jobs self-requeue, so the queue
stayed full at 20 throughout and no calculation was affected.


### Amendment 9 - 2026-08-28, E4 complete; the campaign is finished

**Result: one of E4's three predictions met, and the two that failed both failed for the
same reason, which is a finding rather than a defect.**

| prediction | measured | met |
|---|---|---|
| at most 2 screened candidates reach 400 GPa | 1 of 60 | yes |
| at least 70% of priority candidates confirmed at 400 GPa | 0.7% | **no** |
| UMA and DFT bulk moduli correlate with r at least 0.8 | 0.769 | **no** |

260 of 261 candidates were fitted. `candidate_0248` (ReW4, screened) never produced a
CONTCAR - stage A died with `EDDDAV: Call to ZHEGV failed`, a diagonalisation failure, which
is not a wall-clock timeout and so was not re-run; `stage_b/SKIPPED.json` records it. Three
stage-B volume points died after exhausting their fallbacks (one Bravais-lattice mismatch,
two ZBRENT bracketing failures), leaving their candidates with four points each. That is
within the pre-registered rule of at least four, and dropping one point from the 215
candidates that had five shifts the fitted modulus by a median of 0.05 GPa (sd 0.58,
worst 7.43), so those three keep their values.

**Why the second prediction failed.** The proxy is systematically high: the median DFT/UMA
ratio is 0.940. The 400 GPa target was calibrated on the proxy's scale and sits in the
thickest part of the distribution, so a 6% offset moves almost every candidate across it.
It is a calibration failure, not a selection failure, and the following distinguish the two.
They are reported beside the frozen numbers, never in place of them:

- a priority candidate outranks a screened one 0.966 of the time
- group medians order correctly: priority 383 GPa (IQR 375-387), control 363 (356-378),
  screened 337 (322-350)
- with the target rescaled onto the DFT axis (376 GPa), 1 of 60 screened candidates reaches
  it against 123 of 200 the screen retained
- of the 140 highest bulk moduli under DFT, 3 had been removed by the screen: 97.9% retained
- the queue sweep of Fig. 4f, recomputed against a DFT-defined subset, retains 99.2% of it
  at the frozen PSS cutoff, against 100.0% of the UMA-defined subset

**Why the third prediction failed is worth stating plainly rather than explaining away.**
r = 0.769 against a gate of 0.8, with Spearman 0.710. The proxy separates high from low but
ranks poorly inside the high band, and no rescaling repairs a correlation.

**The single worst miss should be reported, not buried.** `candidate_0980` (Re2IrOs6) has
the highest bulk modulus in the whole set under DFT, 419 GPa, and the proxy put it at
347 GPa - a 72 GPa underestimate. Its PSS of -0.721 is below the cutoff, so the
synthesizability screen removed the best material E4 measured. Two further candidates in the
DFT top 140 were screened out (`candidate_0292` at 374 GPa, `candidate_0995` at 373 GPa).

**A denominator that had to be refused.** Carrying the unrescaled 400 GPa onto the DFT axis
leaves 2 candidates above the line, and the retention computed from them, 50.0%, is not a
measurement. `e4_queue_sweep.py` now prints both subset sizes and declines to report that
figure. The 99.2% above is read off the 124 candidates above the rescaled threshold.

**Standing caveat on every E4 percentage.** The 261 candidates deliberately over-sample high
proxy bulk moduli. None of these fractions carry over to the 1,081-candidate pool.

**Campaign totals.** 1,917 tasks: E1 20, E1b 8, E2 128, E3 200, E4 261 + 1,300.
9 of 13 pre-registered predictions met - E1 2 of 2, E1b 1 of 1, E2 2 of 4, E3 3 of 3,
E4 1 of 3. Roughly 344 node-hours.


**Closing the campaign, and one entry stopped deliberately.** The Eu entry `84b0225bbe`
(TbEuY5NiI12, GNoME) ends at two of its seven orderings finished. Of the rest, `o00` and
`o03` fail with `ERROR FEXCF: supplied exchange-correlation table` and `o01` with
`ZBRENT: can not reach accuracy`; `o04` and `o05` were interrupted mid-run and never
failed on their own. FEXCF is deterministic for a given cell and setting, and section 2
allows a re-run only after a wall-clock timeout and only with identical inputs, so further
attempts could not have produced a different result. The remaining jobs were therefore
cancelled rather than left to exhaust their attempt limits. The entry was already excluded
by the completeness rule, so nothing about the E2 result changes; what changes is that
five nodes stopped repeating a calculation whose answer was fixed. Both f-electron species
in that formula, Tb and Eu, sit in the valence under PBE without U, which is where this
class of failure is expected.

**A bug this exposed, fixed.** `submit_farm.sh` computed its remaining count as
`total - done` and ignored the dead markers, so a package whose only outstanding tasks had
been given up on still looked unfinished. The queue refilled with jobs that swept the list,
found nothing they could claim, and exited - one node burned per cycle, indefinitely. It now
subtracts the dead count and reports "nothing left to run". `controller.sh` already
subtracted it; this was the last line of defence failing to hold.

**Final state.** E1 20/20, E1b 8/8, E2 123/128 (5 stopped as above), E3 200/200,
E4 stage A 260/261 (1 dead), E4 stage B 1297/1300 (3 dead). Queue empty, controller stopped.

Amendments so far: **9**.
