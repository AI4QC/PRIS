# NEXT537--NEXT542: periodic bond-angle affine accommodation

Date frozen: 2026-08-13 (America/Chicago)

## Understanding lock

- Build an additive, interpretable crystal-screening law for generated or
  theoretical structures before any DFT calculation.
- The executable user is a structure generator or screening pipeline; its only
  specimen input is composition plus one raw, initial, fully periodic geometry.
- The purpose is to reject structures likely to undergo large framework change
  while protecting already reasonable x0 structures and preserving useful DFT
  savings.
- DFT outcomes may be opened only as offline labels after feature definitions,
  grids, gates and predictions are frozen.
- No DFT value/calculation, relaxed or later geometry, learned energy/force/
  stress proxy, MLIP, potential, trajectory, same-composition alternative or
  physical/virtual coordinate relaxation may enter the executable formula.
- All existing scripts, artifacts, reports and canonical documents remain
  unchanged.  A new independent report is authorized only after an unseen or
  sealed evaluation passes.
- NEXT534 falsified the SSSP-complement hypothesis, so this branch must use a
  non-duplicate mechanism and may not expand the SSSP weight grid.

The user already selected autonomous continuation (choice A) and explicitly
asked not to be queried again after a scheme is derived.  That instruction is
the confirmation for this understanding lock.

## Assumptions and non-functional requirements

- Exact rigid translations, rotations, site permutations, unimodular cell
  rebasing and exact supercell representations describe the same specimen and
  must not materially change the feature.
- Raw generated cells can be P1 and can contain disconnected guests.  Only
  covalent components with nonzero periodic dimension enter the constraint
  framework; all such components are retained, including interpenetrated nets.
- Covalent-radius graph errors remain a limitation.  Missing/invalid geometry
  fails open to `KEEP`; unsupported structures are never silently treated as
  risky.
- Performance target on the frozen 96-structure probe: median runtime at most
  0.5 s, p95 at most 5 s and maximum at most 30 s per structure on the current
  host.  Full 7,815-row construction uses bounded multiprocessing and publishes
  atomically without replacement.
- Reliability target: deterministic output quantized at `1e-8`; maximum
  representation discrepancy at most `1e-6`; no partial output directory.
- Security/privacy: all data are local project artifacts; no structure payload
  is transmitted.  Literature lookup supplies theory only, never row labels.
- Maintenance: reuse the frozen NEXT49 covalent graph and existing strict
  geometry firewall; keep the pure kernel separate from cohort and endpoint
  orchestration.

## Evidence and alternatives

Periodic rigidity theory identifies affinely periodic flexes with the null
space of a finite rigidity matrix containing atomic and cell degrees of freedom
([Power 2011](https://arxiv.org/abs/1103.1914)).  Rigid-unit modes describe
low-energy collective motions of periodic framework units
([Power 2011 survey](https://arxiv.org/abs/1111.2943)).  A MOF study showed that
a rigid-elements-connected-by-hinges model can separate known rigid and flexible
MOFs and argued that intrinsic flexibility can impede realization
([Sarkisov et al. 2014](https://doi.org/10.1021/ja411673b)).  Framework
experiments and geometric work further distinguish rigid polyhedral units from
softer inter-unit hinges and show that node connectivity and distortion matter
for mechanical response
([Wells and Sartbaeva 2015](https://pmc.ncbi.nlm.nih.gov/articles/PMC5448923/),
[Gładysiak et al. 2023](https://www.nature.com/articles/s42004-023-00981-8)).

Three approaches were reviewed before implementation:

1. **Full periodic bond-angle constraint Jacobian (selected).** Preserve every
   covalent bond length and every local covalent bond angle to first order,
   leaving torsions and collective hinges free.  Measure whether symmetric cell
   strain lies in the atomic constraint column space.  This is continuous,
   collective, template-free and distinct from NEXT75's central-force residual
   features.
2. **SBU rigid-body/hinge graph.** More chemically explicit, but automatic SBU
   and rigid-linker perception introduces many chemistry-specific decisions and
   risks low coverage on generated structures.
3. **Local ideal-polyhedron deviation.** Fast and easy to maintain, but overlaps
   prior coordination-motif/order-parameter features and cannot detect a
   collective periodic mechanism.

## Frozen NEXT537 kernel

### Geometry and graph

Accept strict geometry-only `ase.Atoms` with no calculator, `info`, or extra
arrays.  Reduce exact translational supercells to a primitive representation
with a fixed `spglib` tolerance of `1e-5`; failure to prove a reduction leaves
the input representation unchanged.  This operation changes representation,
not coordinates or physical state.

Build the unchanged NEXT49 periodic covalent-radius graph with ratio cutoff
`1.25`.  Compute quotient components and retain all components whose lattice
translation rank is greater than zero.  Compact retained sites without changing
their Cartesian positions or periodic edge vectors.

### Dimensionless constraints

For each retained edge vector `a`, add the first derivative of `log(|a|)` with
respect to atomic displacement and a symmetric affine cell strain.  At every
retained center having at least two periodic covalent neighbours, add one row
for each unordered neighbour pair: the first derivative of their cosine angle.
The current raw angle is preserved; no ideal angle, force constant, energy or
potential is fitted.

Use the Frobenius-orthonormal Kelvin basis

```text
(E_xx, E_yy, E_zz, sqrt(2) E_yz, sqrt(2) E_xz, sqrt(2) E_xy)
```

and form the dimensionless matrix

```text
J = [A | C]
```

where `A` contains `3N` atomic columns and `C` contains six symmetric cell
strain columns.  Use the analytic derivatives directly: log-length change and
cosine-angle change are both dimensionless.  There is no row normalization,
bond/angle family weight or fitted force constant.  Exact supercell repetition
multiplies both direct and residual Gram matrices by the same factor, which the
generalized ratio below cancels.

Let `P_A` be the orthogonal projector onto the column space of `A`, computed by
a deterministic rank-revealing factorization without applying the resulting
displacements to the structure.  Define

```text
K_direct = C.T C
K_resid  = C.T (I - P_A) C
W        = K_direct^(-1/2) K_resid K_direct^(-1/2)
```

on the positive eigenspace of `K_direct`.  If `K_direct` has rank below six,
the missing strain direction is exactly unconstrained.  Otherwise the six
eigenvalues `mu_i` are clipped only for numerical noise to `[0,1]`.

The sole candidate is

```text
PBAAA(x0) = 1 - min_i(mu_i)
```

with missing direct strain directions assigned `PBAAA=1`.  `PBAAA` lies in
`[0,1]`; higher means that at least one affine strain can be more completely
accommodated by an infinitesimal collective hinge motion and is therefore the
precommitted risk direction.  Quantize the final value at `1e-8`.

The factorization is a first-order kinematic compatibility test.  It does not
generate, move, relax or score an alternative geometry and is not a potential.
Diagnostics may report constraint counts, direct rank, soft-strain fraction,
minimum/median/maximum `mu`, and factorization residual, but no other diagnostic
may become a candidate after labels are viewed.

## Frozen NEXT538 label-blind gates

Select 32 structures per NEXT54 role by deterministic SHA-256 order, with no
endpoint input, for 96 total raw x0 structures.  Require:

- at least 28/32 supported in every role and at least 84/96 overall;
- every supported candidate finite in `[0,1]`;
- at least 10 distinct quantized values per role and 30 overall;
- maximum rigid-representation error `<=1e-6` on eight supported structures;
- median runtime `<=0.5 s`, p95 `<=5 s`, maximum `<=30 s`;
- at least 40 joint finite rows for novelty comparisons and maximum adequate
  absolute Spearman correlation `<0.95` against every numeric NEXT77 feature,
  NEXT533 SSSP, and direct NEXT75 candidate features available on the probe.

Engineering failure stops the branch.  Novelty failure also stops it; neither
failure authorizes changing graph, constraint families, row normalization,
feature direction or tolerances after seeing the probe.

## NEXT539 full label-free freeze

Only after NEXT538 passes, compute PBAAA for all 7,815 NEXT54 geometries across
all three roles before touching replication outcomes.  Each role must have at
least 20 finite distinct values.  Publish a feature table, catalogue, failure
counts, runtime statistics, exact hashes and the unchanged replication
firewall state.

## NEXT540 bounded two-partition search

Use only the already-opened robust discovery and internal-validation endpoints.
Start from the exact six-term NEXT79 score and search only

```text
R_new = R_NEXT79 + w * PBAAA
w in {0, 0.25, 0.5, 1, 2, 4}
```

with one shared observed threshold.  Missing PBAAA contributes zero.  No sign
flip, alternate feature, rescaling, interaction or partition-specific threshold
is allowed.

Discovery and validation must each independently pass coverage lower `>=0.95`,
protected-recall lower `>=0.95`, reject-precision lower `>=0.70`, savings lower
`>=0.02`, pooled extreme AUC `>=0.75`, macro stratum AUC `>=0.65`, and worst
stratum AUC `>=0.55`.  Combined reject-precision Wilson lower must be `>=0.80`.
Use the same deterministic eligible-candidate ranking as NEXT534.

If no candidate passes, do not open replication and do not expand the grid on
these endpoints.

## NEXT541--NEXT542 sealed evaluation

Only if NEXT540 passes, NEXT541 freezes exact internal-replication predictions
from label-free features and verifies the endpoint remains unopened.  NEXT542
then opens the 1,539-row robust endpoint exactly once and applies the same seven
gates without repair.  A passing result authorizes a new independent report;
a failure is preserved and the search continues on a new mechanism or source.

## Decision log

1. **Continue after NEXT534 failure.** Chosen because the active goal requires a
   genuinely better zero-DFT law; rejected expanding SSSP because its validation
   precision lower bound remained near 0.60.
2. **Choose angle-inclusive affine rigidity.** Chosen for collective periodic
   physics and novelty over central-force NEXT75; rejected local motifs as
   duplicate and SBU templates as fragile.
3. **Preserve observed angles rather than fit ideal ones.** Chosen to eliminate
   chemistry tables and fitted targets while retaining rigid-unit kinematics.
4. **Use generalized residual/direct strain eigenvalues.** Chosen because the
   ratio is bounded, dimensionless and invariant to a common constraint scale.
5. **Freeze one scalar candidate.** Chosen to prevent diagnostic feature mining.
6. **Probe before full build, and require novelty before labels.** Chosen to stop
   unsupported, degenerate, slow or duplicate mechanisms cheaply.
7. **Reuse the strict two-partition and sealed-replication gates.** Chosen to
   prevent another discovery-only tail-precision artifact.

## Implementation order

1. Add RED pure-kernel, boundary and invariance tests; implement NEXT537 only
   after the expected import failure.
2. Add RED label-blind probe tests; implement and execute NEXT538.  Stop on any
   gate failure.
3. Conditional on authorization, add tests and NEXT539 full feature build.
4. Conditional on NEXT539, add tests and NEXT540 exact bounded search.
5. Conditional on readiness, freeze NEXT541 predictions and run NEXT542 once.
6. Only after scientific success, write a standalone report and run the full
   repository suite.  Do not edit canonical documents before user review.
