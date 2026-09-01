# Radical-Facet Positive Enclosure Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to
> implement this plan task-by-task.

**Goal:** Test whether the angular balance of every unique radius-aware
radical/power-cell facet supplies a transferable, representation-invariant,
pre-DFT crystal-validity signal and, only if authorized, a stronger bounded
screening law.

**Architecture:** NEXT327 reuses the label-free NEXT279 periodic radical-cell
active-facet incidences and the analytic NEXT295 positive-equilibrium kernel.
For each raw site it deduplicates geometrically coincident facet normals and
solves one uniform-prior convex-balance program. NEXT328 opens discovery
outcomes only as offline audit labels and applies the unchanged NEXT224/NEXT268
source and fold gates. NEXT329/NEXT330 are contingent and may exist only if
NEXT328 authorizes the single frozen hypothesis.

**Tech Stack:** Python 3.11, NumPy, SciPy HiGHS linear programming, pandas,
ASE, the existing NEXT267/NEXT279/NEXT295 geometry kernels, pytest, CodeGraph.

## Scientific and information-boundary freeze

For raw site `i`, let `C_i` be its NEXT279 active radical-facet incidences.
Normalize every displacement to a unit outward direction, quantize each
component on the existing `1e-10` grid, and retain one representative of each
unique direction. Write the resulting population as
`U_i = {u_i1, ..., u_iK}`. Solve

```text
alpha_i* = max alpha
subject to
    sum_k f_ik u_ik = 0,
    sum_k f_ik = 1,
    f_ik >= alpha / K_i,
    f_ik >= 0 and 0 <= alpha <= 1.
```

The sole protected hypothesis is
`rfpe_uniform_equilibrium_q10 = inverse_cdf_0.10({alpha_i*})` in the
`protected_high` direction. The feature asks whether a structure's weaker
site cages can balance all genuine facet directions without assigning an
arbitrarily tiny coefficient to some facet. It is sign-sensitive and
dimensionless. It is not a physical or predicted force, stress, energy,
potential, relaxation, or trajectory, and coordinates and cells are never
moved.

The construction is motivated by Minkowski's polytope theorem: the outward
unit normals of a bounded convex polytope admit strictly positive
area-weighted equilibrium, and conversely balanced spanning normals define a
polytope. The RFPE margin is a new uniform-participation statistic on the
already constructed raw radical cell; the theorem motivates geometric
closure but does not make RFPE a theorem of crystal stability.

RFPE is distinct from NEXT295 because NEXT295 uses a formal-valence-dependent
opposite-sign CrystalNN graph and consequently missed the frozen SCIGEN fold
coverage gate. RFPE uses all unique radius-aware radical facets, requires no
formal-valence assignment, and is designed prospectively to test the graph-
support hypothesis explicitly. RFPE is distinct from NEXT323 because every
site is solved independently: no coefficient is shared across reciprocal
ends and there is no global self-stress constraint.

Executable inputs are limited to element identities, deterministic tabulated
radii, and one initial raw unrelaxed periodic geometry. The branch must not
execute or consume DFT calculations or per-structure DFT values; learned
energy/force/stress proxies; MLIPs; model/proxy potentials; relaxed
structures; trajectories; later geometries; validation outcomes; or
replication outcomes. Discovery outcomes may enter NEXT328 only as offline
labels. Validation and replication files remain physically unopened.

## Alternatives rejected before any new feature or outcome access

1. Preferred-radius Regge/angular-defect curvature was rejected because
   stable close-packed and Frank--Kasper crystals intrinsically require
   nonzero disclination content, so a universal protected-low direction is
   not defensible.
2. Preferred-metric incompatibility projected through a periodic rigidity
   matrix was rejected as a duplicate of NEXT37, which already projects
   centered log radius-sum edge mismatch onto atomic and affine rigidity
   columns.
3. A low-wavevector charge structure factor was rejected as a duplicate of
   NEXT36, which already computes scale-free reciprocal-space Gaussian charge
   spectrum metrics.
4. Maxwell/self-stress dimension was rejected because NEXT20, NEXT37, and
   NEXT323 already cover rank, mismatch compatibility, and global positive
   equilibrium.

No graph change, alternative direction, extra aggregate, weighting, quantile,
threshold, conjunction, or second RFPE feature may be added after the
label-blind probe starts.

## Frozen gates

- Label-blind probe: deterministically select 80 discovery geometries per
  source after the complete identifier-bearing initial-geometry inventory is
  opened in memory; read no endpoint, label, validation, replication, relaxed
  geometry, DFT field, or model-potential field.
- Probe engineering gates: exact schema; finite `[0,1]` values; support at
  least 72/80 per source; at least 20 quantized unique values per source;
  rigid rotation, periodic translation, site permutation, equivalent lattice
  rebasing, and exact integral supercell error at most `1e-8` before `1e-10`
  output quantization.
- Probe novelty gate: maximum absolute Spearman correlation below `0.90`
  against available label-free NEXT267, NEXT279, NEXT295, NEXT299, NEXT303,
  NEXT307, NEXT311, NEXT315, NEXT319, and NEXT323 feature populations on the
  same records. Failure terminates the branch before a formal NEXT327 build
  and before opening discovery outcomes.
- Formal NEXT327 source coverage: at least `0.90` independently in SCIGEN and
  WyFormer, with unsupported structures retained as abstentions and never
  imputed.
- NEXT328 gates, unchanged from NEXT224/NEXT268: minimum cell coverage `0.90`,
  minimum class count `20`, pooled AUC `0.55`, macro AUC `0.53`, and
  worst-fold AUC `0.50` in both sources using reduced-formula folds. Quantile
  normalization remains inverse-CDF `1/16` and `15/16`.
- An empty eligible set sets `next329_search_authorized=false` and
  `rfpe_branch_terminated=true`; NEXT329/NEXT330 must not exist.
- If authorized, NEXT329 may reuse only the unchanged NEXT269 margin-local
  grammar and fixed NEXT224/NEXT135 base score. NEXT330 is discovery-only
  BROAD diagnosis. Validation and replication remain sealed even if BROAD
  passes; any stronger final claim still requires a separately frozen unseen
  validation protocol.

## Task 1: NEXT327 analytic kernel and geometry wrapper

**Files:**

- Create: `src/next327_radical_facet_positive_enclosure.py`
- Create: `tests/test_next327_radical_facet_positive_enclosure.py`
- Reuse without modification: `src/next279_radical_packing_autocorrelation.py`
- Reuse without modification: `src/next295_positive_contact_force_closure.py`

1. Write RED tests for balanced tetrahedral directions (`alpha=1`), an
   imbalanced positive population, a one-sided population (`alpha=0`), and
   duplicate-direction removal.
2. Implement strict direction validation and reuse the frozen NEXT295 HiGHS
   kernel without changing its tolerances.
3. Build unique per-site facet directions from NEXT279 contacts, retaining
   lattice self-image facets but deduplicating coincident unit normals.
4. Add real-structure and invariance tests for NaCl, CsCl, ZnS, rigid
   rotation, periodic translation, site permutation, lattice rebasing, exact
   `2 x 1 x 1` replication, deterministic repeats, and geometry-only input
   refusal.
5. Run the new test file and Python byte compilation.

## Task 2: Label-blind probe and final pre-outcome freeze

**Files:**

- Create: `experiments/next327_rfpe_label_blind_probe.py`
- Create: `tests/test_next327_rfpe_label_blind_probe.py`
- Create only after a passing probe:
  `experiments/next327_rfpe_label_blind_probe_result.json`

1. Reuse the NEXT323 deterministic identity selection and strict complete-
   inventory loader, changing only the feature under test and adding lattice-
   rebasing to the invariance checks.
2. Compute support, range, quantized uniqueness, invariance error, numerical
   diagnostics, and maximum correlation with the frozen prior label-free
   population.
3. Save exact statistics and design/source hashes without reading any outcome.
4. Stop immediately if any probe gate fails.

## Task 3: NEXT327 formal label-free build

**Files:**

- Modify only the two new NEXT327 source/test files as required by RED tests.

1. Add RED builder tests for complete identity, geometry-only reads,
   unsupported-row abstention, exact manifests, source hashes, and false
   boundary flags.
2. Implement the additive multiprocessing builder using the frozen discovery
   geometry loaders and atomic publication pattern.
3. If the probe passes, run the formal build into
   `$PRIS_ARCHIVE/next327_radical_facet_positive_enclosure_v1`.
4. Recompute manifest/output/source hashes and stop before outcomes if either
   source misses coverage.

## Task 4: NEXT328 fixed discovery-only audit

**Files:**

- Create: `src/next328_rfpe_feature_audit.py`
- Create: `tests/test_next328_rfpe_feature_audit.py`

1. Write RED tests for the exact one-hypothesis universe, protected-high
   normalization, unchanged gates/folds, empty-set stop, and validation /
   replication refusal.
2. Reuse NEXT268/NEXT324 audit and provenance helpers without changing the
   cohort, normalization, or gates.
3. Open only physically isolated discovery endpoints as offline labels and
   publish into
   `$PRIS_ARCHIVE/next328_rfpe_feature_audit_v1`.
4. Freeze the eligible-set digest. If empty, stop without NEXT329/NEXT330.

## Task 5: Contingent NEXT329/NEXT330 search

Create NEXT329/NEXT330 scripts and tests only when the exact NEXT328 manifest
sets `next329_search_authorized=true`. Reuse the unchanged NEXT269 grammar,
then the unchanged BROAD diagnostic. Do not add transformations, reverse the
direction, or open validation/replication.

## Task 6: Verification and independent report

**Files:**

- Modify additively:
  `reports/2026-08-08-next115-next117-hcid-no-dft-search.md`
- Do not modify: `paper/`, `tex/`, `notes/`, `README.md`, `PREREG.md`

Run focused tests, adjacent NEXT267/NEXT279/NEXT295/NEXT323 tests, byte
compilation, artifact/boundary assertions, CodeGraph synchronization, and the
complete repository suite. Append a standalone RFPE section with exact hashes
and claim limits. Do not edit canonical paper/report content before user
confirmation.

## Execution note

This is an intentionally dirty shared checkout and all existing scripts and
content must remain. Work additively in the current checkout, make no Git
commit/merge/cleanup, and do not delegate to subagents.
