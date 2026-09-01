# NEXT255--NEXT258 Delaunay Void-Cage Isotropy Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Test whether the angular isotropy and directional closure of complete periodic Delaunay void cages supplies a fully pre-DFT, interpretable correction to the frozen NEXT224 law.

**Architecture:** NEXT255 constructs a fixed, label-free bank of sixteen dimensionless void-centered geometry features from the initial periodic structure. NEXT256 audits the sixteen prospectively directed hypotheses in the exact frozen NEXT224 rejected-extreme cohort. NEXT257 and NEXT258 are conditional search and BROAD-diagnostic stages and remain forbidden unless their preceding gates authorize them.

**Tech Stack:** Python 3.11, NumPy, pandas/Parquet, pymatgen `VoronoiNN`, ASE, pytest, and the unchanged NEXT227/NEXT224 reconstruction and gate evaluators.

Date frozen: 2026-08-09 (America/Chicago)

## Status, novelty, and engineering prerequisite

NEXT251--NEXT252 tested species-conditioned Voronoi face-topology consistency.
All sixteen features were complete and finite, but none passed both discovery-
source raw gates. NEXT253 and NEXT254 were therefore forbidden and remain
uncreated. Those inspected topology statistics and directions may not be
reopened or retuned.

The new mechanism is the geometry of the *dual void cage* at each Voronoi
vertex, not the polygon topology of the atom-centered Voronoi cell. It is also
distinct from NEXT168/NEXT173 atom-centered neighbor-direction Gram matrices,
NEXT179 atom-centered directional closure, NEXT22 bond-valence vector
asymmetry, and the NEXT247 third-order bond-order invariants. Repository
structural inspection found no prior periodic Delaunay or void-centered cage
feature implementation.

The physical and geometric basis is:

- Blatov and Shevchenko, *Acta Crystallographica A* **59**, 34--44 (2003),
  <https://doi.org/10.1107/S0108767302020603>, identify Voronoi vertices as
  void centers in the dual crystal-chemical description.
- Hinuma, *Science and Technology of Advanced Materials: Methods* **2**
  (2022), <https://doi.org/10.1080/27660400.2022.2059336>, treats Delaunay
  polyhedra as void-embracing local structural units and explicitly retains
  non-tetrahedral polyhedra in degenerate co-spherical cases instead of
  forcing an arbitrary tetrahedralization.

Before this freeze, label-free engineering prototypes used ideal FCC, BCC,
NaCl, diamond, cubic perovskite, and a low-symmetry triclinic P1 structure.
The direct SciPy/Qhull tetrahedralization prototype was rejected because its
arbitrary triangulation of co-spherical polyhedra was not invariant. The
complete-cage construction below reproduced the same feature vector after a
rigid rotation, uniform scale, periodic translation, and a `2 x 1 x 1`
supercell to approximately `1e-15` on all six structures. A preliminary FCC
discrepancy was traced to NumPy's linearly interpolated sample quantile, which
changes interpolation indices when an identical population is replicated.
Consequently, all frozen within-structure quantiles below use the empirical
inverse-CDF method. No discovery outcome or endpoint was opened during these
engineering checks.

## Non-negotiable information boundary

Every executable feature and formula may use only composition and the initial,
unrelaxed periodic geometry. Reject any ASE `Atoms` object with a calculator,
nonempty `info`, or arrays other than `numbers` and `positions`. Do not read or
compute a DFT energy, force, stress, hull value, band property, learned energy/
force/stress proxy, model or proxy potential, relaxation step, relaxed
structure, or trajectory.

Discovery outcomes are offline labels only. NEXT255 accepts no label, endpoint,
validation, or replication path. NEXT256--NEXT258 may read only the frozen
SCIGEN and WyFormer discovery endpoints already used by NEXT252. Internal
validation and replication remain physically unopened unless a candidate
passes every frozen discovery gate. No canonical paper, note,
preregistration, README, or prior script/result may be modified.

## NEXT255 fixed complete-cage construction

Use exactly
`VoronoiNN(weight="solid_angle", tol=0, cutoff=13, compute_adj_neighbors=True)`
on the raw periodic structure. At every atomic center `i`, deduplicate a
repeated `(site_index, rounded_integer_image)` face by retaining the largest
positive finite face area, with lexical tie-breaking by the integer Voronoi
vertex tuple and the rounded Cartesian displacement. Sort the retained face
records lexically.

For every Voronoi vertex identifier `v` in that atom-centered cell, collect the
Cartesian displacements `r_j` from center `i` to all neighbor images whose
faces contain `v`. Require at least three incident faces. Determine the void
center `x` relative to atom `i` from the simultaneous perpendicular-bisector
equations

```text
r_j dot x = |r_j|^2 / 2.
```

Solve by `numpy.linalg.lstsq`; require rank three and maximum relative equation
residual no larger than `1e-8`, with denominator
`max(1, max_j ||r_j|^2 / 2|)`. The complete co-spherical Delaunay cage is the
central atom plus every incident neighbor. Its unit directions from the void
center are

```text
u_0 = -x / |x|
u_j = (r_j - x) / |r_j - x|.
```

Require every ray norm to exceed `1e-12`. Do not triangulate a cage with more
than four atoms. Each `(atomic center, incident Voronoi vertex)` is retained as
one atom--void incidence; this intentionally weights a physical void by its
cage size and avoids nonlocal periodic-coordinate canonicalization.

For a cage of size `k`, define

```text
G = (1/k) sum_a u_a u_a^T,
lambda_1 <= lambda_2 <= lambda_3 = eigenvalues(G),

tightness = 3 lambda_1,
volume = 27 lambda_1 lambda_2 lambda_3,
eigenratio = lambda_1 / lambda_3,
closure = 1 - |(1/k) sum_a u_a|.
```

Because `trace(G)=1`, every metric has frozen range `[0,1]`; larger means a
more isotropic or more directionally closed void cage. Accept final roundoff
outside the interval only up to `1e-12`, then clip. Any other nonfinite,
rank-deficient, or out-of-range cage fails the whole record closed.

## NEXT255 fixed structure features

For the complete atom--void incidence population of each metric `m`, compute:

```text
mean(m)
q10(m) = empirical inverse-CDF quantile at 0.10
q25(m) = empirical inverse-CDF quantile at 0.25
lower_quartile_mean(m) = mean of values <= q25(m) + 1e-12
```

The `1e-12` inclusion tolerance is numerical grouping only and is fixed before
outcomes. All quantiles use `numpy.quantile(..., method="inverted_cdf")`.
These population summaries, unlike linearly interpolated quantiles, are
unchanged by exact supercell replication. The exact ordered feature universe
is:

```text
dvci_tightness_mean
dvci_tightness_q10
dvci_tightness_q25
dvci_tightness_lower_quartile_mean
dvci_volume_mean
dvci_volume_q10
dvci_volume_q25
dvci_volume_lower_quartile_mean
dvci_eigenratio_mean
dvci_eigenratio_q10
dvci_eigenratio_q25
dvci_eigenratio_lower_quartile_mean
dvci_closure_mean
dvci_closure_q10
dvci_closure_q25
dvci_closure_lower_quartile_mean
```

All sixteen values must be finite for a supported record. NEXT255 must cover
exactly the same `13,470` SCIGEN and `5,232` WyFormer discovery identities as
NEXT251. Formal publication requires 100% supported, finite rows in both
sources.

## Task 1: NEXT255 TDD and label-free implementation

**Files:**

- Create: `tests/test_next255_delaunay_void_cage_isotropy.py`
- Create: `src/next255_delaunay_void_cage_isotropy.py`
- Preserve: every prior script, test, plan, and result

**Steps:**

1. Write the focused test before the source and run it to observe a missing-
   module RED result.
2. Unit-test a hand-built regular tetrahedral direction set, an anisotropic
   direction set, rank/residual/range failures, and the exact sixteen-feature
   ordering.
3. Unit-test that inverse-CDF `q10/q25` and lower-quartile means are exactly
   invariant when a population is replicated, including the FCC population
   with six `0.6` and eight `1.0` values that fails under linear interpolation.
4. Test ideal FCC and NaCl cage populations plus rigid rotation, uniform scale,
   periodic translation, neighbor ordering, and supercell invariance.
5. Test the exact geometry-only builder interface and fail-closed calculator,
   metadata, array, nonperiodic, missing-input, and existing-output cases.
6. Implement only the code required by the frozen specification, then run
   `pytest -q tests/test_next255_delaunay_void_cage_isotropy.py` to GREEN.
7. Run a nonformal tiny label-free archive smoke test without any endpoint
   path. Inspect only support, finite counts, and invariance diagnostics.
8. Freeze the source/test hashes, run the full formal label-free build into
   `$PRIS_ARCHIVE/next255_delaunay_void_cage_isotropy_v1`,
   and publish manifest, catalogue, Parquet, source, and design hashes.

No commit is authorized on the shared dirty research checkout.

## NEXT256 fixed feature audit

Reconstruct the exact published NEXT224 frontier and rejected-extreme cohort
through unchanged NEXT227 machinery. Use identical reduced-formula folds,
cohort counts, class counts, coverage requirements, and AUC gates. Normalize
each feature using only the finite combined discovery population, with
inverse-CDF `1/16` and `15/16` cutoffs; outcomes do not enter normalization.

Audit exactly the sixteen NEXT255 feature names above, each with the frozen
direction `protected_high`. Opposite directions, transformed variants, feature
combinations, and post-outcome direction changes are forbidden. A hypothesis
is eligible only if both sources pass every frozen raw-feature gate. Reporting
rank is largest minimum worst-fold AUC, largest minimum aggregate AUC, largest
mean aggregate AUC, then lexical hypothesis. If none is eligible, the branch
closes and NEXT257/NEXT258 remain forbidden.

## Task 2: NEXT256 TDD and fixed audit

**Files:**

- Create: `tests/test_next256_dvci_feature_audit.py`
- Create: `src/next256_dvci_feature_audit.py`

**Steps:**

1. Write the focused test first and observe the missing-module RED result.
2. Test the exact sixteen hypotheses and directions, fixed reporting rank,
   source-prefixed material identity alignment, geometry-manifest/source-hash
   provenance, and exclusion of validation/replication interfaces.
3. Implement the audit by reusing NEXT227 reconstruction and gates without
   changing their constants.
4. Run `pytest -q tests/test_next256_dvci_feature_audit.py` to GREEN.
5. Run the formal audit once into
   `$PRIS_ARCHIVE/next256_dvci_feature_audit_v1`.
6. Publish all sixteen source-wise aggregate/macro/worst-fold AUCs, coverage,
   pass/fail gates, eligible count, and exact output hashes. Do not search a
   formula unless the manifest explicitly authorizes NEXT257.

## NEXT257 conditional one-term search

NEXT257 is authorized only if NEXT256 publishes at least one eligible
hypothesis. Start from the exact NEXT224 score. For every eligible hypothesis,
use its frozen NEXT256 `q_lo/q_hi` protection and exactly

```text
h = local_width_fraction * NEXT214_REPAIR_WIDTH
local_weight = max(0, 1 - abs(base_score - NEXT224_THRESHOLD) / h)
delta = amplitude_fraction * h * local_weight * (1 - 2 * protection)
score = max(0, base_score + delta)
```

Widths are exactly `{1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1}` and amplitudes are
exactly `{1/4, 1/2, 1}`. Publish one exact NEXT224 no-op plus
`21 * eligible_hypothesis_count` new candidates. Reporting selection is
restricted to `eligible_new_candidate == true`.

Freeze is authorized only for a new candidate passing source AUC, SAFE, BROAD,
and every discovery gate. If none passes all gates but at least one new
candidate passes AUC+SAFE and fails BROAD, NEXT258 is authorized for exactly
that sorted identity population. Otherwise close the branch.

## NEXT258 conditional BROAD diagnostic

NEXT258 must reproduce every authorized NEXT257 record at evaluator level and
recompute unchanged BROAD threshold tables only for the authorized AUC+SAFE/
non-BROAD population. Rank by fewest failed constraints, smallest normalized
shortfall sum, then lexical candidate key. Compare the closest record with the
frozen NEXT235 reference `(5, 0.12339543654931197)`. Strict improvement
requires a lexicographically smaller tuple. Otherwise close the branch.

## Reporting, verification, and stopping rule

Append results only to the independent report
`reports/2026-08-08-next115-next117-hcid-no-dft-search.md`. Do not modify
canonical manuscript or reporting files. Verify focused tests, full pytest,
all manifest input/output/source hashes, boundary flags, `git diff --check`,
canonical zero-diff, and CodeGraph synchronization.

This discovery branch cannot claim a confirmed law. Even an all-discovery-
gate candidate must first receive a separately frozen unseen-source or still-
sealed internal-validation protocol. All scripts, tests, plans, formal output
directories, and report text are additive.
