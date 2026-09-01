# NEXT291--NEXT294 Radius-Weighted Delaunay-Cage Isotropy Plan

> **For Codex:** use the already-loaded brainstorming, writing-plans,
> test-driven-development, executing-plans, systematic-debugging, and
> verification-before-completion workflows. The active shared checkout is
> intentional. Do not create a branch, commit, PR, or worktree.

**Status:** frozen before opening any NEXT291--NEXT294 discovery outcome.

**Goal:** Test whether the radius-weighted directional closure of the periodic
regular-Delaunay cages dual to the NEXT267 power diagram supplies a genuinely
new, fully pre-DFT correction to the frozen NEXT224 law.

**Architecture:** NEXT291 independently reconstructs the exact NEXT267
periodic power cells from physically isolated discovery geometry and publishes
sixteen label-free radius-weighted cage descriptors. NEXT292 audits exactly
the sixteen preregistered directions in the unchanged NEXT224 rejected-extreme
cohort. NEXT293 searches the unchanged margin-local grammar only if NEXT292
authorizes at least one hypothesis. NEXT294 is an exact BROAD residual
diagnostic only if NEXT293 authorizes an AUC+SAFE/non-BROAD identity set.

**Tech stack:** Python 3.11, ASE, pymatgen, NumPy, pandas, SciPy
`linprog`/`HalfspaceIntersection`/`ConvexHull`, pytest, Parquet, SHA-256, and
the unchanged NEXT227/NEXT224 reconstruction and gate evaluators.

Date frozen: 2026-08-09 (America/Chicago).

## 1. Prior-mechanism audit and scientific motivation

This branch is additive and preserves every prior plan, script, test, output,
report section, and canonical artifact.

The following mechanisms have already been tested and are not reopened:

- ordinary atom-centred Voronoi contacts, topology, face bond order, and local
  rigidity (P2/P10 and NEXT166--NEXT252);
- ordinary unweighted Delaunay void-cage isotropy (NEXT255--NEXT258), periodic
  void-channel bottlenecks (NEXT259--NEXT262), and local angular persistence
  (NEXT263--NEXT266);
- radical/power-cell volume allocation, Chebyshev radius, centroid,
  anisotropy, species summaries, spatial autocorrelation, and exact cell
  sphericity--volume coupling (NEXT267--NEXT286);
- post-outcome variants of the existing PRV score, including one-sided relief
  and confidence deadzones (NEXT287--NEXT290).

The untested object is the *dual cage of a power-cell vertex*. For unequal
atomic radii, this is a regular (weighted-Delaunay) cage, not the ordinary
equal-distance Delaunay cage. Park and Shibutani showed that radius-weighted
Voronoi analysis can change local topology and allocated atomic volume by more
than ordinary equal-bisector analysis in size-disperse atomistic systems
(Intermetallics 15, 187--192, 2007,
<https://doi.org/10.1016/j.intermet.2006.05.005>). Luchnikov et al. introduced
geometric potentials of Delaunay simplices and related them to vibrational
structure in a model glass (Physical Review B 62, 3181, 2000,
<https://doi.org/10.1103/PhysRevB.62.3181>). The CGAL periodic regular-
triangulation definition confirms that regular triangulations are the weighted
Delaunay duals of power diagrams
(<https://doc.cgal.org/latest/Periodic_3_triangulation_3/>).

These sources justify the geometric object only. They do not establish a
crystal-stability law, do not enter the executable formula, and do not replace
the frozen cross-source discovery test.

Three approaches were compared before outcomes:

1. **Selected:** retain each complete co-power-spherical cage without
   triangulating it and weight its generator directions by projected atomic
   angular scale `(r/d)^2`.
2. Force every cage into weighted-Delaunay tetrahedra and score tetrahedral
   perfectness. Rejected because co-power-spherical cages have nonunique
   tetrahedralizations, creating representation-dependent values.
3. Publish only power-vertex clearance/radius summaries. Rejected because
   these mostly re-express NEXT267 sphere crossing/volume ratios and NEXT259
   cavity/bottleneck radii instead of adding cage-direction information.

Before this freeze, label-free engineering prototypes used ideal one-site
simple cubic, FCC, BCC, NaCl, diamond, and an asymmetric triclinic Si--O--Na
cell. All produced finite bounded metrics and power-cell tiling errors below
`5e-16`. On the asymmetric cell, rigid rotation, periodic translation, site
permutation, and exact `2 x 1 x 1` replication changed any metric mean by at
most `1.7e-16`. No endpoint or outcome was opened for these checks.

## 2. Non-negotiable information boundary

Every executable feature and candidate formula receives only element
identities, frozen neutral tabulated radii, and one initial, raw, unrelaxed
three-dimensional periodic geometry. Reject an ASE `Atoms` object with a
calculator, nonempty `info`, arrays other than `numbers` and `positions`,
nonfinite values, nonpositive cell volume, or incomplete PBC.

The executable path must not read, infer, call, or compute:

- a DFT energy, force, stress, charge density, band value, hull value, or any
  other per-structure DFT result;
- a learned energy/force/stress model, MLIP, interatomic potential, proxy
  potential, or model-derived descriptor;
- a relaxed structure, trajectory, later coordinate/cell, physical
  relaxation, or same-composition alternative;
- a validation or replication geometry, endpoint, outcome, or output.

Discovery outcomes are offline labels and may be joined only by NEXT292 after
the NEXT291 source, catalogue, directions, tolerances, input identities, and
this plan are frozen. Internal validation and replication remain physically
sealed even if a discovery candidate passes all gates; opening them requires a
separate future freeze. Unsupported rows abstain/fail open and can never become
automatic rejections.

Every formal manifest must publish exact false values for:

```text
dft_calculation_executed
dft_values_used_by_executable_formula
learned_energy_force_stress_proxy_used
model_or_proxy_potential_used
physical_relaxation_executed
opened_validation_outputs_used
scigen_replication_endpoint_opened
wyformer_replication_endpoint_opened
scientific_improvement_claim
```

## 3. NEXT291 frozen periodic regular-Delaunay construction

### 3.1 Exact power half-spaces

Reuse, without editing NEXT267, its neutral-radius policy, Minkowski-reduced
cell handling, lattice Wigner--Seitz construction, conservative periodic-image
bound, `2,000,000` directed-image guard, radical half-spaces, Chebyshev
interior LP, `Qx` half-space/hull options, vertex feasibility tolerance,
`1e-6` volume-tiling certificate, and `1e10` output grid.

For generator site `i`, position `p_i`, radius `r_i`, competitor image
`(j,T)`, and displacement `d = p_j + T - p_i`, use the exact NEXT267 plane

```text
u_ijT dot y <= c_ijT
u_ijT = d / |d|
c_ijT = (|d|^2 + r_i^2 - r_j^2) / (2 |d|).
```

Track every plane's exact `(site_index, integer_image, displacement, radius)`.
The lattice Wigner--Seitz planes represent periodic images of site `i` and
must be tracked as such. Empty/lower-dimensional power cells add no cage
incidences but retain zero cell volume for the unchanged tiling certificate.

### 3.2 Complete regular-Delaunay cage at a power vertex

For every unique convex-hull vertex `x` of a positive-volume power cell, mark
a plane active when

```text
abs(u dot x - c) <= 1e-8 * max(1, R_WS).
```

Deduplicate active competitors by exact `(site_index, integer_image)` and
require at least three unique competitors plus rank three active normals.
The cage is the central generator plus every retained active competitor;
never tetrahedralize a cage with more than four generators. Require every
center-to-generator distance to exceed `1e-12` and verify the common power
distance equation

```text
|x|^2 - r_i^2 = |x-d_j|^2 - r_j^2
```

to absolute tolerance
`1e-7 * max(1, R_WS^2, max_j(r_j^2))`. A violation fails the record closed.
Each `(central site, power-cell vertex)` remains one incidence. This exact
incidence weighting is invariant to integral periodic replication and avoids
nonlocal, tolerance-sensitive canonicalization of a physical cage.

### 3.3 Radius-weighted cage tensor

For cage generators `a`, let

```text
v_a = p_a - x
d_a = |v_a|
u_a = v_a / d_a
raw_weight_a = (r_a / d_a)^2
omega_a = raw_weight_a / sum_b(raw_weight_b)
G_w = sum_a omega_a u_a u_a^T
m_w = sum_a omega_a u_a.
```

The `(r/d)^2` factor is the projected angular-size scale of a radius-`r`
generator viewed from the cage centre. It is analytic, positive, dimensionless,
and uses no outcome or energetic input. Require finite positive weights. With
eigenvalues `lambda_1 <= lambda_2 <= lambda_3` of `G_w`, define exactly

```text
tightness = 3 lambda_1
volume = 27 lambda_1 lambda_2 lambda_3
eigenratio = lambda_1 / lambda_3
closure = 1 - |m_w|.
```

Because `trace(G_w)=1` and the weights form a convex distribution, all four
metrics lie in `[0,1]`. Permit final roundoff outside that interval only up to
`1e-12`, then clip; otherwise fail closed.

## 4. NEXT291 frozen feature catalogue

For every incidence population and metric `m`, publish:

```text
mean(m)
q10(m)
q25(m)
lower_quartile_mean(m) = mean(values <= q25 + 1e-12).
```

Quantiles use `numpy.quantile(..., method="inverted_cdf")`. The inclusion
tolerance is numerical grouping only. The exact ordered feature universe is:

```text
rwdci_tightness_mean
rwdci_tightness_q10
rwdci_tightness_q25
rwdci_tightness_lower_quartile_mean
rwdci_volume_mean
rwdci_volume_q10
rwdci_volume_q25
rwdci_volume_lower_quartile_mean
rwdci_eigenratio_mean
rwdci_eigenratio_q10
rwdci_eigenratio_q25
rwdci_eigenratio_lower_quartile_mean
rwdci_closure_mean
rwdci_closure_q10
rwdci_closure_q25
rwdci_closure_lower_quartile_mean
```

All sixteen preregistered directions are `protected_high`. Publish every one
of the exact `13,470` SCIGEN and `5,232` WyFormer discovery identities. A
supported row must have a positive cage-incidence count, finite bounded
features, the unchanged NEXT267 tiling certificate, and no malformed cage.
Unsupported rows retain identifiers and NaN features. Formal publication
requires source coverage at least `0.95` and records exact support identities
and failure reasons; no outcome is available at this stage.

## 5. Task 1: NEXT291 TDD and label-free formal build

**Create only:**

- `tests/test_next291_radius_weighted_delaunay_cage_isotropy.py`
- `src/next291_radius_weighted_delaunay_cage_isotropy.py`
- external formal directory
  `$PRIS_ARCHIVE/next291_radius_weighted_delaunay_cage_isotropy_v1`

Steps:

1. Write the focused test first and observe a missing-module RED result.
2. Test exact schema/directions, an analytic regular tetrahedral cage with
   equal radii, unequal-radius weighting, malformed/rank/power-equality guards,
   inverse-CDF replication invariance, and bounded postconditions.
3. Test ideal simple cubic, FCC, BCC, NaCl, and diamond plus rotation,
   translation, site permutation, equivalent lattice rebasing, and integral
   supercell representation.
4. Test the exact geometry-only builder interface and fail-closed calculator,
   metadata, extra-array, PBC, missing-input, and existing-output cases.
5. Implement only the frozen specification, reusing immutable NEXT267 helpers
   where exact and copying only the additional vertex-to-plane provenance
   needed for the dual cage.
6. Run a tiny nonformal geometry-only smoke without any endpoint path; inspect
   only support, finite values, invariance, tiling error, and runtime.
7. Pin the plan and upstream source hashes, then run each physically isolated
   discovery geometry source exactly once and publish manifest, catalogue,
   separate Parquet tables, and hashes atomically.

## 6. NEXT292 prospective feature audit

Reconstruct the exact NEXT224 score/support and rejected-extreme cohort through
unchanged NEXT227 machinery. Normalize each feature using only finite combined
discovery geometry with inverse-CDF `1/16` and `15/16` cutoffs. Audit exactly
the sixteen feature names above and only `protected_high`. Opposite directions,
new quantiles, transformations, conjunctions, and post-outcome direction
changes are prohibited.

Use the unchanged reduced-formula folds, coverage/class-count gates, source
aggregate AUC, macro-fold AUC, and worst-fold AUC gates from NEXT268. A
hypothesis is eligible only if both discovery sources pass every raw gate.
Reporting rank cannot authorize a failing hypothesis. If zero hypotheses are
eligible, close the branch and do not create NEXT293/NEXT294.

## 7. Task 2: NEXT292 TDD and formal audit

**Create only:**

- `tests/test_next292_rwdci_feature_audit.py`
- `src/next292_rwdci_feature_audit.py`
- external formal directory
  `$PRIS_ARCHIVE/next292_rwdci_feature_audit_v1`

Write tests first for the exact hypothesis universe/directions, deterministic
reporting rank, source-prefixed identity alignment, source/input/source hashes,
no validation/replication interface, and missing-input failure. Then implement
by adapting the frozen NEXT268/NEXT284 audit structure without changing its
gates. Run the audit once and freeze the exact eligible identities and digest.

## 8. Conditional NEXT293 margin-local search

Create NEXT293 only if NEXT292 authorizes at least one exact hypothesis. Use
one unchanged NEXT224 reproduction control plus, for every eligible
hypothesis, the exact frozen NEXT269 grid:

```text
local_width_fraction in {1/64,1/32,1/16,1/8,1/4,1/2,1}
amplitude_fraction in {1/4,1/2,1}
h = local_width_fraction * NEXT214_REPAIR_WIDTH
w(s) = max(0, 1 - |s - NEXT224_THRESHOLD| / h)
score = max(0, s + amplitude_fraction*h*w(s)*(1 - 2*protection)).
```

The exact interval edge has zero weight; missing/unsupported/outside rows keep
the NEXT224 score and support. Search only frozen eligible identities. Require
both-source AUC, all twelve SAFE cells, BROAD, and every unchanged discovery
gate. A reporting leader is not a frozen law. If no all-gate candidate exists
but at least one new candidate passes AUC+SAFE and fails BROAD, authorize
NEXT294 only for that exact sorted identity population. Otherwise close.

**Conditional files only:**

- `tests/test_next293_rwdci_margin_local_search.py`
- `src/next293_rwdci_margin_local_search.py`
- `$PRIS_ARCHIVE/next293_rwdci_margin_local_search_v1`

## 9. Conditional NEXT294 exact BROAD diagnostic

Create NEXT294 only if NEXT293 explicitly authorizes it. Reproduce only the
authorized evaluator records and unchanged BROAD threshold tables. Rank by
`(failed_constraint_count, normalized_shortfall_sum, candidate_key)` and
compare with the frozen NEXT270 reference
`(5, 0.0955435292756307)`. No new feature, direction, threshold, coefficient,
formula, or endpoint may be introduced. A strict diagnostic improvement may
justify only a new pre-outcome freeze; it cannot open validation or establish
a law.

**Conditional files only:**

- `tests/test_next294_rwdci_broad_diagnostic.py`
- `src/next294_rwdci_broad_diagnostic.py`
- `$PRIS_ARCHIVE/next294_rwdci_broad_diagnostic_v1`

## 10. Verification, stopping, and reporting

1. Run every new focused test and the complete repository suite.
2. Verify plan/source/test/formal input/output hashes and atomic publication.
3. Verify all no-DFT/no-proxy/no-relaxation/sealed-endpoint flags.
4. Verify `paper/`, `tex/`, `notes/`, `README.md`, and `PREREG.md` remain
   untouched by this branch.
5. Verify CodeGraph has no pending sync.
6. Append only the independent report
   `reports/2026-08-08-next115-next117-hcid-no-dft-search.md`.

Stop at the first failed authorization gate. Preserve all earlier files and
results. Do not claim replacement of or superiority to Pauling unless a
prospectively frozen candidate later passes separately authorized, sealed
validation and replication. The overall goal remains active after a negative
or discovery-only result.
