# Radical-Facet Minkowski Tensor Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` and
> `superpowers:test-driven-development` task by task. Do not delegate.

**Goal:** Test whether the intrinsic surface-normal anisotropy of each raw
radius-aware periodic power cell supplies a robust, transferable pre-DFT
crystal-validity signal and, only if prospectively authorized, a stronger
bounded screening law.

**Architecture:** NEXT335 reconstructs the same labelled power cells as
NEXT267/NEXT283, triangulates only their geometric boundary, and computes the
translation-invariant surface-normal Minkowski tensor. It freezes one
sitewise eigenvalue-ratio statistic and one structure-level q10. A
label-blind probe must pass support, strict domain, nondegeneracy,
representation invariance, and novelty against ordinary Voronoi morphology,
local directional Gram tensors, radical-cell shape, and all recent mechanism
families. Formal NEXT335, fixed NEXT336 audit, and NEXT337/NEXT338 search are
strictly contingent in that order.

**Tech Stack:** Python 3.11, NumPy, SciPy HiGHS/HalfspaceIntersection/
ConvexHull, pandas, ASE, existing isolated discovery inventories, pytest,
CodeGraph.

## Scientific and information-boundary freeze

For a nonempty radius-aware power cell `P_i`, let each oriented boundary
triangle `t` have area `a_it > 0` and unit normal `n_it`. Define the normalized
surface-normal Minkowski tensor

```text
T_i = sum_t a_it (n_it outer n_it) / sum_t a_it,
beta_i = lambda_min(T_i) / lambda_max(T_i),       0 < beta_i <= 1.
```

Coplanar triangulation does not affect `T_i` because all triangles on a facet
share the same normal and their areas sum to the polygon area. The sole
frozen feature is
`rfmt_surface_normal_beta_q10 = inverse_cdf_0.10({beta_i})` in the
`protected_high` direction. The tensor has trace one, is translation and
scale invariant, and rotates by conjugation; its eigenvalues and `beta_i` are
therefore rigid-rotation invariant. A sphere-like/isotropic normal measure
has `beta=1`, whereas boundary normals concentrated along one axis drive it
toward zero.

Tensor-valued Minkowski functionals are established intrinsic morphology and
anisotropy descriptors, including for three-dimensional Voronoi cellular
complexes (Schroeder-Turk et al., arXiv:`1009.2340`; Advanced Materials 23,
2535--2553, 2011, DOI `10.1002/adma.201100562`). The frozen interpretation is
that an extremely directionally collapsed local space allocation is less
protected against raw geometric invalidity. Layered or chain-like stable
crystals may legitimately be anisotropic, so the direction is only a
prospective screening hypothesis, not a theorem of energy or stability.

Executable inputs are limited to element identities, deterministic tabulated
radii, and one initial raw unrelaxed periodic geometry. The branch must not
execute or consume DFT calculations or per-structure DFT values; learned
energy/force/stress proxies; MLIPs; model/proxy potentials; relaxed
structures; trajectories; later geometries; discovery labels during the
probe/build; validation outcomes; or replication outcomes. Discovery
outcomes may enter NEXT336 only as offline audit labels. Coordinates and
cells are never optimized or physically moved.

## Distinction and alternatives frozen before feature evaluation

1. NEXT168 and NEXT173 already compute unweighted and chemistry/distance-
   weighted local direction Gram tensors on formal-valence Voronoi/CrystalNN
   contact graphs, using `3 lambda_min` and determinant. RFMT uses the
   complete radius-aware power-cell boundary surface measure and the standard
   Minkowski `lambda_min/lambda_max` anisotropy ratio. NEXT168 and NEXT173 are
   mandatory first-line novelty controls; high correlation terminates RFMT.
2. NEXT179 and NEXT295/RFPE/NEXT323 measure convex closure or positive
   equilibrium of direction populations. RFMT solves no balance program and
   is continuous in boundary area weights. They remain mandatory controls.
3. NEXT239 computes ordinary Voronoi area-weighted q4/q6 and scalar facet
   evenness, not the rank-two power-cell surface-normal tensor. NEXT239 is a
   mandatory control.
4. NEXT267 vertex covariance anisotropy depends on vertex locations; NEXT283
   total surface area and sphericity are scalar isoperimetric measures; NEXT291
   uses Delaunay cage radial vectors. None is the boundary-normal Minkowski
   tensor, but all remain mandatory controls.
5. RFMP raw minimum facet area was rejected after its frozen support failure.
   RFMT weights every facet by area and is not dominated by a single
   `1e-14` sliver; no minimum, floor, clipping, tolerance relaxation, or
   trimmed facet population is permitted.

Rejected alternatives: surface-area evenness (NEXT239 duplicate), total
sphericity (NEXT283 duplicate), vertex covariance (NEXT267 duplicate),
`3 lambda_min` or determinant (NEXT168/NEXT173 duplicate), and a second
Minkowski tensor/aggregate. No alternative direction, eigenvalue function,
aggregate, quantile, threshold, chemistry condition, conjunction, or second
feature may be added after the label-blind probe starts.

## Frozen gates

- Label-blind probe: deterministically select 80 initial discovery geometries
  per source after loading the complete identifier-bearing inventory; read no
  endpoint, label, validation, replication, relaxed geometry, DFT field, or
  model-potential field.
- Engineering: finite `(0,1]` values; support at least 72/80 per source; at
  least 20 values unique at `1e-10`; surface-triangle area agreement with
  ConvexHull at relative error at most `1e-7`; volume-tiling agreement at the
  unchanged NEXT267 tolerance; rigid rotation, periodic translation, site
  permutation, unimodular lattice rebasing, and exact integral-supercell
  error at most `1e-8` before `1e-10` output quantization.
- Novelty: maximum absolute Spearman correlation below `0.90` independently
  in both sources against available label-free NEXT168, NEXT173, NEXT179,
  NEXT239, NEXT243, NEXT247, NEXT251, NEXT255, NEXT259, NEXT263, NEXT267,
  NEXT271, NEXT275, NEXT279, NEXT283, NEXT291, NEXT295, NEXT299, NEXT303,
  NEXT307, NEXT311, NEXT315, NEXT319, and NEXT323 populations on the same
  records. Failure terminates before a formal NEXT335 build or discovery
  outcomes.
- Formal NEXT335 coverage: at least `0.90` independently in SCIGEN and
  WyFormer, with unsupported rows retained as abstentions and never imputed.
- NEXT336 unchanged audit gates: minimum cell coverage `0.90`, minimum class
  count `20`, pooled AUC `0.55`, macro AUC `0.53`, and worst-fold AUC `0.50`
  in both sources, reduced-formula folds, inverse-CDF `1/16` and `15/16`.
- Empty eligible set means `next337_search_authorized=false` and
  `rfmt_branch_terminated=true`; NEXT337/NEXT338 must not exist.
- If authorized, NEXT337 may reuse only the unchanged NEXT269 margin-local
  grammar and fixed NEXT224/NEXT135 base score. NEXT338 is discovery-only
  BROAD diagnosis. Validation and replication remain sealed.

## Task 1: surface-normal tensor kernel and geometry wrapper

**Files:**

- Create: `src/next335_radical_facet_minkowski_tensor.py`
- Create: `tests/test_next335_radical_facet_minkowski_tensor.py`
- Reuse without modification: NEXT267/NEXT283/NEXT331 source files.

Write RED analytic tests for isotropic normals, an anisotropic rectangular
box, scale/rotation invariance, invalid tensors, and boundary-triangle area
accounting. Implement strict half-space reconstruction and periodic
radius-aware cells. Add NaCl/CsCl/ZnS/distorted-cell, exact geometry-only,
determinism, bounds, and all five representation-equivalence tests.

## Task 2: label-blind novelty probe

**Files:**

- Create: `experiments/next335_rfmt_label_blind_probe.py`
- Create: `tests/test_next335_rfmt_label_blind_probe.py`
- Create only after a passing probe:
  `experiments/next335_rfmt_label_blind_probe_result.json`

Reuse the complete-inventory selection and strict geometry loaders, compute
all frozen engineering/novelty statistics, and record design plus executed
source hashes. Stop immediately if any gate fails.

## Task 3: contingent NEXT335 formal build

Only after a passing probe, add RED builder tests for exact input identity,
geometry-only reads, abstentions, atomic publication, manifests/source
hashes, false boundary flags, and both coverage floors. Publish only into
`$PRIS_ARCHIVE/next335_radical_facet_minkowski_tensor_v1`.

## Task 4: contingent NEXT336 fixed discovery-only audit

Only after formal coverage passes, create `src/next336_rfmt_feature_audit.py`
and tests, reuse unchanged NEXT268/NEXT324 cohort/fold/audit helpers, and open
only physically isolated discovery endpoints as offline labels. Publish into
`$PRIS_ARCHIVE/next336_rfmt_feature_audit_v1` and
freeze the eligible-set digest.

## Task 5: contingent NEXT337/NEXT338 search

Create NEXT337/NEXT338 only when the exact NEXT336 manifest authorizes them.
Reuse unchanged NEXT269 search and BROAD diagnostic. Do not transform/reverse
the feature or open validation/replication.

## Task 6: verification and independent report

Append only to
`reports/2026-08-08-next115-next117-hcid-no-dft-search.md`; do not modify
`paper/`, `tex/`, `notes/`, `README.md`, or `PREREG.md`. Run focused/adjacent
tests, byte compilation, exact hashes/boundary assertions, CodeGraph sync,
and the complete suite. Report exact stop or authorization state and strict
claim limits before any canonical change.

## Execution note

The checkout is intentionally dirty and shared. Preserve all prior content,
work additively, make no Git commit/merge/cleanup, and do not delegate.
