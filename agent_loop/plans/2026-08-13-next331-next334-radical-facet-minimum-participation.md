# Radical-Facet Minimum Participation Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` and
> `superpowers:test-driven-development` task by task. Do not delegate.

**Goal:** Test whether a vanishingly small facet in a radius-aware periodic
power cell is a transferable, representation-invariant, pre-DFT signal of a
geometrically fragile crystal neighborhood and, only if prospectively
authorized, whether it can improve the bounded screening law.

**Architecture:** NEXT331 reconstructs the exact NEXT267/NEXT283 labelled
radical/power cells but retains every distinct active facet area. It reduces
each cell to one minimum-area participation number and a structure to one
frozen lower-tail aggregate. A label-blind probe must first establish support,
nondegeneracy, invariance, and novelty against both ordinary-Voronoi area
evenness and prior radical-cell mechanisms. A formal NEXT331 build and the
NEXT332 fixed discovery-only audit may exist only after that probe passes.
NEXT333/NEXT334 remain contingent on the unchanged audit gates.

**Tech Stack:** Python 3.11, NumPy, SciPy HiGHS/HalfspaceIntersection/
ConvexHull, pandas, ASE, existing NEXT267/NEXT283 geometry and isolated
discovery inventories, pytest, CodeGraph.

## Scientific and information-boundary freeze

For raw site `i`, let its weighted periodic power cell be

```text
P_i = intersection_f {x : n_if . x <= b_if}.
```

For every distinct active supporting plane, retain the geometric polygon
`F_if = P_i intersection {x : n_if . x = b_if}` and compute its Euclidean
area `A_if > 0`. If `K_i` facets are active, define

```text
m_i = K_i * min_f(A_if) / sum_f(A_if),       0 < m_i <= 1.
```

The sole frozen hypothesis is
`rfmp_minimum_area_participation_q10 = inverse_cdf_0.10({m_i})` in the
`protected_high` direction. It equals one only when every facet has equal
area and tends to zero when at least one radical adjacency is represented by
a vanishingly small facet. Facet planes that are geometrically coincident are
deduplicated on the existing `1e-10` plane grid. Lattice self-image facets are
retained because they become ordinary image-neighbor facets under an exact
supercell representation of the same infinite crystal.

Voronoi topology is known to change under infinitesimal perturbations of
ordered point sets (Lazar, Han, and Srolovitz, PNAS 112, E5769--E5776, 2015,
DOI `10.1073/pnas.1505788112`; Leipold et al., 2015,
arXiv:`1509.03280`). Because a Voronoi/power facet is dual to one neighbor
adjacency and disappears at zero area, `m_i` is prospectively interpreted as
a continuous distance-like warning for an adjacency near a topological flip.
That interpretation and the `protected_high` direction are hypotheses, not
theorems of energetic or dynamical crystal stability.

Executable inputs are limited to element identities, deterministic tabulated
radii, and one initial raw unrelaxed periodic geometry. The branch must not
execute or consume DFT calculations or per-structure DFT values; learned
energy/force/stress proxies; MLIPs; model/proxy potentials; relaxed
structures; trajectories; later geometries; discovery labels during the
probe/build; validation outcomes; or replication outcomes. Discovery
outcomes may enter NEXT332 only as offline audit labels. Coordinates and
cells are never moved except in explicit representation-invariance tests.

## Distinction from prior mechanisms, frozen before feature evaluation

1. NEXT239 already measures ordinary, unweighted Voronoi facet-area
   evenness `1/(K sum_f p_f^2)`. Repeating that statistic would not be new.
   RFMP instead uses radius-aware power cells and isolates the weakest single
   facet through `K min_f p_f`; NEXT239 is a mandatory novelty comparator.
2. NEXT243/NEXT247 use ordinary Voronoi facet areas as angular weights, not as
   a minimum-facet participation certificate. They remain comparators.
3. NEXT251 measures species-conditioned ordinary Voronoi topology and facet
   parity/entropy, while NEXT263 applies an ordinary-facet cutoff. Neither
   retains the continuous radius-aware minimum facet area.
4. NEXT267/NEXT271/NEXT275/NEXT279 describe radical-cell volume, centroid,
   anisotropy, species contrast, and autocorrelation. NEXT283 adds total
   surface area and sphericity. None records an individual minimum radical
   facet, so all are mandatory novelty comparators.
5. NEXT295/RFPE/NEXT323 measure angular positive equilibrium of facet/contact
   directions. RFMP does not solve a force-balance program and uses polygonal
   measure rather than directions or fitted coefficients. NEXT295 and NEXT323
   remain mandatory comparators.

Rejected before feature evaluation: global sphericity (NEXT283 duplicate),
facet-area inverse-participation/evenness (NEXT239 duplicate), active-facet
count/parity (NEXT251 duplicate), and another local positive-enclosure margin
(RFPE/PGCE duplicate). No alternative aggregate, area transform, direction,
quantile, threshold, chemistry condition, conjunction, or second RFMP feature
may be added after the label-blind probe starts.

## Frozen gates

- Label-blind probe: deterministically select 80 discovery initial geometries
  per source only after loading the complete identifier-bearing inventory;
  read no endpoint, label, validation, replication, relaxed geometry, DFT
  field, or model-potential field.
- Engineering gates: exact schema; finite values in `(0,1]`; support at least
  72/80 per source; at least 20 values unique at `1e-10`; maximum rigid
  rotation, periodic translation, site permutation, unimodular lattice
  rebasing, and exact integral-supercell error at most `1e-8` before `1e-10`
  output quantization.
- Novelty gate: maximum absolute Spearman correlation below `0.90` in each
  source against available label-free NEXT239, NEXT243, NEXT247, NEXT251,
  NEXT255, NEXT259, NEXT263, NEXT267, NEXT271, NEXT275, NEXT279, NEXT283,
  NEXT291, NEXT295, NEXT299, NEXT303, NEXT307, NEXT311, NEXT315, NEXT319, and
  NEXT323 feature populations on the same records. Failure terminates before
  a formal NEXT331 build or any discovery-outcome access.
- Formal NEXT331 coverage: at least `0.90` independently in SCIGEN and
  WyFormer; unsupported structures remain explicit abstentions and are never
  imputed.
- NEXT332 uses the unchanged NEXT224/NEXT268 source and reduced-formula fold
  gates: minimum cell coverage `0.90`, minimum class count `20`, pooled AUC
  `0.55`, macro AUC `0.53`, and worst-fold AUC `0.50` in both sources, with
  inverse-CDF normalization at `1/16` and `15/16`.
- An empty eligible set sets `next333_search_authorized=false` and
  `rfmp_branch_terminated=true`; NEXT333/NEXT334 must not exist.
- If authorized, NEXT333 may reuse only the unchanged NEXT269 margin-local
  grammar and fixed NEXT224/NEXT135 base score. NEXT334 is discovery-only
  BROAD diagnosis. Validation and replication remain sealed even if BROAD
  passes; a stronger claim still needs a separately frozen unseen validation
  protocol.

## Task 1: analytic facet-area kernel and raw-geometry wrapper

**Files:**

- Create: `src/next331_radical_facet_minimum_participation.py`
- Create: `tests/test_next331_radical_facet_minimum_participation.py`
- Reuse without modification: `src/next267_periodic_radical_voronoi_packing.py`
- Reuse without modification: `src/next283_power_cell_shape_volume_coupling.py`

1. Write RED tests for the analytic participation formula, scale invariance,
   invalid areas, and a cube reconstructed from six half-spaces whose six
   facet areas are equal.
2. Implement a strict half-space cell kernel: Chebyshev interior,
   HalfspaceIntersection, convex-hull/feasibility checks, distinct active
   planes, two-dimensional facet polygon areas, and agreement between summed
   facet areas and convex-hull surface area.
3. Reconstruct the radius-aware periodic cell population with the unchanged
   NEXT267 tabulated radii, candidate radius, neighbor-image, volume-tiling,
   and numerical guards.
4. Add real-structure and geometry-boundary tests for NaCl, CsCl, ZnS, a
   distorted NaCl cell, calculator/info/extra-array refusal, deterministic
   repeats, and output bounds.
5. Add invariance tests for rigid rotation, periodic translation, site
   permutation, unimodular lattice rebasing, and exact `2 x 1 x 1`
   replication. Run the focused tests and byte compilation.

## Task 2: label-blind novelty probe

**Files:**

- Create: `experiments/next331_rfmp_label_blind_probe.py`
- Create: `tests/test_next331_rfmp_label_blind_probe.py`
- Create only after a passing probe:
  `experiments/next331_rfmp_label_blind_probe_result.json`

1. Reuse the already tested deterministic complete-inventory selection and
   strict initial-geometry loaders without adding any outcome-bearing input.
2. Compute support, range, uniqueness, invariance, failure categories, and
   maximum correlation with the exact frozen prior population.
3. Record the frozen design and all executed-source hashes.
4. Stop immediately if any gate fails. Do not save a passing-result
   authorization artifact when the result is negative.

## Task 3: contingent NEXT331 formal build

Only after a passing probe, add RED builder tests for complete identity,
geometry-only reads, abstentions, exact manifests/source hashes, atomic
publication, and false boundary flags. Then implement the additive
multiprocessing builder and publish into
`$PRIS_ARCHIVE/next331_radical_facet_minimum_participation_v1`.
Stop before outcomes if either source misses coverage.

## Task 4: contingent NEXT332 fixed discovery-only audit

Only after formal coverage passes, create
`src/next332_rfmp_feature_audit.py` and its tests. Reuse the unchanged
NEXT268/NEXT324 audit/cohort/fold helpers, open only the physically isolated
discovery endpoints as offline labels, and publish into
`$PRIS_ARCHIVE/next332_rfmp_feature_audit_v1`.
Freeze the eligible-set digest and stop without NEXT333/NEXT334 if empty.

## Task 5: contingent NEXT333/NEXT334 search

Create NEXT333/NEXT334 scripts and tests only if the exact NEXT332 manifest
sets `next333_search_authorized=true`. Reuse the unchanged NEXT269 grammar and
then the unchanged BROAD diagnostic. Do not add transformations, reverse the
direction, or open validation/replication.

## Task 6: verification and independent report

**Files:**

- Modify additively:
  `reports/2026-08-08-next115-next117-hcid-no-dft-search.md`
- Do not modify: `paper/`, `tex/`, `notes/`, `README.md`, `PREREG.md`

Run focused and adjacent geometry tests, byte compilation, exact hash and
boundary assertions, CodeGraph synchronization, and the complete repository
suite. Append an independent RFMP section with exact statistics, stop or
authorization state, and strict claim limits. Make no canonical report or
paper changes before user confirmation.

## Execution note

This is an intentionally dirty shared checkout. Preserve every existing
script and content item, work only additively, make no Git commit/merge/
cleanup, and do not delegate to subagents.
