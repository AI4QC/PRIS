# NEXT259--NEXT262 Periodic Void-Bottleneck Persistence Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Test whether the persistence of raw-geometry crystal cavities relative to their periodic Voronoi-channel bottlenecks supplies a fully pre-DFT, interpretable correction to the frozen NEXT224 law.

**Architecture:** NEXT259 constructs a fixed, label-free bank of sixteen dimensionless features from the quotient graph of periodic Voronoi vertices and edges. NEXT260 audits sixteen prospectively directed hypotheses in the exact frozen NEXT224 rejected-extreme discovery cohort. NEXT261 and NEXT262 are conditional search and BROAD-diagnostic stages and remain forbidden unless their preceding gates authorize them.

**Tech Stack:** Python 3.11, NumPy, pandas/Parquet, pymatgen `VoronoiNN`, ASE, exact integer translation-rank arithmetic, pytest, and the unchanged NEXT227/NEXT224 reconstruction and gate evaluators.

## 1. Scientific question and prior-mechanism audit

This branch is additive. It preserves every prior plan, script, test, output,
report section, and canonical artifact.

The following apparent alternatives are already represented and are not the
new mechanism:

- P10 and NEXT26/NEXT33 measure atom-centred Voronoi free volume, global
  packing, pair contacts, radii, and steric imbalance;
- NEXT49/NEXT166 measure topology and translation rank of *atomic contact*
  graphs;
- NEXT168--NEXT180 measure directional rigidity and closure of atom-centred
  contact graphs;
- NEXT239--NEXT252 measure atom-centred Voronoi bond order and face topology;
- NEXT255--NEXT258 measure angular isotropy and directional closure of a
  complete Delaunay cage at one void centre.

The new physical object is instead the *periodic void-channel graph*. A node is
a local maximum candidate of the nearest-atom distance field at a Voronoi
vertex. A graph edge is the Voronoi edge joining two such voids, and its
capacity is the smallest nearest-generator distance along that segment. The
new information is how much of a cavity radius survives on a path that becomes
periodic in at least one or in all three lattice directions. It is neither a
re-expression of total cell packing nor a retuning of DVCI closure.

The construction is motivated by two primary sources frozen before outcomes:

- Willems et al., *Algorithms and tools for high-throughput geometry-based
  analysis of crystalline porous materials*, Microporous and Mesoporous
  Materials 149 (2012), 134--141,
  <https://doi.org/10.1016/j.micromeso.2011.08.020>, builds a periodic Voronoi
  network and distinguishes largest included voids from bottlenecks on free
  paths.
- Pletzer-Zelgert et al., *LifeSoaks: a tool for analyzing solvent channels in
  protein crystals and obstacles for soaking experiments*, Acta
  Crystallographica D79 (2023), 837--856,
  <https://doi.org/10.1107/S205979832300559X>, defines a node's bottleneck
  radius as the largest threshold at which it belongs to an infinite periodic
  path and notes that it may be much smaller than the local cavity radius.

These sources justify the geometry, not a stability claim. Whether it predicts
the sealed discovery endpoint is the falsifiable question of NEXT260.

## 2. Hard no-DFT and data-firewall boundary

The executable accepts only one raw, unrelaxed periodic structure: element
identities, lattice, and Cartesian coordinates. It may use deterministic
Voronoi tessellation and integer/linear algebra.

It must not read or compute per-structure:

- DFT energy, force, stress, charge density, electronic structure, or any DFT
  derived feature;
- a learned energy/force/stress proxy, MLIP, model potential, or proxy
  potential;
- a relaxed structure, trajectory, same-composition alternative, or any
  coordinate/cell update;
- validation or replication geometry, endpoints, or outcomes.

Discovery outcomes are offline labels and may be joined only by NEXT260 after
the NEXT259 source, catalogue, feature directions, statistics, tolerances, and
this design have been frozen. Validation and replication remain sealed even if
all discovery gates pass; a later separately authorized freeze is required.

Every stage publishes exact false values for:

```text
dft_calculation_executed
dft_values_used_by_executable_formula
learned_energy_force_stress_proxy_used
model_or_proxy_potential_used
physical_relaxation_executed
opened_validation_outputs_used
scigen_replication_endpoint_opened
wyformer_replication_endpoint_opened
```

Unsupported structures abstain/fail open and can never become automatic
rejections.

## 3. Frozen NEXT259 quotient void graph

### 3.1 Strict input and Voronoi faces

Require at least one atom, three-dimensional PBC, no calculator, an empty
`Atoms.info`, and exactly the `numbers` and `positions` arrays. Convert only
that object to pymatgen and run:

```python
VoronoiNN(
    weight="solid_angle",
    tol=0,
    cutoff=13,
    compute_adj_neighbors=True,
)
```

For each atom-centred cell, deduplicate faces by
`(site_index, integer_image)`. Keep the largest positive finite face area;
break an exact tie lexically by sorted unique vertex identifiers and the
neighbor displacement rounded to twelve decimals. Retain the winning face's
original `verts` order, because Qhull/pymatgen supplies the polygon in cyclic
order.

For vertex identifier `v`, collect every incident face-neighbor displacement
`r_j` and solve

```text
r_j dot x_v = |r_j|^2 / 2.
```

Require rank three, a positive finite radius, and maximum absolute residual no
greater than `1e-9 * max(1, max_j ||r_j|^2/2|)`. The absolute node coordinate is
the central atom coordinate plus `x_v`.

### 3.2 Periodic quotient identity

Let the row-vector lattice matrix be `L`, atom count `N`, and

```text
ell = (|det(L)| / N)^(1/3).
```

For absolute node coordinate `y`, calculate fractional coordinate
`f = solve(L.T, y)`. With the frozen integer quantum `Q = 1_000_000_000`, set

```text
q = rint(Q f) in Z^3
node_key = q mod Q
image_shift = (q - node_key) / Q in Z^3.
```

The key defines a quotient-graph node and the image shift defines its lift.
Repeated observations of one quotient node must agree in normalized radius to
`5e-8`; otherwise the structure abstains. Publish the median repeated radius
rounded to eleven decimal places. This fractional identity is invariant to a
common rigid rotation and allows exact integer translations on quotient edges.

### 3.3 Void-edge capacity

Every consecutive pair, including last--first, of reconstructed vertices on a
winning Voronoi face defines one local Voronoi edge. If its absolute endpoints
are `y_0`, `y_1` and the current generator atom is `c`, define

```text
d = y_1 - y_0
t_star = clip(-((y_0-c) dot d) / (d dot d), 0, 1)
rho_e = |y_0 - c + t_star d| / ell.
```

Along a Voronoi edge the three generating atoms are equidistant, so duplicate
cell observations must give the same capacity. Canonicalize an undirected edge
by the lexical minimum of `(key_0,key_1,delta)` and
`(key_1,key_0,-delta)`, where `delta = shift_1-shift_0`. A zero-translation
self-loop or disagreement above `5e-8` causes abstention. Publish the median
duplicate capacity rounded to eleven decimals.

No atomic radius is subtracted. This v1 intentionally studies the parameter-
free nearest-centre distance field and avoids duplicating the prior
radii/packing mechanisms. Uniform scale is removed only by `ell`; no atom or
cell is moved.

### 3.4 Exact periodic bottleneck annotation

Process quotient edges in descending rounded `rho_e`, grouping equal values.
Use a translation-weighted disjoint-set forest. For an oriented edge
`u -> v` with translation `delta`, enforce the exact integer relation

```text
p_v = p_u + delta.
```

When an edge closes a quotient cycle, its residual integer translation is
added to that component's lattice-span basis. Determine rank zero through
three using exact nonzero, cross-product, and triple-product tests; no floating
rank tolerance is allowed.

For each node `v`:

- `b_any(v)` is the largest edge-capacity threshold at which its component
  first has translation rank at least one;
- `b_3d(v)` is the largest threshold at which its component first has rank
  three.

When a finite/lower-rank component joins an already periodic/higher-rank
component, its previously unassigned nodes receive the current threshold.
Unassigned values after all edges are zero. Require
`0 <= b_3d <= b_any <= r_v` within `5e-8`, clipping only final roundoff.

## 4. Frozen NEXT259 feature universe

For every unique quotient node define four higher-is-more-defect quantities:

```text
isolation_any = clip(1 - b_any/r_v, 0, 1)
isolation_3d  = clip(1 - b_3d/r_v, 0, 1)
prominence_any = max(r_v - b_any, 0)
radius = r_v
```

Round node-level values to eleven decimals before aggregation. For each metric
publish exactly:

1. population mean;
2. inverse-CDF 75th percentile;
3. inverse-CDF 90th percentile;
4. mean of all values greater than or equal to the inverse-CDF 75th
   percentile, including all boundary ties.

The sixteen columns are the Cartesian product of:

```text
metrics = (
  pvbp_isolation_any,
  pvbp_isolation_3d,
  pvbp_prominence_any,
  pvbp_radius,
)
statistics = (mean, q75, q90, upper_quartile_mean)
```

All sixteen hypotheses are frozen as `protected_low`: larger isolated voids,
larger radius loss at a bottleneck, or unusually large cell-normalized cavity
radii are prospectively hypothesized to characterize severe rejected
structures. The reverse directions are not searchable after outcomes.

The upper-tail definition, inverse empirical CDF, quotient-node deduplication,
and eleven-decimal rounding are mandatory replication safeguards. Tests must
cover permutation, wrapping, arbitrary rigid rotation, uniform scale,
periodic translation, primitive/conventional representation where available,
and a `2 x 1 x 1` supercell.

Formal directory:

`$PRIS_ARCHIVE/next259_periodic_void_bottleneck_persistence_v1`

Only physically isolated SCIGEN and WyFormer discovery geometry may be read.
Publish the two feature tables, catalogue, manifest, row/support/failure
counts, node/edge counts, feature finite counts, inputs/source/output hashes,
and exact partition-access flags.

## 5. Frozen NEXT260 feature audit

NEXT260 may run only after NEXT259 is immutable. It reconstructs the exact
NEXT224 frontier and its frozen rejected-extreme cohort, joins the sixteen
NEXT259 features by prefixed material identity, and exposes discovery endpoints
only then.

For each feature:

1. calculate combined-discovery inverse-CDF quantiles `1/16` and `15/16`;
2. map the raw `protected_low` feature monotonically to bounded protection in
   `[0,1]`;
3. audit SCIGEN and WyFormer separately using the unchanged pooled, macro-fold,
   worst-fold, minimum-class-count, and minimum-coverage gates;
4. select only hypotheses passing every raw gate in both sources.

The eligible set is sorted and SHA-256 hashed. If it is empty, set
`next261_search_authorized=false`, close the branch, and do not create NEXT261
or NEXT262. No formula, cutoff, width, amplitude, or opposite direction may be
searched in NEXT260.

Formal directory:

`$PRIS_ARCHIVE/next260_pvbp_feature_audit_v1`

## 6. Conditional NEXT261 finite formula search

NEXT261 is forbidden unless the immutable NEXT260 artifact authorizes at least
one exact eligible hypothesis identity. Reproduce NEXT224 once and, for every
eligible hypothesis, enumerate only:

```text
local_width_fraction in {1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1}
amplitude_fraction in {1/4, 1/2, 1}
```

Use the unchanged triangular margin-local protected correction and the
bounded-protection value from NEXT260. Search no interactions, signs, cutoffs,
feature combinations, transforms, or larger amplitudes. Evaluate the exact
cross-source AUC, SAFE12, and BROAD gates. A candidate is freeze-authorized only
if it passes all discovery gates. An AUC+SAFE/non-BROAD candidate may authorize
only NEXT262 diagnosis; it is not a law.

Formal directory:

`$PRIS_ARCHIVE/next261_pvbp_margin_local_search_v1`

## 7. Conditional NEXT262 BROAD diagnostic

NEXT262 is forbidden unless NEXT261 publishes an exact SHA-256 identity list
of AUC+SAFE/non-BROAD candidates. Reproduce exactly those candidates and no
others. Recompute the unchanged BROAD threshold table, count failed constraints,
and sum normalized shortfalls. Compare the closest tuple lexicographically to
the frozen NEXT235 reference:

```text
(failed_constraint_count, normalized_shortfall_sum)
= (5, 0.12339543654931197)
```

If there is no strict improvement, close the PVBP certificate branch and
forbid any post-outcome direction, statistic, threshold, width, or formula
tuning. If there is a strict diagnostic improvement but no all-gate candidate,
record it only as mechanism evidence; validation and replication remain sealed.

Formal directory:

`$PRIS_ARCHIVE/next262_pvbp_broad_diagnostic_v1`

## 8. TDD implementation tasks

### Task 1: Freeze and hash this plan

**Files:**

- Create: `docs/plans/2026-08-09-next259-next262-periodic-void-bottleneck-persistence.md`

**Steps:**

1. Finish all definitions above before any NEXT259 feature is joined to an
   endpoint.
2. Record `sha256sum` of this file in every new source constant and manifest.
3. Do not amend inspected hypotheses or grammar after NEXT260 outcomes exist.

### Task 2: Implement NEXT259 with RED--GREEN TDD

**Files:**

- Create: `tests/test_next259_periodic_void_bottleneck_persistence.py`
- Create: `src/next259_periodic_void_bottleneck_persistence.py`

**Steps:**

1. Write failing tests for exact translation rank, weighted quotient-cycle
   bottleneck annotation, aggregation/tie replication, a hand-built graph,
   known crystals, strict no-calculator input, all invariances, source-firewall
   validation, atomic publication, and manifest flags.
2. Run
   `pytest -q tests/test_next259_periodic_void_bottleneck_persistence.py` and
   preserve the expected missing-module RED result.
3. Implement the minimal source satisfying the frozen interface.
4. Run the same command to GREEN, then materialize the formal directory.
5. Hash and independently verify every input, executed source, and output.

### Task 3: Implement and conditionally run NEXT260

**Files:**

- Create: `tests/test_next260_pvbp_feature_audit.py`
- Create: `src/next260_pvbp_feature_audit.py`

**Steps:**

1. Write tests first for bounded direction, eligibility, frontier
   reconstruction, identity checks, no-search termination, source-specific
   audits, and flags; preserve RED before source creation.
2. Implement by reusing the frozen NEXT256 audit machinery without weakening
   any gate.
3. Run focused tests and the formal NEXT260 audit.
4. Inspect only the authorization fields and exact eligible identities needed
   to decide whether NEXT261 is allowed.

### Task 4: Implement conditional NEXT261/NEXT262 only if authorized

**Files:**

- Conditionally create: `tests/test_next261_pvbp_margin_local_search.py`
- Conditionally create: `src/next261_pvbp_margin_local_search.py`
- Conditionally create: `tests/test_next262_pvbp_broad_diagnostic.py`
- Conditionally create: `src/next262_pvbp_broad_diagnostic.py`

**Steps:**

1. If NEXT260 is ineligible, do not create these files or directories.
2. If eligible, use a separate RED--GREEN cycle for NEXT261 and run exactly the
   frozen grid.
3. If NEXT261 has no authorized diagnostic identities, do not create NEXT262.
4. Otherwise use a separate RED--GREEN cycle for NEXT262 and diagnose exactly
   the frozen identities.

### Task 5: Independent report and verification

**Files:**

- Modify only by append:
  `reports/2026-08-08-next115-next117-hcid-no-dft-search.md`
- Do not modify: `paper/`, `tex/`, `notes/`, `README.md`, `PREREG.md`

**Steps:**

1. Append method, frozen identities, all positive and negative results,
   limitations, hashes, and the exact current conclusion.
2. Run all new focused tests and then the full repository test suite.
3. Run `git diff --check`, protected-path diff checks, manifest/hash
   verification, and CodeGraph status after index synchronization.
4. Do not claim a Pauling replacement unless a separately frozen candidate
   later passes discovery, sealed validation, sealed replication, coverage,
   safety, and compute-saving requirements.

## 9. Frozen stopping rule

The branch stops immediately at the first failed authorization gate. A strong
raw signal, SAFE12 result, or improved BROAD diagnostic is not a confirmed law.
No validation/replication endpoint is opened in NEXT259--NEXT262. Any later
continuation must be a new pre-outcome freeze; it may not retune PVBP after its
outcomes are visible.
