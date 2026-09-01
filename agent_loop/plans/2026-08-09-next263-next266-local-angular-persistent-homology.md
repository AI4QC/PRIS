# NEXT263--NEXT266 local angular persistent-homology plan

## Objective and non-negotiable boundary

Test whether multiscale topology of each raw coordination environment provides
an interpretable cross-source certificate that improves the frozen NEXT224
frontier without using DFT in the executable law.

The executable feature may read only element identities and the initial,
unrelaxed periodic geometry. It must not read or calculate per-structure DFT
energy, force, stress, density, or relaxation data; call a learned
energy/force/stress model or proxy potential; move atoms or the cell; use a
relaxed structure or trajectory; or open validation/replication geometry or
endpoints. SCIGEN and WyFormer discovery outcomes may be attached only after
the feature catalogue, directions, gates, and search grammar below are frozen,
and are offline labels only.

All work is additive. Existing scripts, plans, results, report text, canonical
paper, notes, preregistration, and README remain unchanged. Results are added
only to the independent report after the branch is complete.

## Why this mechanism is independent

Repository and CodeGraph audits found no persistent-homology implementation.
Existing branches cover pair distances, bond valence, atom-centered Voronoi
statistics, contact-graph topology at fixed cutoffs, directional rigidity,
Steinhardt invariants, Delaunay void cages, and void-channel persistence. They
do not follow connected components and one-dimensional holes of a local angular
point cloud across all distance scales.

[Hiraoka et al.](https://doi.org/10.1073/pnas.1520877113) showed that
persistence diagrams extract multiscale rings and cavities from atomic
configurations, and
[Jiang et al.](https://doi.org/10.1038/s41524-021-00493-w) introduced
atom-specific persistent homology for crystalline compounds. This branch uses
that geometric idea only as an analytic descriptor. It does not train the
property-prediction models used in the latter work and never makes their DFT
targets executable.

## Label-free engineering already completed

A pure NumPy prototype constructed Vietoris--Rips persistence over Voronoi
facet normals for FCC, BCC, NaCl, diamond, and a generic triclinic structure.
It verified rigid-rotation, uniform-scale, periodic-translation, and exact
`2 x 1 x 1` supercell invariance to at worst `5.6e-16`. The prototype opened no
endpoint or label. The formal implementation below additionally freezes
integer distance quantization and exact rational aggregation so equality does
not depend on floating summation order.

## NEXT263 frozen feature construction

### Input validation and neighbor point cloud

For each strict ASE `Atoms` input:

- require at least one atom, all three PBC flags, no calculator, empty
  `atoms.info`, and exactly the `numbers` and `positions` arrays;
- convert through `AseAtomsAdaptor` without modifying the geometry;
- use `VoronoiNN(weight="solid_angle", tol=0, cutoff=13)`;
- identify a facet by `(site_index, integer_image)` and keep the largest-area
  record if the backend repeats that identity;
- normalize positive finite facet areas by their site total;
- retain every facet whose normalized area is at least `1/32`;
- require between 4 and 32 retained facets at every site. Any failure makes the
  structure unsupported and therefore fail-open/abstain.

The point cloud for one site is the retained set of unit facet normals. It is
unweighted after the area threshold. No element radius, oxidation state,
endpoint, or learned quantity enters the filtration.

### Exact local Vietoris--Rips filtration

For `m` retained normals, enumerate all vertices, edges, and triangles of the
complete flag complex. Vertex filtration is zero. Edge filtration is its chord
distance on the unit sphere. Triangle filtration is the maximum filtration of
its three edges. Every nonzero filtration value is frozen to

```text
round(distance * 10^11) / 10^11.
```

Sort simplices by `(filtration_integer, dimension, vertex_tuple)` so every face
precedes a cofiltration simplex. Reduce boundary columns over `F_2` with the
standard lowest-pivot algorithm. Retain all finite H0 intervals and strictly
positive H1 intervals. Zero-persistence intervals created by tied symmetric
distances are excluded prospectively.

### Four site quantities and sixteen structure features

For the `m-1` finite H0 death times and positive H1 persistences, define:

```text
h0_death_mean          = mean(H0 death)
h0_death_cv            = population_std(H0 death) / h0_death_mean
h1_persistence_density = sum(H1 persistence) / m
h1_persistence_max     = max(H1 persistence), or 0 if H1 is empty
```

Quantize each site quantity to the same `10^-11` grid. For each quantity,
publish the exact-rational population mean, inverse-CDF 10th percentile,
inverse-CDF 90th percentile, and population standard deviation over sites.
The feature names are the Cartesian product of:

```text
laph_h0_death_mean
laph_h0_death_cv
laph_h1_persistence_density
laph_h1_persistence_max
```

and

```text
mean, q10, q90, std
```

giving exactly sixteen finite values on every supported structure. The
catalogue records the filtration grid, coefficient field, area threshold,
simplex dimensions, quantile convention, and all boundary flags.

### Engineering acceptance tests

Before materialization, tests must cover:

- known H0/H1 barcodes for a triangle, square, and tetrahedral point cloud;
- invariance to point permutation and rigid rotation;
- strict ASE input rejection and fail-open row encoding;
- FCC, BCC, NaCl, diamond, and triclinic rotation/scale/translation/supercell
  equality;
- exact feature schema and no endpoint/validation/replication parameters;
- output no-replace behavior, input/source hashing, and manifest boundary
  flags.

Formal output root:

`$PRIS_ARCHIVE/next263_local_angular_persistent_homology_v1`

## NEXT264 frozen discovery audit

NEXT264 first reproduces the exact NEXT224 frontier and its rejected-extreme
cohort. It then joins NEXT263 rows by exact source-prefixed material identity.
Unsupported rows remain missing and can never be converted to rejection.

Because persistent topology has no reliable universal monotone direction
across tetrahedral, octahedral, close-packed, and low-symmetry crystals, both
directions are prospectively included for every feature:

```text
feature__protected_low
feature__protected_high
```

This freezes exactly 32 hypotheses before outcomes are read; it is not a
post-outcome direction choice. For each feature, normalization cutoffs are the
combined finite-discovery inverse-CDF quantiles `(1/16, 15/16)`. Directional
protection is the same bounded linear transform used by NEXT260.

Each hypothesis must pass the unchanged raw gates in both sources:

- every aggregate and five reduced-formula fold cell has coverage at least
  `0.90` and at least 20 protected and 20 severe rows;
- aggregate AUC at least `0.55`;
- macro-fold AUC at least `0.53`;
- worst-fold AUC at least `0.50`;
- the directed aggregate AUC must not be contradicted by its opposite
  direction.

`eligible_for_search` is exactly `passes_raw_gates`. The eligible identities
are sorted by hypothesis and hashed. If none pass, the branch stops at NEXT264.
NEXT264 never searches or selects a formula.

Formal output root:

`$PRIS_ARCHIVE/next264_laph_feature_audit_v1`

## Conditional NEXT265 frozen local search

NEXT265 is authorized only if the immutable NEXT264 manifest says so. It uses
the exact NEXT224 score `s`, support, threshold `t`, and repair width `W`, and
one eligible bounded certificate `P` at a time:

```text
h = f * W
local_weight = max(0, 1 - abs(s - t) / h)
delta = beta * h * local_weight * (1 - 2P)
score = max(0, s + delta)
```

Missing LAPH values switch only the new term off and keep NEXT224 score and
support unchanged. The frozen grid is

```text
f in {1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1}
beta in {1/4, 1/2, 1}.
```

For `E` eligible NEXT264 hypotheses, the exact universe is one NEXT224
reproduction control plus `21E` new candidates. It is never expanded after
outcomes. Evaluation uses the unchanged source-AUC, twelve-cell SAFE, BROAD,
ranking, and deterministic reproduction rules. Validation remains sealed.

Formal output root:

`$PRIS_ARCHIVE/next265_laph_margin_local_search_v1`

## Conditional NEXT266 residual diagnostic

NEXT266 is authorized only if NEXT265 has no all-gate candidate and publishes
at least one exact new candidate that passes both-source AUC and all SAFE cells
but not BROAD. It evaluates only those sorted candidate identities, reproduces
their NEXT265 records, and recomputes unchanged BROAD threshold tables.

The closest residual is ranked lexicographically by

```text
(failed_constraint_count, normalized_shortfall_sum, candidate_key).
```

It is compared only with the frozen NEXT235 reference
`(5, 0.12339543654931197)`. A continuation is allowed only for a strict
lexicographic improvement. NEXT266 does not search, select, or freeze a new
formula.

Formal output root:

`$PRIS_ARCHIVE/next266_laph_broad_diagnostic_v1`

## Stop rules and claims

- No eligible NEXT264 feature: close the LAPH branch.
- No all-gate NEXT265 candidate: do not freeze a law.
- No strict NEXT266 residual improvement: close the branch.
- Even an all-discovery-gate candidate is only a discovery result; it requires
  a separately frozen authorization before any unopened validation or
  replication endpoint can be accessed.
- No result from this branch may be described as exceeding Pauling or
  approaching DFT screening unless the corresponding sealed evidence later
  supports that statement.
