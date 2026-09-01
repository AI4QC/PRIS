# NEXT387--NEXT390 periodic skeletal vertex-bypass freeze

## Boundary and novelty

This additive branch is frozen before any NEXT387 value or outcome is opened.
Its executable input is exactly composition plus one finite, raw, unrelaxed,
three-periodic geometry.  DFT calculations/values, energies, forces, stresses,
learned or model potentials, relaxations, trajectories and all validation or
replication data are forbidden.  Discovery outcomes may be used only later as
offline labels after frozen label-blind gates and a complete feature build.

Periodic-net vertex connectivity asks whether deleting vertices disconnects a
net.  The repository has translation rank, rigidity, closure, path collision
and multiple geometric/chemical graph operators, but no local deletion test on
the strong atomic skeleton.  NEXT387 tests whether the neighbours of a removed
site have short alternative routes that bypass that site.

## Frozen graph

Use exactly the NEXT383/NEXT379 geometry firewall, NEXT267 Minkowski
reduction, ordinary periodic Voronoi facets, `1e-8` reverse-angle agreement,
mutual local solid-angle salience, `1e-10` quantization, exact-tie filtration,
integer translation-rank calculation and global

    tau = inverted-CDF q10 of site rank-three bottlenecks.

The skeleton contains every edge with salience `>= tau`.

## Sole formula

For each lifted central vertex `(i,0)`, collect its distinct lifted skeleton
neighbours.  Delete only `(i,0)`.  For each unordered neighbour pair, ask
whether a path of at most four skeleton edges connects the pair without the
deleted vertex.  Four is frozen because the deleted two-edge wedge plus a
four-edge bypass detects rings of size at most six, the conventional local
ring range, without searching path radius.  Define the site bypass fraction
as connected pairs divided by all neighbour pairs; sites with fewer than two
distinct neighbours receive zero.

The sole output is

    psvb_skeletal_vertex_bypass4_q10
      = inverted-CDF q10 over site bypass fractions,

quantized at `1e-10`.  The sole direction is `protected_high`.  No alternative
path length, threshold, summary, direction, weight, transform, conjunction or
source-specific choice is allowed.

## Pre-outcome tests and gates

Analytic tests must cover a periodic chain (zero), simple cubic net (one),
tie handling, input order and weight-scale invariance, disjoint exact
replication, malformed reverse facets, distorted NaCl, rotation, translation,
site permutation, unimodular rebasing, a `2x1x1` supercell and all firewall
rejections.

The unchanged deterministic 80+80 discovery-geometry probe must pass per
source: support at least 72, finite `[0,1]` domain, at least 20 values rounded
to `1e-10`, representation error at most `1e-8`, and maximum absolute
Spearman below `0.90` against every prior label-free control through NEXT383
with at least 40 joint-finite rows.  No endpoint is opened.

Only a fully passing probe authorizes an unchanged complete discovery build
with coverage at least `0.95` per source.  Only that build authorizes a
one-hypothesis audit in the exact NEXT224 rejected-extreme cohort, using the
unchanged combined-discovery `(1/16,15/16)` bounded map and source gates:
coverage `>=0.95`, pooled and every fold class count `>=20`, pooled AUC
`>=0.55`, macro-fold AUC `>=0.55`, worst-fold AUC `>=0.50`.  Eligibility is
conjunctive across SCIGEN and WyFormer.  Failure terminates the branch without
NEXT390 search or scientific claim; passing would authorize only a separately
frozen search, not validation or replacement of existing laws.

All scripts, artifacts and report additions remain additive.  Canonical
`paper/`, `tex/`, `notes/`, `README.md` and `PREREG.md` stay untouched pending
user review.
