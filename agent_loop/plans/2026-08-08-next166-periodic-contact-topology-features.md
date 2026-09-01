# NEXT166 Periodic Contact Topology Feature Freeze Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: use `superpowers:test-driven-development`
> and `superpowers:verification-before-completion` while executing this plan.

**Goal:** Freeze cell-basis-invariant 0D/1D/2D/3D topology descriptors of the
raw periodic opposite-sign contact graph for SCIGEN and WyFormer discovery
structures, without opening any endpoint.

**Architecture:** Reuse the analytic formal-valence assignment and the
periodic `image=(nx,ny,nz)` edges already defined by NEXT19. For each connected
component of the finite quotient graph, compute the exact integer rank of
cycle-closure translation vectors, then aggregate component rank by sites.

**Tech stack:** Python, NumPy, pymatgen, ASE, parquet, multiprocessing, pytest.

## Strict boundary

NEXT166 receives only raw, unrelaxed discovery geometry and source manifests.
It has no endpoint argument and may not read discovery, validation, or
replication labels; validation and replication geometry files are also outside
its input inventory. No DFT calculation/value, learned energy/force/stress
proxy, or relaxation is permitted.

Use exactly two already-defined neighbor graphs:

1. `voronoi`: `VoronoiNN(weight="solid_angle", tol=0, cutoff=13)`;
2. `crystalnn`: `CrystalNN(weighted_cn=True, x_diff_weight=0)`.

Both use NEXT19's neutral analytic valence assignment and retain only
opposite-sign contacts. No new distance cutoff or fitted constant is allowed.

## Exact periodic-rank mathematics

For a quotient-graph edge from site `u` to site `v` with integer image
translation `t_e`, choose an arbitrary root in each connected component and
assign integer vertex potentials `p_v` along a spanning tree:

```text
p_v = p_u + t_e
```

Every edge then gives a cycle-closure translation

```text
r_e = p_u + t_e - p_v in Z^3.
```

The component dimensionality is the exact rank over the rationals of the
integer vectors `{r_e}`. Compute rank 0--3 by nonzero-vector, cross-product,
and integer triple-product tests; do not use a floating tolerance. Isolated
sites have rank 0. Rank is invariant to root, spanning tree, edge direction,
site order, duplicate edges, and unimodular lattice-basis changes.

## Frozen features

For each graph mode publish exactly:

```text
pct_<mode>_rank_max        = maximum component rank / 3
pct_<mode>_rank_mean       = site-weighted mean component rank / 3
pct_<mode>_rank0_fraction  = fraction of sites in rank-0 components
pct_<mode>_rank1_fraction  = fraction of sites in rank-1 components
pct_<mode>_rank2_fraction  = fraction of sites in rank-2 components
pct_<mode>_rank3_fraction  = fraction of sites in rank-3 components
```

Also publish one support boolean and failure string per mode. Fractions must be
finite, lie in `[0,1]`, and sum to one on supported rows. No component count,
edge density, or cell-size-dependent cycle density is allowed.

## TDD and formal build

1. Create `tests/test_next166_periodic_contact_topology_features.py` first.
2. Verify exact ranks on synthetic 0D, 1D, 2D, and 3D periodic graphs.
3. Verify edge reversal/order/duplication and site permutation invariance.
4. Verify feature bounds, fraction sum, and a real-structure supercell check.
5. Implement `src/next166_periodic_contact_topology_features.py` only after
   the tests fail for the missing module.
6. Build discovery-only features with 12 workers and atomically publish under
   `$PRIS_ARCHIVE/next166_periodic_contact_topology_features_v1`.
7. Independently verify row identity, hashes, source hashes, and all no-label,
   no-DFT, no-proxy, and no-relaxation flags.

