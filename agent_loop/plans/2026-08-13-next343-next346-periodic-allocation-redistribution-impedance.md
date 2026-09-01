# Periodic Allocation Redistribution Impedance Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` and
> `superpowers:test-driven-development` task by task. Do not delegate.

**Goal:** Test whether the global geometric impedance to redistributing a
radius-inconsistent power-cell volume allocation provides a transferable,
representation-invariant pre-DFT crystal-validity signal, and only if
prospectively authorized, use it in the bounded no-DFT law search.

**Architecture:** NEXT343 reuses the reciprocal `A/d` finite-volume graph
constructed for NEXT339 but replaces the analytically annihilated affine load
with the nontrivial radius-target allocation residual already defined in
NEXT267. It solves one periodic graph Poisson problem, normalizes its global
Green energy by a degree-local reference, and freezes one bounded protection
feature. A label-blind probe precedes all formal or outcome-bearing work.
NEXT344 audit and NEXT345/NEXT346 search are strictly contingent.

**Tech Stack:** Python 3.11, NumPy/SciPy linear algebra and half-space
geometry, pandas, ASE, existing isolated discovery inventories, pytest,
CodeGraph.

## Scientific and information-boundary freeze

For raw site `i`, let `v_i` be its radius-aware periodic power-cell volume,
`V=sum_i v_i` the raw cell volume, and `r_i` the deterministic tabulated
radius. Define observed and target allocation fractions

```text
o_i = v_i / V,
t_i = r_i^3 / sum_j r_j^3,
b_i = o_i - t_i,                    sum_i b_i = 0.
```

On the reciprocal power-facet graph, each undirected edge has
`g_e=A_e/||d_e||`, incidence matrix `B`, weighted Laplacian
`L=B.T G B`, and positive nonself degree `q_i=L_ii`. Solve the gauge-free
periodic Poisson problem `L phi=b` with the symmetric pseudoinverse. Define

```text
E_global = b.T L^+ b,
E_local  = sum_i b_i^2 / q_i,
R        = E_global / E_local,
pari_allocation_redistribution_protection = 1 / (1 + R).
```

When `max_i |b_i| <= 1e-12`, the exact-allocation convention is protection
one. Otherwise the Poisson residual must pass `1e-8`, `E_local>0`, and
`R>=0`. The sole frozen feature is
`pari_allocation_redistribution_protection` in the `protected_high`
direction. Under an exact integral supercell, each allocation residual scales
by the inverse replication count while both quadratic energies scale by the
same factor, so their ratio is representation invariant. Uniform geometric
rescaling of the entire graph also cancels between the two energies.

This is a geometric redistribution certificate, not a real diffusivity,
conductivity, force, energy, stress, interatomic potential, or learned proxy.
The prospective hypothesis is that a large radius-target allocation mismatch
that can be relieved only through a high-resistance periodic facet network is
less protected against raw geometric invalidity. It is not a theorem of
crystal stability.

Executable inputs are limited to element identities, deterministic tabulated
radii, and one initial raw unrelaxed periodic geometry. The branch must not
execute or consume DFT calculations or per-structure DFT values; learned
energy/force/stress proxies; MLIPs; model/proxy potentials; relaxed
structures; trajectories; later geometries; discovery labels during the
probe/build; validation outcomes; or replication outcomes. Discovery
outcomes may enter NEXT344 only as offline audit labels.

## Distinction and alternatives frozen before feature evaluation

1. NEXT267 already measures allocation mismatch amplitude with total
   variation and radius-normalized volume quantiles/CV. PARI adds the
   reciprocal facet-network Green geometry and compares global versus local
   redistribution cost. All NEXT267 features are mandatory novelty controls.
2. NEXT279 correlates radius-normalized volumes over the radical adjacency
   graph but does not solve a Poisson problem or use facet areas. It is a
   mandatory control.
3. NEXT315 solves a Green problem for formal charge on an unweighted radical
   contact graph. PARI has no charge and uses a zero-sum radius-allocation
   residual with `A/d` conductance. NEXT315 is mandatory.
4. NEXT19/NEXT38 transport formal valence through contact capacity, NEXT307
   measures bond-valence Hodge loops, and NEXT323 solves positive global
   contact equilibrium. None transports geometric cell-volume allocation.
5. PGHT is closed by `sum A n=0`; PARI's scalar allocation source is not
   annihilated by facet equilibrium. RFMP/RFMT single-cell area measures and
   NEXT166 winding rank remain mandatory comparators.

Rejected alternatives: raw `b.T L^+ b` (dimensionful and representation
dependent), total variation alone (NEXT267 duplicate), unweighted Laplacian
(NEXT315 overlap), per-species or charge-conditioned sources (post hoc and
chemistry dependent), degree-normalized spectral gap (generic topology), and
any second statistic. No alternative source, conductance, normalization,
transform, direction, threshold, conjunction, or feature may be added after
the probe starts.

## Frozen gates

- Probe 80 deterministic discovery initial geometries per source after the
  complete identifier inventory is loaded; read no outcome-bearing field.
- Engineering: reciprocal facet area error at most `1e-7`, surface-area
  certificate at most `1e-7`, volume tiling at the unchanged NEXT267
  tolerance, connected nonself Laplacian for nonzero source, Poisson residual
  at most `1e-8`, finite output `(0,1]`, support at least 72/80, at least 20
  values unique at `1e-10`, and rotation/translation/permutation/unimodular
  rebasing/exact-supercell error at most `1e-8`.
- Novelty: maximum absolute Spearman below `0.90` independently in both
  sources against NEXT166, NEXT168, NEXT173, NEXT179, NEXT239, NEXT243,
  NEXT247, NEXT251, NEXT255, NEXT259, NEXT263, NEXT267, NEXT271, NEXT275,
  NEXT279, NEXT283, NEXT291, NEXT295, NEXT299, NEXT303, NEXT307, NEXT311,
  NEXT315, NEXT319, and NEXT323 label-free populations. Failure stops before
  formal NEXT343 or outcomes.
- Formal NEXT343 coverage: at least `0.90` independently in both sources;
  unsupported rows remain abstentions without imputation.
- NEXT344 unchanged gates: cell coverage `0.90`, class count `20`, pooled AUC
  `0.55`, macro AUC `0.53`, worst-fold AUC `0.50` in both sources,
  reduced-formula folds, inverse-CDF `1/16` and `15/16`.
- Empty eligible set means `next345_search_authorized=false` and
  `pari_branch_terminated=true`; NEXT345/NEXT346 must not exist.
- If authorized, NEXT345 reuses only unchanged NEXT269 margin-local grammar
  and fixed NEXT224/NEXT135 base; NEXT346 is discovery-only BROAD diagnosis.
  Validation and replication remain sealed.

## Task 1: allocation-impedance kernel and geometry wrapper

Create `src/next343_periodic_allocation_redistribution_impedance.py` and its
tests. RED analytic tests cover exact allocation protection one, a two-site
graph with protection `2/3`, a path/triangle showing topology dependence,
global rescaling, source/edge orientation/order/gauge invariance, disconnected
or invalid sources, and residual checks. Reuse NEXT339 reciprocal facet graph
without modification, attach its site volumes, and add NaCl/CsCl/ZnS/
distorted-cell, geometry-only, and all five representation tests.

## Task 2: label-blind probe

Create `experiments/next343_pari_label_blind_probe.py` and tests. Reuse the
strict complete-inventory loaders and exact comparator population. Record the
design and every executed-source hash. Retain a probe-result JSON only if all
gates pass; otherwise stop.

## Task 3: contingent formal build/audit/search

Only after a passing probe, TDD and publish the formal NEXT343 builder into
`$PRIS_ARCHIVE/next343_periodic_allocation_redistribution_impedance_v1`.
Only after both coverage gates pass, create NEXT344 fixed audit. Create
NEXT345/NEXT346 only if the exact NEXT344 manifest authorizes them. Do not
open discovery outcomes earlier, and keep validation/replication sealed.

## Task 4: verification and independent report

Append only to
`reports/2026-08-08-next115-next117-hcid-no-dft-search.md`; do not modify
`paper/`, `tex/`, `notes/`, `README.md`, or `PREREG.md`. Run focused/adjacent
tests, byte compilation, exact hashes/boundaries, CodeGraph sync, and the
complete suite before handoff.

## Execution note

The checkout is intentionally dirty and shared. Preserve all prior content,
work additively, make no Git commit/merge/cleanup, and do not delegate.
