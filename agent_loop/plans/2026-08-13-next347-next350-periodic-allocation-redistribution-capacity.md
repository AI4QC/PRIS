# Periodic Allocation Redistribution Capacity Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` and
> `superpowers:test-driven-development` task by task. Do not delegate.

**Goal:** Prospectively repair the representation defect exposed by closed
PARI and test whether an infinite-periodic facet-capacity normalization turns
radius-allocation redistribution into a transferable, no-DFT pre-screening
signal and, only if authorized, a stronger bounded law.

**Architecture:** NEXT347 retains PARI's radius-allocation source, reciprocal
`A/d` power-facet graph, Laplacian, and global Green energy, but replaces the
invalid quotient-Laplacian diagonal reference by the exact infinite-periodic
incident capacity. Self-image facet orbits count twice, matching their two
incident copies in every explicit integral supercell. A label-blind probe
precedes formal publication and outcome access. NEXT348 fixed audit and
NEXT349/NEXT350 search remain strictly contingent.

**Tech Stack:** Python 3.11, NumPy/SciPy linear algebra and half-space
geometry, pandas, ASE, existing isolated discovery inventories, pytest,
CodeGraph.

## Scientific and information-boundary freeze

Reuse PARI's site allocations and source

```text
o_i = v_i / V,
t_i = r_i^3 / sum_j r_j^3,
b_i = o_i - t_i,                    sum_i b_i = 0,
```

and its reciprocal power-facet edge conductance `g_e=A_e/||d_e||`, incidence
matrix `B`, weighted quotient Laplacian `L=B.T G B`, and global energy
`E_global=b.T L^+ b`.

For the local reference, define the infinite-periodic incident capacity

```text
c_i = sum_(e nonself incident to i) g_e
      + 2 sum_(e self-image at i) g_e.
```

The factor two is not fitted: one undirected self-image edge orbit has two
incidences at every lifted node, and in an explicit supercell becomes two
edge incidences to neighboring replicas. Define

```text
E_capacity = sum_i b_i^2 / c_i,
R_capacity = E_global / E_capacity,
parc_allocation_redistribution_protection = 1 / (1 + R_capacity).
```

When `max_i |b_i| <= 1e-12`, protection is exactly one. Otherwise the
Poisson residual must pass `1e-8`, every nonzero-source site must have
positive capacity, and both energies must be valid. The sole feature is
`parc_allocation_redistribution_protection` in the `protected_high`
direction.

Under an `N`-fold exact supercell, each source component becomes `b_i/N`,
each lifted site retains the same infinite capacity, and there are `N`
copies. Both global and capacity-reference energies scale by `1/N`, so the
ratio and protection are exactly representation invariant. Uniform geometric
scaling multiplies every conductance and capacity by the same factor and also
cancels.

PARC is a geometric redistribution certificate only. It is not a real
diffusivity, conductivity, energy, force, stress, potential, MLIP, or learned
proxy, and no physical relaxation occurs. The hypothesis that difficult
radius-allocation redistribution marks raw invalidity remains prospective,
not a theorem of crystal stability.

Executable inputs are limited to element identities, deterministic tabulated
radii, and one initial raw unrelaxed periodic geometry. The branch must not
execute or consume DFT calculations or per-structure DFT values; learned
energy/force/stress proxies; MLIPs; model/proxy potentials; relaxed
structures; trajectories; later geometries; discovery labels during the
probe/build; validation outcomes; or replication outcomes. Discovery
outcomes may enter NEXT348 only as offline audit labels.

## Distinction and alternatives frozen before feature evaluation

1. PARI is a closed, invalid representation: its local degree omitted
   self-image incidences in a primitive quotient and failed exact supercell
   invariance by `0.1098164761`. PARC is a separate pre-outcome branch with a
   mathematically fixed infinite-periodic capacity, not a retrospective
   reinterpretation of PARI results. PARI had no valid population and is not
   an empirical novelty comparator.
2. NEXT267 measures allocation mismatch amplitude only; NEXT279 measures
   graph autocorrelation of radius-normalized volume; NEXT315 solves charge
   Green resistance on an unweighted graph. PARC combines a radius-allocation
   source with reciprocal facet measure and a global-to-local capacity ratio.
   They are mandatory novelty controls.
3. NEXT166 winding rank, NEXT168/NEXT173 local direction tensors, NEXT307
   bond-valence Hodge loops, and NEXT323 positive global equilibrium do not
   solve this allocation/capacity problem. They remain controls.
4. PGHT's affine load was annihilated by `sum A n=0`; PARC's scalar volume-
   allocation source is not. RFMP/RFMT cell area measures remain controls.

Rejected alternatives: changing `A/d`, changing the source, omitting or
fitting the self-image factor, using raw Green energy, using total variation
alone, adding spectral-gap/topology terms, or adding a second feature. No
alternative normalization, transform, direction, threshold, conjunction,
or chemistry condition may be introduced after the label-blind probe starts.

## Frozen gates

- Probe 80 deterministic initial discovery geometries per source after the
  complete identifier inventory is loaded; read no outcome-bearing field.
- Engineering: reciprocal facet area error at most `1e-7`, surface-area
  certificate at most `1e-7`, volume tiling at unchanged NEXT267 tolerance,
  positive infinite capacity on every nonzero-source site, Poisson residual
  at most `1e-8`, finite output `(0,1]`, support at least 72/80, at least 20
  values unique at `1e-10`, and rotation/translation/permutation/unimodular
  rebasing/exact-supercell error at most `1e-8`.
- Novelty: maximum absolute Spearman below `0.90` independently in both
  sources against NEXT166, NEXT168, NEXT173, NEXT179, NEXT239, NEXT243,
  NEXT247, NEXT251, NEXT255, NEXT259, NEXT263, NEXT267, NEXT271, NEXT275,
  NEXT279, NEXT283, NEXT291, NEXT295, NEXT299, NEXT303, NEXT307, NEXT311,
  NEXT315, NEXT319, and NEXT323 label-free populations. Failure stops before
  formal NEXT347 or outcomes.
- Formal NEXT347 coverage: at least `0.90` independently in both sources;
  unsupported rows remain abstentions without imputation.
- NEXT348 unchanged gates: cell coverage `0.90`, class count `20`, pooled AUC
  `0.55`, macro AUC `0.53`, worst-fold AUC `0.50` in both sources,
  reduced-formula folds, inverse-CDF `1/16` and `15/16`.
- Empty eligible set means `next349_search_authorized=false` and
  `parc_branch_terminated=true`; NEXT349/NEXT350 must not exist.
- If authorized, NEXT349 reuses only unchanged NEXT269 margin-local grammar
  and fixed NEXT224/NEXT135 base; NEXT350 is discovery-only BROAD diagnosis.
  Validation and replication remain sealed.

## Task 1: infinite-capacity kernel and geometry wrapper

Create `src/next347_periodic_allocation_redistribution_capacity.py` and its
tests. RED tests cover exact allocation, the analytic two-site case,
self-image factor two, exact equality between a primitive quotient and its
explicit two-copy cover, conductance scaling, edge orientation/order/gauge,
invalid/disconnected sources, and residuals. Reuse NEXT339 reciprocal graph
and NEXT267 site volumes without modification. Add NaCl/CsCl/ZnS/distorted
geometry, strict geometry-only input, and all five representation tests.

## Task 2: label-blind probe

Create `experiments/next347_parc_label_blind_probe.py` and tests. Reuse strict
complete-inventory loaders and the exact comparator population. Record the
design and every executed-source hash. After an initial passing probe,
complete the formal builder, rerun the still-label-blind probe against final
source hashes, and retain a result JSON only if every gate still passes.

## Task 3: contingent NEXT347 formal builder

Only after the initial probe passes, TDD the atomic multiprocessing builder,
exact identity/manifests, abstentions, output/source hashes, false boundary
flags, and coverage certificates. Rerun the final probe, then publish only
into
`$PRIS_ARCHIVE/next347_periodic_allocation_redistribution_capacity_v1`.

## Task 4: contingent NEXT348 audit and NEXT349/NEXT350 search

Only after both formal coverage gates pass, create NEXT348 fixed discovery-
only audit with unchanged NEXT268/NEXT324 cohort/folds/gates. Create
NEXT349/NEXT350 only if the exact NEXT348 manifest authorizes them. Do not
open discovery outcomes earlier; keep validation and replication sealed.

## Task 5: verification and independent report

Append only to
`reports/2026-08-08-next115-next117-hcid-no-dft-search.md`; do not modify
`paper/`, `tex/`, `notes/`, `README.md`, or `PREREG.md`. Run focused/adjacent
tests, byte compilation, exact hashes/boundaries, CodeGraph sync, and the
complete suite before handoff.

## Execution note

The checkout is intentionally dirty and shared. Preserve all prior content,
work additively, make no Git commit/merge/cleanup, and do not delegate.
