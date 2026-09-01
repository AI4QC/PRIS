# Periodic Geometric Homogenized Transmissivity Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` and
> `superpowers:test-driven-development` task by task. Do not delegate.

**Goal:** Test whether a periodic, facet-measure-weighted graph cell problem
detects internally short-circuitable raw crystal allocations that prior local
shape, topology, charge, and equilibrium rules miss, and only if
prospectively authorized, use it in the bounded no-DFT screening loop.

**Architecture:** NEXT339 reconstructs reciprocal radius-aware power-cell
facets with their polygon areas and exact periodic generator displacements.
Each shared facet becomes one undirected finite-volume edge. A deterministic
periodic corrector problem eliminates all cell-internal scalar modes and
compares the homogenized quadratic response with its uncorrected affine
response. A label-blind probe precedes any formal build or outcome access.
NEXT340 fixed audit and NEXT341/NEXT342 search are strictly contingent.

**Tech Stack:** Python 3.11, NumPy/SciPy linear algebra and half-space
geometry, pandas, ASE, existing isolated discovery inventories, pytest,
CodeGraph.

## Scientific and information-boundary freeze

Let each reciprocal radius-aware power-cell facet define one undirected
periodic edge `e=(i,j,s)` from raw site `i` to image `j+s`, physical generator
displacement `d_e`, facet area `A_e`, and purely geometric finite-volume
transmissibility

```text
g_e = A_e / ||d_e|| > 0.
```

Choose one canonical representative from the two directed reciprocal facets;
their independently reconstructed areas must agree within the frozen
relative tolerance. Let `B` be the edge-by-site incidence matrix (`-1` at
`i`, `+1` at `j`; a self-image edge has a zero row) and let `D` contain the
physical displacement rows. For imposed affine vector `E`, solve the periodic
cell problem

```text
min_phi sum_e g_e [d_e . E + (B phi)_e]^2.
```

The uncorrected and corrected tensors are

```text
K0 = D.T G D / V,
K  = (D + B Phi).T G (D + B Phi) / V,
H  = K0^(-1/2) K K0^(-1/2).
```

`Phi` contains the three minimum-energy correctors with one gauge-free
pseudoinverse solve. The sole frozen feature is
`pght_affine_retention_floor = lambda_min(H)` in the `protected_high`
direction. Variationally `0 <= H <= I`; one-site Bravais networks have
exactly one, while a multi-site basis with an internal scalar mode that
cancels much of an affine gradient yields a smaller value. Values at or below
the `1e-10` output grid are unsupported rather than clipped.

`g=A/d` is the standard orthogonal Voronoi finite-volume transmissibility;
periodic graph homogenization has a variational effective tensor. These facts
motivate the cell problem, not a claim that PGHT predicts real electrical,
ionic, thermal, or mass conductivity. It is a dimensionless geometric
certificate only, not a force, energy, stress, potential, or learned proxy.

Executable inputs are limited to element identities, deterministic tabulated
radii, and one initial raw unrelaxed periodic geometry. The branch must not
execute or consume DFT calculations or per-structure DFT values; learned
energy/force/stress proxies; MLIPs; model/proxy potentials; relaxed
structures; trajectories; later geometries; discovery labels during the
probe/build; validation outcomes; or replication outcomes. Discovery
outcomes may enter NEXT340 only as offline audit labels.

## Distinction and alternatives frozen before feature evaluation

1. NEXT166 retains only the integer rank of periodic contact-component
   winding. PGHT is a continuous area/distance-weighted corrector energy;
   NEXT166 is a mandatory topology comparator.
2. NEXT315 solves a formal-charge Green resistance on an unweighted radical
   contact graph. PGHT uses no charge, source/sink vector, or voltage-drop
   statistic and instead applies three affine periodic cell problems. NEXT315
   is mandatory.
3. NEXT168/NEXT173 are local direction Gram tensors; NEXT323 is a global
   positive equilibrium LP; NEXT307 is bond-valence Hodge-loop frustration.
   None eliminates internal node modes in a facet-measure-weighted affine
   graph response. All remain mandatory comparators.
4. RFMP and RFMT use individual-cell area statistics. PGHT uses the reciprocal
   cross-cell network and normalizes by its own affine tensor, removing the
   single-cell second-moment anisotropy that made RFMT redundant.
5. NEXT37 projects radius-sum edge mismatch through an atomic-plus-affine
   rigidity matrix. PGHT uses scalar graph correctors and facet conductance,
   no preferred-radius mismatch or mechanical rigidity matrix. NEXT37 is not
   available in the recent formal label-free population but is a conceptual
   exclusion.

Rejected alternatives: raw conductivity trace/eigenvalue (cell-shape and
scale confounding), conductivity anisotropy (RFMT/NEXT267 overlap), unweighted
graph conductance (NEXT315 overlap), topology rank alone (NEXT166 duplicate),
and any second statistic. No alternative weight, normalization, eigenvalue
function, direction, aggregate, quantile, threshold, chemistry condition, or
conjunction may be added after the probe starts.

## Frozen gates

- Probe 80 deterministic discovery initial geometries per source after the
  complete identifier inventory is loaded; read no outcome-bearing field.
- Engineering: reciprocal facet pairing and area relative error at most
  `1e-7`; cell facet/surface certificate at most `1e-7`; volume tiling at the
  unchanged NEXT267 tolerance; corrector residual at most `1e-8`; generalized
  spectrum in `[0,1]` up to `1e-8`; output strictly `(0,1]`; support at least
  72/80 and at least 20 `1e-10`-unique values per source; rotation,
  translation, permutation, unimodular lattice rebasing, and exact integral
  supercell error at most `1e-8`.
- Novelty: maximum absolute Spearman below `0.90` independently in both
  sources against available NEXT166, NEXT168, NEXT173, NEXT179, NEXT239,
  NEXT243, NEXT247, NEXT251, NEXT255, NEXT259, NEXT263, NEXT267, NEXT271,
  NEXT275, NEXT279, NEXT283, NEXT291, NEXT295, NEXT299, NEXT303, NEXT307,
  NEXT311, NEXT315, NEXT319, and NEXT323 label-free populations. Failure stops
  before formal NEXT339 or outcomes.
- Formal NEXT339 coverage: at least `0.90` independently in both sources;
  unsupported structures remain abstentions without imputation.
- NEXT340 unchanged gates: minimum cell coverage `0.90`, class count `20`,
  pooled AUC `0.55`, macro AUC `0.53`, worst-fold AUC `0.50` in both sources,
  reduced-formula folds, inverse-CDF `1/16` and `15/16`.
- Empty eligible set means `next341_search_authorized=false` and
  `pght_branch_terminated=true`; NEXT341/NEXT342 must not exist.
- If authorized, NEXT341 reuses only unchanged NEXT269 margin-local grammar
  and fixed NEXT224/NEXT135 base; NEXT342 is discovery-only BROAD diagnosis.
  Validation and replication remain sealed.

## Task 1: periodic conductance/corrector kernel

Create `src/next339_periodic_geometric_homogenized_transmissivity.py` and its
test file. RED tests cover one-site cubic retention one, an analytic two-site
periodic loop with retention `0.8`, gauge/edge orientation/order invariance,
rank-deficient refusal, and strict spectrum/residual checks. Reconstruct
labelled radical facets with reciprocal area certificates, canonicalize one
undirected edge, and add real NaCl/CsCl/ZnS/distorted-cell plus all five
representation-invariance and geometry-only tests.

## Task 2: label-blind novelty probe

Create `experiments/next339_pght_label_blind_probe.py` and its test file.
Reuse the strict complete-inventory loaders, calculate all frozen diagnostics,
and record the design and every executed-source hash. Create a retained probe
result JSON only if all gates pass; otherwise stop immediately.

## Task 3: contingent formal build and audit

Only after a passing probe, TDD the atomic multiprocessing NEXT339 builder and
publish into
`$PRIS_ARCHIVE/next339_periodic_geometric_homogenized_transmissivity_v1`.
Only after both formal coverage gates pass, create NEXT340 fixed audit/tests
and publish its immutable eligible-set digest. No discovery outcomes before
this point.

## Task 4: contingent search and report

Create NEXT341/NEXT342 only when the exact NEXT340 manifest authorizes them.
Reuse unchanged search/BROAD code. Append independently to
`reports/2026-08-08-next115-next117-hcid-no-dft-search.md`; do not modify
`paper/`, `tex/`, `notes/`, `README.md`, or `PREREG.md`. Run focused/adjacent
tests, byte compilation, hashes/boundaries, CodeGraph sync, and full suite.

## Execution note

The checkout is intentionally dirty and shared. Preserve all prior content,
work additively, make no Git commit/merge/cleanup, and do not delegate.
