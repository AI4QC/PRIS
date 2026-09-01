# Topological Bond-Strength Angular-Territory Correspondence Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to
> implement this plan task-by-task. Execute additively and test first. Preserve
> every prior artifact. Keep validation and replication sealed.

**Goal:** Test whether a crystal is more plausible when the topology-only,
path-constrained a-priori bond strength carried by each periodic contact agrees
with that contact's share of the local Voronoi angular territory in the one raw
initial geometry.

**Architecture:** NEXT500 constructs the unchanged opposite-sign ordinary
periodic Voronoi multigraph, solves Hawthorne's charge-conserving curl-free
edge field, and compares that topological allocation independently at both
ends of every translated edge with a solid-angle allocation of the site's
formal charge. NEXT501--NEXT504 are conditional full label-free build,
discovery outcome audit, bounded search and BROAD diagnostic.

**Tech Stack:** Python, NumPy, ASE, pymatgen, pandas, pytest, existing sealed
SCIGEN/WyFormer cohort readers.

## 1. Scientific choice and novelty boundary

Three mechanisms were considered before opening new results:

1. Spherical Riesz/Thomson repulsion was rejected because an executable
   inverse-chord energy could reasonably be interpreted as a model or proxy
   potential and overlaps existing angular-spacing features.
2. Direction-frame isotropy and largest-empty-cap tests were rejected because
   NEXT239/NEXT243/NEXT295/NEXT335/NEXT359/NEXT416 already cover their local
   geometric content.
3. The frozen candidate couples two previously separate representations:
   Hawthorne's a-priori bond strength in set-theoretic topology and the
   ordinary Voronoi solid angle of its embedding in Euclidean space.

CHARDI distributes formal charge from bond-length weights; O'Keeffe's solid
angle defines a geometric coordination weight. NEXT500 is neither formula: it
uses the topology-only path field as the quantity to be checked and the raw
Voronoi angular territory as an independent embedding constraint. No existing
repository feature directly compares these two edgewise shares at every site.

## 2. Frozen graph and formula

Use the ordinary periodic Voronoi graph with every translated opposite-sign
contact retained as a separate edge. Orient every edge from positive to
negative formal charge. Let `B` be its signed site-edge incidence matrix and
`q` the neutral composition-only formal charge vector. Freeze

```text
L = B B^T,
phi = L^+ q,
s = B^T phi.
```

This is the unchanged NEXT440 unit-conductance, charge-conserving, zero-loop-
sum a-priori field. For each edge `e` use its cation-centred raw ordinary
Voronoi solid angle `Omega_e > 0`. The shared ordinary Voronoi facet makes this
an edge property; using the cation-centred evaluation gives one deterministic
orientation even when two numerical evaluations of the same facet differ.

For every site-edge incidence `(i,e)`, allocate the site's absolute formal
charge by local angular territory,

```text
t_ie = |q_i| Omega_e / sum_(f incident to i) Omega_f.
```

Freeze the sole candidate

```text
N = sum_(i,e incident to i) |s_e - t_ie|,
D = sum_(i,e incident to i) (|s_e| + t_ie),
TBAC(x0) = round_1e-10(1 - N/D).
```

The feature is
`tbac_topological_bond_angular_correspondence`, direction `protected_high`,
range `[0,1]`. A positive path field that exactly matches both endpoint angular
allocations scores one. A negative edge contributes maximal mismatch at both
ends. A charge-infeasible disconnected contact graph receives supported
physical zero. Malformed input, a missing site incidence, nonpositive angle or
failed formal-valence inference fails closed.

No alternate norm, site quantile, facet-area definition, radical radius,
solid-angle cutoff, fitted exponent, chemistry-specific exception or feature
direction may be selected after the probe is seen.

## 3. Hard no-DFT boundary and invariance

The executable reads only composition and one raw, initial, unrelaxed periodic
geometry. It may use deterministic composition-only formal-valence inference,
ordinary periodic Voronoi topology and solid angles, and linear algebra on the
resulting graph. It must not run or read DFT; energy, force or stress; learned
proxies; MLIPs or model potentials; relaxation; trajectories; later geometry;
same-composition alternatives; discovery outcomes; validation; or replication.

The pure kernel must be invariant to charge scale, edge order and exact
disjoint replication. The raw wrapper must be invariant within `1e-8` to rigid
rotation, translation, site permutation, unimodular rebasing and exact
supercells. A calculator or any extra `Atoms.info`/array payload fails closed.

## 4. Frozen ordered label-blind gates

Use the unchanged deterministic 80+80 discovery probes. Before opening any
prior feature table, require in each source:

```text
support >= 72/80,
all finite values in [0,1],
at least 20 values distinct at 1e-10,
maximum representation error <= 1e-8.
```

Only if all engineering gates pass, compare TBAC with every available prior
label-free formal feature and direct recomputations through NEXT495. Require
at least 40 joint finite rows and maximum adequate absolute Spearman `<0.90`
in each source. The direct set must explicitly include PCRL, periodic CHARDI
return consistency, PCABP, PCABSM, PFPU, CCLAB and CCLAB-CDE.

Only if all pass may NEXT501 build the full discovery tables and require
coverage `>=0.95` in each source. Only that certificate may authorize NEXT502
to open discovery outcomes under the unchanged NEXT224/NEXT413 gates.
Validation and replication remain sealed until one formula later passes every
discovery gate and is frozen.

## 5. Test-first artifact order

### Task 1: RED kernel and boundary tests

**Files:**
- Add: `tests/test_next500_topological_bond_angular_correspondence.py`

1. Assert frozen schema, direction and all-false boundary flags.
2. Assert exact match, negative-edge maximal mismatch, infeasible graph zero,
   charge-scale/edge-order/replication invariance and malformed fail-closed.
3. Assert raw periodic representation invariance and geometry firewall.
4. Run this test and observe failure because NEXT500 does not exist.

### Task 2: GREEN minimal NEXT500 implementation

**Files:**
- Add: `src/next500_topological_bond_angular_correspondence.py`

1. Implement the pure edge-incidence formula with strict domain checks.
2. Implement ordinary periodic Voronoi cation-side solid-angle extraction.
3. Implement the raw-geometry wrapper and row adapter.
4. Run the focused kernel suite to green.

### Task 3: RED then GREEN ordered engineering probe

**Files:**
- Add: `tests/test_next500_tbac_label_blind_engineering_probe.py`
- Add: `experiments/next500_tbac_label_blind_engineering_probe.py`

1. Test the frozen gate thresholds, no-label interface and hash coverage.
2. Implement only the 80+80 raw-geometry engineering probe.
3. Run focused tests, then execute the probe and save its complete record.
4. Stop the branch if any engineering gate fails.

### Task 4: Conditional novelty and downstream work

Only if Task 3 authorizes it:

- Add novelty-probe tests and `experiments/next500_tbac_label_blind_novelty_probe.py`.
- Run the frozen prior-feature comparisons and save the complete record.
- Only if novelty passes, add NEXT501 full discovery build and coverage gate.
- Only if NEXT501 passes, add NEXT502 discovery outcome audit.
- Only if NEXT502 passes, add the bounded NEXT503 search and NEXT504 BROAD
  diagnostic. Do not open validation or replication here.

### Task 5: Independent report and verification

**Files:**
- Modify additively: `reports/2026-08-13-hawthorne-characteristic-cn-no-dft-search.md`

1. Record the physical derivation, literature/repository distinction, hashes,
   gates and exact stopping point without editing canonical artifacts.
2. Run focused tests and the complete repository suite from fresh output.
3. Verify boundary flags, result hashes, manifest integrity, working-tree scope
   and that paper/README/notes/preregistration/TeX targets were untouched.
