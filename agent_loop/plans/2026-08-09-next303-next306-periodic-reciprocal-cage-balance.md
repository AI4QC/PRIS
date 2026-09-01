# NEXT303--NEXT306 Periodic Reciprocal Cage Balance Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: use `superpowers:test-driven-development`
> task by task and `superpowers:verification-before-completion` before reporting.
> The active shared checkout is intentional; do not create a worktree, branch,
> commit, PR, or modify any canonical manuscript/report artifact.

**Status:** frozen before computing or joining any NEXT303 feature outcome.

**Goal:** Test whether full-cell reciprocity between the dimension-minimal
opposite-sign cages adds a transferable, DFT-free protection law beyond the
site-independent NEXT299 MOSPC descriptors.

**Architecture:** NEXT303 reconstructs each site's certified NEXT299 cage,
canonicalizes its periodic images into one undirected edge-orbit graph, assigns
each edge one shared amplitude, and publishes thirteen continuous reciprocal
closure/mutual-selection descriptors. NEXT304 audits those frozen directions on
the unchanged discovery cohort. NEXT305 reuses the unchanged finite
margin-local grammar only if NEXT304 authorizes at least one hypothesis;
NEXT306 is run only for an AUC+SAFE/non-BROAD identity set.

**Tech Stack:** Python 3.11, ASE, pymatgen, NumPy, pandas, pytest, Parquet,
SHA-256, immutable NEXT19/NEXT267/NEXT295/NEXT299 geometry helpers, and the
unchanged NEXT268/NEXT227 audit/search gates.

---

Date frozen: 2026-08-09 (America/Chicago).

## Scientific freeze

This is sequential discovery: NEXT299--NEXT302 discovery outcomes have already
been seen. They showed that local inverse-square cage closure transferred as an
AUC signal but failed the fixed BROAD gate. NEXT303 therefore changes the
mechanism rather than its threshold: a periodic edge selected by either endpoint
is represented once, carries one amplitude at both endpoints, and contributes
equal-and-opposite vectors. No validation or replication geometry, endpoint, or
outcome may be opened.

The executable object receives only element identities and one initial raw,
unrelaxed, fully periodic geometry. It must reject calculators, nonempty ASE
`info`, extra arrays, incomplete PBC, nonfinite geometry, and singular cells. It
must never read or compute DFT values, use a learned energy/force/stress proxy,
use a model or proxy potential, execute physical relaxation, inspect a later
structure/trajectory, or read validation/replication outputs. Formal valences
are the immutable composition-only NEXT19 assignment. The word "force" below
means a dimensionless analytic edge amplitude, not a DFT/model force.

Before this freeze, a label-free 60+60 discovery-geometry probe supported 60/60
SCIGEN and 59/60 WyFormer records. The nine closure candidates were continuous
and nondegenerate. Mutual-selection candidates had Spearman correlation only
about 0.28--0.56 with `mospc_inverse_square_closure_q10`. An exact positive
self-stress floor was zero in 91.7% of the SCIGEN probe and is therefore
excluded before outcome access. The probe loaded full discovery inventories to
satisfy the cohort identity firewall, then sampled in memory; it read no
endpoint field.

## Frozen NEXT303 construction

1. Apply NEXT295's exact geometry-only guard, NEXT267's Minkowski reduction and
   wrapping, NEXT19's formal signs, and NEXT299's certified fourth-nearest
   opposite-sign cage including every fourth-distance tie.
2. NEXT303 must retain, in addition to each vector and distance, the neighbor
   basis index and integral periodic shift. For a directed image
   `(i, j, T)`, canonicalize the physical edge orbit as the lexicographically
   smaller of `(i, j, T)` and `(j, i, -T)`. Reverse the vector when the latter
   representation is selected. Conflicting duplicate vectors fail closed.
3. Use the union of all site-selected edge orbits. For edge `e=(i,j,T)`, let
   `u_e` point from canonical endpoint `i` to `j+T`, with distance `d_e`.
   Endpoint `i` receives `+u_e` and endpoint `j` receives `-u_e`.
4. Freeze three shared positive edge-amplitude priors, each normalized only in
   the site closure denominator:

```text
uniform:                  w_e = 1
inverse_square:           w_e = 1 / d_e^2
charge_inverse_square:    w_e = |z_i z_j| / d_e^2
```

5. For every site `i` and prior `w`, define the threshold-free reciprocal
   closure

```text
closure_i(w) = clip(1 - ||sum_{e incident i} s_ie w_e u_e||
                         / sum_{e incident i} w_e, 0, 1),
s_ie = +1 at the canonical first endpoint and -1 at the second.
```

6. An edge is mutually selected exactly when the same canonical orbit occurs
   in both endpoint cages. For each site, define its mutual fraction as mutual
   incident union edges divided by all incident union edges. Define global
   mutual edge fraction analogously. No fitted distance/contact threshold is
   permitted.
7. Round final values to the immutable NEXT299 `1e10` grid. Require every
   supported value finite in `[0,1]`, every site incident to at least one edge,
   and all canonical duplicates consistent to `1e-8` Cartesian absolute error.

## Frozen feature catalogue

Publish `min`, inverse-CDF `q10`, and arithmetic `mean` for each of the three
site closure priors, plus the same three summaries of site mutual fraction and
one global mutual-edge fraction:

```text
prcb_uniform_closure_min
prcb_uniform_closure_q10
prcb_uniform_closure_mean
prcb_inverse_square_closure_min
prcb_inverse_square_closure_q10
prcb_inverse_square_closure_mean
prcb_charge_inverse_square_closure_min
prcb_charge_inverse_square_closure_q10
prcb_charge_inverse_square_closure_mean
prcb_mutual_site_fraction_min
prcb_mutual_site_fraction_q10
prcb_mutual_site_fraction_mean
prcb_mutual_edge_fraction
```

All directions are frozen as `protected_high`. Publish all exact 13,470
SCIGEN and 5,232 WyFormer discovery identities. Unsupported rows retain IDs and
NaN features. Formal publication requires per-source coverage at least 0.97.

## Task 1: NEXT303 TDD and label-free build

**Files:**

- Create: `tests/test_next303_periodic_reciprocal_cage_balance.py`
- Create: `src/next303_periodic_reciprocal_cage_balance.py`
- Create atomically: `$PRIS_ARCHIVE/next303_periodic_reciprocal_cage_balance_v1`

1. Write focused tests first for exact schema, NaCl/CsCl/ZnS unit symmetry,
   distorted-cell nontriviality, mutual canonicalization, fourth-shell ties,
   rigid rotation/translation/site permutation/equivalent cell/replication
   invariance, strict input rejection, fail-open unsupported chemistry, and a
   discovery-only builder smoke.
2. Run the test module and record the missing-module RED failure.
3. Implement only the frozen graph, feature computation, fail-open result, and
   atomic publication interface. Reuse immutable upstream routines rather than
   changing them.
4. Run the focused test until GREEN, then run all NEXT267/NEXT295/NEXT299 tests
   as regressions.
5. Freeze exact plan/source/test/input hashes, run the formal discovery-only
   build once, and verify row counts, coverage, ranges, identity uniqueness,
   manifests, and boundary flags. Do not commit in this shared dirty checkout.

## Task 2: NEXT304 fixed feature audit

**Files:**

- Create: `tests/test_next304_prcb_feature_audit.py`
- Create: `src/next304_prcb_feature_audit.py`
- Create atomically: `$PRIS_ARCHIVE/next304_prcb_feature_audit_v1`

1. Write RED tests for the exact thirteen hypotheses, protected-high bounded
   transform, source-prefix identity checks, provenance/boundary rejection, and
   eligible-hypothesis ranking.
2. Implement by reusing the exact NEXT300/NEXT268 reconstruction, rejected-
   extreme population, quantiles, source/fold AUC gates, coverage gates, and
   deterministic ranking. No extra candidate or direction may be introduced.
3. Freeze hashes, run once on discovery outcomes, and publish every hypothesis,
   not only winners. Authorization requires the unchanged cross-source gates.

## Task 3: Conditional NEXT305 search

**Files:**

- Create only if NEXT304 authorizes at least one feature:
  `tests/test_next305_prcb_margin_local_search.py`
- Create only if authorized: `src/next305_prcb_margin_local_search.py`
- Create atomically only if authorized:
  `$PRIS_ARCHIVE/next305_prcb_margin_local_search_v1`

Reuse NEXT301's exact frozen candidate grammar, baseline score, threshold grid,
SAFE/AUC/BROAD gates, tie-breaking, and publication schema. Only replace the
authorized MOSPC feature catalogue with the exact NEXT304-authorized PRCB
identities. If none are eligible, stop this task and record the predeclared
stop without inventing a rescue.

## Task 4: Conditional NEXT306 diagnostic

**Files:**

- Create only for an AUC+SAFE/non-BROAD identity set:
  `tests/test_next306_prcb_broad_diagnostic.py`
- Create only if authorized: `src/next306_prcb_broad_diagnostic.py`
- Create atomically only if authorized:
  `$PRIS_ARCHIVE/next306_prcb_broad_diagnostic_v1`

Reuse NEXT302's exact BROAD failure decomposition without changing a threshold,
candidate, or score. This is diagnosis, not a new search. If NEXT305 yields an
all-gate pass, omit NEXT306.

## Task 5: Independent report and verification

Append a clearly separated NEXT303--NEXT306 section only to
`reports/2026-08-08-next115-next117-hcid-no-dft-search.md`. State sequential-
discovery status, exact information boundary, coverage, all fixed audit/search
results, negative results, and whether validation remains sealed. Do not edit
`paper/`, `tex/`, `notes/`, `README.md`, or `PREREG.md`.

Before handoff, run the focused modules, the full pytest suite, every formal
artifact/hash/boundary verifier, CodeGraph health, and an exact canonical-path
diff. A discovery pass is only a candidate; no scientific replacement claim
is allowed without separately authorized sealed validation and replication.
