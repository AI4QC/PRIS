# Positive Valence-Transport Margin Implementation Plan

> **For Codex:** Execute additively and test first. Preserve every prior
> script/result and all canonical documents. Keep validation and replication
> sealed.

**Goal:** Measure whether an opposite-sign periodic bond graph can carry exact
formal charge while keeping every edge strictly active, without DFT,
distance-derived bond strengths, relaxation, or learned potentials.

**Architecture:** NEXT435 defines one bounded linear-programming margin on the
frozen NEXT19 periodic graph and subjects it to the unchanged 80+80
label-blind gates. NEXT436--NEXT439 are conditional full label-free build,
cross-source discovery audit, bounded formula search, and BROAD residual
diagnostic.

## 1. Scientific question and prior-work boundary

Hawthorne's a-priori bond-strength rule requires incident bond strengths to
sum to formal site charges, while bond topology supplies the unknown edge
field (<https://doi.org/10.1180/mgm.2026.10215>). NEXT425 showed that a unique
strictly positive maximum-entropy completion is label-blind novel but lacks
the frozen `0.95` full WyFormer coverage: many graphs are infeasible, contain
isolated charged sites, or force the optimum to the boundary with zero edge
strength. That failure motivates measuring the boundary margin itself rather
than treating it as missing data.

Existing quantities do not compute this margin. P3 Hawthorne records NNLS and
minimum-norm residuals plus incidence rank deficiency on a collapsed simple
graph. NEXT19 minimizes overload and geometry-prior reallocation. NEXT38
projects a distance-derived correction, while NEXT347 measures redistribution
capacity for another allocation problem. NEXT435 instead maximizes a uniform
strict-positive lower bound under exact charge marginals on the full periodic
multigraph. Label-blind correlation with all these controls remains a hard
gate.

Rejected alternatives before computing a NEXT435 value are the maximum-entropy
iteration count/residual, which is numerical; raw minimum edge flow, which is
charge-scale dependent; and a fitted edge floor, which adds a tunable constant.

## 2. Hard no-DFT boundary

The executable formula may read only element identities, deterministic NEXT19
formal valences, and one raw initial unrelaxed fully periodic geometry. It may
use the unchanged opposite-sign NEXT19 Voronoi graph. It may solve a linear
program but may not change coordinates or cell.

It must not run DFT; read energy, force, stress, hull, or any outcome; use an
ML energy/force/stress proxy, MLIP, or potential; relax a structure; read a
trajectory, later geometry, or same-composition alternative; or access
validation/replication data. Discovery outcomes are permitted only offline
after a successful frozen full label-free build. Canonical `paper/`, `tex/`,
`notes/`, `README.md`, and `PREREG.md` remain untouched.

## 3. Frozen PVTM linear program

Let `b_i=|q_i|` for finite nonzero neutral formal charges. For edge
`e=(c,a,image)` in the periodic graph let `CN_i` count translated incident
opposite-sign contacts and define a symmetric positive reference

`r_e=sqrt((b_c/CN_c)(b_a/CN_a))`.

Let `C` be the unsigned site-edge incidence matrix. Freeze

`lambda*=max lambda`

subject to

`C x=b`, `x_e >= lambda r_e` for every edge, `x_e>=0`, `lambda>=0`.

Solve with SciPy HiGHS using its deterministic default feasibility machinery;
verify the returned equality and lower-bound residuals independently at
`1e-8`. If exact nonnegative conservation is infeasible, a charged site is
isolated, or the periodic graph has no usable opposite-sign edge, define
`lambda*=0` as a supported physical violation rather than a missing value.
Formal-valence inference failure and malformed numeric inputs remain
unsupported. A solver status other than optimal or proven infeasible fails
closed.

Global charge scaling multiplies both `x` and `r` and leaves `lambda*`
unchanged. Exact cell replication duplicates the feasible system without
changing its optimum. Freeze the bounded feature

`pvtm_positive_transport_margin = round_1e-10(lambda*/(1+lambda*))`.

Its sole direction is `protected_high`; zero means no strict positive
charge-conserving field, and larger values mean greater all-edge interior
margin. Require a finite nonnegative optimum and output in `[0,1)`. There is no
alternate reference mean, regularizer, graph, cutoff, solver objective,
transform, direction, or companion feature.

## 4. Frozen label-blind gates

Use the same deterministic 80 discovery rows from each source, reading only
`x0`, base label-free features, all 32 prior formal families, and recomputed
ZBVVG, BECNS, SSSP, OBS, P4BSS, APRBS, and ECSLO controls. No outcome field may
be read.

For each source require support at least `72/80`, output in `[0,1]`, at least
20 values distinct at `1e-10`, maximum rigid/permutation/rebasis/supercell
error at most `1e-8`, and maximum absolute Spearman strictly below `0.90`
against every adequate control with at least 40 joint finite rows.

Any failure records `next436_formal_build_authorized=false` and terminates the
branch. No later artifact may run.

## 5. Conditional full loop

Only after all label-blind gates pass:

1. NEXT436 processes 13,470 SCIGEN and 5,232 WyFormer discovery rows in label
   isolation and requires at least `0.95` support in both sources.
2. NEXT437 uses the unchanged NEXT224/NEXT413 rejected-extreme cohorts,
   reduced-formula five-fold split, inverted-CDF normalization, and frozen
   source AUC/coverage gates for this one `protected_high` feature.
3. NEXT438 runs only after a two-source NEXT437 pass and reuses exactly the
   NEXT261/NEXT414 width and amplitude grids.
4. NEXT439 runs only for AUC+SAFE12 candidates missing BROAD and reuses the
   frozen NEXT415 strict residual-improvement rule.

Validation and replication remain sealed; discovery success is not a
confirmed law.

## 6. Test and artifact order

1. Add RED tests for analytic feasible/interior/boundary/infeasible graphs,
   scaling/order/replication invariance, and malformed inputs.
2. Implement the pure LP kernel, periodic wrapper, zero-margin graph-failure
   semantics, and geometry firewall.
3. Add and run the 80+80 label-blind probe with APRBS/ECSLO controls.
4. Continue mechanically only under Sections 4--5.
5. Append all results to the independent report
   `reports/2026-08-08-next115-next117-hcid-no-dft-search.md`.
6. Run focused and full tests and verify hashes/boundary flags before closeout.
