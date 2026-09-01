# Path-Constrained A-Priori Bond Positivity Implementation Plan

> **For Codex:** Execute additively and test first. Preserve every prior
> script/result and all canonical documents. Keep validation and replication
> sealed.

**Goal:** Test whether a raw periodic contact topology admits a predominantly
positive, unique path-constrained a-priori bond-strength field under exact
formal-charge conservation, without DFT, bond lengths in the field solve,
relaxation, or learned potentials.

**Architecture:** NEXT440 defines one bounded graph-Poisson feature on the
unchanged NEXT19 opposite-sign periodic graph and subjects it to the unchanged
80+80 label-blind gates. NEXT441--NEXT444 are conditional full label-free
build, cross-source discovery audit, bounded formula search, and BROAD residual
diagnostic.

## 1. Scientific question and prior-work boundary

Hawthorne's 2026 a-priori bond-strength rule states that incident directed
bond strengths conserve formal charge and that directed strengths sum to zero
along non-degenerate paths between symmetrically equivalent ions
(<https://doi.org/10.1180/mgm.2026.10215>). The paper explicitly separates
this set-theoretic, topology-and-charge field from observed-distance
bond-valence fields.

NEXT425 instead selected a positive maximum-entropy field from all exact
unsigned transports and then compared its order with raw lengths. NEXT435
maximized a positive all-edge floor over the same unconstrained transport
polytope. Neither enforced the zero-path-sum/zero-loop condition. P3 Hawthorne
solved unsigned least-squares and NNLS fields, NEXT307 projected a
distance-derived bond-valence field onto cycle space, NEXT323 solved 3D vector
contact equilibrium, and NEXT347 solved a radical-volume redistribution
Poisson problem. NEXT440 is distinct: it obtains the unique curl-free
formal-charge bond field itself and measures whether its physical
cation-to-anion orientation stays positive.

A minimum-cut version is rejected prospectively because it is dual to the
NEXT435 max-min flow. A fitted edge conductance is rejected because it would
add a parameter or import Euclidean distance into the a-priori field. A raw
minimum edge value is rejected because it is charge-scale and system-size
dependent.

## 2. Hard no-DFT boundary

The executable formula may read only element identities, deterministic NEXT19
formal valences, and one raw initial unrelaxed fully periodic geometry. The
geometry is used only to construct the unchanged opposite-sign NEXT19 Voronoi
multigraph; all field conductances are exactly one and no edge length enters
the solve.

It must not run DFT; read energy, force, stress, hull, or any outcome; use an
ML energy/force/stress proxy, MLIP, or potential; relax a structure; read a
trajectory, later geometry, or same-composition alternative; or access
validation/replication data. Discovery outcomes are permitted only offline
after a successful frozen full label-free build. Canonical `paper/`, `tex/`,
`notes/`, `README.md`, and `PREREG.md` remain untouched.

## 3. Frozen PCABP graph-Poisson field

Let `q_i` be finite, nonzero, neutral formal charges. Orient every translated
opposite-sign contact edge `e=(c,a,image)` from its cation to its anion. Let
`B` be the signed site-edge incidence matrix, with `+1` at `c` and `-1` at
`a`. Freeze the unit-conductance gradient field

```text
L = B B^T,
phi = L^+ q,
s = B^T phi.
```

`s` is the unique minimum-Euclidean-norm edge field satisfying both exact
charge conservation `B s=q` and the zero directed sum around every graph loop;
equivalently, path sums depend only on their endpoint potentials and vanish
between topologically equivalent equal-potential endpoints. The pseudoinverse
gauge does not affect `s`. Independently verify `B s=q` and the loop-free
projection residual at relative tolerance `1e-9`.

Define

```text
S+ = sum_e max(s_e, 0),
S- = sum_e max(-s_e, 0),
PCABP(x0) = round_1e-10(S+/(S+ + S-)).
```

The sole feature is
`pcabp_path_constrained_bond_positivity`, with frozen direction
`protected_high`. It equals one when every path-constrained a-priori bond
points physically from cation to anion and decreases only when the unique
curl-free charge-conserving solution requires reversed bonds. A proven
component-charge obstruction, isolated charged site, or absent usable
opposite-sign edge is a supported physical violation with value zero. Formal
valence inference failure, malformed inputs, or an indeterminate solver fails
closed.

Positive global charge scaling, edge order, disjoint exact replication, rigid
motions, site permutation, unimodular rebasing and exact supercells must leave
the feature unchanged within `1e-8`. There is no alternate conductance,
regularizer, graph, cutoff, norm, quantile, direction, transform, or companion
feature.

## 4. Frozen label-blind gates

Use the same deterministic 80 discovery rows from each source, reading only
`x0`, base label-free features, all 32 prior formal families, and recomputed
ZBVVG, BECNS, SSSP, OBS, P4BSS, APRBS, ECSLO and PVTM controls. No outcome
field may be read.

For each source require support at least `72/80`, output in `[0,1]`, at least
20 values distinct at `1e-10`, maximum rigid/permutation/rebasis/supercell
error at most `1e-8`, and maximum absolute Spearman strictly below `0.90`
against every adequate control with at least 40 joint finite rows.

Any failure records `next441_formal_build_authorized=false` and terminates the
branch. No later artifact may run.

## 5. Conditional full loop

Only after all label-blind gates pass:

1. NEXT441 processes 13,470 SCIGEN and 5,232 WyFormer discovery rows in label
   isolation and requires at least `0.95` support in both sources.
2. NEXT442 uses the unchanged NEXT224/NEXT413 rejected-extreme cohorts,
   reduced-formula five-fold split, inverted-CDF normalization, and frozen
   source AUC/coverage gates for this one `protected_high` feature.
3. NEXT443 runs only after a two-source NEXT442 pass and reuses exactly the
   NEXT261/NEXT414 width and amplitude grids.
4. NEXT444 runs only for AUC+SAFE12 candidates missing BROAD and reuses the
   frozen NEXT415 strict residual-improvement rule.

Validation and replication remain sealed; discovery success is not a
confirmed law.

## 6. Test and artifact order

1. Add RED tests for analytic positive and reversed-edge fields,
   component-charge obstruction, scaling/order/replication invariance, and
   malformed inputs.
2. Implement the pure graph-Poisson kernel, periodic wrapper, supported-zero
   physical-obstruction semantics, and geometry firewall.
3. Add and run the 80+80 label-blind probe with PVTM and every earlier control.
4. Continue mechanically only under Sections 4--5.
5. Append all results to the independent report
   `reports/2026-08-08-next115-next117-hcid-no-dft-search.md`.
6. Run focused and full tests and verify hashes/boundary flags before closeout.
