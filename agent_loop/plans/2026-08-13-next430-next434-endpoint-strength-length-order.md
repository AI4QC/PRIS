# Endpoint-Strength Length-Order Implementation Plan

> **For Codex:** Execute additively with tests first. Preserve all prior
> scripts, results, and canonical documents. Do not open validation or
> replication data within this plan.

**Goal:** Test whether bonds with larger local endpoint characteristic strength
are shorter in one raw periodic geometry, using only formal charges and
coordination topology to define strength.

**Architecture:** NEXT430 replaces the coverage-limited global interior
transport of NEXT425 with a closed-form local edge field, then applies the same
local length-order protection statistic. The candidate faces the unchanged
80+80 label-blind gates. NEXT431--NEXT434 are conditional full label-free
construction, cross-source discovery audit, bounded formula search, and BROAD
residual diagnostic.

**Tech stack:** Python 3.11, NumPy, pandas/Parquet, ASE, pymatgen, existing
NEXT19/NEXT267/NEXT295 guards, pytest.

## 1. Motivation and novelty boundary

Hawthorne defines characteristic bond strength as charge divided by
characteristic coordination number and states that stable structures favor
Lewis-acidity/basicity matching
(<https://doi.org/10.1180/mgm.2026.10215>). Pauling's electrostatic bond
strength uses the corresponding local `charge/CN` form. This branch tests a
narrow, raw-geometry consequence: within a coordination star, a bond whose two
endpoints require a larger characteristic strength should not be longer than a
locally weaker bond.

The candidate differs from existing features as follows:

1. P3 Hawthorne tests graph conservation/rank and Pauling-gap quantities but
   has no raw length-order comparison and collapses periodic images.
2. NEXT19/38 begin from a distance-dependent prior and measure conservation
   reallocation or differential compatibility.
3. P4BSS tests whether strong cation stubs co-occur at anions, not whether edge
   strength ranks agree with lengths.
4. NEXT425 used a unique global maximum-entropy conserved field. Its label-blind
   probe passed, but full WyFormer support was only `0.932339`, below the frozen
   `0.95` gate, so no formal table or outcome audit was authorized. NEXT430 does
   not change that failed definition; it asks a simpler predeclared question.

Rejected alternatives before computing a NEXT430 value are cation-only
strength, because it ignores the anion endpoint and overlaps P4BSS; arithmetic
mean, because it lets one endpoint dominate; and fitted inverse-distance or
bond-valence transforms, because NEXT38 already covers them.

## 2. Hard no-DFT boundary

The executable candidate may read only element identities, deterministic
NEXT19 formal valences, and one raw initial unrelaxed periodic geometry. It may
use the unchanged opposite-sign NEXT19 Voronoi graph.

It must not run DFT; read any DFT energy, force, stress, hull value, or outcome;
use an ML energy/force/stress proxy, MLIP, or potential; relax coordinates or
cell; read later geometry, trajectories, or same-composition alternatives; or
access validation/replication data. Discovery outcomes may be used only
offline after a successful frozen full label-free build. Canonical `paper/`,
`tex/`, `notes/`, `README.md`, and `PREREG.md` remain untouched.

## 3. Frozen endpoint characteristic field

Let `q_i` be a finite nonzero neutral formal charge and let `CN_i` be the number
of translated opposite-sign NEXT19 contacts incident to reference-cell site
`i`. For periodic edge `e=(c,a,image)`, define endpoint characteristic strengths

`s_c = |q_c|/CN_c`, `s_a = |q_a|/CN_a`,

and the symmetric edge strength

`g_e = sqrt(s_c s_a)`.

This is closed-form and strictly positive whenever the charged periodic graph
has no isolated site. It is not required to satisfy every global site marginal
simultaneously and is not claimed to reproduce Hawthorne's path-equation
solution. Global charge scaling rescales all `g_e` and cannot affect the final
formula. No Euclidean metric, radius, bond-valence parameter, or learned value
enters `g_e`.

## 4. Frozen ECSLO formula

For every unordered pair of distinct periodic edges `(e,f)` incident to the
same reference-cell cation or rebased anion, define

`Delta_g=(g_e-g_f)/(g_e+g_f)`,

`Delta_d=(d_e-d_f)/(d_e+d_f)`,

and `p_ef=Delta_g Delta_d`. Positive `p_ef` means the topology-only stronger
edge is longer and violates the proposed order. Freeze

`W=sum |p_ef|`, `V=sum max(p_ef,0)`,

`P=1-V/W` for `W>1e-15`, and `P=1/2` otherwise.

Freeze exactly one feature:

`ecslo_endpoint_strength_length_order_protection = round_1e-10(P)`.

Its sole direction is `protected_high`. It is invariant to edge ordering,
uniform length/charge scale, rigid motions, site permutation, cell rebasing,
and exact cell replication. Require neutral nonzero charges with both signs,
positive finite distances and endpoint strengths, exact edge orientation, at
least one contact per site, at least one local edge pair, and output in `[0,1]`.
Malformed input fails closed. There is no alternate mean, exponent, quantile,
species subset, graph, cutoff, direction, or companion feature.

## 5. Frozen label-blind gates

Use the unchanged deterministic 80 discovery records per source chosen from
`(natoms, chemical_system, material_id)`. Read only discovery `x0`, base
label-free features, the 32 prior formal feature families, and recomputed
ZBVVG, BECNS, SSSP, OBS, P4BSS, and APRBS controls. No endpoint field may be
read.

For both SCIGEN and WyFormer require:

- support at least `72/80`;
- all finite values in `[0,1]`;
- at least 20 values distinct at `1e-10`;
- maximum error at most `1e-8` across rigid rotation/translation, site
  permutation, unimodular rebasing, and exact `2 x 1 x 1` supercell;
- maximum absolute Spearman strictly below `0.90` against every adequate prior
  label-free control with at least 40 joint finite rows.

Any failure records `next431_formal_build_authorized=false` and
`ecslo_branch_terminated=true`; NEXT431--434 must not run.

## 6. Conditional full loop

Only after all label-blind gates pass:

1. NEXT431 builds all 13,470 SCIGEN and 5,232 WyFormer discovery rows in label
   isolation and requires at least `0.95` support in both sources.
2. NEXT432 uses the unchanged NEXT224/NEXT413 rejected-extreme cohorts,
   reduced-formula five-fold split, inverted-CDF normalization, and frozen
   per-source AUC/coverage gates for this one `protected_high` feature.
3. NEXT433 is authorized only if NEXT432 passes both sources; it reuses exactly
   the NEXT261/NEXT414 width and amplitude grids with no post-label additions.
4. NEXT434 runs only for AUC+SAFE12 candidates that miss BROAD and reuses the
   frozen NEXT415 strict residual-improvement test.

Validation and replication remain sealed. Discovery performance can nominate
a candidate signal but cannot establish a confirmed law.

## 7. Test and artifact sequence

1. Add failing exact-kernel tests for endpoint strengths, agreeing/reversed/tie
   orders, scale/order/replication invariance, and malformed inputs.
2. Implement NEXT430 core, periodic wrapper, raw-geometry firewall, and
   structure-equivalence tests.
3. Add and run the frozen 80+80 label-blind probe with all prior controls,
   including recomputed APRBS where defined.
4. Continue or stop mechanically under Section 5; if authorized, build NEXT431
   and enforce Section 6.
5. Append all outcomes to the independent additive report
   `reports/2026-08-08-next115-next117-hcid-no-dft-search.md`.
6. Run focused and full tests; verify hashes, no-DFT flags, sealed later
   partitions, and no canonical-document changes.
