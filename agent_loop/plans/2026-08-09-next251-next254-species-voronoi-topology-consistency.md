# NEXT251--NEXT254 Species-Conditioned Voronoi Topology Consistency Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Test whether repeated atoms of the same element having coherent local Voronoi face topology supplies a fully pre-DFT, interpretable correction to the frozen NEXT224 law.

**Architecture:** NEXT251 computes a fixed discovery-only feature bank from initial periodic geometry. NEXT252 audits fixed feature directions in the exact NEXT224 rejected-extreme cohort. NEXT253 and NEXT254 are conditionally authorized search and BROAD-diagnostic stages and must not run unless their preceding frozen gates authorize them.

**Tech Stack:** Python 3.11, NumPy, pandas/Parquet, pymatgen `VoronoiNN`, pytest, the existing NEXT227/NEXT224 reconstruction and gate evaluators.

Date frozen: 2026-08-09 (America/Chicago)

## Status and distinct mechanism

NEXT247--NEXT248 tested normalized third-order Steinhardt invariants. All
features were complete and finite, but none of sixteen prospectively directed
hypotheses passed both discovery-source raw gates. NEXT249 and NEXT250 were
therefore forbidden and remain uncreated. Those inspected angular directions
and normalizations may not be reopened or retuned.

The next mechanism is the polygon topology of each atom's Voronoi cell. It is
not the periodic translation-rank topology tested by NEXT166--NEXT167, which
classified whole contact-graph components as 0D, 1D, 2D, or 3D. Here the
local signature counts cell faces by their number of polygon edges and asks
whether repeated atoms of the same element share a small, coherent set of
signatures. This is motivated by Lazar, Han, and Srolovitz, *PNAS* **112**,
E5769--E5776 (2015), <https://doi.org/10.1073/pnas.1505788112>, which develops
Voronoi-cell topology as a local structural classifier for ordered and
disordered condensed matter.

Three alternatives were considered before freezing. Radial shell-gap order
was rejected because P6/P6c already tests coordination-shell gaps.
Centrosymmetry was rejected because a globally low score would systematically
penalize valid polar and tetrahedral environments. A new sweep over spherical-
harmonic orders was rejected as too close to retuning the failed MVBO/CMVBO/
TVBO family. Species-conditioned Voronoi face topology adds discrete local
packing information absent from those branches.

## Non-negotiable information boundary

Every executable feature and formula may use only composition and the initial,
unrelaxed periodic geometry. Reject any ASE `Atoms` object with a calculator,
nonempty `info`, or arrays other than `numbers` and `positions`. Do not read or
compute a DFT energy, force, stress, hull value, band property, learned energy/
force/stress proxy, model or proxy potential, relaxation step, relaxed
structure, or trajectory.

Discovery outcomes are offline labels only. NEXT251 accepts no label, endpoint,
validation, or replication path. NEXT252--NEXT254 may read only the frozen
SCIGEN and WyFormer discovery endpoints already used by NEXT248. Internal
validation and replication remain physically unopened unless a new candidate
passes every frozen discovery gate. No canonical paper, note,
preregistration, README, or previous script/result may be modified.

## NEXT251 fixed local topology

Use exactly `VoronoiNN(weight="solid_angle", tol=0, cutoff=13)` on the raw
periodic structure. For every site, obtain its Voronoi polyhedron and require
every retained face to have a finite positive area and integer `n_verts >= 3`.
Deduplicate a repeated `(site_index, rounded_integer_image)` face by retaining
the record with the largest area, with lexical tie-breaking. Sort the retained
records lexically before aggregation.

Compute two preregistered modes:

- `raw`: retain every positive finite face;
- `robust`: retain only faces whose area divided by the site's total raw face
  area is at least `1/64`; renormalize retained areas after filtering.

The robust threshold is fixed before outcomes and may not be swept. A site
with no retained face or a nonfinite normalization fails closed. Map each
retained face degree to one of seven bins `(3, 4, 5, 6, 7, 8, 9+)`. Its
integer signature is the ordered tuple of face counts in those bins. From the
area weights define

```text
odd_area_fraction(i) = sum area_fraction(face)
                       for face degrees 3, 5, 7, or 9+

degree_entropy(i) = -sum_b p_b log(p_b) / log(7),
```

where `p_b` is the retained face-area fraction in degree bin `b` and `0 log 0`
is zero. Both quantities must lie in `[0, 1]` up to `1e-12` final-roundoff
tolerance.

For each mode, group site signatures by atomic number. Define four whole-
structure consistency quantities for `N` sites:

```text
species_modal_agreement = sum_s max_signature count(s, signature) / N

species_signature_entropy = sum_s (N_s / N) * H_s,
H_s = 0                                      if N_s <= 1
H_s = -sum_t p_st log(p_st) / log(N_s)       otherwise

species_singleton_fraction = (1 / N) * sum_s count of sites whose signature
                             occurs once within species s, only for N_s >= 2

species_excess_signature_density =
    (1 / N) * sum_s max(number_unique_signatures_s - 1, 0).
```

Use population statistics and NumPy linear quantiles across sites. The exact
ordered feature universe is:

```text
svtc_raw_odd_area_mean
svtc_raw_odd_area_q90
svtc_raw_degree_entropy_mean
svtc_raw_degree_entropy_q90
svtc_raw_species_modal_agreement
svtc_raw_species_signature_entropy
svtc_raw_species_singleton_fraction
svtc_raw_species_excess_signature_density
svtc_robust_odd_area_mean
svtc_robust_odd_area_q90
svtc_robust_degree_entropy_mean
svtc_robust_degree_entropy_q90
svtc_robust_species_modal_agreement
svtc_robust_species_signature_entropy
svtc_robust_species_singleton_fraction
svtc_robust_species_excess_signature_density
```

All sixteen values must be finite for a supported record. NEXT251 must cover
exactly the same `13,470` SCIGEN and `5,232` WyFormer discovery identities as
NEXT247. Any construction failure is explicit; formal publication requires
100% support in both sources.

## NEXT251 engineering tests and TDD order

Create `tests/test_next251_species_voronoi_topology_consistency.py` before
`src/next251_species_voronoi_topology_consistency.py` and observe the missing-
module failure. Tests must then cover:

1. exact face-degree binning, area fractions, and `1/64` robust filtering;
2. exact species-modal agreement, normalized species entropy, repeated-species
   singleton fraction, and excess-signature density on hand-built signatures;
3. known simple-cubic/NaCl and FCC Voronoi face signatures;
4. rigid rotation, uniform scale, neighbor-order, and supercell invariance;
5. exact material identity and fail-closed geometry-only input checks;
6. range and finite-value guards;
7. absence of label, endpoint, validation, and replication arguments.

Run the focused test file to GREEN before the formal label-free build. Publish
the source/test hashes and all formal output hashes.

## NEXT252 fixed feature audit

Create the focused NEXT252 test first and verify RED. Reconstruct the exact
published NEXT224 frontier and rejected-extreme cohort through the unchanged
NEXT227 machinery. Use the identical reduced-formula folds, cohort counts,
class counts, coverage requirements, and AUC gates used by NEXT248.
Normalizations use only the finite combined discovery population, with
inverted-CDF `1/16` and `15/16` cutoffs; outcomes do not enter normalization.

Audit exactly these sixteen hypotheses. Opposite directions and post-outcome
direction changes are forbidden:

```text
svtc_raw_odd_area_mean__protected_low
svtc_raw_odd_area_q90__protected_low
svtc_raw_degree_entropy_mean__protected_low
svtc_raw_degree_entropy_q90__protected_low
svtc_raw_species_modal_agreement__protected_high
svtc_raw_species_signature_entropy__protected_low
svtc_raw_species_singleton_fraction__protected_low
svtc_raw_species_excess_signature_density__protected_low
svtc_robust_odd_area_mean__protected_low
svtc_robust_odd_area_q90__protected_low
svtc_robust_degree_entropy_mean__protected_low
svtc_robust_degree_entropy_q90__protected_low
svtc_robust_species_modal_agreement__protected_high
svtc_robust_species_signature_entropy__protected_low
svtc_robust_species_singleton_fraction__protected_low
svtc_robust_species_excess_signature_density__protected_low
```

A hypothesis is eligible only if both sources pass every frozen raw-feature
gate. Reporting rank is largest minimum worst-fold AUC, largest minimum
aggregate AUC, largest mean aggregate AUC, then lexical hypothesis. If none is
eligible, the branch closes and NEXT253 is forbidden.

## NEXT253 conditional one-term search

NEXT253 is authorized only if NEXT252 publishes at least one eligible
hypothesis. Start from the exact NEXT224 score. For every eligible hypothesis,
use its frozen NEXT252 `q_lo/q_hi` protection and exactly

```text
h = local_width_fraction * NEXT214_REPAIR_WIDTH
local_weight = max(0, 1 - abs(base_score - NEXT224_THRESHOLD) / h)
delta = amplitude_fraction * h * local_weight * (1 - 2 * protection)
score = max(0, base_score + delta)
```

Widths are exactly `{1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1}` and amplitudes are
exactly `{1/4, 1/2, 1}`. Publish one exact NEXT224 no-op plus
`21 * eligible_hypothesis_count` new candidates. Reporting selection is
restricted to `eligible_new_candidate == true`.

Freeze is authorized only for a new candidate passing source AUC, SAFE, BROAD,
and every discovery gate. If none passes all gates but at least one new
candidate passes AUC+SAFE and fails BROAD, NEXT254 is authorized for exactly
that sorted identity population. Otherwise the branch closes.

## NEXT254 conditional BROAD diagnostic

NEXT254 must reproduce every authorized NEXT253 record at evaluator level and
recompute unchanged BROAD threshold tables only for the authorized AUC+SAFE/
non-BROAD population. Rank by fewest failed constraints, smallest normalized
shortfall sum, then lexical candidate key. Compare the closest record with the
frozen NEXT235 reference `(5, 0.12339543654931197)`. Strict improvement
requires a lexicographically smaller tuple. Otherwise close the branch.

## Reporting, verification, and stopping rule

Append results only to the independent report
`reports/2026-08-08-next115-next117-hcid-no-dft-search.md`. Do not modify
canonical manuscript or reporting files. Verify focused tests, full pytest,
all manifest input/output/source hashes, boundary flags, `git diff --check`,
canonical zero-diff, and CodeGraph synchronization.

This discovery branch cannot claim a confirmed law. Even an all-discovery-
gate candidate must first receive a separately frozen unseen-source or still-
sealed internal-validation protocol. All scripts, tests, plans, formal output
directories, and report text are additive. The shared dirty research branch
must not be committed automatically.
