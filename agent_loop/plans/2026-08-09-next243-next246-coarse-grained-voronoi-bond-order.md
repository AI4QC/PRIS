# Frozen NEXT243--NEXT246 coarse-grained Voronoi bond-order plan

Date frozen: 2026-08-09 (America/Chicago)

## Status and reason for a new mechanism

NEXT239--NEXT242 tested single-site, Voronoi-facet-area-weighted `q4/q6`
magnitudes and facet evenness. One of seven directed raw hypotheses transferred
across the two discovery sources, but none of its frozen one-term corrections
passed BROAD and the closest diagnostic record was worse than NEXT235.
Consequently, this plan must not retune the inspected NEXT239 feature
directions, quantiles, local widths, amplitudes, or threshold records.

The repository contains no implementation of complex `q_lm` vectors,
neighbor-to-neighbor bond-order correlations, solid-bond correlations, or
Lechner--Dellago neighbor averaging. Literal and CodeGraph audits found only
the scalar NEXT239 `q4/q6` implementation. This plan therefore tests a distinct
physical mechanism: spatial coherence of adjacent local angular environments,
not another scalar single-site bond-order summary.

The primary scientific basis is:

- Steinhardt, Nelson, and Ronchetti, *Phys. Rev. B* **28**, 784 (1983),
  <https://doi.org/10.1103/PhysRevB.28.784>;
- Lechner and Dellago, *J. Chem. Phys.* **129**, 114707 (2008),
  <https://doi.org/10.1063/1.2977970>;
- Mickel et al., *J. Chem. Phys.* **138**, 044501 (2013),
  <https://doi.org/10.1063/1.4774084>.

## Non-negotiable information boundary

Every executable quantity may use only composition and the initial,
unrelaxed periodic geometry. The implementation must reject any `Atoms` object
with a calculator, nonempty `info`, or arrays other than `numbers` and
`positions`. It must not read or compute a DFT energy, force, stress, hull
value, band property, learned energy/force/stress proxy, model or proxy
potential, relaxation step, relaxed structure, or trajectory.

Discovery outcomes are offline labels only. NEXT243 must not accept any label,
endpoint, validation, or replication path. NEXT244--NEXT246 may read only the
already frozen SCIGEN and WyFormer discovery endpoints. Internal validation and
replication remain physically unopened unless a frozen discovery candidate
passes every required gate. No canonical paper, note, preregistration, README,
or prior script/result may be changed.

## Fixed raw geometry construction

For each periodic site `i`, use exactly
`VoronoiNN(weight="solid_angle", tol=0, cutoff=13)` to obtain the same neighbor
support used by NEXT239. For each returned facet, read only `site_index`,
`image`, `poly_info["normal"]`, and `poly_info["area"]`. Deduplicate on
`(site_index, rounded_integer_image)` by retaining the largest positive finite
area. Sort those keys lexicographically. Normalize the retained facet areas at
each center so that `sum_j w_ij = 1`.

For `l` in `{4, 6}` and `m = -l, ..., l`, define

```text
q_lm(i) = sum_j w_ij * Y_lm(u_ij)
q_l(i)  = sqrt(4*pi/(2*l+1) * sum_m |q_lm(i)|^2)
```

where `u_ij` is the normalized facet normal. Use
`scipy.special.sph_harm_y(l, m, theta, phi)` with polar angle
`theta = arccos(clip(u_z, -1, 1))` and azimuth
`phi = atan2(u_y, u_x) mod 2*pi`. Only even `l` values are used, so a possible
global reversal of facet-normal sign leaves the construction unchanged.

After all site vectors are available, define the directed facet-weighted
neighbor average including the central site with exactly half of the weight:

```text
bar_q_lm(i) = 0.5 * (q_lm(i) + sum_j w_ij * q_lm(j))
bar_q_l(i)  = sqrt(4*pi/(2*l+1) * sum_m |bar_q_lm(i)|^2)
```

Periodic images reuse the `q_lm` vector of their underlying `site_index`;
translation does not rotate the local environment.

Two bounded coherence measures are fixed. Let `norm(v)` be the complex
Euclidean norm and let `eps = 1e-14`.

```text
coherence_l(i) = norm(bar_q_lm(i)) /
                 (0.5 * (norm(q_lm(i)) +
                         sum_j w_ij * norm(q_lm(j))))

neighbor_corr_l(i) = sum_j w_ij *
    Re(sum_m q_lm(i) * conj(q_lm(j))) /
    (norm(q_lm(i)) * norm(q_lm(j)))
```

If a coherence denominator is `<= eps`, set the coherence to `0`. If either
norm in one correlation denominator is `<= eps`, that directed correlation
contribution is `0`. Clip only final floating-point roundoff:
`coherence_l` to `[0, 1]` and each normalized pair correlation to `[-1, 1]`.
No structure-dependent threshold is permitted.

## NEXT243 fixed feature universe

Aggregate over sites with NumPy population statistics and linear quantiles.
The complete and ordered feature list is:

```text
cmvbo_bar_q4_mean
cmvbo_bar_q4_q10
cmvbo_bar_q4_std
cmvbo_bar_q6_mean
cmvbo_bar_q6_q10
cmvbo_bar_q6_std
cmvbo_coherence_q4_mean
cmvbo_coherence_q4_q10
cmvbo_coherence_q6_mean
cmvbo_coherence_q6_q10
cmvbo_neighbor_corr_q4_mean
cmvbo_neighbor_corr_q4_q10
cmvbo_neighbor_corr_q6_mean
cmvbo_neighbor_corr_q6_q10
cmvbo_neighbor_corr_joint_q10
```

The joint site value is exactly
`0.5 * (neighbor_corr_4(i) + neighbor_corr_6(i))`. All fifteen structure
features must be finite for a supported record. NEXT243 must cover exactly the
same `13,470` SCIGEN and `5,232` WyFormer discovery material identities as
NEXT239. Any construction failure is recorded explicitly; formal publication
requires 100% support in both sources.

Engineering tests fixed before the formal run:

- a synthetic spherical-harmonic kernel test against the scalar addition
  theorem used by NEXT239;
- coherence and correlation range/identity tests;
- rigid rotation, uniform scale, and supercell invariance tests;
- exact material-identity and fail-closed geometry-only interface tests;
- no label/endpoint/validation/replication argument in the NEXT243 interface.

## NEXT244 fixed feature audit

Reconstruct the exact published NEXT224 frontier and its rejected-extreme
cohort. Use the identical reduced-formula fold assignment, cohort counts,
minimum class counts, coverage rules, and AUC gates used by NEXT227 and
NEXT240. Normalizations use the finite combined discovery population only,
with inverted-CDF `1/16` and `15/16` cutoffs. Outcome labels must not enter
normalization.

Audit exactly these fifteen directed hypotheses, with no opposite-direction
search and no post-outcome direction change:

```text
cmvbo_bar_q4_mean__protected_high
cmvbo_bar_q4_q10__protected_high
cmvbo_bar_q4_std__protected_low
cmvbo_bar_q6_mean__protected_high
cmvbo_bar_q6_q10__protected_high
cmvbo_bar_q6_std__protected_low
cmvbo_coherence_q4_mean__protected_high
cmvbo_coherence_q4_q10__protected_high
cmvbo_coherence_q6_mean__protected_high
cmvbo_coherence_q6_q10__protected_high
cmvbo_neighbor_corr_q4_mean__protected_high
cmvbo_neighbor_corr_q4_q10__protected_high
cmvbo_neighbor_corr_q6_mean__protected_high
cmvbo_neighbor_corr_q6_q10__protected_high
cmvbo_neighbor_corr_joint_q10__protected_high
```

A hypothesis is eligible only if both sources pass every frozen raw feature
gate. Reporting rank is, in order: largest minimum worst-fold AUC, largest
minimum aggregate AUC, largest mean aggregate AUC, then lexical hypothesis.
If no hypothesis is eligible, the branch closes and NEXT245 is forbidden.

## NEXT245 conditional one-term search

NEXT245 is authorized only if NEXT244 publishes at least one eligible
hypothesis. It starts from the exact NEXT224 score, not NEXT235 or NEXT241.
For every eligible hypothesis, use its frozen NEXT244 `q_lo/q_hi` protection
and the exact triangular signed local update:

```text
h = local_width_fraction * NEXT214_REPAIR_WIDTH
local_weight = max(0, 1 - abs(base_score - NEXT224_THRESHOLD) / h)
delta = amplitude_fraction * h * local_weight * (1 - 2 * protection)
score = max(0, base_score + delta)
```

The only permitted width fractions are
`{1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1}` and the only amplitudes are
`{1/4, 1/2, 1}`. Support and missing-value behavior remain exactly those of
NEXT224/NEXT214. Publish one exact NEXT224 no-op reproduction control plus
`21 * eligible_hypothesis_count` new candidates. Reporting selection must be
restricted to `eligible_new_candidate == true`.

Freeze is authorized only for a new candidate passing source AUC, SAFE,
BROAD, and every discovery gate. If no candidate passes all gates but at least
one new candidate passes AUC+SAFE and fails BROAD, NEXT246 is authorized for
exactly that sorted identity population. Otherwise the branch closes.

## NEXT246 conditional BROAD diagnostic

NEXT246 must reproduce the exact published NEXT245 records byte-for-byte at
the evaluator level and recompute the unchanged BROAD threshold tables for
only the authorized AUC+SAFE/non-BROAD population. Rank residuals
lexicographically by:

1. fewest failed constraints;
2. smallest normalized shortfall sum;
3. lexical candidate key.

Compare the closest record only against the frozen NEXT235 reference:
`failed_constraint_count = 5` and
`normalized_shortfall_sum = 0.12339543654931197`. A strict diagnostic
improvement requires a lexicographically smaller tuple. If there is no strict
improvement, close the branch. If there is a strict improvement but still no
all-gate law, any continuation requires a new pre-outcome plan and may not
retune an inspected NEXT245 record.

## Reporting and stopping rule

All scripts, tests, plans, external formal outputs, and report additions are
additive. Append exact results, hashes, failures, and boundary receipts to the
independent report only after the branch reaches its frozen stopping point.
Do not modify canonical manuscript/reporting artifacts. A discovery success is
still not a confirmed law until an independently frozen unseen-source or
unopened internal validation protocol also succeeds.
