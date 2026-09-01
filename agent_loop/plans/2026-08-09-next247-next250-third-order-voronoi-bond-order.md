# Frozen NEXT247--NEXT250 third-order Voronoi bond-order plan

Date frozen: 2026-08-09 (America/Chicago)

## Status and distinct physical mechanism

NEXT243--NEXT246 tested quadratic bond-order magnitudes, neighbor-averaged
quadratic magnitudes, norm coherence, and pairwise complex-vector
correlations. One of fifteen directed hypotheses transferred across both
discovery sources, but none of its frozen local corrections passed BROAD and
the closest residual was worse than NEXT235. Those inspected directions,
normalizations, widths, amplitudes, and candidate records must not be retuned.

The original Steinhardt construction includes third-order rotational
invariants as well as the quadratic `q_l` power spectrum. The third-order
invariant couples three angular modes with a Wigner 3-j tensor and therefore
contains angular information that cannot be reconstructed from a quadratic
magnitude or a pairwise normalized dot product. The repository contains no
scientific implementation of a Wigner-3j bond-order invariant; the unrelated
file name `next3_targeted_w4.py` denotes a workflow variant, not `W_4` bond
order. This plan tests the missing higher-order mechanism without reopening or
retuning the NEXT243--NEXT246 feature universe.

Primary scientific sources:

- Steinhardt, Nelson, and Ronchetti, *Phys. Rev. B* **28**, 784 (1983),
  <https://doi.org/10.1103/PhysRevB.28.784>;
- Lechner and Dellago, *J. Chem. Phys.* **129**, 114707 (2008),
  <https://doi.org/10.1063/1.2977970>;
- Mickel et al., *J. Chem. Phys.* **138**, 044501 (2013),
  <https://doi.org/10.1063/1.4774084>.

## Non-negotiable information boundary

Every executable quantity may use only composition and the initial,
unrelaxed periodic geometry. The implementation must reject any `Atoms`
object with a calculator, nonempty `info`, or arrays other than `numbers` and
`positions`. It must not read or compute a DFT energy, force, stress, hull
value, band property, learned energy/force/stress proxy, model or proxy
potential, relaxation step, relaxed structure, or trajectory.

Discovery outcomes are offline labels only. NEXT247 must not accept a label,
endpoint, validation, or replication path. NEXT248--NEXT250 may read only the
already frozen SCIGEN and WyFormer discovery endpoints. Internal validation
and replication remain physically unopened unless a frozen discovery
candidate passes every required gate. No canonical paper, note,
preregistration, README, or prior script/result may be changed.

## Fixed complex vectors and third-order invariant

Use exactly the NEXT243 periodic neighbor construction:
`VoronoiNN(weight="solid_angle", tol=0, cutoff=13)`, deduplication on
`(site_index, rounded_integer_image)` by the largest positive finite facet
area, lexicographic ordering, and per-center facet-area normalization. For
`l` in `{4, 6}`, compute the same complex `q_lm(i)` vectors with
`scipy.special.sph_harm_y(l, m, theta, phi)` and the same half-central
coarse-graining:

```text
q_lm(i) = sum_j w_ij Y_lm(u_ij)
bar_q_lm(i) = 0.5 * (q_lm(i) + sum_j w_ij q_lm(j))
```

For any complex vector `v_m`, define the normalized third-order invariant

```text
W_l(v) = sum_{m1+m2+m3=0}
         wigner3j(l,l,l; m1,m2,m3) * v_m1 * v_m2 * v_m3

hat_W_l(v) = Re(W_l(v)) / (sum_m |v_m|^2)^(3/2)
```

If the denominator is `<= 1e-14`, set `hat_W_l` to `0`. The imaginary part
must be zero to a frozen absolute tolerance of `1e-12`; otherwise the record
fails closed. Guard the final normalized value in `[-1, 1]` with numerical
tolerance `1e-10` and clip only final roundoff.

Wigner 3-j coefficients must be computed without an outcome-dependent table.
The production implementation shall use a deterministic integer-factorial
Racah sum for the equal integer orders `l=4,6` and cache its nonzero
`(m1,m2,m3,coefficient)` tuples. Engineering tests must compare every cached
coefficient against `sympy.physics.wigner.wigner_3j` before the formal run;
SymPy is test-only and must not be imported by the executable feature module.

For each site and order define

```text
raw_hat_w_l(i) = hat_W_l(q_lm(i))
bar_hat_w_l(i) = hat_W_l(bar_q_lm(i))
delta_l(i) = abs(bar_hat_w_l(i) - raw_hat_w_l(i))
```

The absolute invariant magnitude measures third-order angular organization
without assuming that all valid crystal motifs share one sign. The delta
measures whether the third-order local symmetry is spatially consistent under
the already frozen neighbor averaging.

## NEXT247 fixed feature universe

Aggregate over sites with NumPy population statistics and linear quantiles.
The complete ordered feature list is:

```text
tvbo_w4_abs_mean
tvbo_w4_abs_q10
tvbo_w4_abs_std
tvbo_w6_abs_mean
tvbo_w6_abs_q10
tvbo_w6_abs_std
tvbo_bar_w4_abs_mean
tvbo_bar_w4_abs_q10
tvbo_bar_w4_abs_std
tvbo_bar_w6_abs_mean
tvbo_bar_w6_abs_q10
tvbo_bar_w6_abs_std
tvbo_w4_coarse_delta_mean
tvbo_w4_coarse_delta_q90
tvbo_w6_coarse_delta_mean
tvbo_w6_coarse_delta_q90
```

All sixteen values must be finite for a supported record. NEXT247 must cover
exactly the same `13,470` SCIGEN and `5,232` WyFormer discovery identities as
NEXT243. Any construction failure is recorded explicitly; formal publication
requires 100% support in both sources.

Engineering tests frozen before the formal run:

- exhaustive `l=4,6` production Wigner-3j coefficients against SymPy;
- normalized third-order invariant range, reality, scale invariance, and
  known identity tests;
- rigid rotation, uniform scale, neighbor-order, and supercell invariance;
- exact material-identity and fail-closed geometry-only interface tests;
- no label/endpoint/validation/replication argument in the NEXT247 interface.

## NEXT248 fixed feature audit

Reconstruct the exact published NEXT224 frontier and its rejected-extreme
cohort. Use the identical reduced-formula folds, cohort counts, class counts,
coverage requirements, and AUC gates used by NEXT227, NEXT240, and NEXT244.
Normalizations use the finite combined discovery population only, with
inverted-CDF `1/16` and `15/16` cutoffs. Outcomes must not enter
normalization.

Audit exactly the following sixteen directed hypotheses. Opposite directions
and post-outcome direction changes are forbidden:

```text
tvbo_w4_abs_mean__protected_high
tvbo_w4_abs_q10__protected_high
tvbo_w4_abs_std__protected_low
tvbo_w6_abs_mean__protected_high
tvbo_w6_abs_q10__protected_high
tvbo_w6_abs_std__protected_low
tvbo_bar_w4_abs_mean__protected_high
tvbo_bar_w4_abs_q10__protected_high
tvbo_bar_w4_abs_std__protected_low
tvbo_bar_w6_abs_mean__protected_high
tvbo_bar_w6_abs_q10__protected_high
tvbo_bar_w6_abs_std__protected_low
tvbo_w4_coarse_delta_mean__protected_low
tvbo_w4_coarse_delta_q90__protected_low
tvbo_w6_coarse_delta_mean__protected_low
tvbo_w6_coarse_delta_q90__protected_low
```

A hypothesis is eligible only if both sources pass every frozen raw feature
gate. Reporting rank is, in order: largest minimum worst-fold AUC, largest
minimum aggregate AUC, largest mean aggregate AUC, then lexical hypothesis.
If no hypothesis is eligible, the branch closes and NEXT249 is forbidden.

## NEXT249 conditional one-term search

NEXT249 is authorized only if NEXT248 publishes at least one eligible
hypothesis. Start from the exact NEXT224 score. For every eligible hypothesis,
use its frozen NEXT248 `q_lo/q_hi` protection and exactly

```text
h = local_width_fraction * NEXT214_REPAIR_WIDTH
local_weight = max(0, 1 - abs(base_score - NEXT224_THRESHOLD) / h)
delta = amplitude_fraction * h * local_weight * (1 - 2 * protection)
score = max(0, base_score + delta)
```

Widths are exactly `{1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1}` and amplitudes
exactly `{1/4, 1/2, 1}`. Support and missing behavior remain those of
NEXT224/NEXT214. Publish one exact NEXT224 no-op plus
`21 * eligible_hypothesis_count` new candidates. Reporting selection is
restricted to `eligible_new_candidate == true`.

Freeze is authorized only for a new candidate passing source AUC, SAFE,
BROAD, and every discovery gate. If none passes all gates but at least one new
candidate passes AUC+SAFE and fails BROAD, NEXT250 is authorized for exactly
that sorted identity population. Otherwise the branch closes.

## NEXT250 conditional BROAD diagnostic

NEXT250 must reproduce the exact published NEXT249 records at evaluator level
and recompute unchanged BROAD threshold tables only for the authorized
AUC+SAFE/non-BROAD population. Rank residuals by fewest failed constraints,
then smallest normalized shortfall sum, then lexical candidate key. Compare
the closest record against the frozen NEXT235 reference:
`failed_constraint_count = 5` and
`normalized_shortfall_sum = 0.12339543654931197`.

A strict diagnostic improvement requires a lexicographically smaller tuple.
If there is no strict improvement, close the branch. If there is a strict
improvement but still no all-gate law, continuation requires a new
pre-outcome plan and may not retune an inspected NEXT249 record.

## Reporting and stopping rule

All scripts, tests, plans, external formal outputs, and independent-report
additions are additive. Do not modify canonical manuscript/reporting
artifacts. A discovery success is not a confirmed law until an independently
frozen unseen-source or unopened internal-validation protocol also succeeds.
