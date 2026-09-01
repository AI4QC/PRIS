# NEXT239--NEXT242 Morphometric Voronoi Bond-Order Plan

**Goal:** Test whether a continuous, facet-weighted description of local bond
orientation supplies a transferable pre-DFT protection certificate that is
missing from the current distance-, valence-, topology-, and second-moment
feature families.

Date: 2026-08-09

Status: frozen after NEXT238 branch closure and before computing any NEXT239
feature value, NEXT240 AUC, cutoff, or NEXT241 candidate score.

## Why this is a distinct mechanism

The existing catalogue includes CrystalNN coordination fingerprints, local
directional second moments, convex directional closure, periodic graph rank,
and motif dispersion. It does not contain the Voronoi-facet-area-weighted
bond-orientational invariants introduced to make local order metrics robust to
neighbor-selection ambiguity. Conventional Steinhardt invariants are standard
rotational descriptors (Phys. Rev. B 28, 784, DOI 10.1103/PhysRevB.28.784),
while Mickel et al. identified discontinuous nearest-neighbor selection as a
major failure mode and proposed morphometric Voronoi-weighted invariants
(J. Chem. Phys. 138, 044501, DOI 10.1063/1.4774084).

This branch uses that morphometric idea only as analytic geometry. It contains
no energy expression, force, stress, learned model, or potential.

## Immutable no-DFT and data boundary

- Executable inputs are element identities, cell, and Cartesian coordinates of
  one initial, unrelaxed periodic structure.
- No DFT calculation or value; learned energy, force, or stress proxy; model or
  proxy potential; relaxed structure; trajectory; or physical relaxation may
  enter feature construction or the executable law.
- NEXT239 has no endpoint path and may read only physically isolated SCIGEN and
  blind-WyFormer discovery geometry plus their frozen metadata/manifests.
- Discovery outcomes are offline labels used only after NEXT239 is atomically
  published.
- Validation and replication geometry and endpoints remain physically sealed
  unless a frozen candidate passes every discovery gate.
- All files and results are additive. Do not edit `paper/`, `tex/`, `notes/`,
  `README.md`, or `PREREG.md`.

## NEXT239 frozen morphometric feature definition

For every crystallographic site, use exactly

```text
VoronoiNN(weight="solid_angle", tol=0, cutoff=13)
```

on the undecorated initial structure. Retain every finite Voronoi face returned
by `get_nn_info`, including periodic images and same-element neighbors. The
face normal must be a finite unit vector and the face area must be strictly
positive. Duplicate `(site_index, image)` entries retain the largest area.
Any invalid or empty site makes only this feature family unsupported.

For site `i`, normalize face areas as

```text
w_ij = A_ij / sum_k A_ik.
```

For `l` in `{4, 6}`, compute the rotationally invariant morphometric bond order
without complex spherical-harmonic implementation dependence:

```text
q_l(i)^2 = sum_j sum_k w_ij w_ik P_l(u_ij dot u_ik)
q_l(i)   = sqrt(max(0, q_l(i)^2))
```

where `P_l` is the exact Legendre polynomial evaluated in float64 after clipping
the dot product to `[-1,1]`. Values outside `[0,1]` beyond `1e-10` fail closed;
roundoff inside that guard is clipped.

Also define facet-area evenness

```text
E_i = 1 / (n_i * sum_j w_ij^2),
```

which lies in `(0,1]` and equals one for equal-area faces. For each element,
form the centroid of its site vectors `(q_4,q_6)` and the Euclidean residual of
each same-element site from that centroid.

Publish exactly eleven structure features:

```text
mvbo_q4_mean
mvbo_q4_std
mvbo_q6_mean
mvbo_q6_std
mvbo_facet_evenness_min
mvbo_facet_evenness_q10
mvbo_facet_evenness_mean
mvbo_facet_evenness_std
mvbo_same_element_q46_dispersion_rms
mvbo_same_element_q46_dispersion_q95
mvbo_same_element_q46_dispersion_max
```

The 0.10 and 0.95 quantiles use NumPy `inverted_cdf`. Site order, rigid
rotation, translation, lattice representation, and uniform common scaling must
not change the features within frozen numerical tolerances. A primitive cell
and exact integer supercell must agree.

NEXT239 reports all eleven quantities, but the four raw `q_l` mean/std fields
are diagnostics only because no universal monotone stability direction exists
across coordination archetypes.

## NEXT240 frozen audit

Audit exactly seven prospectively directed hypotheses:

```text
protected_high:
  mvbo_facet_evenness_min
  mvbo_facet_evenness_q10
  mvbo_facet_evenness_mean

protected_low:
  mvbo_facet_evenness_std
  mvbo_same_element_q46_dispersion_rms
  mvbo_same_element_q46_dispersion_q95
  mvbo_same_element_q46_dispersion_max
```

Reconstruct the exact NEXT224 exploratory frontier. Audit only its frozen
rejected extreme cohort:

```text
supported
AND finite NEXT224 score
AND score >= 0.1520033762332462
AND (endpoint <= 1 OR endpoint >= 2)
```

Reuse the exact source/fold gates: minimum support `0.90`, minimum protected and
severe count `20`, aggregate AUC `0.55`, macro-fold AUC `0.53`, and worst-fold
AUC `0.50` in both sources. Feature normalization is endpoint-blind inverted-
CDF `1/16` and `15/16` over all finite combined discovery rows. Rank for
reporting only by minimum worst-fold AUC, minimum aggregate AUC, mean aggregate
AUC, then hypothesis identity. NEXT241 must use every eligible hypothesis.

If none is eligible, close this mechanism without running NEXT241.

## NEXT241 frozen fresh one-term grammar

NEXT241 starts from the exact NEXT224 score, not NEXT235 or NEXT238. For one
eligible bounded protection certificate `P`, exact NEXT224 score `s`, threshold
`t=0.1520033762332462`, and original NEXT214 repair width `W`, define

```text
h = f * W
local_weight = max(0, 1 - abs(s - t) / h)
local_delta = beta * h * local_weight * (1 - 2 * P)
score = max(0, s + local_delta)
```

The term is zero at and beyond distance `h`; a missing certificate turns only
the new term off; support remains exactly NEXT214 support.

Frozen grids:

- `f` in `{1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1}`;
- `beta` in `{1/4, 1/2, 1}`.

For `K` eligible NEXT240 hypotheses, the complete catalogue contains one exact
NEXT224 reproduction control and `21*K` eligible new candidates. Use the
unchanged evaluator and ranking. If any candidate passes every discovery gate,
stop before opening validation and freeze a separate validation protocol.
Otherwise report the best eligible AUC+SAFE candidate without using BROAD
residual for selection.

## NEXT242 residual and stopping rule

If NEXT241 has no all-gate candidate and has an eligible AUC+SAFE/non-BROAD
population, reproduce that exact population, verify its sorted-key digest and
evaluator records, and compute the unchanged BROAD residual. Rank by failed
constraint count, normalized shortfall, and candidate key.

Compare against the best overall exploratory NEXT235 tuple
`(5, 0.12339543654931197)`. A strict improvement is reporting evidence only and
requires a different pre-outcome continuation freeze. A non-improvement closes
the branch. No inspected feature, quantile, direction, grid value, source, fold,
or failure component may be retuned.

## Verification

Use TDD for the Legendre invariant, bounds, rotation/ordering/scaling and
supercell invariance, exact schema, discovery-only interfaces, candidate
completeness, base reproduction, and fail-closed provenance. Publish formal
artifacts atomically outside the repository; independently verify SHA-256
identities and false boundary flags; run focused and full pytest; check
`git diff --check`, forbidden canonical paths, and CodeGraph status; append
results only to the independent report.
