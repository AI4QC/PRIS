# next11 PHSC-v0 design

Status: **frozen before any next11 feature generation or label-byte opening**.

This note is additive.  It does not alter any prior script, artifact, report,
paper source, README, or preregistration.  Historical-test payloads, OMat24,
and the existing development-label bytes remain closed.

## Scientific question

Can a fixed-cell, as-provided-cell, Gamma-point atomic Hessian supply a robust
negative-mode signal that is orthogonal to the frozen M5/AGREE995 x0-energy
rules?

PHSC-v0 means **Projected Hessian Spectral Criterion v0**.  It is an MLIP
local-curvature diagnostic, not a certificate of DFT stability, a full phonon
calculation, or a complete crystal-stability criterion.

## Why this is new information

LRRC-v0 evaluates one Rayleigh quotient along the translation-projected x0
force direction.  It therefore misses a negative mode orthogonal to that
direction and returns a stationary fallback at an exact saddle.  PHSC-v0
examines every internal atomic direction and is able to expose both cases.

The earlier fixed-cell FIRE route predominantly follows directions excited by
the initial gradient and its frozen candidate uses short-trajectory energy
gaps.  It is not a spectral minimum-curvature test.

## Frozen geometry and finite differences

For a strictly three-dimensionally periodic N-atom structure with N >= 2, a
finite nonsingular cell, and no MIC-coincident atom pair, first canonicalize
fractional coordinates into `[0,1)`.  Let x be its atom-major Cartesian
coordinate vector and F(x) the MatterSim 5M force vector.  Define

```text
d_star = median over atoms of the nearest positive finite MIC distance
h      = 2^-8 d_star
```

For every one of the 3N Cartesian coordinates j, evaluate paired forces at
`x +/- h e_j` and `x +/- (h/2) e_j`, wrapping each perturbed geometry back into
the same cell before prediction.  Construct float64 matrices

```text
H_h[:, j]  = -(F(x + h e_j)   - F(x - h e_j))   / (2h)
H_h2[:, j] = -(F(x + h/2 e_j) - F(x - h/2 e_j)) / h
```

The force cost is exactly `12N` structure evaluations.  Cartesian probes are
used rather than finite differences directly in a chosen internal basis so
that the truncation pattern is invariant to the arbitrary Helmert-basis
ordering.

## Translation projection and spectrum

Symmetrize first:

```text
S_h  = (H_h  + H_h.T)  / 2
S_h2 = (H_h2 + H_h2.T) / 2
```

Construct a deterministic normalized Helmert contrast matrix `C_N` and
`Q = kron(C_N, I_3)`.  The columns of Q are an orthonormal basis for the
`3N-3` dimensional subspace orthogonal to uniform Cartesian translations.
Do not obtain the internal spectrum by deleting three near-zero eigenvalues
from the full matrix.

```text
A_h  = Q.T S_h  Q
A_h2 = Q.T S_h2 Q
A_R  = (4 A_h2 - A_h) / 3
e_num = norm((A_h2 - A_h) / 3, 2)
lambda_h  = min eigval(A_h)
lambda_h2 = min eigval(A_h2)
lambda_R  = min eigval(A_R)
U_num = lambda_R + e_num
L_num = lambda_R - e_num
```

`e_num`, `U_num`, and `L_num` are deterministic two-scale numerical-
consistency proxies.  They are not confidence bounds and are not rigorous
truncation-error bounds.

The raw antisymmetric and acoustic-translation diagnostics are frozen as

```text
B_delta = (H_delta - H_delta.T) / 2
T = kron(ones(N)/sqrt(N), I_3)
skew_delta = norm(B_delta, 2)
translation_delta = norm(S_delta T, 2)
```

They do not enter the v0 decision because neither is a derived bound on the
internal symmetric eigenvalue error; the two-scale projected operator
difference is the sole frozen decision proxy.

## Non-tunable algorithmic tolerance

For internal dimension `d = 3N-3`, use

```text
tau_alg = 64 d eps64 max(1 eV/A^2,
                         norm(A_h,2), norm(A_h2,2), norm(A_R,2))
```

This protects strict sign comparisons at float64 eigensolver roundoff.  It is
not a physical curvature threshold and must not be scanned, fitted, or changed
after observing any cohort result.

## Frozen outcome states

```text
resolved_negative:
    lambda_h < -tau_alg
    and lambda_h2 < -tau_alg
    and U_num < -tau_alg

resolved_nonnegative:
    lambda_h > tau_alg
    and lambda_h2 > tau_alg
    and L_num > tau_alg

near_zero_or_inconsistent:
    every other finite, successfully evaluated case
```

Only `resolved_negative` may add a rejection.  Both other successful states
retain the baseline decision.  Unsupported geometry found before inference
produces a zero-call row ABSTAIN; a numerical failure after all probes produces
a `12N`-call row ABSTAIN.  Because one production predictor call mixes several
structures, any predictor exception, missing/misaligned output, invalid force
shape/value, or incomplete coordinate set is fatal to the entire run: there is
no retry and no partial artifact is published.  Failures are never converted
into evidence of instability.

The unweighted Hessian in eV/A^2 is the primary object because the question is
the sign of the static potential-energy curvature.  A positive-mass dynamical
matrix is a congruence transform and preserves inertia by Sylvester's law, but
mass-weighted frequency diagnostics would require the corresponding
sqrt(mass)-weighted translation basis.  They are outside v0.

## Scope boundaries

PHSC-v0 includes only atomic displacements in the supplied fixed periodic
cell.  It does not include:

- cell strain, elastic stiffness, or atom-strain coupling;
- finite-q modes or a Brillouin-zone phonon dispersion;
- primitive/supercell equivalence claims;
- finite-temperature, anharmonic, magnetic, electronic, or decomposition
  stability;
- a guarantee that a locally negative x0 mode relaxes to a worthless DFT
  endpoint.

Global rotations are not projected because the periodic cell is fixed.

## Label-free stopping rule

The sealed development gate contains 2,171 rows.  Matching the earlier
three-percentage-point necessary savings increment requires at least

```text
ceil(0.03 * 2171) = 66
```

net deterministic rejections in `M5_PHSC_OR` relative to the frozen
primary-track M5 decision:

```text
net_reject_delta
  = count(composed == REJECT) - count(baseline == REJECT)
  = nonreject_to_reject - reject_to_nonreject
```

The complete baseline-to-composed KEEP/REJECT/ABSTAIN 3-by-3 transition matrix
must be published, so any PHSC failure that changes an existing M5 REJECT into
ABSTAIN is deducted rather than hidden.  Comparator-track or AGREE995 counts are
descriptive only and cannot replace the primary-track M5 stop.

After the formal feature artifact passes provenance and runtime verification,
reconstruct M5/AGREE995 decisions only from their frozen label-free features,
cutoffs, and thresholds.  Do not read endpoint labels.  If `M5_PHSC_OR` adds
fewer than 66 rejects over M5, stop this branch before any label opening.

Even if the count reaches 66, the exposed ELEMENTA development labels must not
be reopened for tuning or a scientific-improvement claim.  Outcome validation
requires a newly frozen, physically isolated x0-to-DFT cohort that retains all
success, failure, timeout, and nonconvergence outcomes.

## Acceptance checks before the formal GPU run

- exact quadratic positive, negative, and zero/near-zero spectra;
- an exact stationary saddle that LRRC cannot see;
- a negative mode orthogonal to a positive force direction;
- explicit translation-mode removal with a deterministic Helmert basis;
- atom-order covariance of the final spectrum and outcome;
- two-scale inconsistency routed to `near_zero_or_inconsistent`;
- malformed geometry and oracle outputs routed to explicit abstention;
- exact `12N` force-call count and exact batch alignment;
- synthetic-only manifest marked engineering evidence and scientific claim
  false;
- production checkpoint identity, source closure, input rehash, CUDA device,
  evaluation count, and atomic no-overwrite publication verified.
