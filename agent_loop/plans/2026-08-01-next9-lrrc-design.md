# Next9: LRRC-v0 and group-level quota design

## 1. Objective and evidence boundary

This round starts from next8's post-hoc results, but does not go back to the already-inspected
development labels to retune. next8's `AGREE995` carries signal over M5 at formula selection, yet
on the independent development gate only about `+0.18` percentage points remain, the confidence
interval crosses zero, and the valuable-recall safety gate fails. The main problem is not how the
threshold is written but that M1 and 5M both belong to the MatterSim family, so their
disagreement does not supply enough orthogonal information.

next9 uses two strictly separated components:

1. `Quota-CRC` is a group-level risk policy: it only redistributes the rejection budget, and is
   not called a new physical law;
2. `LRRC-v0` (local restoring-response criterion) adds the second-order response of the local
   potential-energy surface, and is the only candidate that could add new physical information.

This round permits synthetic mathematical, numerical and interface verification only. Until there
is a new complete `x0 -> DFT endpoint` cohort, no result may be written as a scientific
performance improvement, as surpassing Pauling's rules, or as approaching or exceeding DFT.

## 2. Why not simply keep enlarging the same-family committee

next8 already showed the M1/5M gap to be highly correlated across the development stages; the
main effect of an extreme disagreement gate is to increase ABSTAIN rather than to raise
high-energy candidate identification consistently. Adding further checkpoints from the same
family, sweeping disagreement quantiles or adjusting thresholds would all enlarge researcher
degrees of freedom over the same development evidence, without addressing the shared training
bias.

A genuinely heterogeneous committee needs at least three models differing substantially in
training data or architecture, with a conformal lower bound constructed on a wholly new DFT
calibration set. This round keeps only the future interface for that direction, and does not pass
M1/5M off as heterogeneous models.

## 3. Quota-CRC: a safety policy, not a new law

For a supported composition group `G` whose scores are finite, let:

\[
k_G=\lceil\sqrt{|G|}\rceil,\qquad q_G=s_{(k_G)}.
\]

where `s_i` is the pre-frozen M5 or committee risk score and `s_(k_G)` is the `k_G`-th smallest
score. A candidate may be REJECTed only when:

\[
s_i>\tau \quad\text{and}\quad s_i>q_G
\]

Ties at the quota boundary are all KEEP; unsupported rows and rows with non-finite scores stay
ABSTAIN.

At the same threshold,

\[
R_{quota}(\tau)\subseteq R_{base}(\tau),
\]

so it cannot create additional savings. Its only legitimate use is to permit a more aggressive
threshold on a future new calibration while preventing a small group from being over-rejected.
Future empirical work must report both the fixed-threshold and refit-threshold ablations; if the
gain comes only from the latter, it may be called a policy gain and nothing more.

## 4. LRRC-v0: the local restoring response

### 4.1 Direction

Remove the overall translation from the M5 atomic forces of a fixed-cell structure:

\[
f'_i=f_i-\frac{1}{N}\sum_j f_j,
\qquad
u_i=\frac{f'_i}{\sqrt{N^{-1}\sum_j\|f'_j\|^2}}.
\]

so that `mean(u)=0` and `mean(||u_i||^2)=1`. If the projected RMS falls below the frozen purely
numerical floor of `1e-12 eV/angstrom`, LRRC constructs no direction, marks
`STATIONARY_FALLBACK` and falls back to the base rule; this also makes the known blind spot at
exact stationary saddles explicit.

### 4.2 A label-free step size

Let `d_star` be the median per-atom nearest-neighbour distance computed with the minimum-image
convention, and fix:

\[
h=2^{-8}d_\star.
\]

`2^-8` is a frozen numerical discretisation choice and is not swept over any old label or real
checkpoint. `N<2`, no finite positive nearest-neighbour distance, or an invalid periodic cell are
all unsupported.

### 4.3 Two-scale directional curvature

Take a central difference at both `h` and `h/2`:

\[
\kappa_h=-\frac1N\sum_i u_i\cdot
\frac{F_i(x+h u)-F_i(x-h u)}{2h}.
\]

then define:

\[
\kappa_R=\frac{4\kappa_{h/2}-\kappa_h}{3},
\qquad
e_{num}=\frac{|\kappa_{h/2}-\kappa_h|}{3},
\qquad
U_{num}=\kappa_R+e_{num}.
\]

`U_num` is only a deterministic two-scale conservative proxy; no statistical confidence bound or
rigorous remainder bound is claimed. An LRRC signal requires `kappa_h < 0`, `kappa_h2 < 0` and
`U_num < 0` -- the two scales agreeing in sign and the Richardson proxy still negative.

### 4.4 Combining the decisions

The future candidate decision is:

\[
REJECT \iff g_{5M}>\tau_E \quad\lor\quad LRRC\_negative.
\]

When LRRC succeeds and is non-negative, the base decision stands; when the LRRC force oracle
fails, produces a non-finite value, or the geometry is unsupported, the result is ABSTAIN.
`STATIONARY_FALLBACK` is a known, diagnosable state with no added signal, and keeps the base
decision. Quota-CRC must be applied as the last layer and may turn a REJECT within the quota back
into a KEEP; ABSTAIN is never rewritten.

## 5. The synthetic verification matrix

The following must be verified:

- a positive-definite quadratic potential gives positive curvature and adds no rejection;
- an inverted quadratic potential gives negative curvature consistently at both scales away from
  the origin;
- translation, rigid rotation, atom permutation and periodic wrapping leave the result unchanged;
- on a quadratic potential, `kappa_h` and `kappa_h2` converge to the analytic value;
- when the force oracle raises, returns a wrong shape, or is non-finite, it fails open to ABSTAIN;
- the LRRC OR combination adds genuine REJECTs rather than manufacturing nominal savings by
  increasing ABSTAIN;
- an exact `F=0` saddle falls back explicitly to the base rule;
- Quota-CRC keeps `ceil(sqrt(n))` fixed, keeps boundary ties as KEEP, and leaves ABSTAIN
  unchanged;
- the manifest stores the formula, the constants, the hash of the executed source and the
  synthetic case results, and contains no real label or protected identifier.

## 6. The data route

A local audit found no cohort satisfying all of "labels unopened, physically isolated, complete
candidate groups, x0/DFT endpoint paired". The old extraction of ELEMENTA Spin additionally
merged `(material, structure, spin)` wrongly by material, leaving only 78,027/207,185
trajectories, and cannot be used.

The preferred external option is the `m3gnet/rng` control cohort in the 2025 Alexandria
expansion. The official paper reports 29,671 successful DFT endpoints, later in time than
MatterSim 2024 and drawn at random rather than filtered by a stability model. It still has two
boundaries: the inputs are M3GNet pre-relaxed structures rather than the rawest generated state;
and MatterSim's training details are not public, so any training overlap can only be recorded as
unknown. The roughly 100 GB of full relaxation paths is neither downloaded nor parsed until the
source tags, candidate-group completeness and shard mapping are confirmed.

References:

- [Alexandria 2025 expansion](https://arxiv.org/html/2512.09169v2)
- [Alexandria geometry-optimization paths](https://alexandria.icams.rub.de/data/geo_opt_paths/2025.07.02/pbe/)
- [Conformal Risk Control](https://research.google/pubs/conformal-risk-control/)

## 7. Success gates

Passing the synthetic tests means only that the engineering and algebraic contracts hold. Genuine
scientific success requires a pre-split new cohort to satisfy all of:

1. the increase in DFT savings over the frozen base rule meets the pre-registered gate, with a
   positive paired 95% CI lower bound;
2. valuable recall is non-inferior, and the one-sided 95% lower bound of exact/protected group
   retention is at least 0.95;
3. there is no all-reject group, and the increase in abstention is within the pre-registered
   limit;
4. the fixed-threshold, refit-threshold, LRRC-only and quota-only ablations are all reported;
5. the outer gate is opened once only, with no retuning after a failure.

Until those gates pass, no existing report, paper, README or pre-registration file is modified.
