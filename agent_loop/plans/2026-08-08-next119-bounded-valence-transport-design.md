# NEXT119 bounded bond-valence transport budget-deficit design

Date: 2026-08-08

## Boundary and relationship to existing work

This additive branch follows the failed NEXT118 AMPP adaptive probe. It does
not replace NEXT38/BVTC, NEXT80/PRLR, NEXT115/HCID, any earlier result, or any
canonical document.

NEXT38 already projects the unique minimum-norm bond-valence edge correction
onto the column space of the analytic bond-valence Jacobian. That answers a
topological, unbounded linear question: which part of the correction is
infinitesimally compatible with some generalized atomic/cell motion?

NEXT119 asks the missing finite-budget question: how much correction remains
along a frozen, physically interpretable minimum-norm response path when its
largest dimensionless atomic/cell coordinate is limited to a small budget?
It consumes only one raw unrelaxed `x0`, analytic oxidation/bond-valence data,
and geometry. It performs no coordinate update or relaxation. DFT values,
DFT calculations, relaxed structures, trajectories, learned potentials, and
same-composition alternatives are forbidden.

## Mathematics

Let `c` be the unique site-conserving edge-valence correction from NEXT38 and
let `J` be its analytic edge-valence Jacobian with `3N` Cartesian atomic
columns and six symmetric cell-strain columns. With characteristic bond length
`L`, define the dimensionless response matrix

```text
A = J diag(L I_(3N), I_6).
```

The unbounded least-squares floor is

```text
r_inf = min_z ||A z - c||_2 / ||c||_2.
```

Let the required path budget be

```text
t_* = ||z_*||_inf.
```

For each frozen budget `tau` in `{0.01, 0.03, 0.10}`, truncate only along the
minimum-norm ray,

```text
alpha_tau = min(1, tau / t_*)
z_tau = alpha_tau z_*
r_tau = ||A z_tau - c||_2 / ||c||_2.
```

This is a deterministic closed-form path descriptor, not the global optimum of
a coordinate-wise box-constrained least-squares problem. The path debt beyond
the unbounded incompatibility floor is

```text
d_tau = sqrt(max(r_tau^2 - r_inf^2, 0)).
```

`d_tau` is zero when the minimum-norm path endpoint is already within budget.
It is positive when uniformly truncating that frozen response path leaves a
compatible component unresolved. No numerical optimization loop is executed.

The discovery metadata freeze imposes `N <= 64` sites. This covers all SCIGEN
discovery structures and
5,206/5,232 (99.50%) WyFormer discovery structures. Larger or invalid systems
fail open. The cap is an executable pre-screen latency boundary, not a
label-derived scientific threshold. The discarded iterative TRF prototype was
never published as a formal feature artifact.

The minimum-norm unbounded solution also yields dimensionless motion
diagnostics: generalized RMS, maximum per-site atomic displacement, and cell
strain Frobenius norm (with off-diagonal tensor components counted twice).

Zero correction maps exactly to all-zero features. All errors fail open through
a structured result; unsupported structures are not converted into risk.

## Frozen kernel feature schema

- `bvtbd_unbounded_residual_fraction`
- `bvtbd_required_linf_budget`
- `bvtbd_minimum_motion_rms`
- `bvtbd_atomic_motion_max`
- `bvtbd_cell_strain_frobenius`
- `bvtbd_residual_fraction_tau01`
- `bvtbd_residual_fraction_tau03`
- `bvtbd_residual_fraction_tau10`
- `bvtbd_deformation_debt_tau01`
- `bvtbd_deformation_debt_tau03`
- `bvtbd_deformation_debt_tau10`

## Verification before any dataset build

The pure kernel must pass exact scalar box oracles, an incompatible-subspace
oracle, zero-correction handling, row-permutation invariance, joint
length/Jacobian scale invariance, monotonic residuals across budgets, and
fail-open validation. Only then may an additive raw-x0 cross-source feature
builder be designed.
