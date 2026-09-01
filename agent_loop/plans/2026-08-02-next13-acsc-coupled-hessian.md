# NEXT13 ACSC-v0 Coupled Atomic–Cell Hessian Plan

## Scope and frozen boundary

This is an additive, label-free experiment. It does not modify or replace PHSC-v0,
CHSC-v0, M5, any previous output, the existing NEXT12 report, or canonical paper
material. No DFT endpoint, generator identity, or exposed evaluation label may be
used to choose a threshold.

ACSC-v0 tests a missing physical direction: a simultaneous internal atomic
displacement and homogeneous cell strain. It is intended to add evidence only
when the separate atomic and cell blocks are each resolved nonnegative but their
coupling produces a resolved negative mode.

## Frozen generalized coordinates

For a structure with `N >= 2`, median nearest-neighbour distance `d*`, Helmert
internal basis `Q`, reference fractional positions `f`, and the existing
Frobenius-orthonormal strain basis `B_a`, define

```text
r(z, eta) = f A exp(sum_a eta_a B_a)^T + d* Q z
e(z, eta) = E(r(z, eta), A(eta)) / N.
```

Both `z` and `eta` are dimensionless. The Hessian of `e` therefore has one unit,
eV/atom, and block form

```text
K = [[d*^2/N Q^T H_RR Q,  d*/N Q^T C],
     [d*/N C^T Q,          H_eta_eta/N]],
C = -dF/deta.
```

`H_RR` is reconstructed from the existing PHSC Cartesian probes.
`H_eta_eta/N` is reconstructed from the existing CHSC energy probes. `C` uses
the force outputs of the six axial CHSC directions, so it adds no structures to
a combined PHSC+CHSC evaluation.

## Two-scale numerical rule

- Atomic steps: `h_R = 2^-8 d*` and `h_R/2` (unchanged PHSC-v0).
- Strain steps: `h_eta = 2^-7` and `h_eta/2` (unchanged CHSC-v0).
- Form `K_h` and `K_h2`, then use `K_R = (4 K_h2 - K_h)/3`.
- Use `e_num = ||(K_h2-K_h)/3||_2`, `u_num=lambda_min(K_R)+e_num`,
  and the existing strict PHSC comparisons and floating-point tolerance.
- A coupling-only rejection requires atomic PHSC and cell CHSC to both be
  `resolved_nonnegative` and ACSC to be `resolved_negative`.
- Existing PHSC/CHSC resolved negatives remain separate rejection reasons. ACSC
  never downgrades them.
- Every unsupported geometry, malformed prediction, missing complete probe
  group, or numerical failure abstains; it is never converted into rejection.

The two-scale interval is a numerical-consistency proxy, not a statistical
confidence interval, rigorous truncation bound, or DFT stability certificate.

## Execution order

1. Contract-test scaling, translation removal, cross-force finite differences,
   two-scale classification, and a PSD-block/coupling-saddle counterexample.
2. Implement the numerical core without model or dataset imports.
3. Add a synthetic deterministic artifact and manifest.
4. Implement a MatterSim batch runner that preserves complete probe groups and
   reuses axial CHSC forces for the cross block.
5. Run a small engineering smoke, then the frozen old cohort label-free. Report
   only prespecified incremental counts among PHSC+CHSC resolved keeps.
6. If and only if the incremental signal is nonzero and numerically resolved,
   write a new standalone NEXT13 report. Do not edit earlier reports or papers.
7. Treat all MLIP outcomes as provisional until the physically isolated DFT
   endpoint evaluator is complete.
