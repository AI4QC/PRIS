# NEXT13b ACSC-DIRECT-v0 Plan

## Additive boundary

This validation does not edit any NEXT13/NEXT12 report, prior source file,
paper, README, or prior output. It selects only the 123 already sealed ACSC-v0
coupling-only negatives and opens no DFT endpoint or label artifact.

## Frozen direct path

For each candidate, reconstruct the same two-scale coupled matrices `K_h` and
`K_h2`, form `K_R=(4*K_h2-K_h)/3`, and take its normalized minimum eigenvector
`v=(z,eta)`. Canonicalize its sign by making the largest-absolute component
positive.

Using the exact ACSC generalized coordinates, evaluate

```text
A(t) = A0 exp(t * sum_a eta_a B_a)^T
r(t) = f0 A(t) + d_star Q (t z)
e(t) = E(r(t), A(t)) / N
```

at `t = 0, +/-2^-8, +/-2^-9`. The primary step is no larger than the frozen
PHSC atomic displacement norm and is half the frozen CHSC strain norm.

Compute

```text
q_h  = (E(+h)-2E(0)+E(-h))/(N*h^2)
q_h2 = (E(+h/2)-2E(0)+E(-h/2))/(N*(h/2)^2)
q_R  = (4*q_h2-q_h)/3
e_num = abs((q_h2-q_h)/3)
u_num = q_R+e_num
```

Direct confirmation requires strict negativity of `q_h`, `q_h2`, and `u_num`
under the same floating-point tolerance family. There is no fitted threshold.

## Interpretation

This is a distinct numerical route because it evaluates actual mixed displaced
structures using energies, instead of inferring the cross term from strained
forces. It still uses the same MatterSim checkpoint, so confirmation is not an
independent scientific model and cannot replace blinded DFT validation.

## Execution

1. TDD the minimum-mode sign convention, mixed geometry path, and scalar
   two-scale classification.
2. Reconstruct all 123 sealed candidates with the frozen combined probes.
3. Directly evaluate five mixed-mode structures per successfully reconstructed
   candidate.
4. Atomically publish a feature table and manifest with source/input hashes,
   complete telemetry, mode vectors, and confirmation counts.
5. Do not modify the existing NEXT13 report before user confirmation.
