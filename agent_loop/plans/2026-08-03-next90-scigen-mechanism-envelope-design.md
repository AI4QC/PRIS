# NEXT90: SCIGEN coupled mechanism-envelope law

Date: 2026-08-03

Status: discovery-development design freeze written after the negative NEXT87
discovery result and before computing any NEXT90 composite score. Internal
validation and internal replication endpoints remain physically unopened.

## Motivation

NEXT87's analytic family had strong pooled and lattice discrimination, but no
exact three-term list won more than two of five composition-group searches.
The fold winners repeatedly exchanged highly correlated proxies of the same
physical mechanisms. The next hypothesis is that proxy identity is a nuisance
coordinate: a stable law should first collapse interchangeable measurements
into fixed physically named envelopes and then couple the envelopes.

This is a new additive law family, not a reinterpretation or rescue of NEXT87.
NEXT87 remains a failed frozen protocol.

## Executable boundary

The law accepts one raw unrelaxed generated `x0` structure and the frozen
NEXT85/NEXT86 analytic pipeline. It may not use any DFT value, relaxed
structure, trajectory, learned energy/force/stress proxy, ML interatomic
potential, physical relaxation, same-composition alternative, material ID,
formula, chemical system, lattice-class ID, or element-identity shortcut at
execution time. Any missing constituent produces `KEEP`.

## Fixed mechanism envelopes

For every NEXT86 term `t`, retain its already frozen transform, direction,
center, and scale and define `h_t(x) = max(0, z_t(x))`. No term parameter is
refit. Define four envelopes using a maximum so that correlated measurements
act as alternative witnesses rather than being counted repeatedly.

### B: local bond-valence disequilibrium

- `scbv_anion_mismatch_rms__high`
- `scbv_mismatch_q95__high`
- `scbv_mismatch_max__high`

`B(x)` is the maximum of these three hinges.

### V: valence-rigidity incompatibility

- `sivr_edge_mismatch_rms__high`
- `sivr_edge_mismatch_max__high`
- `sivr_stiffness_min__low`

`V(x)` is the maximum of these three hinges.

### E: analytic electrostatic residual

- `aefi_residual_rms__high`
- `aefi_residual_q95__high`
- `aefi_residual_max__high`

`E(x)` is the maximum of these three hinges.

### L: unresolved repulsive/self-stress load

- `sscp_load_rms__high`
- `sscp_load_q95__high`
- `sscp_load_fraction__low`
- `prlr_residual_fraction__high`
- `prlr_cell_residual_fraction__high`
- `prlr_risk__high`

`L(x)` is the maximum of these six hinges.

## Finite formula family

Every candidate contains all four envelopes:

`M(x) = B(x) + w_V V(x) + w_E E(x) + w_L L(x)`.

Each of `w_V`, `w_E`, and `w_L` is in `{0.25, 0.5, 1, 2, 4}`, producing
exactly 125 candidates. Fixing the coefficient of `B` to one removes the
common-multiplier degeneracy. The rejection threshold is selected on discovery
only with the exact NEXT87 extreme-precision semantics and frozen Wilson gates.

Whole reduced-formula groups use the existing deterministic five-fold
assignment. For each held-out fold, select a weight vector and threshold using
the other four folds only. The exact weight vector must win at least four of
five foldwise searches. The final candidate must be supported in every fold,
and every held-out fold must retain raw severe-rejection precision >= 0.70 and
protected recall >= 0.93.

## Unchanged performance gates

- support coverage Wilson lower >= 0.90;
- protected recall Wilson lower >= 0.95;
- severe-rejection precision Wilson lower >= 0.80;
- total rejection/savings Wilson lower >= 0.02;
- pooled protected-versus-severe AUC >= 0.75;
- macro lattice AUC >= 0.65;
- worst eligible lattice AUC >= 0.55;
- at least eight lattice classes with protected and severe examples;
- all five held-out raw fold gates above;
- reject more severe discovery rows than the frozen Pauling P2--P5 union and
  have a higher severe-rejection precision Wilson lower bound.

The middle stratum is excluded from precision but included in total savings.
`ABSTAIN` is not a Pauling rejection.

## Lockbox transition

If and only if every discovery gate passes, publish the full formula and
freeze predictions for discovery, internal validation, and internal
replication using label-free feature files. The runner has no validation or
replication endpoint argument. Then open internal validation exactly once.
Failure at discovery publishes a negative result and no locked predictions.
Failure at validation leaves replication unopened. No parameter or subgroup
exception may change after a prediction freeze.

## Alternatives considered and rejected

1. Re-run NEXT87 with a weaker exact-term stability definition: rejected as a
   post-result gate change.
2. Fit PCA or a learned group projection: rejected because the executable law
   would be less transparent and would introduce unnecessary fitted degrees of
   freedom.
3. Use a max or mean over every eligible term in a broad physics group:
   rejected because physically backward-but-predeclared terms can dilute the
   mechanism. NEXT90 uses only the recurrent, explicitly named proxy sets above
   and freezes them before measuring the composite.

