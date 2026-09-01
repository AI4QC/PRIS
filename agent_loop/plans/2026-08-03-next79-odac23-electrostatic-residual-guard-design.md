# NEXT79 single electrostatic-residual guard and stopping protocol

## Motivation

NEXT78 tested the 27 classical electrostatic descriptors in the same broad
finite catalogue as all earlier analytic families.  No candidate passed, but
`aefi_residual_q95` was the highest-ranked electrostatic single correction and
ranked third among all 211 usable guard features.  This adaptive final test asks
one narrower physical question: can the upper tail of the charge- and
length-normalized Ewald force residual remove false rejects from the already
fixed NEXT72 five-term formula?

## Frozen catalogue

Use the exact five terms of the published NEXT72 candidate as the anchor and
discard its threshold.  The only permitted additional feature is
`aefi_residual_q95`.  Evaluate the anchor alone and that one standardized term
with direction -1 or +1 and weight 0.125, 0.25, 0.5, 1, 2, 4, or 8.  Use the
unchanged 0.02--0.30 rejection-fraction thresholds, missing=KEEP rule, seven
scientific gates, strata, ranking, and deterministic tie break.  Do not add a
second guard and do not perform exact post-hoc threshold calibration.

Because this hypothesis was selected after NEXT78 discovery diagnostics, it is
not allowed to reach the final replication lockbox merely by passing the old
0.70 precision gate.  It must pass all seven original gates and achieve a
discovery reject-precision Wilson lower bound of at least 0.80.  Failure is the
precommitted stop for further additive discovery search in this ODAC23 cohort;
the result then becomes diagnostic evidence in a standalone report and the
replication labels remain unopened.

## Boundary

The executable formula uses only one raw unrelaxed x0, frozen elemental tables,
deterministic graph/geometry descriptors, and the analytic point-charge Ewald
residual.  No DFT calculation or value, relaxed geometry, opened validation
result, learned energy/force/stress proxy, physical relaxation, or
same-composition alternative is an input.  Search reads robust discovery labels
only.
