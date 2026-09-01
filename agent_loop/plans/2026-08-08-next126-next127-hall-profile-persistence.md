# NEXT126--127 Hall-profile persistence plan

## Objective

Resolve the sole remaining NEXT125 discovery failure: no common threshold
strictly Pareto-dominates Pauling in all 12 source/fold cells, although 260
frozen laws already pass all six AUC gates and all 12 SAFE cells.

This branch remains additive. It must not edit any prior script, artifact,
report section, paper, README, preregistration, or canonical document.

## Scientific operand

NEXT124 already materialized the endpoint-free multiscale Hall deficit gains

`g05 <= g10 <= g25 <= g50`

for the fixed Voronoi contact thresholds 0.05, 0.10, 0.25, and 0.50. Define one
new operand on the expanded-catalogue negative Hall direction:

```text
HPP = 0                                               if g50 <= 1e-12
HPP = (0.05*g05 + 0.15*g10 + 0.25*g25)/(0.45*g50)   otherwise
```

HPP is the normalized left-step persistence of the tau50 Hall deficit over
the weak-contact interval [0.05, 0.50). Monotonicity gives `0 <= HPP <= 1`.
High HPP means that the eventual tau50 Hall obstruction already appears when
only weaker contacts are removed, rather than emerging only at the final
moderate-contact cutoff.

Only `mhpp_expanded_negative_weak_contact_persistence` is retained. The choice
is fixed from endpoint-free coverage and redundancy diagnostics: it has the
largest cross-source nonzero support among the four sign/mode profiles and is
not rank-equivalent to the corresponding tau50 gain. No endpoint, label,
validation geometry, DFT value, relaxation, MLIP, or learned energy/force/
stress proxy may be read while materializing NEXT126.

Unsupported expanded MHCR rows remain unsupported. In the executable optional
guard, unsupported or nonfinite HPP deactivates the guard and keeps the base
law unchanged.

## NEXT126 artifact

Create a cross-source discovery feature artifact by reading only the frozen
NEXT124 feature tables, catalogue, and manifest. It must:

- reproduce all material identifiers and row counts exactly;
- verify NEXT124 output hashes and its no-label/no-endpoint boundary;
- recompute HPP independently from the four frozen gains;
- record source support, nonzero counts, quantiles, and Spearman correlation to
  tau50 without opening any outcome;
- publish atomically with source and output SHA-256 identities.

## NEXT127 finite search

After NEXT126 is published, freeze the search before scoring:

- bases: all 260 published NEXT125 candidates that pass both all six source AUC
  gates and all 12 SAFE cells;
- flatten each NEXT125 base plus its one/two MHCR terms into a complete physical
  formula;
- candidates per base: pure-base reproduction and one HPP guard at weights
  `0.10`, `0.25`, `0.50`, and `1.00`;
- expected candidate count: `260 * 5 = 1,300`;
- optional direction: high HPP means higher structural risk;
- optional missing policy: `OPTIONAL_GUARD_OFF_KEEP_BASE`;
- all six AUC, SAFE12, and BROAD Pauling-dominance gates remain unchanged;
- validation and replication remain closed unless at least one frozen candidate
  passes every discovery gate.

Candidate identities, base identities, the input feature hash, and the complete
weight grid must be hashed before the formal scoring run. No formula, threshold,
weight, or winner-selection rule may change after results are visible.

## Verification

- TDD for HPP bounds, monotonic profile semantics, zero handling, fail-open
  handling, row accounting, and serial/parallel search equivalence.
- Exact pure-base reproduction of the 260 NEXT125 diagnostics.
- Independent output SHA-256 verification.
- No validation/replication access and no DFT/proxy/relaxation execution.
- Append the formal outcome only to the standalone research report.
