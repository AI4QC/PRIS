# NEXT118 analytic metallic-applicability protection design

Date: 2026-08-08

## Status and scientific boundary

This is an additive, adaptive discovery branch after NEXT117. It does not
replace any earlier script, formula, result, report, or manuscript text. The
executable operand remains one raw, unrelaxed `x0` structure plus frozen
analytic elemental and geometric tables. DFT quantities, relaxed structures,
learned energies/forces/stresses, MLIPs, physical relaxation, trajectories,
and same-composition alternatives are forbidden.

The two discovery endpoints were already opened by NEXT117. Consequently any
threshold, width, combination rule, or weight examined here is adaptive
discovery and cannot be described as independent validation. Internal
validation and replication remain unopened unless every frozen discovery gate
is passed.

## Hypothesis

Ionic/bond-valence risk can over-penalize metallic or multicentre networks. A
bounded protection operand should be active only when both of the following
analytic conditions hold:

1. the mean elemental electronegativity is below a frozen upper shoulder; and
2. the lower-tail covalent distance ratio is above a frozen lower shoulder,
   indicating that the raw geometry is not dominated by compressed contacts.

For electronegativity mean `X` and covalent-ratio lower tail `C`, define

```text
M_X = clip((X_upper - X) / X_width, 0, 1)
M_C = clip((C - C_lower) / C_width, 0, 1)
P_min = min(M_X, M_C)
P_product = M_X * M_C
```

Both variants are bounded in `[0, 1]`. Missing or non-finite operands turn the
optional protection off rather than changing base-law support.

For a nonnegative base risk `R`, support mask `S`, and nonnegative weight
`lambda`, compose

```text
R_protected = max(0, R - lambda * P)   on S
R_protected = NaN                       off S
```

This is an applicability correction, not negative evidence for stability. It
cannot create negative risk and it cannot turn an abstention into a decision.

## Implementation and test contract

The new module will be `src/next118_metallic_applicability_protection.py` and
will expose two pure functions:

- `metallic_packing_protection(...)`
- `compose_bounded_protection_score(...)`

Tests must cover both conjunction rules, exact shoulders, missing-data
fail-open behaviour, unchanged support, nonnegative output, input immutability,
shape/schema rejection, and non-finite/negative parameter rejection.

## Search discipline

The first search using this operand is a disposable adaptive probe. A formal
run requires a separate frozen JSON catalogue written before rejoining the
already-open discovery endpoints. No validation or replication endpoint may be
opened on a failed discovery result.
