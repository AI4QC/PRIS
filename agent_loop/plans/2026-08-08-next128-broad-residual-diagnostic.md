# NEXT128 BROAD residual diagnostic

## Purpose

Diagnose why none of the 260 frozen NEXT125 AUC+SAFE12 laws admits a common
threshold that strictly Pareto-dominates Pauling in every source/fold cell.
This is an offline discovery-label diagnostic, not an executable law and not a
feature search.

## Frozen inputs

- the 260 NEXT125 records satisfying all six AUC gates and SAFE12;
- the same label-free feature tables and discovery endpoints used by NEXT125;
- the exact nested fail-open MHCR score semantics reconstructed through the
  tested reversible virtual-base representation;
- the unchanged NEXT98 SAFE and BROAD gate definitions and Pauling baselines.

Validation and replication outputs remain unopened. No DFT calculation, DFT
value, relaxation, MLIP, or learned energy/force/stress proxy is permitted.

## Diagnostic rule

For every law, enumerate every distinct observed score threshold below its
published SAFE threshold. At each threshold evaluate, in all 12 cells:

1. coverage lower bound strictly above Pauling;
2. protected structures kept at least Pauling;
3. severe structures rejected strictly above Pauling;
4. severe-rejection precision lower bound strictly above Pauling;
5. savings lower bound strictly above Pauling.

Also require severe-precision lower bound at least 0.45 in each source-
aggregate cell. A threshold is ranked first by the number of failed scalar
constraints, then by the summed normalized positive deficit, then by the lower
threshold. Equality remains a failure wherever BROAD uses a strict inequality.

## Outputs

- minimum failed-constraint count for each of the 260 laws;
- the globally closest threshold and its exact failing cell/components;
- frequency of each limiting cell/component among per-law optima;
- immutable input, source, and output SHA-256 identities;
- explicit discovery-only and no-DFT/no-validation provenance.

The diagnostic may motivate one new label-free physical mechanism only if a
stable repeated residual is found. It must not weaken BROAD or tune a new
formula inside NEXT128.
