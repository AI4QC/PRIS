# NEXT129--130 high-coordination protection plan

## Residual evidence

NEXT128 proves that every one of the 260 NEXT125 AUC+SAFE12 laws has the same
closest BROAD residual: only the six SCIGEN `protected_kept` constraints fail;
all other cellwise Pareto constraints pass at that law's closest threshold.

Within the globally closest law's fixed rejected SCIGEN set, the endpoint-free
feature `cov_coord110_mean` distinguishes falsely rejected protected structures
from correctly rejected severe structures with pooled AUC 0.75346 and worst-
fold AUC 0.71564. Protected and severe medians are 10.3077 and 7.1667. This is
interpreted as evidence that dense covalent coordination is a protection signal
against the existing positive-risk score.

## NEXT129 label-free operand

Materialize one bounded protection operand from discovery feature tables only:

```text
z = (log1p(cov_coord110_mean) - 2.1671471220989416)
    / 0.5873264716128193
coordination_protection = clip(max(0, z), 0, 0.9209581129860017)
```

The center is the pooled endpoint-free median, the scale is the pooled
endpoint-free IQR, and the clip is the pooled endpoint-free 99.5th-percentile
distance from the median in IQR units. Nonfinite or invalid raw values disable
protection and keep the base score. No endpoint, validation/replication output,
DFT value, relaxation, MLIP, or learned energy/force/stress proxy may be read.

## NEXT130 finite search

- bases: all 260 frozen NEXT125 AUC+SAFE12 laws, preserving nested fail-open
  MHCR semantics through the tested reversible virtual-base encoding;
- candidates per base: no protection and subtractive protection weights
  `0.10`, `0.25`, `0.50`, `1.00`, and `2.00`;
- expected candidate count: `260 * 6 = 1,560`;
- active score: `max(0, base_score - weight * coordination_protection)`;
- inactive/missing protection: keep `base_score` exactly;
- base support mask remains unchanged;
- all six AUC, SAFE12, and strict BROAD Pauling-dominance gates remain
  unchanged;
- freeze all base, formula, weight, and candidate identities before scoring;
- do not open validation or replication unless a frozen candidate passes every
  discovery gate.

This is a protection correction, not a risk term. Positive protection weights
must never increase any row's base score. No threshold or gate may be relaxed.
