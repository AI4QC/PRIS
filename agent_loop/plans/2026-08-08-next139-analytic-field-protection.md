# NEXT139 Analytic-Field Protection Freeze

Protocol: `2026-08-08-next139-analytic-field-protection-v1`

## Mechanism

Protect structures whose largest dimensionless analytic Ewald field is small.
`aefi_field_max` is computed from the zero-step periodic structure and formal
oxidation-state charges with an analytic Ewald derivative, normalized by the
cell length and charge scales. It uses neither a DFT calculation/value nor a
learned energy/force/stress proxy.

The frozen protection is

```text
t = log1p(max(0, aefi_field_max))
z = (1.060255159285863 - t) / 1.1838398971398365
analytic_field_balance_protection = clip(max(0, z), 0, 0.8956068821868941)
```

The center is the pooled discovery-feature median, the scale is the pooled
discovery-feature IQR, and the clip is the endpoint-free 99.5th percentile of
the positive normalized tail. These constants were derived without opening an
endpoint column.

Support requires both `next43_analytic_field_supported == true` and a finite,
nonnegative `aefi_field_max`. If unsupported, the protection term is off and
the base score is retained.

## Discovery rationale only

Offline discovery labels showed that lower `aefi_field_max` distinguishes
protected from severe structures in the same direction across SCIGEN and
WyFormer and within every fold. Those outcomes motivate testing the frozen
feature but do not enter the executable formula or its constants.

## Boundaries

- Read only the two published discovery feature tables; do not read validation
  or replication feature tables or endpoints while materializing NEXT139.
- Do not execute DFT or physical relaxation.
- Do not use DFT values or learned energy/force/stress proxies.
- Preserve all prior scripts and artifacts; NEXT139 is additive.
