# NEXT148 Charge-Order Spectrum Protection Freeze

Protocol: `2026-08-08-next148-charge-order-spectrum-protection-v1`

Materialize one parameter-free, label-free protection from the bounded
long-wavelength formal-charge spectrum fraction:

```text
charge_order_spectrum_protection = 1 - csf_long_fraction
```

The raw feature is supported only when `next43_charge_spectrum_supported` is
true and the raw value is finite in `[0, 1]`; unsupported rows fail open in
later score composition.

`csf_long_fraction` is the ratio of Gaussian-smoothed reciprocal formal-charge
intensity at dimensionless smoothing scales 0.60 and 0.25. It is calculated
from the input lattice, fractional coordinates, and neutral formal charges.
The formula contains no endpoint-fitted center, scale, threshold, or clip.

The selection basis is frozen NEXT142 threshold-local evidence: lower
`csf_long_fraction` has SCIGEN pooled/macro/worst-fold AUC
`0.598915/0.597993/0.570485` and WyFormer pooled/macro/worst-fold AUC
`0.649294/0.652237/0.615480` in the same protected direction. This is a new
physical axis after NEXT145 and NEXT147 terminated both smooth and sparse ACSB
protection.

No discovery endpoint or label is opened during materialization. No DFT
calculation or DFT value, learned energy/force/stress proxy, or physical
relaxation is used. Validation and replication remain unopened.
