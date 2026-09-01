# NEXT133/NEXT134 compactness protection continuation

## Motivation

At the globally closest NEXT131 threshold, the remaining SCIGEN error set has
570 rejected protected structures and 5,103 rejected severe structures. Across
all five formula-group folds, protected errors have smaller `geom_volume_pa`
and larger normalized covalent packing. These are structure-only quantities.

## Endpoint-free features

Constants are frozen from the pooled SCIGEN+WyFormer discovery feature tables
without reading any endpoint.

### Low-volume protection

- raw: `geom_volume_pa`, valid only when finite and positive
- transformed: `x = log(raw)`
- `center = 3.0858220121448285`
- `scale = 0.6305067898025083`
- `clip = 1.5310711399624055`
- feature: `clip(max(0, (center - x) / scale), 0, clip)`

### Covalent-packing protection

- raw: `geom_covalent_packing`, valid only when finite and nonnegative
- transformed: `x = log1p(raw)`
- `center = 0.5102962511091282`
- `scale = 0.15390578713507463`
- `clip = 1.9773347262377292`
- feature: `clip(max(0, (x - center) / scale), 0, clip)`

Invalid inputs turn only that protection term off and keep the current base
score. Neither feature may change base support.

## Planned search

- Bases: the 11 NEXT130 formulas that pass AUC+SAFE12 at coordination
  protection weight 2.
- New term weights: `0.1, 0.25, 0.5, 1, 2, 4`.
- Configurations per base: the unchanged base, 12 single-term variants, and 36
  two-term variants.
- Score: `max(0, coordination_protected_base - sum(w_i * compactness_i))`.
- Gates and winner ranking remain identical to NEXT130.

## Boundaries

- Discovery outcomes are offline labels only.
- No validation or replication endpoint may be opened before all frozen
  discovery gates pass.
- No DFT value, learned energy/force/stress proxy, or relaxation may enter the
  executable law.
