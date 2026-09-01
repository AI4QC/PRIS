# NEXT143 Coulomb--Steric Balance Protection Freeze

Protocol: `2026-08-08-next143-coulomb-steric-balance-protection-v1`

## Frozen mechanism

`acsb_site_residual_max` is the maximum sitewise residual after matching the
zero-step analytic periodic Coulomb vector field with the short-range steric
repulsion vector field by their globally optimal nonnegative scalar. It is
already normalized to `[0,1]`.

Define, without fitted constants,

```text
coulomb_steric_balance_protection = 1 - acsb_site_residual_max
```

Support requires `next43_coulomb_steric_balance_supported == true` and a
finite raw value in `[0,1]`. Unsupported rows turn the protection off and keep
the base score.

## Selection rationale

NEXT142's frozen SAFE-to-BROAD shell audit identified low
`acsb_site_residual_max` as a fold-stable, cross-source-concordant local
retention marker: SCIGEN pooled/worst-fold AUC `0.6478259409556923` /
`0.6261704397297618`, WyFormer pooled AUC `0.54807760484986` under the fixed
SCIGEN sign. It was selected over higher-ranked source-specific size/count
features because those reverse direction on WyFormer, and over
`nm_site_min` because its WyFormer shell AUC is only `0.5113181125252597`.

These discovery outcomes select the mechanism for testing but do not appear
in the executable definition.

## Boundaries

- Materialization reads discovery feature tables only, without endpoint data.
- The analytic Coulomb and steric vector fields are zero-step classical
  calculations, not DFT and not learned energy/force/stress proxies.
- Validation and replication remain unopened; no relaxation is executed.
- All artifacts are additive and canonical documents remain unchanged.
