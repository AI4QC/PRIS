# NEXT201 motif-weight floor repair search

## Purpose

Search one additive, interpretable, pre-DFT repair of the frozen NEXT164 base
score after NEXT200 authorized exactly one x0-only motif hypothesis:
`motif_weight_sum_min__protected_high`.

The executable candidate may use only the original unrelaxed crystal geometry,
the existing no-DFT base law, and the CrystalNN order-parameter weight sum. It
must not use DFT values, learned energy/force/stress proxies, relaxed structures,
trajectories, or physical relaxation. Discovery outcomes are offline labels only.

## Frozen physical certificate

Let `w = clip(motif_weight_sum_min, 0, 1)`. For a fixed floor `tau < 1`, define

`P_tau = clip((w - tau) / (1 - tau), 0, 1)`.

`P_tau` is the degree to which the least coherent atomic site approaches the
ideal unit total CrystalNN coordination confidence above a fixed floor. Missing
or unsupported motif values disable the correction and preserve the base score.

The exact floor grid is:

`(0, 1/2, 3/4, 7/8, 15/16, 31/32, 63/64, 127/128, 255/256, 511/512, 1023/1024)`.

No direction reversal, second motif feature, or endpoint-derived quantile is
allowed.

## Frozen repair law

Let `B` and `S` be the existing frozen BROAD and SAFE thresholds and let
`R = S - B`. For the original base score `s0`, the candidate score is

`s = max(0, s0 - alpha * R * P_tau)`

only when `B <= s0 < S`, base support is true, and `P_tau` is finite. Outside
that interval the candidate equals `s0` exactly. The interval decision always
uses the original base score, never the corrected score.

The exact attenuation grid is `(1/4, 1/2, 3/4, 1, 3/2, 2)`. The candidate
universe is the unchanged base plus all 11 x 6 floor/attenuation pairs: 67 exact
candidates.

## Frozen evaluation and selection

- Reconstruct and exactly reproduce the NEXT164 base candidate.
- Use only SCIGEN and WyFormer discovery endpoint cohorts.
- Reuse the existing cross-source source-AUC, SAFE-cell, and BROAD-cell gates
  without modification.
- Rank/select with the existing deterministic evaluator; a new law is successful
  only if one candidate passes every discovery gate.
- Keep internal validation and replication endpoint paths physically absent from
  the interface and unopened.
- Publish a standalone catalogue, evaluation, candidate formula, full candidate
  table, and manifest atomically outside the repository.

If no candidate passes all gates, terminate this branch and do not open any
validation or replication endpoint. If a candidate passes, freeze it first and
still keep validation/replication sealed pending a separate preregistered step.
