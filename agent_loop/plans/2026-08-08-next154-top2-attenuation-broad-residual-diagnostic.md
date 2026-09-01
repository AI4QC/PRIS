# NEXT154 top-two attenuation BROAD residual diagnostic freeze

## Purpose and boundary

Diagnose, without searching a new formula, whether the NEXT153 `gamma=0.1`
homotopy materially closes the frozen BROAD constraints relative to its
`gamma=0` endpoint.  This is additive and discovery-label-only.  It opens no
internal-validation or replication endpoint and performs no DFT calculation,
uses no DFT value in an executable formula, invokes no learned
energy/force/stress proxy, and performs no relaxation.

## Frozen population

Read the immutable NEXT153 candidate table and retain exactly candidates that
already pass both all source-AUC gates and all SAFE12 gates.  Do not select on a
BROAD residual.  The frozen expected population is:

- 22 candidates total;
- 11 at `gamma=0` and 11 at `gamma=0.1`;
- candidate-key SHA-256 over newline-joined sorted keys:
  `6b065381ec7fc40d60cc1074eb477480ffedac09faa51250ba2fe6d714a83838`.

Reconstruct all NEXT153 scores from their analytic physical terms and prove
that their published source AUC and SAFE status are unchanged before running
the diagnostic.

## Frozen diagnostic

For each retained candidate, scan the same threshold table used by the prior
BROAD residual diagnostics, subject to that candidate's published SAFE
threshold ceiling.  At each eligible threshold evaluate the frozen BROAD
constraints against the same per-source and per-fold Pauling baselines.  Rank
thresholds and candidates deterministically by:

1. failed constraint count;
2. summed normalized shortfall;
3. threshold;
4. candidate key.

Report the global closest candidate, closest candidate separately for
`gamma=0` and `gamma=0.1`, exact failures, and frequency of each failed
cell/component.  The diagnostic must not change gamma, protection weights,
physical terms, score, SAFE threshold, BROAD gates, or candidate ordering.

If `gamma=0.1` does not reduce the best normalized shortfall at the same minimum
failure count relative to `gamma=0`, terminate the attenuation mechanism.  If
it does reduce the residual, only that pre-existing mechanism may motivate a
separately frozen next experiment; NEXT154 itself never authorizes validation
and makes no scientific improvement claim.

## Outputs

Atomically publish the manifest, JSON diagnostic, and per-candidate parquet to
`$PRIS_ARCHIVE/next154_top2_attenuation_broad_residual_diagnostic_v1`.
Only the standalone research report may be updated afterward.
