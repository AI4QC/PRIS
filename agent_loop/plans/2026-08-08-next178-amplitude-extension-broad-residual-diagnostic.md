# NEXT178 Amplitude-Extension BROAD Residual Diagnostic

## Purpose

Close the frozen NEXT177 amplitude-extension branch by measuring its exact
remaining BROAD constraint residual. This is an additive discovery-label
diagnostic, not a formula search, threshold search, validation run, or claim.

## Frozen population

- Reconstruct and reproduce all 13 NEXT177 candidate scores and published
  discovery records from hash-pinned inputs.
- Retain exactly the candidates passing source AUC and all SAFE cells while
  failing BROAD.
- Expected retained count: 8.
- SHA-256 of newline-joined, sorted candidate keys:
  `1947c3c1a3cbb17a1472cc0f0dc665c2bc731da4f6dc07f6cb11c5d48a753b1c`.
- Expected feature counts: base 1; weighted CrystalNN tightness minimum 4;
  weighted CrystalNN tightness q10 3.

## Diagnostic and boundaries

Use the unchanged NEXT164/NEXT176 per-threshold BROAD residual procedure and
unchanged Pauling cell baselines. Publish all failed components and normalized
shortfalls. No candidates, thresholds, gates, features, or amplitudes change.

Discovery outcomes remain offline labels. No DFT calculation/value, learned
energy/force/stress proxy, or physical relaxation enters the executable law.
The runner exposes no validation or replication endpoint, and all such
endpoints remain sealed.

## Outputs

Publish atomically under
`$PRIS_ARCHIVE/next178_amplitude_extension_broad_residual_diagnostic_v1`:

- `MANIFEST.json`;
- `NEXT178_AMPLITUDE_EXTENSION_BROAD_RESIDUAL_DIAGNOSTIC.json`;
- `next178_amplitude_extension_broad_residual_per_candidate.parquet`.

Append the result only to the standalone investigation report. Do not edit
canonical paper, notes, TeX, README, or preregistration files.
