# NEXT47 OMC25 energy-response replication design

## Objective

Run a second, zero-refit confirmation of the frozen NEXT31 analytic law on every
OMC25 validation archive member that followed the first 24 main members used by
NEXT26--31.  The executable law remains a one-shot, unrelaxed-x0 geometry rule;
DFT values are opened only as offline evaluation labels after predictions are
sealed.

## Frozen law

The only eligible rule artifact is
`NEXT31_FROZEN_ENERGY_RULE.json` with SHA-256
`993d64b851c755fc5cc0d4b68ca7ca6994d4bdb7ed666f860d43a04925e254a8`.
No coefficient, threshold, feature, support rule, or endpoint threshold may be
changed.

## Cohort selection before labels

- Source: the official `omc_val_250802.tar.gz` archive.
- Main-member selection: archive indices 24 through 39 (`skip_main=24`,
  `take_main=16`).  This is the complete tail after the 24 members used by
  NEXT26--31 and is fixed before member names or scientific records are read.
- Begin with the cumulative 1,732-refcode exclusion set sealed after NEXT31
  `data0035` (SHA-256
  `f05c044297c2287ec18abf3a91bfe57ad3016f32b395f780d7768d0748e7ff3e`).
- Process tail members in archive order and extend the exclusion set after each
  label-free x0 export, preventing CSD-refcode overlap both with historical data
  and between new shards.
- Decode only trajectory identity metadata and frame-zero numbers, positions,
  cell, and PBC before prediction.  Do not parse energy, force, stress, relaxed
  coordinates, or any other endpoint field.

## Prediction and opening order

1. Freeze the NEXT47 protocol, binding the rule, exclusion list, source URL,
   archive selection, and unchanged NEXT31 gates.
2. Extract raw LMDB members as opaque bytes.
3. Export one geometry-only x0 per eligible complete trajectory.
4. Compute the existing deterministic periodic nonbonded features.
5. Apply the frozen NEXT31 law and publish identity-locked predictions.
6. Only then decode initial/final DFT records for the exact prediction IDs.
7. Evaluate the second cohort and a no-refit pooled view with the first NEXT31
   confirmation.

## Frozen endpoints and gates

- Protected: DFT relaxation energy drop at most 0.01 eV/atom.
- Energy-positive: DFT relaxation energy drop at least 0.04 eV/atom.
- The second cohort must independently pass the original NEXT31 gates:
  coverage lower bound >= 0.95, protected-recall lower bound >= 0.95,
  rejection-precision lower bound >= 0.70, DFT-savings lower bound >= 0.02,
  and energy-positive AUC >= 0.85.
- The pooled cohort and per-shard results are descriptive migration diagnostics;
  they cannot rescue a failed second confirmation.

## Claim boundary

A pass supports conservative, DFT-free-at-execution prescreening for large DFT
relaxation-energy response within OMC25.  It does not establish formation
energy, convex-hull stability, general thermodynamic stability, external-source
transfer, or replacement of DFT.  The raw files contain accessible labels, so
this is a sealed-order confirmation, not a physically isolated never-read
lockbox.
