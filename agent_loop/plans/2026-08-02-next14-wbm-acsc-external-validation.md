# NEXT14: frozen ACSC validation on an external WBM source

## Evidence class

This is an external-source retrospective validation, not a fresh project-wide lockbox. The WBM test labels were opened previously by an older geometry-only rule, but PHSC/CHSC/ACSC and their thresholds were fixed without using WBM labels. No NEXT14 sample, threshold, mode, or decision may be changed after the NEXT14 label opening.

## Additive and storage boundary

- Add new scripts, tests, manifests, and aggregate results only.
- Do not edit prior scripts, reports, papers, notes, README, or preregistration files.
- Keep identifier-bearing geometries, feature tables, predictions, and label joins outside the repository under the external feature store.
- Keep only aggregate metrics and cryptographic manifests in the repository.

## Frozen cohort

- Source population: the physically isolated WBM `test_x0_features.parquet` partition, bound to its existing manifest and official initial-structure ZIP.
- Eligibility: periodic initial structures with 2 through 12 atoms, inclusive. This matches the small-cell regime in which ACSC was developed and avoids post-label computational filtering.
- Selection: the 2,048 eligible material IDs with the smallest SHA-256 of `next14-wbm-acsc-external-v1|material_id`.
- Selection uses no label, relaxed structure, final energy, or initial-final fingerprint.
- If fewer than 2,048 structures are eligible, fail rather than changing the criterion.

## Frozen methods

1. Classical Pauling 2--5 combined rule, with oxidation/topology failures recorded as `ABSTAIN`.
2. PHSC negative curvature.
3. PHSC or CHSC negative curvature.
4. PHSC/CHSC plus formal coupling-only ACSC negative curvature.
5. Conservative gate: PHSC/CHSC negative, or a coupling-only ACSC mode confirmed negative at amplitudes `2^-7`, `2^-8`, and `2^-9`.
6. The already-frozen WBM Born/packing rule as an external comparator, applied without recalibration.

All MatterSim probes use the already-frozen 5M checkpoint and the existing numerical definitions. No margin or amplitude is fitted on WBM.

## Labels and endpoints opened only after features are sealed

- Primary protection: official WBM `stable` label (`corrected e_above_hull <= 0`).
- Valuable retention: corrected hull energy `<= 0.05 eV/atom`.
- High-energy rejection: corrected hull energy `>= 0.20 eV/atom`.
- Structural-change diagnostic: `site_stats_fingerprint_init_final_norm_diff`, reported continuously without choosing a new cutoff.

## Frozen metrics

For every method report:

- coverage, reject fraction/DFT savings, stable recall, valuable recall;
- reject precision for `e_above_hull > 0`;
- high-energy rejection recall;
- exact binomial confidence intervals;
- median and mean initial-final fingerprint among rejected versus non-rejected rows.

Compare every new gate to Pauling using composition-cluster bootstrap differences. A `complete_superiority_over_pauling` result requires all of:

1. stable-recall lower 95% confidence bound at least 0.95;
2. coverage no lower than Pauling;
3. lower 95% bound of the paired high-energy-recall difference above zero;
4. lower 95% bound of the paired DFT-savings difference above zero;
5. lower 95% bound of reject-precision difference no worse than -0.02.

Failure of any clause is a negative scientific result, not a reason to refit.

## Verification

- Geometry archive contains exactly 2,048 deterministic, metadata-free initial structures.
- Every label-free artifact states `labels_opened=false` and binds all inputs and executed sources by SHA-256.
- The isolated evaluator is implemented and tested before label opening.
- Label opening is logged once with pre-opening feature and evaluator hashes.
- Full tests and output-manifest verification pass.

