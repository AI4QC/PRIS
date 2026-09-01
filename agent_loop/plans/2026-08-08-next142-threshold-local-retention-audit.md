# NEXT142 Threshold-Local Incremental-Retention Audit Freeze

Protocol: `2026-08-08-next142-threshold-local-retention-audit-v1`

## Frozen score shell

- Reconstruct the formal NEXT136 global-closest product candidate.
- Candidate-key SHA-256:
  `44b9eabae5e1ff3014ef4746758bbc3a79a4f193bad94507dd17c7db0edd1919`.
- Candidate mechanism: packing-product protection, weight `0.5`.
- Published SAFE12 threshold: `3.4014264642057306`.
- Closest BROAD threshold: `0.8669460357541353`.
- Incremental shell: rows with
  `closest_broad_threshold <= score < safe_threshold`.

The shell contains structures newly rejected when lowering the threshold from
SAFE12 toward BROAD. The audit target is protected versus severe discovery
extremes inside this shell, not all discovery rows and not all rows rejected at
the BROAD threshold.

## Frozen feature audit

1. Reconstruct the combined label-free feature table and exact score without
   opening validation or replication data.
2. Consider numeric structure/composition features only.
3. Exclude identifiers, bookkeeping, endpoints, labels, targets, Pauling
   decisions, DFT/energy/force/stress/relaxation fields, learned-proxy fields,
   virtual-score columns, and all already-tested coordination, volume,
   packing, product, bottleneck, and analytic-field protection columns.
4. Require at least 80% finite coverage separately for shell-protected and
   shell-severe rows and at least 10 distinct finite values.
5. Choose the sign by SCIGEN pooled shell AUC, then report that fixed sign on
   SCIGEN pooled and five reduced-formula folds and on WyFormer pooled/folds.
6. Record support, counts, pooled/macro/worst-fold AUC, source-direction
   concordance, and rank candidates by:
   - all five SCIGEN folds evaluable;
   - descending SCIGEN worst-fold AUC;
   - descending SCIGEN macro AUC;
   - descending SCIGEN pooled AUC;
   - descending WyFormer pooled AUC under the SCIGEN-fixed sign;
   - feature name.

This is a diagnostic, not a formula search. Any next executable feature and
all of its constants, grammar, weights, gates, and candidate identities must
be frozen in a later stage before evaluation.

## Boundaries

- Discovery outcomes are offline audit labels only.
- Validation and replication remain unopened.
- No DFT calculation/value enters any executable formula.
- No learned energy/force/stress proxy or physical relaxation is used.
- Existing scripts and canonical documents remain unchanged; NEXT142 is
  additive.
