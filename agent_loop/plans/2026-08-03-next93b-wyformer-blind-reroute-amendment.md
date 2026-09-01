# NEXT93b WyFormer blind reroute amendment

## Reason for amendment

The first NEXT93 router correctly separated per-row endpoint payloads, but its
top-level manifest exposed partition-specific counts of DFT successes,
failures, protected rows, middle rows, and severe rows.  No per-row validation
or replication label was inspected and no formula was fitted.  Nevertheless,
those aggregate disclosures invalidate the intended never-read claim for the
first validation and replication partitions.

The first NEXT93 artifacts are retained as contaminated audit evidence and
must never be used for formal validation or replication.

## Blind reroute fixed before execution

- New protocol: `2026-08-03-next93b-wyformer-blind-reroute-v1`.
- New whole-reduced-formula split salt:
  `NEXT93B_WYFORMER_BLIND_REDUCED_FORMULA_SPLIT_V1`.
- Buckets remain 55.00% discovery, 22.50% internal validation, and 22.50%
  internal replication.
- The salt and bucket boundaries are fixed here before the new partitions are
  routed.
- The old partition aggregates are not inputs to the new assignment.

## Non-disclosure requirements

1. The cohort manifest may publish only label-independent row counts per role,
   endpoint file hashes, and provenance identities.
2. It must not publish partition-specific DFT success/failure counts, endpoint
   stratum counts, energies, or any summary derived from them.
3. Endpoint manifests may publish only the role, total row count, payload hash,
   and fixed endpoint definition; they must not publish observed label counts.
4. The router return value follows the same restriction so a formal command
   cannot print hidden aggregate labels accidentally.
5. Feature computation receives the cohort directory only and is run before
   opening even the new discovery endpoint.

If any non-disclosure test fails, NEXT93b is not published.
