# NEXT27 periodic contact-pressure implementation plan

1. Add failing tests for periodic-image enumeration, exact 1-2/1-3/1-4 path
   exclusion, translation/permutation invariance, self-image contacts, and
   denser-cell pressure monotonicity.
2. Implement a new additive periodic-contact feature module without changing
   NEXT26 scripts or artifacts.
3. Build NEXT27 feature tables for the two labelled development shards and
   verify cross-shard AUC, correlations, support, and feature distributions.
4. Add tests and implementation for a two-shard-stable analytic candidate
   search.  Freeze only if pooled and per-shard Wilson gates all pass.
5. If eligible, locate and byte-extract the third archive member, sanitize x0,
   compute features, and freeze predictions before endpoint opening.
6. Evaluate once, run focused and full tests, verify hashes, and write a new
   standalone report.  Leave every existing report, paper, README, and prior
   script untouched.

