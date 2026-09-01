# NEXT43 analytic law search implementation plan

1. Add contract tests for a single-row analytic feature bank, strict geometry-only input validation, fail-open family support, output hashing, and no-replace publication.
2. Implement `src/next43_analytic_feature_bank.py` by composing the existing pure-analytic kernels directly on sanitized NEXT42 x0 structures.
3. Run focused tests, then build the full 2,285-row feature bank outside the repository.
4. Add contract tests for deterministic hash splitting, Wilson metrics, finite formula construction, missing-value fail-open behavior, discovery-only selection, and frozen output serialization.
5. Implement `src/next43_finite_law_search.py` and run the fixed catalogue on NEXT42 development labels.
6. If a formula passes both internal splits, freeze its full executable specification before acquiring/opening any new endpoints. Otherwise report the strongest negative result and feature-family diagnostics without relaxing the gates.
7. Add an independent NEXT43 report. Do not edit paper, README, canonical reports, or older scripts.
8. Run focused and full repository tests, audit the CodeGraph pending-sync state, and record artifact hashes.
