# NEXT42 converged Alexandria source qualification implementation plan

Date: 2026-08-03

1. Add failing unit tests for deterministic provenance classification, benchmark exclusion, exact atom-order validation, convergence checks, fail-closed manifests, and no-replace publication.
2. Implement a provenance-only scanner that binds the two formal trajectory shards, official benchmark list, and Alexandria final-database shards while emitting only material identity, source family, and location.
3. Run the scanner and freeze a source-qualification inventory.  Review every surviving source family against the primary-source workflow documentation before setting an allowlist.
4. If and only if at least one family has defensible raw-x0 provenance, add failing tests and implement a geometry-only cohort freezer.  The freezer must not emit DFT energy, force, stress, or final geometry.
5. Apply the unchanged NEXT23 B+E rule and Pauling controls; freeze predictions before any final structure is opened.
6. Add failing tests and implement a convergence-and-structure-change evaluator using the published `0.005 eV/angstrom` maximum-force criterion and the historical Matbench Discovery fingerprint implementation already sealed by NEXT39.
7. Run focused tests, the full suite in the existing `newpauling` environment, manifest/source hash verification, and CodeGraph status.
8. Write a standalone report.  Do not modify canonical reports or manuscript material without user confirmation.
