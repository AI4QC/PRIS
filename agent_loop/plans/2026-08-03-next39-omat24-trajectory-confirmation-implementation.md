# NEXT39 OMat24 trajectory confirmation implementation plan

1. Write failing tests for trajectory identity parsing, parent-unique selection,
   step-0-only cohort publication, hash locking, and no-overwrite behavior.
2. Implement `src/next39_omat24_trajectory_cohort.py` and make those tests pass.
3. Write failing tests for exact frozen-rule validation, B+E scoring, fail-open
   decisions, Pauling controls, and prediction publication.
4. Implement `src/next39_next23_predictions.py` and make those tests pass.
5. Write failing tests for post-freeze later-geometry opening, exact fingerprint
   distance, atom-identity checks, Wilson metrics, and evaluation publication.
6. Implement `src/next39_trajectory_evaluate.py` and make those tests pass.
7. Run focused tests, freeze the full OMat24 cohort, publish predictions, verify
   all hashes, and only then run the later-geometry evaluation.
8. Run the full test suite, audit CodeGraph health and canonical-file diffs, and
   write a standalone NEXT39 report without modifying canonical documents.
