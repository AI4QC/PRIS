# NEXT40 OMat24 short-horizon implementation plan

1. Add failing tests for identity-only exclusion, 1--19 horizon filtering,
   opaque record copying, source hashing, and no-overwrite publication.
2. Implement `src/next40_omat24_short_source.py` and pass focused tests.
3. Build the full filtered raw source from the existing OMat24 LMDB while later
   geometry and DFT values remain unopened.
4. Reuse the unchanged NEXT39 cohort and prediction runners to freeze step-0
   structures and predictions for every eligible parent.
5. Verify hashes, then run the unchanged exact structure-change evaluator.
6. Publish a separate NEXT40 report, run the complete regression suite, and
   leave every canonical document unchanged.
