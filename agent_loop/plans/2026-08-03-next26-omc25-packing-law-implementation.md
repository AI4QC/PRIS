# NEXT26 OMC25 packing-law implementation plan

1. Add tests for ASE-LMDB decoding, trajectory identities, complete-group
   selection, chronological endpoints, and x0 sanitization that strips all DFT
   labels.
2. Implement a dependency-light OMC25 reader and source-audit command with
   SHA-256 manifests and create-once output directories.
3. Add tests for deterministic periodic molecular-crystal descriptors,
   covalent-neighbour exclusions, permutation invariance, fail-open handling,
   and forbidden-column checks.
4. Implement the DFT-free packing/contact feature builder.
5. Add tests for the fixed DFT-response endpoint, Wilson gates, small analytic
   candidate catalogue, deterministic selection, and freeze-before-open
   contracts.
6. Run development search on complete `data0031` trajectories and publish a
   frozen rule only if an eligible physically interpretable candidate exists.
7. If eligible, extract and sanitize `data0037`, remove development CSD
   refcodes, compute predictions, hash/freeze them, and only then evaluate the
   DFT labels once.
8. Compare with Pauling controls, run focused and full regression tests, verify
   all manifests/hashes, and write a new standalone NEXT26 report.  Do not edit
   any earlier report, paper, README, or canonical content.

