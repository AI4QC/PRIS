# NEXT101b Expanded-Catalogue DOBVR Amendment

**Status:** frozen after label-free composition/support audit and before reading any discovery endpoint labels for NEXT101/NEXT101b.

## Why an additive amendment is needed

NEXT101 v1 remains unchanged as the strict common+ICSD, element-uniform mechanism. A ten-structure raw-x0 smoke test supported 3/10 structures. A complete label-free composition audit then measured the following row coverage on physically isolated discovery metadata only:

| catalogue/allocation | SCIGEN discovery | WyFormer discovery |
|---|---:|---:|
| common+ICSD, uniform | 0.173125 | 0.361047 |
| common+ICSD, at most one mixed element | 0.222643 | 0.433486 |
| all tabulated states, uniform | 0.557981 | 0.600535 |
| all tabulated states, at most one mixed element | 0.649369 | 0.687309 |

No stability, relaxation, DFT, Pauling-decision, or success/failure endpoint column was loaded during this audit. The mixed-valence branch is not adopted now because a site allocation rule that is simultaneously deterministic, permutation invariant, supercell invariant, and computationally bounded has not yet been frozen.

NEXT101b is therefore a separate expanded-catalogue uniform-assignment mechanism. It does not replace or alter NEXT101.

## Frozen NEXT101b rules

1. Candidate states are the nonzero finite integer values in pymatgen `Element.oxidation_states`.
2. Neutrality, exact enumeration bounds, deterministic ordering, graph construction, and no-DFT execution boundary are inherited from NEXT101.
3. Chemically reversed assignments are removed by a label-free electronegativity orientation gate: the site-count-weighted mean Pauling electronegativity of negative sites must be strictly greater than that of positive sites. Missing/nonfinite electronegativity makes that assignment unsupported.
4. Each surviving state receives catalogue tier 0 if it appears in that element's frozen common+ICSD union, otherwise tier 1. An assignment's tier is the maximum site tier.
5. The winning explanation is ranked lexicographically by assignment tier, then SCBV RMS, q95, maximum mismatch, then element-state tuple. Thus an exotic state cannot displace an available common/ICSD explanation merely by fitting geometry better.
6. NEXT101b exposes the winning tier, the fraction of surviving assignments that are tier 0, and the winning electronegativity margin in addition to the NEXT101-style realizability aggregates.
7. Unsupported rows abstain and are counted in unconditional coverage; they cannot be silently dropped or treated as successful rejection.
8. The all-state catalogue and actual per-element options are hashed. Pymatgen version remains recorded.

## TDD and evaluation sequence

1. Add `tests/test_next101b_expanded_oxidation_bv_realizability.py` and verify import failure.
2. Add `src/next101b_expanded_oxidation_bv_realizability.py` minimally.
3. Test NaCl reversed-charge removal, NaO expanded-tier fallback, TiO2 tier-0 priority, structure purity, finite schema, determinism, and supercell invariance.
4. Run NEXT101/NEXT101b/NEXT19/NEXT22 regressions.
5. Run label-free discovery smoke tests for both sources. Only if support and runtime are acceptable may NEXT102 materialize both v1 and v1b discovery features.
6. Freeze NEXT103 candidate grammar before joining endpoint labels. Do not open either replication endpoint unless every cross-source discovery gate passes.

Protected paper/report paths and all prior scripts remain unchanged. This amendment is additive and does not reinterpret NEXT101 v1 as successful.
