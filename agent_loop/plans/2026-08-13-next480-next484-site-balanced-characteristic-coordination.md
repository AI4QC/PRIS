# Site-Balanced Characteristic Coordination Implementation Plan

> **For Codex:** Execute additively and test first. Preserve every prior
> artifact. Keep validation and replication sealed.

**Goal:** Test the equal-local-environment aggregation of Hawthorne's
characteristic-coordination prior, without allowing high-CN cations to dominate
the structural score.

**Architecture:** NEXT480 averages the frozen NEXT475 per-cation compatibility
over cation sites. NEXT481--NEXT484 are conditional full label-free build,
cross-source discovery audit, bounded formula search, and BROAD diagnostic.

## 1. Scientific distinction

NEXT470 pools raw deviations and denominators, hence weights cation site `c`
in proportion to `CN_c+t_c`. NEXT475 takes the worst local compatibility and
failed novelty against NEXT470 in WyFormer. NEXT480 gives every positive site
exactly one vote. It therefore asks whether the typical cation environment,
rather than the high-CN population or single worst site, is compatible with
an experimentally characteristic coordination.

Reuse exactly the NEXT475 table policy: Appendix 3 characteristic CN sets, the
unambiguous printed H+ CN `2.03`, and conservative site compatibility zero for
an element absent from the source. No new constant, interpolation, fitted
weight, subset, tolerance, quantile or fallback is allowed.

## 2. Frozen SBCC formula and no-DFT boundary

For positive site `c`, choose nearest characteristic `t_c` when available and
define `k_c=1-|CN_c-t_c|/(CN_c+t_c)`; for an element absent from the table use
`k_c=0`. Freeze

```text
SBCC(x0) = round_1e-10((1/n_cation) sum_c k_c).
```

The sole feature is `sbcc_site_balanced_characteristic_coordination`, direction
`protected_high`, bounded `[0,1]`. The executable reads only element identities,
charge signs, one raw initial periodic geometry, contact multiplicities and
published constants. It does not run/read DFT, energy/force/stress, learned
proxy/MLIP/potential, relaxation, trajectory, later geometry,
same-composition alternatives, validation or replication. The same failure and
representation-invariance semantics as NEXT475 apply.

## 3. Frozen ordered blind gates

Use the unchanged 80+80 discovery probes. First require support `>=72/80` per
source. Only then compare with all 32 prior formal families, ZBVVG through PFPU,
CLAM, MV-CLAM, ECCC and CCCB. Require `[0,1]`, at least 20 values distinct at
`1e-10`, invariance error `<=1e-8`, and maximum adequate absolute Spearman
`<0.90` with at least 40 joint finite rows.

Only if all pass: NEXT481 requires full discovery coverage `>=0.95` per source;
NEXT482 applies unchanged NEXT224/NEXT413 outcome gates; NEXT483 and NEXT484
remain conditionally authorized exactly as in prior cycles. Validation and
replication stay sealed.

## 4. Artifact order

1. RED pure-kernel/invariance/firewall tests.
2. Independent core/wrapper; do not alter NEXT475.
3. Ordered support and full novelty probes.
4. Continue mechanically only on authorization.
5. Append independent report and run complete verification.
