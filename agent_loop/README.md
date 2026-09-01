# The agent loop

The manuscript's methodological claim is not that a search found eight laws. It is
that the search **kept refuting itself**, and that what survived is what withstood
the refutations. This directory is that record.

The loop's driver is not here. Task assignment, parallel execution and result
exchange were handled by our own `codex-api` client
(<https://github.com/szl666/codex-api>), which called Codex directly; the
multi-agent loop never invoked a language model itself. What this directory holds
is the trace the loop left, and the trace is what a reader can check.

```
investigations/   546 scripts, one or more per numbered investigation
plans/            329 task specifications written before the code
tests/            574 checks written against the investigations
frozen/           the sealed rules, scores and contact logs of four chains
index.json        every numbered stage -> its plans, scripts and tests
refutation_ledger.csv   the eleven conclusions that were published and then overturned
```

`index.json` covers 478 numbered stages between NEXT2 and NEXT568: 414 have a
script, 360 a plan, 410 a test. Gaps are real. Some numbers were reserved and
never used; the early stages ran before plans were written down; several plans
cover a range and only the first number produced code.

## These are not eight successes and 560 footnotes

Most of what is here failed, and several whole lines were later corrected. Reading
any single script as a result would be a mistake. The stages group as follows.

| stage | what it asked | how it ended | standing |
|---|---|---|---|
| NEXT2–NEXT5 | anion corrections, valence guards, lockbox and temporal checks | local gains, but guard domains and parameter fallbacks did not hold | main line; feeds the refutation ledger |
| NEXT6–NEXT16 | MatterSim, few-step relaxation, coupled Hessians, DFT queues | **incompatible with the boundary frozen later**: a law may not need DFT, a machine-learned potential, or a relaxation | **corrected side branch** |
| NEXT17–NEXT25 | analytic relaxation change, cross-source prediction freezes | in-domain signal, but endpoint-dependent or poor transfer | boundary exploration |
| NEXT26–NEXT48 | OMC25 periodic contact and energy response, then QMOF | confirmed twice within source, failed to transfer to QMOF | main line; the clearest in-domain-success / external-failure case |
| NEXT49–NEXT79 | ODAC23 framework topology, coordination, electrostatic tails | candidates failed one-shot validation or the safety margin | **side branch, negative** |
| NEXT80–NEXT98 | PRLR, SCIGEN, WyFormer, cross-source combinations | source-specific signal, no cross-source candidate | **side branch, negative**; shows the source fingerprint |
| NEXT101–NEXT117 | DOBVR, CMVF, CMVO, CMVOM, HCID | mathematical and engineering gates passed, SAFE and BROAD did not | negative |
| NEXT118–NEXT455 | hundreds of protective transforms, Voronoi and force-closure quantities, spectral and bond-valence families | BROAD-qualified count stayed at zero throughout | **not new laws**; the evidence that search was never the bottleneck |
| NEXT460–NEXT524 | Hawthorne characteristic coordination and Lewis acid–base | wide coverage, weak AUC; 0 of 21 corrections passed BROAD | systematic refutation of a literature-inspired route |
| NEXT525–NEXT540 | SSSP sequential validation, BVC, ODAC guards | internal validation succeeded, the replicated operating gate failed | failure ledger |
| NEXT541–NEXT550 | JARVIS-CDVAE, Li–Si, OMC25 two-sided contact | class degeneracy, near-random, retrospective failure | **side branch, not a result** |
| NEXT551–NEXT562 | high-entropy alloys: endpoints, packing, EPCU | endpoints degenerate; blind validation failed; 192 combinations, none pre-qualified | **side branch**; the full failure chain is kept deliberately |
| NEXT565–NEXT568 | mechanism decomposition, three-candidate freeze, independent confirmation | all 13 pre-registered confirmation clauses passed | the one new result of that round, and narrow |

The eight PRIS laws come from the plausibility line, not from these branches. The
branches are here because a search that only publishes its successes cannot be
audited, and because the diagnostics that killed a claim outlive the claim.

## Reading an investigation

Scripts are numbered, not named for what they found. Start from `index.json`,
open the plan first, then the script, then the test. Most scripts declare a
`PROTOCOL` constant naming the frozen design they execute; 486 of 546 do.

They are **an audit trail, not a supported library**. They import each other
several layers deep, several still expect the derived feature store that this
repository does not carry (`PRIS_FEATURES`), and none is maintained. Nothing in
`src/` depends on them.

## The eleven

`refutation_ledger.csv` holds the conclusions that were written down as results
and then overturned by the same loop, each with the reason it looked true and the
cheap diagnostic that exposed it. They are Supplementary Figs. S1 and S2 of the
manuscript, and Supplementary Note S6 tells the story.
