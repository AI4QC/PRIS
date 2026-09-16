# Pre-registration: replacing Pauling's rules with a minimal law set found by AI search

**Version v1.0 · 2026-07-28**
**This file is committed and git-tagged before the lockbox is opened for the first time. The commit hash of that tag goes into the Methods section of the paper.**
**Any later change may only append a "Revision log" entry; existing entries may not be rewritten.**

Corresponding research plan: `new-pauling-laws_AI-discovery_research-plan.md` v1.3
Data seal: `$PRIS_ARCHIVE/lockbox/LOCKBOX.sealed.json`

---

## 0. Why pre-register

Searching for rules over 38,307 structures will inevitably turn up spurious ones. The whole
persuasive force of this project rests on "the search never sees part of the data, and the
criteria are fixed in writing before the data are seen". Adjusting the criteria afterwards is
HARKing, and once is enough to void every conclusion.

---

## 1. Data and splits (frozen)

- **Analysis set**: experimental structures (from ICSD 73,823 + COD 25,339) with a single anion
  and no H and no C; measured at **38,307** entries.
  Anion distribution O 19,833 / S 4,783 / F 2,839 / Se 2,745 / N 1,673 / P 1,663 / Te 1,558 / Cl 1,434 / I 909 / Br 870.
- **Strict oxide subset**: 23,728 entries. **Note that it is not a subset of the analysis set**;
  the 3,895 entries in the difference all contain P (a disagreement over how phosphates are counted).
- **Split**: `split_of(sid, seed) = sha256(f"{seed}:{sid}")[:8] as uint64 / 2**64`,
  seed = `20260728`, discovery 60% / calibration 25% / lockbox 15%.
  Measured **22,925 / 9,634 / 5,748**. A pure function, independent of row order.
- **Lockbox opening quota: 3.** Each opening must state a reason of ≥10 characters and record it
  in `openings.log`.
  **The paper must report the actual number of openings and the law-set hash at each one.**
- **What each partition is for**: discovery may be inspected and searched freely; calibration may
  be used to set thresholds, choose N and select models, **but not to search for laws**; the
  lockbox is used only for the final one-shot evaluation.

## 2. Primary hypothesis and primary target (frozen)

**H1 (primary)**: there exists a law set of size N ≤ 12, each law passing the eight hard gates
G1–G8 of §4.3, that predicts coordination environments better than a modal look-up table baseline
**at matched coverage**.

**The primary target is T4**: predict the coordination environment from composition-level features
alone (element, formal charge, electronegativity, radius sum, stoichiometry).
**Input tier ≤ T0.** Any quantity that requires a structure before it can be evaluated (bond
length, CSM, BVS) may not enter a Guard or a Body.

**Hard lower bound**: Waroquiers 2017's "look up the most common coordination environment by
(element, oxidation state)" reaches about 80%; Pauling's rule 1 reaches 66%.
**Failing to beat the modal look-up table means it is not a law**, whatever the significance.

**Comparisons must be made at matched coverage**: compare against the look-up table only on the
instances the set actually triggers on. Comparison along the full-coverage axis is a metric a
guarded set is bound to lose, and may not be used as a criterion.

## 3. Three degrees of freedom that must be fixed before looking at the data

### 3.1 λ and `N_eff` (§13.5 decision item 1)

The measured leverage on the headline number N is **12-fold** (59 → 5), and this is the only
degree of freedom in the project that can destroy credibility **silently**.

**Frozen decisions**:
- Pre-register **λ = 1**; additionally report three curves for λ ∈ {3, 10, 30} as a sensitivity
  analysis, **all of them, with no cherry-picking**.
- `N_eff = N_data / deff`, `deff = 1 + (m−1)ρ`, with the **clustering unit taken to be the
  structure prototype**. `ρ` is measured by MPU-1 on the feature table grouped by `proto_id`, and
  **must be measured and written into the revision log of this file before any candidate law is
  seen**.
- If the confidence interval on `ρ` is wide enough that `argmin L_total(N)` spans more than 3
  laws, **abandon argmin** and switch to a fixed criterion (cumulative compression ≥90% / ≥80%,
  two levels), **reporting N as an interval**.

### 3.2 Freezing the vocabulary

The following three are frozen together with the lockbox; after the freeze, new Tier C primitives
may only enter the next round of pre-registration:
- the vocabulary Σ of 47 feature primitives, with the three columns `(tier, locality, cost)` for each
- the vocabulary of 250 Guard selectors
- the composition-level implementation of `lewis_base_env` and the base-strength table it references

**Without this freeze G3 is a paper gate**: adding one primitive costs only 0.03 bit, so any fitted
expression can be smuggled in under the name of "it is a primitive".

### 3.3 Combination semantics

**The main line uses conjunction** (an instance satisfies the set ⟺ it satisfies every applicable
member), which has the same semantics as George's 13% and is directly comparable. Union/coverage
semantics is reported as a secondary line. **The cost is known and accepted**: under conjunctive
semantics the `1−1/e` guarantee for greedy selection does not hold, and optimality may not be
claimed in the paper.

## 4. The shape of the headline number (frozen)

**The primary headline is the curve `L_total(N)`, not a percentage.** Nine curves are delivered
(3 Tiers × 3 compression targets); Pauling's five rules and Hawthorne's three are each points on
those curves.

**It is forbidden** to report "N laws satisfied simultaneously by X%" on its own and set it beside
George's 13%. Reasons:
(a) the units differ (his is unguarded, structure-level, 5,000 oxides; ours is guarded, site-level,
38,307 across all anions);
(b) under conjunctive semantics X is monotonically non-increasing in N and systematically rewards
narrowing the guard (measured: 20 guards covering only 2% give a worthless rule with 96.1%
conjunctive satisfaction), so the direction that inflates the number is exactly the degenerate one;
(c) `0.13^(1/4) = 0.60`, so beating 13% amounts only to "60% → 95% per law on average", while the
look-up table is already at 80%.

Where a percentage must be reported, it must be an equal-N comparison under the **five "sames"**
(the same 23,728-oxide subset, the same neighbour algorithm ChemEnv, the same oxidation-state
source, the same structure granularity, the same law count N=4), and it must use **`SET-P25`
recomputed through our own pipeline** rather than George's 13%, and it must **report the pair
(coverage, restrictiveness)**.

## 5. Oxidation-state provenance (frozen)

**Oxidation states derived from `BVAnalyzer` are excluded wholesale from the main statistics for
rules 2/4 and for every Hawthorne quantity.**
Reason: it back-solves valence from bond lengths and is then used to test laws about bond valence,
which is deriving the premise from the conclusion. Its measured failure rate is also 18–35%, and
the failures are not random: they concentrate on mixed valence, uncommon oxidation states and
large unit cells — exactly the most informative samples.

Only two provenance levels are allowed: `cif` (native to ICSD) and `guess` (pure compositional
inference). Derived entries carry an `ox_source` flag in the feature table.

## 6. go/no-go and stop-loss (frozen)

| gate | when | criterion | what if it fails |
|---|---|---|---|
| G-A | Week 3 | reproduce George's five hit rates to ±3 pt | **stop and hunt the bug; do not continue** |
| G-B | Week 3 | is the pairwise AUC on same-composition polymorph pairs ≈0.5 | do not spend GPU; the ΔE column becomes "the laws are unrelated to 0 K stability", **but a number must still be reported** |
| **G-C** | Week 6–7 | does `L_total(N)` show a knee (knee moves ≤ ±3 laws across the four λ levels) | **pivot to "the regularity of crystal chemistry is not low-rank"**, report N as an interval under the fixed criterion, and make the negative result the main result |
| G-D | Week 8 | can the pipeline rediscover Pauling's third rule | **stop and hunt the bug; do not continue** |
| G-E | Week 9–10 | does a T0 set beat the modal look-up table at matched coverage | aim lower, at *Sci. Adv.* / *Angew.*; the paper still stands |

## 7. Graded success criteria (frozen; may not be raised or lowered afterwards)

| level | content | self-assessed probability |
|---|---|---|
| L0 | the set-level objective is well defined and `N*` is not an artefact of λ | 70% |
| L1 | George's five rules reproduced + 60 numbers of methodological sensitivity | ≥95% |
| L2 | all-anion statistical map + first large-scale solution of Hawthorne's a-priori bond strengths + `L_total` baseline | 90% |
| L3 | a pure-CN form of the fourth rule + a distortion correction to the second rule + the ΔE column | 68% |
| L4 | one Tier-0 law set beats the modal look-up table | 35% |
| L5 | a **single universal** Tier-0 law beats the modal look-up table | 12% |

**The L4 criterion has been lowered from v1.0's "one law" to "one law set", and the old L4 becomes
L5. This lowering is recorded here and is therefore part of the pre-registration; no further
adjustment may be made in the paper.**

## 8. Known and accepted limitations (to be written into the paper's Limitations)

1. **Disordered structures are systematically absent**: 87,237 disordered ICSD entries (about 36%
   of valid ICSD entries) were discarded when the database was built. The domain of applicability
   is declared to be "ordered stoichiometric phases". Solid solutions, high-entropy phases and
   A-site mixing are outside the scope of the conclusions.
2. **Positives only**: each of the four negative-sample channels carries its own prior
   contamination. In particular ELEMENTA's compositional enumeration rules are themselves a layer
   of prior, and its flat anion distribution is artificial and may not be used as a natural
   frequency.
3. **Tool circularity**: ChemEnv's weights and the BVS R0 table are both fitted from the same body
   of data, so a "new law" may be no more than a restatement of those tools' internal assumptions.
   Agreement across three neighbour algorithms (G6) is the strongest mitigation; **there is no
   complete solution**.
4. **MLIP noise floor**: the measured median ΔE over ELEMENTA's 111,527 polymorph pairs is only
   0.0203 eV/atom, and only 32.7% exceed the MLIP's MAE of 0.036 eV/atom. The detectable-effect
   lower bound for S1/S2/S4/S5 is therefore pinned at 0.036, and nothing below it is reported.
   S3-A switches to DFT energies and is not subject to this limit.
5. **Time extrapolation stands on a thin leg**: COD post-2019 yields only about 700–1,100
   analysable structures, with a severely skewed anion distribution
   (O 750 / S 112 / Se 84 / N 6 / Br 2). **Only the oxide family alone supports a statistically
   powered temporal hold-out**; for the rest, report the trend but not confidence intervals.
6. **Neither ELEMENTA nor ICSD may be redistributed** (CC-BY-NC-4.0 / FIZ copyright plus the EU
   sui generis database right). A public benchmark can only be built on COD (CC0).

---

## Revision log

*(append only from here; nothing above may be rewritten)*

### Revision R1 · 2026-07-28 · measurement of `deff` / `ρ` (the §3.1 requirement to finish "before any candidate law is seen")

Script `src/measure_deff.py`, B=500 whole-cluster bootstrap. Output `features/deff.json`.

**`ρ` depends on the compression target, so it is registered per target rather than as a single
value** (which costs nothing, since `L_total` is only comparable within one compression target
anyway):

| compression target | K | m (Kish) | ρ [95% CI] | deff | N_eff |
|---|---|---|---|---|---|
| T_CE (ce_symbol departs from the mode) | 7,685 | 73.3 | 0.204 [0.191, 0.216] | 15.7 | 7,658 |
| T_CONN (site participates in edge/face sharing) | 8,018 | 72.3 | 0.608 [0.569, 0.647] | 44.3 | 2,897 |
| T_BV (\|bvs_dev\| > 0.2 vu) | 7,764 | 77.9 | 0.204 [0.194, 0.216] | 16.7 | 7,617 |

`m` uses the Kish weighted mean `Σm²/Σm` (with unequal clusters this is what m must be in
`deff=1+(m−1)ρ`; the arithmetic mean underestimates deff by a factor of 4). The negative control
(independent Bernoulli) gives ρ = −0.0005 and deff = 1.0, so the estimator is unbiased.

**Key §3.1 determination: the fallback clause is triggered, the argmin point estimate is abandoned,
and N is reported as an interval.** Three grounds:
(1) for the two site-level targets the argmin CI spans exactly 3 laws (8–11), sitting right on the
threshold with no safety margin;
(2) changing how missing prototypes are handled moves N\* from 3 (typed_only) to 13 (strict) —
**a modelling choice moves N\* more than sampling error does**, which is exactly what the §3.1
clause is meant to guard against;
(3) taking λ from 1 to 30 drops N\* from 10 to 1, far more than ±3.

**Registered N intervals (λ=1, hybrid as the main criterion)**: T_CE **N ∈ [5, 11]**, T_BV
**N ∈ [5, 11]**, T_CONN **N ∈ [3, 5]**. The lower end is the fixed criterion N80, the upper end is
the upper bound of the argmin CI.

**Control**: without the clustering correction (deff=1), N\* = 101. The clustering correction is
what compresses N from three digits to two, and it is the only one of the four layers of defence
with statistical justification. Confirmed by measurement.

**Two upstream numbers corrected in passing**:
- `icsd_meta.structure_type` hits **54.77%** (20,966/38,245) on the **analysis set**, not 78.13% —
  the latter was computed over all 203,830 entries, a different denominator. COD hits 0%, ICSD hits
  76.50%. There are 3,380 unique values, not 9,015.
- The `deff=20.2` assumed in plan §4.5.1 holds up numerically (typed_only measures 20.1–22.2),
  **but only because two errors cancel**: ρ was assumed 4× too high (0.80 vs 0.20) and m was
  assumed 4× too low (arithmetic 25 vs Kish 99).

### Revision R2 · 2026-07-28 · clarification of the §5 oxidation-state provenance

Measured: `_atom_type_oxidation_number` in the raw ICSD CIFs covers **27,408/27,408 = 100%**, and
the decoration in the blob agrees with the raw CIF at the site level to **100.000%** (847,608 sites
compared), with mixed valence preserved case by case rather than averaged away. But **only 1.76% of
COD entries carry a decoration, and its provenance cannot be verified** (no raw COD CIFs are held
locally); it may have been filled in upstream with BVAnalyzer — which PREREG §5 explicitly
excludes, so **what cannot be falsified is not trusted, and all of COD goes through `guess`**.

The `cif` level therefore actually covers **27,374/38,307 = 71.5%**, not all of it.
`cif + guess` together cover **96.11%**, above the 85% expected in §6.2. **BVAnalyzer was never
called at any point.**
