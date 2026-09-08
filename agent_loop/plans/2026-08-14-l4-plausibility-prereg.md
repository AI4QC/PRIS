# PREREG-L4: an interpretable law set reaching >=0.80 on both plausibility axes (extending L3 with new physical predicates)

Date frozen: 2026-08-14 (written before the augmented matrix features were computed and before
any predicate search).
Status: **frozen**; only a "revision log" may be appended.

> **Translation note.** This file was translated into English on 2026-09-08. The frozen
> Chinese original is the version hashed as `199788c4988da1b25d1a8c960f200e92fad91ebe84c85f707d0a2be8fa0c2ba1`
> in `agent_loop/frozen/20260814_f3_synth/PREREG_SHA256`, and remains recoverable from the git
> history. Nothing below has been changed apart from the language.

## 0. Motivation

L3 stands at 0.9171 / 0.7004 on calibration: its exclusion power is below 0.80. A certified
depth-3 tree proves (at cost ratio 1.0) that an operating point of satisfaction 0.8895 /
exclusion 0.8553 exists within this feature space, but the tree is a fingerprint of the damage
operators (it collapses under LOPO) and cannot be written as a one-line law. This chain tests
whether adding the classical Born and symmetry features of PREREG-F3 section 4 to L3 as
**one-line physical predicates** can reach >=0.80 on both axes in an interpretable form.

## 1. Data (in existence before this pre-registration)

`next20260801/law_{real,bad}.parquet` (audited isolated, no lockbox: 17,929 real / 12,202
perturbed; discovery 12,632+8,590, calibration 5,297+3,612). The augmented features are computed
from the blob records of `records_{real,bad}.parquet` plus a deterministic regeneration through
`make_negatives.perturb` (exactly the recipe of `phys_law._bad`: `seed_of(sid)`, the fixed S1-S5
order, a shared rng, `swapped_val`), with definitions word for word those of PREREG-F3 section 4,
written to the new files `law_real_aug.parquet` / `law_bad_aug.parquet`. Missing values count as
satisfied, following the paper's convention.

## 2. Procedure (frozen)

- **L4 = L3 (the five predicates unchanged, thresholds untouched) + at most 4 new predicates.**
- A candidate predicate is (feature, direction, threshold, guard): the feature is drawn from the
  augmented features together with the existing 79; the threshold from that feature's quantile
  grid over the discovery **real** rows
  {0.5,1,2,3,4,5,10,20,...,90,95,96,97,98,99,99.5}%; and the guard from {none, fi>0.50, fi>0.55}.
- Greedy: at each step take the candidate with the largest gain in pooled exclusion, subject to
  discovery satisfaction >= 0.81; stop when the gain falls below +0.005 or 4 predicates have been
  added. Once selected it is frozen and its SHA-256 recorded.
- Also reported (but not used for selection): a beam-search comparison with the same number of
  rules starting from scratch, and the frontier points of the certified trees.

## 3. Gates (calibration, one frozen evaluation; any reuse of calibration is disclosed on the paper's convention)

- **C1**: satisfaction >= 0.80
- **C2**: pooled exclusion >= 0.80
- **C3**: exclusion >= 0.55 on each of the five classes (preserving the "no blind spot" property)
- **C4 (disclosure)**: LOPO for each new predicate: reselect with its target class removed and
  measure exclusion on that class; report it together with the denominator of each class
- On failure, report the operating point actually reached; adjusting a threshold afterwards and
  re-evaluating is forbidden.

## 4. Boundary statements (frozen)

- This chain addresses exclusion power against synthetic perturbations; the LOPO disclosure makes
  the "operator fingerprint" risk transparent, and no claim of generality across operators is
  made.
- A new predicate must be expressible as a one-line physical statement (Born like-charge
  repulsion sum, contact wall, elastic strain, Wyckoff parsimony); an uninterpretable feature is
  not selected even at a higher gain (this defines the model class and is not a post-hoc choice).
- The lockbox is not touched.
