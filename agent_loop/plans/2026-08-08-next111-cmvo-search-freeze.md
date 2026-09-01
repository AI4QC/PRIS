# NEXT111 CMVO Discovery Search Freeze

Date: 2026-08-08

## Boundary and ordering

This file freezes the complete NEXT111 optional-term grammar before either
SCIGEN or WyFormer discovery endpoint payload is opened by NEXT111.  All term
eligibility, centers, scales, caps, directions, weights, base selection rules,
candidate order, and gates below were determined from raw-structure features
and immutable prior search provenance only.  Validation and replication remain
physically unopened.

The executable remains strictly pre-DFT.  DFT outcomes are used only as the two
already-isolated discovery labels after this freeze; they are not executable
features.  No DFT calculation, relaxed structure, trajectory, energy, force,
stress, learned proxy, MLIP, or same-composition alternative may enter a term.

## Immutable inputs

| input | SHA-256 |
|---|---|
| NEXT109 design | `fb74b3ed4ac1aef153891e5f93ac025af0d745567d57d02ef04fd1133a204c3e` |
| NEXT110 manifest | `06000213e80de7afa2e13f9dd67561ff2b56a9a10fede90260c269ad57dc03b3` |
| NEXT110 feature catalogue | `5a9e66a87779555f91019ac1873a5b2974154e51b2911986b3911c5d69b5ac01` |
| NEXT110 SCIGEN features | `023c0662fabd73df0a7f47c1e10dc7e229fb0b5cde6f2d76c34c3c6efc1bb31e` |
| NEXT110 WyFormer features | `fd225555c8cadd2219df6fec679c74c78a9a5c15065f23553d7e6d1eec681c94` |
| NEXT108 manifest | `005f276ab2627863f8f942e2e15ab8a3c2b9868439ca00c4782d8ff94eeefda5` |
| NEXT98 manifest | `5fcd924b125767e52ac1826203595692af868ab35366899e12b82aea2726e32c` |
| NEXT98 term catalogue | `f2165f548a56cda04559a11a0d575f0654d3e8a17cf3b85b76e7974ea65dee41` |
| NEXT98b manifest | `b20d2f500ce74a6fd8b1a8a992bca3fff3ee5952fc38c09d3ad34ca317c3084d` |
| NEXT98b search records | `748a4623ecfc725636837f3944b70482a97b2df39a495a81e3f8e09f5d09a4e4` |
| SCIGEN analytic discovery features | `7031d86e4fb6e469c674d208f680ace1dbe5e11e45f3d4b2befefd747efdde16` |
| SCIGEN discovery endpoint | `f86cff6f5e9124ee82aae13911ffe55a125c6fe111fc1f64122a610febf67958` |
| WyFormer analytic discovery features | `c515baec0fccef5bc03c7672f1d4e1aca278f5ed4d7b6f1bf7f66c734e2b87f7` |
| WyFormer discovery endpoint | `f39836e62a1da03ed823479e87d6f75fc0d01da60a8c0a2faa696638cc2fb9d7` |

The endpoint hashes are prior provenance identities only.  Their contents were
not read to choose anything in this file.

## Label-free eligibility audit

All eight core/expanded CMVO columns exceeded the frozen 15% finite-coverage
floor in each source and had at least eight unique finite values.  The frozen
transform also requires a positive pooled IQR because the risk is divided by
that IQR.  Five terms have IQR exactly zero and are therefore arithmetically
ineligible; no epsilon or endpoint-informed fallback is allowed.

| feature | SCIGEN coverage / unique | WyFormer coverage / unique | pooled IQR | decision |
|---|---:|---:|---:|---|
| `cmvo_core_min_interval_slack` | 0.571343726800297 / 1,151 | 0.7113914373088684 / 463 | 0.24999999999999986 | eligible |
| `cmvo_core_global_balance_gap` | 0.571343726800297 / 269 | 0.7113914373088684 / 120 | 0.4666666666666667 | eligible |
| `cmvo_core_component_balance_gap` | 0.571343726800297 / 249 | 0.7113914373088684 / 103 | 0.5 | eligible |
| `cmvo_core_unserved_site_fraction` | 0.571343726800297 / 53 | 0.7113914373088684 / 39 | 0 | excluded |
| `cmvo_expanded_min_interval_slack` | 0.8310319227913883 / 717 | 0.8686926605504587 / 304 | 0 | excluded |
| `cmvo_expanded_global_balance_gap` | 0.8310319227913883 / 187 | 0.8686926605504587 / 82 | 0 | excluded |
| `cmvo_expanded_component_balance_gap` | 0.8310319227913883 / 174 | 0.8686926605504587 / 68 | 0 | excluded |
| `cmvo_expanded_unserved_site_fraction` | 0.8310319227913883 / 39 | 0.8686926605504587 / 37 | 0 | excluded |

## Frozen physical terms and transform

For eligible raw feature `x`, define a high-is-risk optional term

\[
h(x)=\min\left[c,\max\left(0,\frac{x-m}{s}\right)\right].
\]

Unsupported or nonfinite rows have `h=0`; the base law support mask is copied
unchanged.  The three exact constants are:

| term ID | raw feature | `m` | `s` | raw p99.5 | normalized cap `c` |
|---|---|---:|---:|---:|---:|
| `cmvo_core_min_interval_slack__high` | `cmvo_core_min_interval_slack` | 0.07142857142857142 | 0.24999999999999986 | 0.5587653898768803 | 1.9493472737932367 |
| `cmvo_core_global_balance_gap__high` | `cmvo_core_global_balance_gap` | 0.14285714285714285 | 0.4666666666666667 | 0.9 | 1.6224489795918366 |
| `cmvo_core_component_balance_gap__high` | `cmvo_core_component_balance_gap` | 0.14285714285714285 | 0.5 | 1.0 | 1.7142857142857144 |

The implementation may use a reversible `sinh`/`asinh` encoding only to pass
`h(x)` through the already-validated evaluator.  That encoding must recover the
numbers above to numerical precision and is not an additional transform.

## Frozen base pool and candidate grammar

Apply the unchanged NEXT108 near-miss rule to immutable NEXT98b records: all six
source AUCs must be no more than 0.01 below their gates.  Exactly 353 bases must
result.  Each base retains its old one-to-three terms and coefficients exactly.

For every base enumerate:

- the unchanged base once;
- each of three CMVO terms singly with weight `{0.25, 0.5, 1, 2, 4}`;
- each unordered pair of distinct CMVO terms with independent weights
  `{0.25, 0.5, 1, 2}`.

No triple, negative sign, expanded term, unserved term, CMVF reweighting, base
coefficient change, or continuous optimization is permitted.  The count is

`1 + 3*5 + C(3,2)*4*4 = 64` per base and `353*64 = 22,592` total candidates.

Configurations and candidate keys are sorted lexicographically.  Any count or
identity mismatch aborts before endpoints are read.

## Unchanged discovery gates and stop rule

Use the exact NEXT98/NEXT103 evaluator and gates:

- both-source pooled, macro-crystal-system, and worst-crystal-system AUC gates;
- SAFE gates in the two source aggregates and ten fixed reduced-formula folds;
- one shared BROAD threshold that Pareto-dominates Pauling in all 12 cells.

Candidate selection and tie breaks are unchanged.  If no candidate passes every
gate, NEXT111 stops, keeps all validation and replication outputs closed, and
writes a standalone negative/partial report.  If one passes, its full formula,
thresholds, calibration, candidate hash, and row predictions must be frozen
before any isolated replication endpoint can open.
