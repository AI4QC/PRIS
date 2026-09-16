# MatterSim two-capacity consensus and recovery gating design

Date: 2026-08-01

## 1. Objective and non-objectives

This round tests one limited question: at the x0 inference stage, without reading any DFT
endpoint label, can the same-composition relative-energy consensus and disagreement of MatterSim's
1M and 5M checkpoints screen out high-energy candidates more safely than the current 5M x0
baseline? If quality really does improve, the group-level adaptive few-step strategy already found
in next7 is then added as a cost-reduction layer.

This round does not call two highly correlated checkpoints an independent ensemble, and does not
claim MLIP consensus to be a new universal physical law. Success means only finding a pre-DFT
screening candidate worth taking to a new source for validation. The old scripts, old outputs,
existing reports, the paper, the README and PREREG are all preserved; only `next8` files and a
standalone report are added.

## 2. Existing evidence and the rationale for the design

next7 showed that a fixed 8-step FIRE raises the historical screening rate by only `0.0685`
percentage points over 5M x0, far below the preset `+3` percentage-point gate; but group-level
adaptive replay reproduces the S8 decisions 100% and cuts the mean force evaluations from 9 to
1.815. So a few-step trajectory is suited to being a conditional-computation layer, and is not by
itself a better quality criterion.

The literature also does not support "high Fmax means a bad structure". A high force may indicate
that relaxation still has room to descend; and a single universal MLIP may additionally soften
systematically on high-energy or OOD configurations. This round therefore rejects only when both
models give a high relative energy and their disagreement is bounded; high disagreement always
ABSTAINs and goes to DFT.

## 3. Data and isolation

### 3.1 Development data

Continue with the fixed three-stage ELEMENTA development, but before any endpoint label is read,
sort the composition groups of `threshold_calibration` by the SHA-256 of a frozen salt and split
them physically into two disjoint subsets:

- `search_calibration`: fixes the label-free disagreement quantiles and each candidate's safety
  thresholds only;
- `formula_selection`: selects one formula from the frozen candidate catalogue;
- `threshold_calibration/threshold_fit`: recalibrates the final thresholds for the selected
  formula and M5 only;
- `threshold_calibration/development_gate`: applies the final thresholds and computes the
  improvement gates only, with no further fitting or selection.

The frozen salt is `next8-threshold-fit-gate-v1-20260801`. After a stable sort on
`sha256(salt + "\0" + rk)`, the first half of the composition groups go to `threshold_fit` and the
second half to `development_gate`; complete `rk` groups are kept on both sides. Valid groups for
the primary track must be counted by the current formula itself: the group has at least one row
that is `protected & supported & finite(score)`. If `threshold_fit` has fewer than 99 such valid
groups for the primary track, the primary-track threshold can only be keep-all-supported; the
`-inf` of unsupported groups may not be used to make up the count, and gate data may not be merged
in to supplement the sample. Rows where the model failed or the value is non-finite still ABSTAIN
and may not become KEEP through keep-all.

The feature process may read only x0 species, cell, PBC, coordinates, `sid/rk/material/stage` and
the strict input flags. It may not read the endpoint energy, the final ionic step, the suffix, DFT
forces/stresses or the stability labels.
A formal freeze must validate the feature manifest before any endpoint label is read or hashed:
`production_protocol_eligible` must be exactly boolean `true`, `evidence_role` must be
`protocol_feature_generation`, the adapter must be the source-reviewed `builtin_mattersim`, and
its actual implementation source path and SHA must lie within the executed-source hash closure;
artefacts from a test-injected predictor may not enter a formal freeze. The hashing and the parsing
of the feature parquet, the feature manifest and the label parquet must each come from the same
immutable byte snapshot (or the same fixed file descriptor); hashing by path and then reparsing a
mutable path afterwards is not allowed. The label snapshot may be created only after full feature
validation, derivation of the search cutoffs and the threshold-role split. Before publication, the
original paths are rehashed and must equal the snapshot hashes.

### 3.2 The historical test set

The ELEMENTA test set has been exposed repeatedly, and may be used for one historical
counter-check only after the code, formula, thresholds and checkpoints are all frozen. However
good the numbers, they may only be written as retrospective discovery, never as confirmatory.

### 3.3 External data

- Bartel's `matgen_baselines`, commit
  `770129797a9919955d84f3c3e59cc389e3b04315`, contains 500 DFT decomposition-energy labels for
  each of six generation/template methods, but this round's audit has already opened the CSV and
  the CIFs are very likely DFT-relaxed final states; it can serve only as an external
  ceiling/falsification.
- The 8 ASE-LMDBs and 955,135 configurations of the OMat24 validation set have not yet had their
  DFT payload opened; they suit an audit of the energy, force, recovery and uncertainty mechanisms
  only, and are not a confirmation of hull stability.
- Genuine outer confirmation should use a physically isolated Alexandria 2025 time slice, or a
  new-generator x0 -> uniform DFT endpoint batch with randomised IDs. Every candidate must have a
  label; computing only the structures the model selects is not allowed.

## 4. Runtime and model identity

Development features must recompute 1M and 5M in the same process and the same Torch/CUDA
environment; new 1M output may not simply be concatenated with 5M output from an older runtime.
The models are fixed as:

```text
MatterSim package = 1.2.3
1M checkpoint SHA-256 = 28b0b0b0f13efefee06b47ea4c9105a26bd3e2c8396da193430da96b3b49a8be
5M checkpoint SHA-256 = e3df9fa708725e3d453140646c7d1838324b347a3d1214cf1440522146f872b5
```

The manifest must record Python, Torch, CUDA, the GPU, package versions, both checkpoint paths and
hashes, the input hashes, the inference counts, wall time, peak CUDA memory and the source-code
hashes. Output is published atomically with no-replace; if the target directory already exists it
fails closed.

## 5. The frozen candidate catalogue

For each supported structure and each checkpoint, first compute within the same `rk` composition
group:

\[
g_m(i)=e_m(i)-\min_{j\in rk(i)}e_m(j),\qquad m\in\{1M,5M\}.
\]

Define the energy disagreement `u_E=|g_1-g_5|`, and use inter-model force disagreement -- not high
absolute force -- as the OOD proxy:

\[
u_F=\max\left(|F_{\max,1}-F_{\max,5}|,
               |F_{{\rm rms},1}-F_{{\rm rms},5}|\right).
\]

Only the following eleven candidates with no fitted weights are permitted. The first nine keep
their original order; before any endpoint label is opened, two candidates that aggregate first and
zero afterwards are appended on the basis of a label-free mathematical audit:

1. `M5 = g5`;
2. `M1 = g1`;
3. `MIN = min(g1,g5)`;
4. `MEAN = (g1+g5)/2`;
5. `MAX = max(g1,g5)`;
6. `LCB = max(0, (g1+g5)/2-u_E)`;
7. `AGREE99 = MEAN`, but ABSTAIN when `u_E` exceeds the label-free q99 of search-calibration;
8. `AGREE995 = MEAN`, but ABSTAIN when `u_E` exceeds the label-free q99.5;
9. `AGREE_EF995 = MEAN`, but ABSTAIN when `u_E` exceeds the label-free q99.5 or `u_F` exceeds the
   label-free q99.5;
10. `CMEAN = c`, where
    \(h_i=0.5g_1(i)+0.5g_5(i)\) and
    \(c_i=h_i-\min_{j\in rk(i)}h_j\). This is equivalent to averaging the two models' absolute
    per-atom energies first and zeroing within the same `rk` afterwards, and is not equivalent to
    `MEAN`, which retains the group-common offset produced by a disagreement in the two models'
    argmin;
11. `CMEAN_JOINT99 = CMEAN`, but ABSTAIN when the label-free joint disagreement score `J` below
    exceeds the q99 of search-calibration.

`CMEAN_JOINT99` uses only the joint-complete rows of search-calibration, building a row-weighted,
right-continuous empirical CDF separately for `d_E=|g1-g5|`, `d_Fmax=|Fmax1-Fmax5|` and
`d_Frms=|Frms1-Frms5|`:

\[
H_k(x)=n^{-1}\sum_{\ell=1}^{n}\mathbf 1[d_k(\ell)\le x],\qquad
J=\max(H_E,H_{F\max},H_{F\mathrm{rms}}).
\]

Freeze `q_J=quantile(J,0.99,method=higher)`, and ABSTAIN only when `J>q_J`, with equality KEEP.
`n`, the joint-complete `n_rk`, `weighting=row`, `side=right`, the quantile method, `q_J`, and the
three ordered reference distributions with their hashes must all be serialised; storing `q_J`
alone and then guessing the thresholds on the original quantities is not allowed. With `n` eligible
rows in search, this gate's additional empirical abstentions are at most
`floor(0.01*(n-1))`, a fraction strictly below 1%; that excludes incomplete-group abstentions and
gives no equivalent guarantee for later stages.

Once the candidate catalogue is frozen, no continuous weight, per-element threshold, suffix,
material identity or post-hoc model may be added. The two models share a training regime, so
`u_E/u_F` is only a proxy for capacity disagreement, not a calibrated epistemic uncertainty.
Both cutoffs of `AGREE_EF995` use `method=higher`; the union of the two 0.5% tails caps the
additional abstention on the search-calibration empirical sample at about 1%.

## 6. Selection, thresholds and the success gates

The primary track keeps `valuable <= 50 meV/atom`, `within_group=max`, `alpha=0.01`. The
historical comparison track keeps `near_min <= 1 meV/atom`, `within_group=min`, `alpha=0.035`.
Unknowns, a failure of any required model, and incomplete composition groups all fail open to
ABSTAIN: `M5` depends only on a complete 5M group and `M1` only on a complete 1M group, while the
remaining formulas require both models to be complete -- an M1 failure must not drag down the M5
baseline.
`AGREE99/995/EF995/CMEAN_JOINT99` ABSTAIN only the target rows whose disagreement is out of
bounds; the gaps of complete groups and the `CMEAN` re-zeroing are still computed from every
finite model output, and the within-group minimum may not change because high-disagreement rows
were removed first.

Each candidate calibrates its thresholds on search-calibration; on formula-selection it must first
satisfy: exact-min retention 95% lower bound `>=0.95`; the primary track's valuable-group (the
comparison track's near-min) retention 95% lower bound `>=0.95`; `regret_p95<=0.05 eV/atom`; and
`all_rejected_groups=0`. It is then ordered by:

```text
dft_savings descending
measured/evaluation cost ascending
formula complexity ascending
catalog order ascending
```

The deterministic cost units are fixed as `M1=1`, `M5=5` and both models `=6`; complexity units
are fixed as single model `=1`, `MIN/MEAN/MAX=2`, `LCB`/a single disagreement gate/`CMEAN=3`, an
energy+force two-cutoff gate `=4`, and the three-ECDF joint gate `CMEAN_JOINT99=5`. A noisy wall
time from this run is not used as a tie-break. The primary track selects the single production
formula; the comparator only rechecks that same formula and does not independently select a second
winner.

After selection, `threshold_fit` recalibrates only the selected formula and the M5 baseline, and
`development_gate` applies only those two final rules. The improvement gates against M5 are
computed on development_gate alone and must all hold:

- DFT savings increase by at least `0.03` in absolute terms, with a composition-paired 95% CI
  lower bound above 0;
- the 95% CI lower bound of the valuable-item recall difference is not below `-0.005`;
- exact/near-min retention continues to clear its own lower bounds;
- the increase in abstention rate is at most `0.01`;
- no composition group is rejected in its entirety.

If development does not clear the `+3` percentage-point gate, stop the quality line: do not open
the OMat24 payload, do not add a third model, and report the negative result only. Only if
development passes is the evaluator frozen and one historical test run; and only if an outer test
on a new source simultaneously reaches savings lower bound `>=30%`, stable recall lower bound
`>=99%` and group-min retention lower bound `>=95%`, while beating the Pauling/BVS/Ewald/geometric
and single-model MLIP baselines, may a change to the formal paper be requested.

## 7. The adaptive recovery layer

Adaptive few-step plays no part in this round's quality-formula selection. It is enabled only
after the committee quality gate passes: the whole `rk` group advances in step, and at k=0/2/4 it
advances to the next snapshot only if some structure lies within the `1.0/0.75/0.5 meV/atom`
boundary band of some frozen threshold, or if the current `Fmax>0.15 eV/Angstrom`; otherwise it
stops. Those thresholds come from an already-exposed historical replay and must be pre-registered
before a new batch. Cost conclusions report the actual eval count and the wall time separately; a
saving in count may not be presented as a saving in GPU time.

## 8. Artefacts and stopping conditions

The new root directory is `outputs/20260801_mattersim_committee/`, containing at least the
development features, the development freeze, an optional historical test, the diagnostics and a
fully hashed manifest. The standalone report goes to
`reports/2026-08-01-mattersim-committee-followup.md`.

Any of the following fails closed immediately: a stage boundary violation, an inconsistent input,
model or code hash, a duplicate `sid`, an incomplete group, a non-finite energy, an existing target
output, a label read before the freeze, or a test result triggering a new candidate or a threshold
sweep.
