# NEXT32 design: a law for pre-screening the inorganic DFT response on OMat24

Date: 2026-08-03  
Status: frozen before implementation; new files only, with no modification to existing scripts,
reports or the paper

## Objective

NEXT32 goes straight for the inorganic crystal domain NEXT31 does not yet cover: given only a
generated or theoretically predicted unrelaxed periodic structure `x0`, can structures that would
produce a severe initial DFT force or stress be screened out with high precision before any DFT
calculation?

While the law executes it may use only the elements, the cell, the coordinates, the periodic
boundaries, the frozen element table, and deterministic geometry, Voronoi, bond-valence,
electrostatic and linear-algebra operations. It may not read or call DFT numbers, relaxed
structures, trajectories, same-composition candidates, MatterSim/MLIP, or any learned energy,
force or stress surrogate. DFT forces and stresses serve as labels only during development of the
law, or in the evaluation stage after the predictions are frozen.

Even if NEXT32 passes, it demonstrates only pre-screening of severe DFT single-point response;
that is not formation energy, the hull, kinetic stability, synthesizability, or a replacement for
DFT.

## Choice of data

The official [OMat24 data card](https://huggingface.co/datasets/facebook/OMAT24/blob/main/README.md)
states that the dataset contains DFT total energies, forces and stresses for inorganic
non-equilibrium structures in an ASE-compatible LMDB; the paper is
[Barroso-Luque et al., OMat24](https://arxiv.org/abs/2410.12771). The data licence is CC BY 4.0.

The three routes compare as follows.

1. **Independent perturbation sources within OMat24 -- adopted.** `rattled-relax` serves as
   exposed development only, and `rattled-300/500/1000` as three confirmation sources. Their
   endpoints agree, their material domain is inorganic bulk, and the archives can be sealed
   independently.
2. **Complete Alexandria/MP relaxation pairs -- later.** Energy descent is closer to the goal,
   but the download, trajectory identity and initial/final-state completeness cost more, and
   should not be mixed into the same round as the present single-point task.
3. **A NEXT23+NEXT31 routing cascade -- not adopted as scientific confirmation.** It can form an
   engineering pre-screener, but it adds no evidence about inorganic DFT response.

The `rattled-relax` validation source has 95,206 records and has already been opened for a schema
audit, so the whole source can only be called an exposed development source. It is a sample of
non-equilibrium frames and does not retain each complete relaxation trajectory; NEXT32 may not
construct an energy-descent primary endpoint from the first and last of incomplete frames.

The development cohort is drawn from `rattled-relax` by ordering on

```text
sha256("NEXT32-DEV-v1|" + parent_id + "|" + sid)
```

keeping at most one record per `parent_id` and taking the first 4,096. The selection reads
identity and geometry only, and uses no DFT label.

The confirmation archives are the as-yet-unopened `rattled-300`, `rattled-500` and
`rattled-1000`. Each source is first read with a geometry-only projecting parser for `sid`,
`parent_id`, atomic numbers, coordinates, cell and PBC, skipping the top-level
`energy/forces/stress` values. Every development `parent_id` is excluded, and previously selected
parents are excluded cumulatively across the three confirmation sources; each source takes 2,048
unique parents under a fixed salt, 6,144 in total. All three sources must complete their
features, Pauling comparators and prediction sealing together before any label is opened.

The raw LMDB holds geometry and labels together, so this remains a procedural isolation and the
manifest must record `physical_never_read_lockbox=false`; no physical never-read lockbox may be
claimed.

## The offline endpoint

For one DFT single-point record, define

\[
F_{\max}=\max_i\|\mathbf F_i\|,\qquad
F_{\rm rms}=\sqrt{N^{-1}\sum_i\|\mathbf F_i\|^2},\qquad
S=\|\boldsymbol\sigma\|_2.
\]

The severe-response label is fixed as

\[
y_+=1\quad\Longleftrightarrow\quad
F_{\max}\ge1.0\ {\rm eV/\AA}
\;\lor\;F_{\rm rms}\ge0.40\ {\rm eV/\AA}
\;\lor\;S\ge0.030\ {\rm eV/\AA^3}.
\]

The low-response structures to be protected are fixed as those satisfying all of

\[
F_{\max}\le0.50,\qquad F_{\rm rms}\le0.20,\qquad S\le0.015.
\]

These thresholds follow the severe-response magnitudes of NEXT26-NEXT28 and are frozen before the
confirmation labels are opened. Energy is a post-opening diagnostic only and enters neither the
formula, the selection, nor the primary gates.

## Analytic candidates

### Absolute periodic contact terms

Take the frozen tabulated covalent radius \(r_i\) for each atom. Enumerate the unique periodic
atom pairs satisfying \(d_{ij\mathbf n}/(r_i+r_j)\le1.60\), with no molecular 1-4 path exclusion.
Define

\[
q_{ij\mathbf n}=\frac{d_{ij\mathbf n}}{r_i+r_j},\qquad
\delta_{ij\mathbf n}=\max(0,1-q_{ij\mathbf n}).
\]

Emit dimensionless geometric quantities only: `cov_q01`, `cov_q05`, the per-atom count of
contacts with `q<0.85`, the per-atom squared overlap, and the 95th percentile and maximum of the
site overlap load. They are measures of geometric crowding; no potential energy is computed, no
force is derived and no virtual relaxation is performed.

### Already-validated analytic terms

Reused without modification from NEXT20-NEXT22:

- SIVR: `sivr_edge_mismatch_q95`, `sivr_site_imbalance_rms`, `sivr_cell_anisotropy`;
- normalized Madelung: the weak-binding direction of `nm_total_reduced`, and `nm_site_spread`;
- SCBVE: `scbv_mismatch_q95`, `scbv_vector_asymmetry_rms`;
- `abs(log(scbv_global_scale / median_dev))` as a closed-form two-sided scale mismatch.

Each term is robust-z scored using the development median and IQR alone. The risk direction is
written into the candidate table in advance; no arbitrary continuous weight is fitted, and no
tree, neural network or kernel model is used.

The candidates comprise single terms and equally weighted two-term sums with a clear mechanism,
at most two terms. The rejection fractions permitted are only
`{0.025, 0.05, 0.075, 0.10, 0.15}`; a missing required term fails open and rejects nothing.

## The development promotion gate

On the development set, the unique formula must satisfy all of these one-sided 95% Wilson lower
bounds:

- analytic coverage `>=0.95`;
- protective recall on low-response structures `>=0.98`;
- rejection precision on severe response `>=0.90`;
- DFT savings `>=0.05`;
- ROC AUC of the continuous risk against severe response `>=0.85`;
- the rejection-precision lower bound minus the one-sided 95% upper bound of the overall
  severe-response base rate `>=0.20`.

If several candidates pass, the unique formula is fixed in order by precision lower bound,
savings lower bound, AUC, fewer terms, smaller rejection fraction, and lexicographic order. If no
candidate passes, stop; do not download or open the confirmation labels in search of a second
chance.

## Confirmation and the comparison with Pauling

Once the unique formula is promoted, freeze the formula, the normalisation constants, the
thresholds, the missing-value policy, the confirmation IDs and the evaluation protocol. The three
confirmation sources first generate non-overwritable predictions and the fixed Pauling 2-5
controls together, and only then are the DFT forces and stresses opened.

Overall confirmation uses the same six development gates; in addition, each source must satisfy:
coverage lower bound `>=0.90`, protective recall on low response lower bound `>=0.95`, rejection
precision lower bound `>=0.75`, savings lower bound `>=0.02`, and AUC `>=0.75`. The per-source
gates prevent one high-positive-rate source from masking a transfer failure.

Only if NEXT32 clears the overall gates and every per-source gate, and Pauling 2, 3, 4 and 5 and
the joint control all fail the overall primary gate on the same cohort under the same fail-open
semantics, may we write:

> On the severe-initial-DFT-response endpoint across three inorganic perturbation sources in
> OMat24, NEXT32 surpasses this project's fixed operational Pauling 2-5 comparators.

Even then, it may not be written as surpassing Pauling across the board, as predicting hull
stability, or as reaching or exceeding DFT energies.

## Handling failure, and the artefacts

- Every directory is published once and never overwritten; SHA-256 hashes are stored for the
  inputs, code, rules, predictions and endpoints.
- A failure before the labels are opened may be fixed as an engineering defect and a new version
  directory regenerated; after the labels are opened, the formula, thresholds, confirmation
  cohort and gates may not be changed.
- A development failure keeps the sweep results; a confirmation failure keeps the frozen formula
  and the evidence of failure, with no refitting.
- In the end only NEXT32 code, tests, external data artefacts and a standalone report are added;
  the paper, README, PREREG and old reports are not modified before the user confirms.
