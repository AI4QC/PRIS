# NEXT39 OMat24 trajectory confirmation design

## Objective

Test whether the already frozen NEXT23 B+E rule transfers, without refitting,
from WBM to an independent OMat24 structure-optimization source.  The rule must
remain executable from one raw step-0 structure and frozen elemental tables.

## Scientific boundary

- Prediction may read only identity metadata and the step-0 composition, cell,
  species, and coordinates.
- Prediction may use deterministic analytic geometry, valence assignment,
  SIVR, SCBVE, and fixed Pauling 2--5 controls.
- Prediction may not read DFT values, later structures, trajectories, learned
  energies/forces/stresses, physical relaxation, or alternative structures.
- Later coordinates are opened only after the prediction artifact and all
  source/input hashes are sealed.
- The later frame is the latest observed sampled frame, not a claimed converged
  relaxed endpoint.

## Frozen cohort

Parse OMat24 trajectory identities using the exact suffix pattern
`<trajectory_stem>_<integer_step>`.  Include a trajectory only when:

1. its task type is exactly `Structure Optimization`;
2. it contains step 0;
3. its maximum observed step is at least 20; and
4. the selected trajectory is parent-unique.

When more than one eligible trajectory exists for a parent, select the minimum
SHA-256 ordering under a fixed salt.  Store step-0 geometry only.  Store the
record key and step number of the latest observed frame as opaque future
evaluation identity, without decoding that frame's geometry or numeric fields.

## Frozen prediction

Use the immutable NEXT23 rule artifact and verify its SHA-256 plus the hashes of
the frozen SIVR, SCBVE, and scoring kernels.  Compute:

`score = (B - median_B) / IQR_B + (E - median_E) / IQR_E`

where `B` is the charge-exponent-zero Voronoi SIVR cell anisotropy and `E` is
SCBVE vector-asymmetry RMS.  Reject only when both terms are supported and the
score is at least the frozen threshold.  Unsupported rows fail open.  Apply the
unchanged Pauling P2--P5 controls on the same step-0 structures.

## Blind evaluation

After verifying the sealed prediction and cohort hashes, decode only the latest
observed frame geometry for each selected trajectory.  Compute the exact
Matbench Discovery structure fingerprint used by NEXT23:

- `CrystalNNFingerprint.from_preset("ops")`;
- `SiteStatsFingerprint(..., stats=("mean", "std_dev", "minimum", "maximum"))`;
- Euclidean distance between step-0 and latest-observed fingerprints.

Use the frozen operational cutoffs: protected `<= 0.10`, changed `> 0.10`,
substantial `>= 0.20`, severe `>= 0.50`.  The primary success decision is the
conjunction of the existing one-sided 95% Wilson lower-bound gates:

- coverage `>= 0.90`;
- protected recall `>= 0.95`;
- rejection precision for changed structures `>= 0.90`;
- savings `>= 0.10`.

No threshold, feature, cohort, or cutoff may change after later geometries are
opened.  Pauling P2--P5 is reported as a fixed comparator, not used to modify
the NEXT23 decision.

## Deliverables

All work is additive: three new scripts, focused tests, immutable external data
artifacts, and one standalone report.  Existing scripts, reports, paper,
README, and preregistration files remain unchanged.
