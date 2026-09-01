# NEXT15 Basin-Hull additive design

## Outcome sought

Test a deployable DFT-before-screening rule that addresses the failure mode of
NEXT14: local negative curvature alone rejects too many structures that relax
into a stable basin.

## Frozen rule before the 2048-row run

1. Start only from the sealed NEXT14 geometry-only WBM holdout.
2. Relax positions and cell with MatterSim 1.2.3, the sealed 5M checkpoint,
   FIRE, `FrechetCellFilter`, `fmax=0.05 eV/A`, and at most 64 predictions.
   Inference is packed with MatterSim's default 512-atom batch budget; this is
   an execution-memory parameter and does not alter per-structure trajectories.
3. Evaluate the last finite predicted total energy against the uncorrected
   Materials Project reference `PatchedPhaseDiagram` built only from chemical
   subspaces needed by the holdout.
4. Define the Basin-Hull score

   `B64 = E_MatterSim,64/N - E_MP-hull(composition)`.

5. `REJECT` exactly when `B64 >= 0.20 eV/atom`; `KEEP` below the boundary;
   numerical, composition, cell, or support failures are `ABSTAIN`.

The `0.20 eV/atom` threshold is the already frozen high-energy boundary, not a
threshold selected from NEXT14 WBM labels. No PHSC/CHSC/ACSC or prior artifact
is replaced.

## Evidence boundary

NEXT14 already opened the full WBM test label table before NEXT15 was proposed.
Therefore the NEXT15 WBM result is retrospective even though its feature script
must not read any WBM endpoint label. A positive WBM result is a promotion
signal only; it requires validation on another external source before a new
scientific report is written.
