# NEXT533--NEXT536: SSSP framework-risk complement with sealed ODAC replication

Date frozen: 2026-08-13 (America/Chicago)

## Motivation and evidence accounting

NEXT31/NEXT47 established a repeatable, x0-only, zero-DFT short-contact law for
OMC25 molecular-crystal relaxation-energy response.  Its frozen transfer to
QMOF failed.  NEXT73 and NEXT79 established useful ODAC23 framework-displacement
ranking, but their high-risk tail did not clear the adaptive safety margin;
NEXT73 also failed the already-opened internal-validation tail-precision gate.

ODAC23 discovery and internal validation are now development data.  The 1,539
row robust internal-replication endpoint remains sealed.  No new law may open
that endpoint unless it first passes every two-partition readiness gate below.

The new physical hypothesis is narrow and non-duplicate: the framework risk
axis describes metal--donor extension, directional underconstraint and
heteroatomic topology, while SSSP measures whether local periodic charge-sign
shells are prematurely crowded.  Low SSSP may supply an independent bounded
risk increment without replacing the framework axis or treating an unsupported
formal-charge partition as evidence of safety or danger.

## Immutable executable boundary

Every executable feature receives composition and one raw, initial, fully
periodic x0 geometry only.  Frozen elemental/radius/bond-valence tables,
formal-valence inference, periodic graph/Voronoi operations and deterministic
linear algebra are allowed.

Forbidden executable inputs are DFT values or calculations, relaxed/later
geometry, trajectories, learned energy/force/stress, MLIP or proxy potentials,
and any physical or virtual relaxation.  DFT-relaxed displacement is an offline
label only after a prediction is frozen.

## NEXT533 label-free SSSP feature freeze

Compute unchanged NEXT411 SSSP for all 7,815 NEXT54 representative x0
geometries, including all three roles, before touching the internal-replication
endpoint.  Publish exact support/failure counts by role, source hashes and a
single feature table.  The robust-endpoint firewall must still state
`internal_replication_endpoint_values_summarized_or_inspected=false`.

This stage is authorized if each role has at least 20 finite distinct SSSP
values.  Coverage is reported, not required to be 0.95, because SSSP is an
optional bounded increment and missing SSSP contributes exactly zero.

## NEXT534 bounded two-partition development search

Use only already-opened robust discovery and internal-validation endpoints.
Never deserialize the internal-replication label file.

Start from the exact six-term NEXT79 framework formula, including its frozen
centers, scales, directions and weights.  Define the independent SSSP deficit

```text
D_S = max(0, 0.5231805323 - SSSP) / 0.5231805323
```

when SSSP is supported and `D_S=0` otherwise.  Search only

```text
R_new = R_NEXT79 + w * D_S
w in {0, 0.25, 0.5, 1, 2, 4}
```

and shared observed score thresholds.  No sign flip, feature substitution,
partition-specific threshold or additional term is allowed.

Each development partition must independently satisfy:

- coverage Wilson lower bound `>= 0.95`;
- protected-recall Wilson lower bound `>= 0.95`;
- severe-rejection precision Wilson lower bound `>= 0.70`;
- savings Wilson lower bound `>= 0.02`;
- pooled extreme AUC `>= 0.75`;
- macro defective/OMS-stratum AUC `>= 0.65`;
- worst stratum AUC `>= 0.55`.

In addition, the combined discovery+validation severe-rejection precision
Wilson lower bound must be `>= 0.80`.  Rank eligible formulas by minimum
partition precision lower bound, then minimum partition AUC, minimum protected
recall lower bound, combined severe recall, fewer nonzero added terms, smaller
weight and deterministic JSON order.

If no formula passes, stop without opening replication.  This failure does not
authorize expanding the grid on the same endpoint.

## NEXT535 replication feature/prediction freeze

Only if NEXT534 passes, freeze the exact formula, apply it to the already-frozen
internal-replication x0 features, publish predictions and hashes, and verify
again that replication labels have not been summarized or inspected.

## NEXT536 one-shot replication

Only after NEXT535 publication may the evaluator open the robust
internal-replication endpoint once.  The exact same seven per-partition gates
and combined-safety interpretation apply, with no repair or threshold change.
Pauling controls are reported if a compatible x0-only control is available;
they cannot rescue a failure.

If NEXT536 passes, write a new independent report distinguishing molecular,
framework and inorganic evidence.  Do not edit a canonical report, README,
preregistration, notes, paper or manuscript before user review.  If it fails,
preserve every artifact and continue with a non-duplicate mechanism or a new
evidence source.
