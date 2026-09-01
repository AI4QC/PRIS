# NEXT173--174 Graph-Weighted Local Directional Rigidity

## Physical hypothesis

NEXT168 treated every edge in a fixed Voronoi or CrystalNN graph equally even
though NEXT19 already freezes a positive `neighbor_weight` for each edge.
Equal weighting makes a weak peripheral contact contribute as much as the
graph algorithm's dominant contacts. The new hypothesis is that protected
structures have a locally complete *salient* contact frame, while severe
structures may obtain an artificial unweighted 3D frame from weak contacts.

## NEXT173 label-free feature freeze

For each site `i`, incident unit direction `u_ie`, and frozen graph weight
`w_e > 0`, define

```text
G_i^w = sum_e w_e u_ie u_ie^T / sum_e w_e
T_i^w = 3 lambda_min(G_i^w)
V_i^w = 27 det(G_i^w).
```

The trace of `G_i^w` is one, so both certificates remain bounded in `[0,1]`.
They are invariant to rigid rotation, uniform scaling of all graph weights,
edge orientation, and input ordering. Isolated sites receive zero. No new
neighbor cutoff, fitted exponent, endpoint-derived constant, DFT value,
learned proxy, or relaxation is used.

Freeze five summaries for each unchanged NEXT19 graph mode (Voronoi and
CrystalNN): site minimum, inverted-CDF q10, and site mean of `T_i^w`; q10 and
mean of `V_i^w`. Build discovery feature tables without accepting any endpoint
path. Fail open independently by graph mode.

## NEXT174 discovery-only audit

Only after NEXT173 features and manifests are frozen, audit exactly ten
predeclared high-direction hypotheses in the unchanged NEXT164 repair shell:

- higher weighted tightness minimum, q10, and mean;
- higher weighted volume q10 and mean;
- each for Voronoi and CrystalNN.

Use the same eligibility gates as NEXT169:

- feature support at least 0.90 in both sources;
- SCIGEN shell AUC at least 0.55 in every fixed fold;
- WyFormer shell pooled AUC at least 0.55;
- full protected-versus-severe pooled AUC at least 0.50 in each source.

No law, feature cutoff, graph mode, direction, or weight is selected before
the audit. If no hypothesis is eligible, terminate the weighted-rigidity
branch. If one or more are eligible, freeze a separate finite formula-search
plan before evaluating any formula.

## Boundaries and outputs

- Executable descriptors are structure-only and pre-DFT.
- Discovery outcomes may be used only as offline audit labels in NEXT174.
- No validation or replication endpoint or geometry is opened.
- No DFT calculation/value, learned energy/force/stress proxy, or physical
  relaxation.
- Preserve every prior script and artifact; add new NEXT173/NEXT174 files.
- Append results only to the standalone report before any canonical edits.

NEXT173 publishes atomically under
`$PRIS_ARCHIVE/next173_weighted_local_directional_rigidity_v1`.
NEXT174, if executed, publishes under
`$PRIS_ARCHIVE/next174_weighted_local_directional_rigidity_audit_v1`.
