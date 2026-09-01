# NEXT565--NEXT568 HEA mechanism-calibrated composition laws

## Rationale and execution boundary

The opened 4,400-row development set shows that the fixed extreme-waste OR
endpoint contains two weakly overlapping mechanisms: `e_above_hull >= 0.4`
and large DFT relaxation response.  High ideal composition entropy predicts
the first, while high coefficient of variation of tabulated atomic mass or
atomic number predicts the second.  The executable laws use only composition;
they never use DFT values, final structures, trajectories, model potentials,
or coordinate changes.

For a frozen candidate batch let `u_H` be the midrank percentile of ideal
composition entropy, `u_M` the percentile of weighted atomic-mass coefficient
of variation, and `u_Z` the percentile of weighted atomic-number coefficient
of variation.  Freeze exactly three candidates:

1. `MEPU24 = 1-(1-u_H^2)(1-u_M^4)`;
2. `ZEPU24 = 1-(1-u_H^2)(1-u_Z^4)`;
3. `MEMAX = max(u_H,u_M)`.

No further formula, exponent, threshold, or feature may be introduced after a
new endpoint is opened.

## NEXT566 selection cohort and one-time opening

Exclude all NEXT551 and NEXT560 identities.  From remaining label-free source
rows, select the lowest `SHA256("NEXT566-v1|"+fid)` 2,000 ordered and 2,000 SQS
identities.  Publish sanitized x0 geometries and all three predictions before
opening endpoints.  Then open endpoints only for those 4,000 FIDs and apply
the unchanged NEXT555 extreme-waste definition.

Each class must contain at least 200 rows and at least 75 rows must be
protected.  A candidate is selectable only with coverage at least 0.99;
overall AUC at least 0.72 and chemical-system bootstrap lower bound at least
0.68; ordered and SQS AUC at least 0.66; severity Spearman at least 0.35 with
bootstrap lower bound at least 0.25; top-15% lift at least 1.50 with protected
fraction at most 0.02; overall AUC at least 0.02 above each raw component; and
the NEXT561 Pauling comparison rule.  Select by the worse ordered/SQS AUC,
then overall AUC, then lexical candidate name.  This selection is not final
confirmation.

## NEXT567--NEXT568 independent confirmation

Exclude NEXT551, NEXT560, and NEXT566 identities.  Select the next 2,000
ordered and 2,000 SQS identities by the lowest
`SHA256("NEXT567-v1|"+fid)`, freeze only the selected formula, then open only
those endpoints once.  Apply exactly the same class, coverage, AUC, cluster,
family, severity, top-tail, component-margin, and Pauling gates as NEXT566.

Passing authorizes an independent same-source HEA-domain report only.  It does
not authorize canonical report/paper edits, a universal law claim, or a claim
that the formula reproduces DFT energies.
