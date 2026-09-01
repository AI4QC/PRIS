# First-principles results (E1-E4)

Estimators and thresholds are quoted from `PREREG-DFT.md`; this report only evaluates them.

## Pre-registered predictions

| experiment | prediction | measured | met |
|---|---|---|---|
| E1 | the reduced coordinate localises the 0.1 eV per atom crossing at least 1.5x more tightly than angstroms | 1.80x | yes |
| E1 | the median crossing lies in (0.7, 1.0), bracketing the D1 floors | 0.927 | yes |
| E1b | hard potentials move the excess at the floor by less than 25% | 0.7% | yes |
| E2 | GNoME median T_od below 300 K | 0 K | yes |
| E2 | control median T_od above 1000 K | 995 K | **no** |
| E2 | at least 60% of GNoME entries below 300 K | 100.0% | yes |
| E2 | at most 20% of controls below 300 K | 30.0% | **no** |
| E3 | DFT and MatterSim rank correlation at least 0.7 | 0.953 | yes |
| E3 | S5 median exceeds the parent median by 0.3 eV per atom | 0.3976 | yes |
| E3 | the S5-to-parent ratio is larger under DFT than under MatterSim | DFT 460.69 vs MatterSim 164.38 | yes |
| E4 | at most 2 screened candidates reach 400 GPa | 1 | yes |
| E4 | at least 70% of priority candidates confirmed | 0.7% | **no** |
| E4 | UMA and DFT bulk moduli correlate with r at least 0.8 | 0.769 | **no** |

9 of 13 predictions met.

## E1 — the reduced-contact energy landscape

- curves fitted: 20, excluded: 0
- reduced contact at which compression costs 0.1 eV per atom: median 0.927, range 0.712-1.024
- the same crossing in angstroms: median 1.921, range 1.259-2.371 A
- relative spread across chemistries: 0.152 in the reduced coordinate against 0.274 in angstroms, a factor of 1.80
- energy excess: 5.357 eV per atom at rho_c=0.735, 2.084 at 0.804, 0.210 at the D2 ceiling 1.05 (medians)

### E1b — hard-potential control

- compounds replicated: 8, excluded: 0
- relative change in the excess at rho_c=0.735 when the small-core potentials replace the standard ones: median 0.7%, worst 16.3%

## E2 — is GNoME's low-symmetry excess thermodynamically real

- entries analysed: 23 GNoME, 10 experimental controls; excluded: 6
    - E2-84b0225bbe-o00: stage relax_cell exited 1
    - E2-84b0225bbe-o01: stage relax_cell exited 1
    - E2-84b0225bbe-o03: stage relax_cell exited 1
    - E2-84b0225bbe-o04: stage static never ran
    - E2-84b0225bbe-o05: stage static never ran
    - 84b0225bbe: the released ordering did not finish
- GNoME ordering energy: median 0.0000 eV per atom; order-disorder temperature: median 0 K
- control ordering energy: median 0.0238 eV per atom; order-disorder temperature: median 995 K
- below 300 K: 100.0% of GNoME entries, 30.0% of controls


Reported alongside, not part of the predictions: the pre-registered dE measures how far the released ordering sits above the best one, which says whether the release picked the ground state rather than whether the compound is ordered. The cost of disordering, mean over orderings minus the minimum, is the quantity that decides that:
- disordering energy: GNoME median 0.00010 eV per atom, controls 0.03642, a factor of 358
- the temperature that implies: GNoME median 11 K, controls 1524 K
- below 300 K on that measure: 18 of 23 GNoME, 0 of 10 controls

## E3 — do the controlled damages read as damage to DFT

- cells analysed: 200, excluded: 0

| variant | n | DFT release, median (eV per atom) | MatterSim, median |
|---|---|---|---|
| P0 | 50 | 0.0009 | 0.0024 |
| S1 | 30 | 0.3032 | 0.3012 |
| S2 | 30 | 0.3953 | 0.3135 |
| S3 | 30 | 3.1537 | 3.3265 |
| S4 | 30 | 0.5545 | 0.4756 |
| S5 | 30 | 0.3985 | 0.3884 |

- DFT vs MatterSim rank correlation over 200 paired cells: 0.953
- S5 cation-anion exchange exceeds the undamaged parents by 0.3976 eV per atom

## E4 — does the inverse-design screen survive first principles

- candidates fitted: 260 (60 screened, 140 priority, 60 control); excluded: 3
- screened candidates at or above 400 GPa under DFT: 1 of 60
- priority candidates confirmed at or above 400 GPa: 0.7%
- UMA vs DFT bulk modulus correlation: r = 0.769
- DFT bulk modulus: median 374 GPa, max 419 GPa

Reported alongside, not part of the predictions:
- the proxy runs high: median DFT/UMA ratio 0.940, so the 400 GPa target sits at 376 GPa on the DFT scale
- rank correlation (Spearman) 0.710, against Pearson 0.769
- a priority candidate outranks a screened one 0.966 of the time (0.5 would be no signal)
- at that rescaled 376 GPa: 1 of 60 screened candidates reach it, against 123 of 200 the screen retained
- of the 140 highest bulk moduli under DFT, 3 had been removed by the screen: it retains 97.9%
    - lost: candidate_0980 Re2IrOs6, DFT 419 GPa, proxy 347 GPa, PSS -0.721
    - lost: candidate_0292 Re4Os, DFT 374 GPa, proxy 337 GPa, PSS -0.644
    - lost: candidate_0995 Re7Os3, DFT 373 GPa, proxy 360 GPa, PSS -0.638
- priority: n = 140, median 383 GPa, IQR 375-387
- control: n = 60, median 363 GPa, IQR 356-378
- screened: n = 60, median 337 GPa, IQR 322-350

The candidates chosen for E4 deliberately over-sample high proxy bulk moduli, so none of these percentages carry over to the full candidate pool; they describe this set only.

