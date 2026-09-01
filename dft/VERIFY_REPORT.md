# DFT task package verification

- packages checked: E1_rho_curve, E1b_paw_control, E2_ordering, E3_crosscheck, E4_design
- tasks: 617, VASP runs: 1854, atoms across all cells: 6861
- errors: 0, warnings: 29

| package | tasks | VASP runs | atoms | shortest contact (A) |
|---|---|---|---|---|
| E1_rho_curve | 20 | 340 | 179 | 1.031 |
| E1b_paw_control | 8 | 136 | 69 | 1.031 |
| E2_ordering | 128 | 256 | 1621 | 1.401 |
| E3_crosscheck | 200 | 600 | 1954 | 0.926 |
| E4_design | 261 | 522 | 3038 | 1.763 |
## PAW core overlap (shortest contact below 0.75 of the radius sum)

Overlap by itself is normal in dense solids, so only tight cells are listed. E1 compresses on purpose and is expected here, which is why it carries the E1b hard-potential control; anywhere else a tight cell is worth a second look.

- E1_rho_curve: 51 cells below 0.75, worst 0.54 (F-Nb at 1.213 A in E1_rho_curve/E1-cod-1507759)
- E1b_paw_control: 13 cells below 0.75, worst 0.59 (F-Nb at 1.213 A in E1b_paw_control/E1b-cod-1507759)
- E2_ordering: 4 cells below 0.75, worst 0.66 (Tl-O at 1.401 A in E2_ordering/E2-cod-1511279-o01)
- E3_crosscheck: 24 cells below 0.75, worst 0.41 (Ba-O at 0.928 A in E3_crosscheck/E3-cod-1001469-S3)


## Warnings

- E1_rho_curve: collected.csv is present but not recorded in the manifest
- E1_rho_curve: collected.json is present but not recorded in the manifest
- E1_rho_curve: curves.json is present but not recorded in the manifest
- E1_rho_curve/E1-cod-1004057: POSCAR.v00 O-W at 0.56 of the PAW radius sum
- E1_rho_curve/E1-cod-1008795: POSCAR.v00 Zr-F at 0.58 of the PAW radius sum
- E1_rho_curve/E1-cod-1010285: POSCAR.v00 F-Si at 0.59 of the PAW radius sum
- E1_rho_curve/E1-cod-1010286: POSCAR.v00 F-Ge at 0.55 of the PAW radius sum
- E1_rho_curve/E1-cod-1010286: POSCAR.v01 F-Ge at 0.60 of the PAW radius sum
- E1_rho_curve/E1-cod-1507759: POSCAR.v00 F-Nb at 0.54 of the PAW radius sum
- E1_rho_curve/E1-cod-1507759: POSCAR.v01 F-Nb at 0.58 of the PAW radius sum
- E1_rho_curve/E1-cod-1511203: POSCAR.v00 B-Ni at 0.56 of the PAW radius sum
- E1b_paw_control: collected.csv is present but not recorded in the manifest
- E1b_paw_control: collected.json is present but not recorded in the manifest
- E1b_paw_control: paw_shift.json is present but not recorded in the manifest
- E1b_paw_control/E1b-cod-1507759: POSCAR.v00 F-Nb at 0.60 of the PAW radius sum
- E3_crosscheck/E3-cod-1001469-S3: POSCAR.init shortest contact 0.928 A
- E3_crosscheck/E3-cod-1001469-S3: POSCAR.init Ba-O at 0.41 of the PAW radius sum
- E3_crosscheck/E3-cod-1010175-S3: POSCAR.init Sn-Cl at 0.56 of the PAW radius sum
- E3_crosscheck/E3-cod-1010182-S3: POSCAR.init Pt-Cl at 0.50 of the PAW radius sum
- E3_crosscheck/E3-cod-1010286-S3: POSCAR.init F-Ge at 0.58 of the PAW radius sum
- E3_crosscheck/E3-cod-1011311-S3: POSCAR.init K-Cl at 0.60 of the PAW radius sum
- E3_crosscheck/E3-cod-1100971-S3: POSCAR.init V-O at 0.52 of the PAW radius sum
- E3_crosscheck/E3-cod-1501467-S3: POSCAR.init Tc-O at 0.51 of the PAW radius sum
- E3_crosscheck/E3-cod-1507759-S3: POSCAR.init Mn-F at 0.58 of the PAW radius sum
- E3_crosscheck/E3-cod-1507767-S3: POSCAR.init shortest contact 0.926 A
- E3_crosscheck/E3-cod-1507767-S3: POSCAR.init Na-F at 0.47 of the PAW radius sum
- E3_crosscheck/E3-cod-1509211-S3: POSCAR.init shortest contact 0.997 A
- E3_crosscheck/E3-cod-1509211-S3: POSCAR.init B-F at 0.59 of the PAW radius sum
- E3_crosscheck/E3-cod-1509287-S3: POSCAR.init S-V at 0.47 of the PAW radius sum

## Notes

- E1_rho_curve: 20 tasks, 340 VASP runs, 179 atoms, rule: discovery split, in_analysis_set, n_sites<=12, ordered with a shortest contact o...
- E1b_paw_control: 8 tasks, 136 VASP runs, 69 atoms, rule: The 8 E1 compounds whose shortest contact at the D1 floor sits furthest inside t...
- E2_ordering: 128 tasks, 256 VASP runs, 1621 atoms, rule: GNoME: merge_test status ok, merged_any, econ_raw_01>2/3, primitive cell <=20 at...
- E3_crosscheck: 200 tasks, 600 VASP runs, 1954 atoms, rule: 30 discovery-split experimental parents with <=16 sites for which all five damag...
- E4_design: 261 tasks, 522 VASP runs, 3038 atoms, rule: All 61 PSS-screened candidates (synthesis_score < -0.6368790173149083), all 140 ...
- E2: {'gnome': 24, 'experimental': 10}, control classes ['G13', 'RE']
- E4: roles {'control': 60, 'priority': 140, 'screened': 61}
