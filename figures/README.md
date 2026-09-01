# Every figure, and the code that draws it

Thirty figures: five in the main text, twenty-five in the Supplementary
Information. `manifest.json` is the machine-readable version of the table below;
`make.py` reads it so you never have to look a script name up by hand.

```bash
python figures/make.py --list        # the table
python figures/make.py "Fig. 1"      # one figure
python figures/make.py S7            # the S prefix is enough
python figures/make.py --all         # every generator once
```

**Three script names disagree with the figure they draw.** The names were fixed
before the figures were renumbered for submission, and renaming them now would
break the tests and the archived provenance. `manifest.json` is the authority:

| script | actually draws |
|---|---|
| `src/fig6_deployment.py` | **Fig. 5** |
| `src/figS7_amplitude_response.py` | **Fig. S5** |
| `src/figS17_charge_coverage.py` | **Fig. S23** |

## Main text

| | file | generator | inputs |
|---|---|---|---|
| Fig. 1 | `fig1_agentic_law_learning.pdf` | `src/paper_figs.py` | `paper/data/fig1_*` |
| Fig. 2 | `fig2_rules.pdf` | `src/paper_figs.py` | `paper/data/fig2_*`, `band_*` |
| Fig. 3 | `fig3_anatomy.pdf` | `src/fig3_anatomy.py` | builds the spinel itself; `dft/E1_rho_curve/` |
| Fig. 4 | `fig4_validation_synthesis.pdf` | `experiments/pu_synthesizability_20260821/plot_merged_fig45_nature.py` | see *What is not here* |
| Fig. 5 | `fig5_deployment.pdf` | `src/fig6_deployment.py` | `paper/data/fig6_*`, `dft/E2_ordering/`, `dft/E3_crosscheck/` |

## Supplementary Information

| | generator | | | generator |
|---|---|---|---|---|
| S1 | `src/paper_figs.py` | | S14 | `…/render_moved_si_panels.py` |
| S2, S3, S4, S6, S8, S9, S10, S11, S13 | `src/si_figs.py` | | S15, S16 | `src/pss_stability_figs.py` |
| S5 | `src/figS7_amplitude_response.py` | | S17 | `…/plot_si_l4_contribution.py` |
| S7 | `experiments/pris_composition_holdout_20260829/plot_figures.py` | | S18 | `experiments/pu_model_performance_audit/plot_roc_confusion.py` |
| S12, S20, S21, S24, S25 | `src/si_dft_figs.py` | | S19, S22 | `…/render_moved_si_panels.py` |
| | | | S23 | `src/figS17_charge_coverage.py` |

## What is not here

Twenty-six of the thirty figures redraw from data committed in `paper/data/`,
`paper/si_data/` and `dft/`. Four do not, and saying so is more useful than
shipping a script that fails on a missing path:

- **Fig. 4, Fig. S17** read the positive–unlabelled score shards for 8,125,976
  unlabelled structures (48 GB of parquet across `binary_v1/experimental` and
  `binary_v1/pu_negative`) and the MatterGen inverse-design run. Neither is
  redistributed: the shards are keyed to structures we may not republish, and
  the run is reproducible from `experiments/property_design_20260821/`.
- **Fig. S19, Fig. S22** reuse panels from the same pipeline.

The aggregate numbers those panels display are in `paper/data/`, so the figures
can be checked against the manuscript even where they cannot be re-rendered.

The derived feature store is the other absent input. It is gigabytes of parquet
built from structures under licences that forbid redistribution; scripts that
need it read `$PRIS_FEATURES`.
