# Paper-to-code guide

This guide covers all main-text and supplementary figures within the Python repository’s scope. Filenames follow the supplied final manuscripts. The MATLAB state-network analyses are a separate work and are not included here.

## Figures

| Figure file | Stage | Code |
| --- | --- | --- |
| `figure1.pdf` | `demo or simulations` | [figure1.py](../scripts/figure1.py) |
| `figure3.pdf` | `figures` | [paper_figures.py](../scripts/paper_figures.py) |
| `figure4.pdf` | `figures` | [plot_publication.py](../scripts/plot_publication.py) |
| `figure5.pdf` | `figures` | [paper_figures.py](../scripts/paper_figures.py) |
| `figure6.pdf` | `figures` | [paper_figures.py](../scripts/paper_figures.py) |
| `figureS2a_part1.pdf` | `simulations` | [plot_simulations.py](../scripts/plot_simulations.py) |
| `figureS2a_part2.pdf` | `simulations` | [plot_simulations.py](../scripts/plot_simulations.py) |
| `figureS2b_star_seeding.pdf` | `simulations` | [plot_simulations.py](../scripts/plot_simulations.py) |
| `figureS2c_unrescaled_plateau.pdf` | `simulations` | [plot_simulations.py](../scripts/plot_simulations.py) |
| `figureS2d_inference.pdf` | `simulations` | [simulation_inference.py](../scripts/simulation_inference.py) |
| `figureS2e_proxy.pdf` | `simulations` | [simulation_inference.py](../scripts/simulation_inference.py) |
| `figureS3_graph_choice.pdf` | `empirical` | [supplement_checks.py](../scripts/supplement_checks.py) |
| `figureS4_refinement.pdf` | `figures` | [paper_figures.py](../scripts/paper_figures.py) |
| `figureS5_null_permutation.pdf` | `empirical` | [supplement_checks.py](../scripts/supplement_checks.py) |
| `figureS6_core_thresholds_lag.pdf` | `figures` | [paper_figures.py](../scripts/paper_figures.py) |
| `figureS6b_core_geography_lag_delta.pdf` | `figures` | [paper_figures.py](../scripts/paper_figures.py) |
| `figureS6b_core_geography_lag_lockdown.pdf` | `figures` | [paper_figures.py](../scripts/paper_figures.py) |
| `figureS6c_matched_support.pdf` | `figures` | [paper_figures.py](../scripts/paper_figures.py) |
| `figureS7_dynamic_robustness.pdf` | `figures` | [plot_empirical.py](../scripts/plot_empirical.py) |
| `figureS8.pdf` | `figures` | [plot_empirical.py](../scripts/plot_empirical.py) |
| `figureS9_mixing_vs_distortion.pdf` | `figures` | [plot_empirical.py](../scripts/plot_empirical.py) |
| `figureS10_deaths_lag.pdf` | `figures` | [paper_figures.py](../scripts/paper_figures.py) |
| `figureS10_deaths_incoherence.pdf` | `figures` | [paper_figures.py](../scripts/paper_figures.py) |
| `figureS10b_deaths_gradient.pdf` | `figures` | [plot_empirical.py](../scripts/plot_empirical.py) |
| `figureS11_algorithm_lag.pdf` | `figures` | [paper_figures.py](../scripts/paper_figures.py) |
| `figureS12_sensitivity_lag.pdf` | `figures` | [paper_figures.py](../scripts/paper_figures.py) |
| `figureS12b_lockdown_lag42.pdf` | `figures` | [paper_figures.py](../scripts/paper_figures.py) |
| `figureS13a_gradient_incoherence.pdf` | `figures` | [paper_figures.py](../scripts/paper_figures.py) |
| `figureS13b_frontier_incoherence.pdf` | `figures` | [paper_figures.py](../scripts/paper_figures.py) |
| `figureS13c_dynamic_incoherence.pdf` | `figures` | [paper_figures.py](../scripts/paper_figures.py) |
| `figureS13d_incoherence_decomposition.pdf` | `figures` | [paper_figures.py](../scripts/paper_figures.py) |

All figure files are generated in `results/figures/`.

## Tables and numerical claims

| Paper content | Generated data | Producing code |
| --- | --- | --- |
| S1 definitions and preprocessing | Case panel; NYC alignment; source validation | `prepare_data.py`, `prepare_nyc_panel.py`, `validate_nys_alignment.py`, `validate_pipeline.py` |
| S2 canonical inference | `results/analysis/simulation_inference.csv`, `simulation_validation.csv`, `bridge_cluster_lags.csv` | `simulation_inference.py`, `plot_simulations.py`, `figure1.py` |
| S3 graph weights/support | `results/analysis/graph_statistics.csv`, `retained_supports.csv`; `results/paper/M8_units.csv`, `M8_geography.json` | `supplement_checks.py`, `recompute_empirical.py`, `geography.py` |
| S4 ten-seed ranges | `results/paper/seed_ranges.csv`, `seed_matched.csv`, `seed_frontiers.csv` | `seed_robustness.py` |
| S5 gradient, county tails, bootstrap ranges | `results/analysis/corrected_gradient.csv`; `results/paper/M3_gradient.csv`, `M3_counties.csv`, `M3_units.csv`; `results/analysis/null_summary.csv` | `paper_analysis.py`, `additional_checks.py`, `supplement_checks.py` |
| S5/S10/S13 matched compression and bands | `results/paper/matched_comparisons.csv` | `paper_tables.py` |
| S6 support intersections | `results/paper/S-F_common_frontiers.csv`, `common_support_reclustered_comparisons.csv`, `retention_matched_lags.csv`; `results/analysis/retained_supports.csv` | `paper_analysis.py`, `recompute_empirical.py`, `common_support_recluster.py`, `paper_tables.py` |
| S7 admissible scale by tolerance/period | `results/analysis/phase_medians.csv`, `dynamic_summary.csv`; `results/paper/admissible_below_csa.csv` | `summarize_results.py`, `paper_tables.py` |
| S8 full rolling curves | `results/analysis/corrected_dynamic.csv` | `recompute_empirical.py` |
| S9 holdout errors | `results/analysis/prediction_scores.csv`, `prediction_holdout.csv` | `predictive_check.py` |
| S10 death gradient | `results/analysis/deaths_gradient.csv`; `results/paper/death_frontiers.csv` | `recompute_empirical.py`, `paper_analysis.py` |
| S11 Leiden/Louvain | `results/paper/algorithm_comparison.csv` | `paper_tables.py` |
| S12 smoothing, revision and date checks | `results/analysis/sensitivity.csv`, `revision_handling_comparisons.csv`, `nyc_date_sensitivity.csv`, `nyc_death_timing_comparisons.csv` | `additional_checks.py`, `revision_handling_check.py`, `nyc_date_sensitivity.py`, `nyc_death_timing_check.py` |
| S12 lag-bound saturation/comparisons | `results/paper/lag_bound_saturation.csv`, `matched_lag_bounds.csv` | `paper_analysis.py`, `paper_tables.py` |
| S13 anticorrelation counts | `results/paper/incoherence_above_one.csv` | `paper_tables.py` |
| S13 timing/shape decomposition | `results/paper/S-C4_decomposition.csv`, `county_metrics.parquet` | `paper_figures.py`, `paper_analysis.py` |

Intermediate filenames such as `M3`, `S-F`, and `corrected_…` retain the labels used during the analysis.
