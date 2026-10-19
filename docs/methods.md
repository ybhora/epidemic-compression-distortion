# Methods and code guide

## From a county curve to a distortion value

Each county has a daily case/death signal. Cases are daily cumulative differences with negative revisions set to zero; primary deaths retain negative revisions. Apply a centered seven-day mean to each county's observed row history **before** slicing analysis phases or filling missing panel entries with zero. NYC cases use observed state-source borough counts shifted one day later. NYC deaths use observed borough counts without a date shift. `validate_nys_alignment.py` checks shifts −3 through +3 against the other 57 New York counties and compares borough totals with the separate NYT city record.

Sum the unstandardized smoothed counts of counties in a unit. Compare each county with its own unit aggregate, including itself:

- **Lag:** absolute integer shift maximizing overlapping Pearson correlation, searched over ±28 days for empirical curves or ±90 days for simulations. Ties within `1e-12` favor the smallest absolute shift, then the negative shift.
- **Incoherence:** `1 − Pearson correlation` at zero lag, with range 0–2. Values above one are meaningful and retained.

Undefined correlations remain missing, including constant signals or fewer than three paired values. First average county distortions within each unit, then weight units equally. Frontiers use the mean of unit summaries; Figure 3 uses the median of unit summaries. Compression is the population of all declared members divided by nonempty units, including members with undefined signals.

`scripts/validate_pipeline.py` defines these shared metrics. `tests/test_science.py` checks them against a separate scalar calculation. The bootstrap fixes partitions and aggregates, resamples county summaries within units 500 times, and summarizes unit means or medians. Its bands describe conditional composition variation, not all epidemic or reporting uncertainty. Permutation controls instead shuffle unit labels and recompute aggregates (200 replicates, seed 42, Holm adjustment across 16 comparisons).

## Spatial partitions

`prepare_data.py` sums directed `pop_flows` across 52 weeks, averages by week, and symmetrizes. G2 uses absolute flow; G1 first divides by origin population. Edges are stored in lexical upper-triangle order. `recompute_empirical.py` builds weighted undirected Leiden RBConfiguration partitions with seed 42 and iterations until convergence. Preserve edge order, package versions, and grid when reproducing memberships.

For retained cores, rank continental counties by induced-G2 node strength, breaking ties by FIPS, then include counties until cumulative population reaches 85%, 90%, or 95%. The 85% core contains 849 counties. Isolated vertices remain included. Canonical partitions come from the registry's State, CSA, and CBSA columns; CBSA≥2/≥3 excludes units with fewer members.

The descriptive coarse Leiden partition is selected from full support using its absolute log-compression distance to States, with no epidemic outcome in the selection. Intermediate and fine examples use core85 at resolutions 33.471236 and 111.186983. Main frontiers compare each candidate on its own support. Common-support checks first intersect memberships, then recompute aggregates and compression. `common_support_recluster.py` additionally reruns Leiden on each induced common-support graph for the separate S6 table column. These answer distinct questions.

## Curves and time windows

`summarize_results.py` interpolates distortion linearly in **log compression**, averages duplicate compression coordinates, and does not extrapolate. Admissible scale is the largest feasible compression over **all** curve segments at the stated lag tolerance; the curve is not forced to be monotonic. The 103 eight-week windows advance weekly with fixed candidate memberships. Overlapping windows are descriptive observations, not independent replicates.

`predictive_check.py` uses expanding chronological training sets and only predictor/target windows available at the forecast origin. It retains retrospective revisions; it is a temporal holdout check, not real-time operational validation.

`paper_analysis.py` evaluates all three static phases, lag-bound alternatives, county lag tails, conditional bootstrap draws, and timing/shape components. `seed_robustness.py` reconstructs the ten-seed check described in S4. `geography.py` computes exact shared-vertex queen contiguity and signed-ring centroids using the supplied geometry.

## Canonical SIR simulations

`sir_model.py` implements the resident SIR derivative from the paper. `topologies.py` builds row-stochastic star, fully connected, directed-cycle, and two-cluster bridge matrices. `figure1.py` and `plot_simulations.py` use RK45, relative tolerance `1e-7`, absolute tolerance `1e-9`, β=0.3/day, γ=0.1/day, 1,000 people per node, and one infectious seed. The 100-node star is infeasible at mobility 0.02 and is explicitly omitted.

Figure 1 node curves share the largest node peak as divisor; aggregates use their own peak, a display-only normalization. The simulation lag calculation uses daily interpolated, unnormalized trajectories. `simulation_inference.py` fits a homogeneous model to total infectious prevalence and illustrates susceptible-fraction proxies; those proxies are not estimates of the coupled-network reproduction number.

## Adapting the method

To evaluate your own data, provide a DataFrame with counties as rows and ordered equally spaced days as columns, and a Series mapping those county identifiers to units. Apply your chosen preprocessing explicitly, then call `metrics(raw, membership)` and summarize the returned unit table. For a new region or input snapshot, declare the support, populations, date windows, graph weights, and tolerances, and invalidate the membership cache. Treat changed inputs as a new analysis rather than an exact reproduction.
