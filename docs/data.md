# Input data

The repository contains small reference inputs that define the paper's analysis. Large raw data are downloaded separately. `data/sources.csv` records source URLs, sizes, and SHA-256 hashes of the actual raw files used. `data/reference_manifest.csv` records the bundled reference inputs.

## Downloaded inputs

| Input | Source | Use |
| --- | --- | --- |
| County cases/deaths, 2020–2022 annual CSVs | [The New York Times archive](https://github.com/nytimes/covid-19-data) | County surveillance trajectories |
| Archived NYS county cases | [New York State Department of Health](https://health.data.ny.gov/Health/New-York-State-Statewide-COVID-19-Testing-Archived/xdss-u53e) | Five NYC county case trajectories |
| NYC daily borough deaths | [NYC DOHMH coronavirus-data](https://github.com/nychealth/coronavirus-data/tree/master/trends) | Five NYC county death trajectories |
| Census county population, `POPESTIMATE2019` | [Census Vintage 2019](https://www2.census.gov/programs-surveys/popest/datasets/2010-2019/counties/totals/co-est2019-alldata.csv) | July 1, 2019 populations |
| 52 weekly 2019 population-flow tables | [GeoDS weekly-flow repository](https://github.com/GeoDS/COVID19USFlows-WeeklyFlows/tree/909c07d616365c112532dba1fdec4162a64e2ec1/weekly_flows/county2county) | G1/G2 and retained population cores |

`python run.py download` fetches the analysis group. The weekly GeoDS source is pinned to commit `909c07d616365c112532dba1fdec4162a64e2ec1`. The other URLs identify public archives; hashes pin the required content. If an archive changes, the downloader fails rather than treating the new data as identical. The flow tables contain derived `pop_flows` estimates, not individual device records.

The 2019 weekly files are dated January 7 through December 30. Dates refer to Mondays; intervals cover Monday–Sunday. The last week extends into January 2020. See the upstream [GeoDS project](https://github.com/GeoDS/COVID19USFlows) and Kang et al. (2020), *Multiscale dynamic human mobility flow dataset in the U.S. during the COVID-19 epidemic*, for data attribution.

## Bundled references

- `county_registry.csv`: declared 3,108 continental county-equivalent identifiers, state/CBSA/CSA membership, and coordinates. Population is rebuilt from Census. Statistical areas use the original 2023 delineations, a retrospective crosswalk.
- `statistical_areas_2023.csv`: crosswalk and area names used to select the named Figure 4 examples. The county memberships used analytically are fixed in `county_registry.csv`; the full title crosswalk is used for names and region selection.
- `window_resolution_registry.csv`: exact 103 original eight-week windows and 25-resolution grid. It contains analysis design, not epidemic results.
- `rolling_mobility.csv`: the archived 149-window national mobility predictor for the temporal holdout check. It includes pandemic-period mobility and cannot be reconstructed from the 52 baseline 2019 files alone.
- `rolling_mobility_population.csv`: original county support and population normalization used to build that predictor. This historical denominator is retained solely to reproduce the archived predictor. A positive constant rescaling leaves an OLS regression with an intercept unchanged. Partition compression uses the corrected Census population instead.
- `maps/`: exact generalized county geometry used for the paper, with the original source metadata. The identifier correction maps Shannon County's obsolete code 46113 to Oglala Lakota County 46102. Geometry is unchanged. Retaining this vintage matters for shared-vertex contiguity and centroid-distance diagnostics; it is not a current boundary database.

## Rebuild the archived predictor, optionally

The full empirical workflow uses the small bundled predictor, so additional pandemic flow downloads are unnecessary for reproducing the holdout fits. To rebuild that predictor from all 156 weekly 2019–2021 flow tables (approximately 10 GB total raw flows):

```bash
python scripts/fetch_data.py --group all
python scripts/rebuild_mobility_index.py
```

The rebuilt series is written to `results/analysis/rebuilt_rolling_mobility.csv` and compared with every archived predictor value. The original reference table is preserved.

## Derived inputs

`prepare_data.py` builds the current Census county master, NYT daily case panel, and both ordered mobility edge tables. `prepare_nyc_panel.py` adds the five observed NYC case series. Other scripts build retained supports, memberships, and death panels. No historical pickles or analysis outputs are required as distributed inputs.

Important source limitations remain: NYT's separate Missouri city reports and unknown-location counts leave partial county reporting; constant series have undefined correlations; NYC deaths are dated by occurrence while NYT deaths elsewhere are dated by report. The sensitivity scripts implement the date/revision checks described in the supplement.

Input data retain the providers' terms. See [third-party notices](../THIRD_PARTY_NOTICES.md).
