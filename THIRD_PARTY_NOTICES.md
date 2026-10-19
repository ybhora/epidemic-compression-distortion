# Input attribution and terms

The MIT license in this repository applies to the authors' analysis code and documentation. It does not replace data-provider terms.

## GeoDS mobility

The downloader obtains weekly population-flow estimates from [COVID19USFlows-WeeklyFlows](https://github.com/GeoDS/COVID19USFlows-WeeklyFlows), linked from [COVID19USFlows](https://github.com/GeoDS/COVID19USFlows). Cite Kang et al. (2020), *Multiscale dynamic human mobility flow dataset in the U.S. during the COVID-19 epidemic*, Scientific Data 7, 390, [doi:10.1038/s41597-020-00734-5](https://doi.org/10.1038/s41597-020-00734-5). The source projects declare MIT licensing; the supplied upstream notice is retained verbatim in [GeoDS-LICENSE.txt](third_party/GeoDS-LICENSE.txt). Raw flow files are downloaded separately and not distributed in this code repository. The bundled rolling mobility predictor is a derived aggregate of these population flows.

## Surveillance and Census data

County COVID-19 records are from [The New York Times](https://github.com/nytimes/covid-19-data); retain its attribution and consult its [data license](https://github.com/nytimes/covid-19-data/blob/master/LICENSE). NYS case records are from the [New York State Department of Health archived testing dataset](https://health.data.ny.gov/Health/New-York-State-Statewide-COVID-19-Testing-Archived/xdss-u53e). NYC deaths are from the [New York City Department of Health and Mental Hygiene](https://github.com/nychealth/coronavirus-data). Population estimates are from the U.S. Census Bureau, Vintage 2019 `POPESTIMATE2019`. Statistical-area memberships derive from the 2023 OMB/Census delineations. These raw surveillance/population files are obtained from their sources rather than bundled here.

## Generalized county geometry

`data/reference/maps/` retains the geometry used to generate the paper's maps and geographic diagnostics. Source: **ESRI, U.S. Counties (Generalized), ESRI Data & Maps 2008**, with the underlying source/copyright attributions and full terms preserved in [`source_metadata.xml`](data/reference/maps/source_metadata.xml). The metadata state that redistribution of this geodata without a value-added application is permitted with proper metadata and source/copyright attribution. This repository retains that metadata; it does not relicense the geometry under MIT.

The analysis copy corrects the obsolete Shannon County identifier 46113 to Oglala Lakota County 46102. The polygon geometry is unchanged. See the metadata for the source's geographic vintage, scale, accuracy limitations, vendor attributions, and redistribution conditions.

## Software libraries

NumPy, pandas, SciPy, Matplotlib, Apache Arrow, igraph, leidenalg, and PyShp are installed from their own distributions under their respective licenses. They are not vendored into this repository.
