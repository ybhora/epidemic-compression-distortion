# Choosing the spatial scale of aggregation in epidemic models

Code for **“Choosing the spatial scale of aggregation in epidemic models: compression versus distortion”**, by Bhora et al.

This repository reproduces the canonical network SIR simulations and US county analyses in the paper and supplement. The MATLAB state-network simulations are maintained separately.

## Install

You need **Python 3.14**. Download and extract the repository, then open a terminal in its folder.

On macOS or Linux:

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows (PowerShell):

```powershell
py -3.14 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Keep this environment activated when running the commands below.

## Try a single figure

```bash
python run.py demo
```

This creates **Figure 1** as `results/figures/figure1.pdf` and `figure1.png`, without downloading data.

## Reproduce the paper

```bash
python run.py download
python run.py all
```

The first command downloads approximately **4 GB** of public data. The second runs the analyses and generates the figures and numerical results. Allow several hours, about **15 GB of free disk space**, and preferably **16 GB of RAM**; runtime depends on your computer. Once the download is complete, the analyses run offline.

Outputs are saved in:

- `results/figures/`: figures as PDFs and PNGs, numbered as in the paper.
- `results/paper/`: numerical comparisons, gradients, and supplementary tables.
- `results/analysis/`: intermediate results and community memberships.

Downloaded data and generated outputs are excluded from Git.

## Guides

- [Run individual stages or resume an interrupted analysis](docs/reproduction.md)
- [Find the code and outputs for each figure and table](docs/paper-map.md)
- [Understand the input data](docs/data.md)
- [Understand the calculations](docs/methods.md)

## Citation and license

Please cite the accompanying paper; author metadata are in [CITATION.cff](CITATION.cff). The code uses the [MIT license](LICENSE). Data retain their providers’ terms; see [third-party notices](THIRD_PARTY_NOTICES.md).
