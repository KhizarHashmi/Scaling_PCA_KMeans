# Does Scaling Decide the Clusters?

Feature scaling and PCA in k-means on the UCI Dry Bean data. Semester project for Mathematical and Statistical Foundations, Fall 2026, FAST-NUCES.

Khizar Mohammad Hashmi (26I-8009), Alishba Abbas (26I-8032), MDS-1A.

## Question

How do z-score scaling and PCA change the clusters k-means finds on the Dry Bean data, and does the silhouette score favour the same pipeline as the hidden variety labels?

## Current stage: feasibility checkpoint

`notebooks/checkpoint.ipynb` shows that the data loads, the raw k-means baseline runs, and the planned held-out evaluation (silhouette, ARI with a bootstrap interval, restart stability) can be computed. It does not run the full comparison yet. That comes in the next stage.

## How to run

```
conda env create -f environment.yml
conda activate msf_clustering
python -m pytest -q tests
jupyter nbconvert --to notebook --execute notebooks/checkpoint.ipynb --output checkpoint.ipynb
```

Runs on a laptop CPU in under a minute. No GPU is needed. The notebook is self-contained, so it can also be uploaded to Colab. If `data/Dry_Bean_Dataset.xlsx` is not next to it, the data is downloaded from UCI, which can take a few minutes.

## Layout

| Path | What it is | Owner |
|---|---|---|
| `src/config.py` | Seeds and settings fixed before the study | Khizar |
| `src/data.py` | Loading and the 80/20 split | Khizar |
| `src/pipelines.py` | The four recipes, the shared k-means fit, the rotation check | Khizar |
| `src/selection.py` | Choosing k with silhouette and the gap statistic | Khizar |
| `src/evaluation.py` | Held-out silhouette, ARI with interval, shuffled-data check | Alishba |
| `src/stability.py` | Restart and bootstrap stability | Alishba |
| `tests/` | Checks on leakage, PCA, the rotation property, the gap rule and the shuffle | Both |
| `notebooks/` | The checkpoint notebook and the script that builds it from `src/` | Alishba |

## Data

UCI Dry Bean Dataset, Koklu and Ozkan (2020), DOI 10.24432/C50S4B, licence CC BY 4.0. See `data/README.md`.
