"""Build the notebooks from src/, so a notebook can never drift from the code.

The full-study notebook is built only once src/run_study.py and src/figures.py exist."""
import re
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
LOCAL_IMPORT = re.compile(r"^(from (config|data|pipelines|selection|evaluation|stability|figures) import .*|import config as cfg)$")


def source(name):
    text = (SRC / f"{name}.py").read_text(encoding="utf8")
    text = re.sub(r'^""".*?"""\n', "", text, count=1, flags=re.S)
    text = text.split('if __name__ == "__main__":')[0].rstrip() + "\n"
    kept = [line for line in text.splitlines() if not LOCAL_IMPORT.match(line)]
    body = "\n".join(kept).strip() + "\n"
    return body.replace('Path(__file__).resolve().parent.parent / "data" / "Dry_Bean_Dataset.xlsx"',
                        'next((p for p in (Path("data/Dry_Bean_Dataset.xlsx"), Path("../data/Dry_Bean_Dataset.xlsx")) if p.exists()), Path("data/Dry_Bean_Dataset.xlsx"))')


def md(text):
    return nbf.v4.new_markdown_cell(text)


def code(text):
    return nbf.v4.new_code_cell(text.strip())


def save(cells, name):
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nbf.write(nb, ROOT / "notebooks" / name)
    print(name, len(cells), "cells")


SHOW = """
from IPython.display import Image, display
import json, sklearn, platform
print("Python", platform.python_version(), "| scikit-learn", sklearn.__version__, "| numpy", np.__version__, "| pandas", pd.__version__)
"""

def full_notebook():
    return [
        md("# Does Scaling Decide the Clusters?\n\nFeature scaling and PCA in k-means on the UCI Dry Bean data.\n\n"
           "Khizar Mohammad Hashmi (26I-8009), Alishba Abbas (26I-8032), MDS-1A.\n\n"
           "Runs on CPU with no GPU. If `data/Dry_Bean_Dataset.xlsx` is not next to this notebook, the data is downloaded from UCI, which can take a few minutes.\n\n"
           "Data: Koklu & Ozkan (2020), UCI Machine Learning Repository, DOI 10.24432/C50S4B, licence CC BY 4.0."),
        code("import warnings\nwarnings.filterwarnings('ignore')\nimport os\nos.environ.setdefault('OMP_NUM_THREADS', '4')"),
        md("## Settings fixed before the study"),
        code(source("config") + "\nimport types\ncfg = types.SimpleNamespace(**{k: v for k, v in globals().items() if k.isupper()})"),
        md("## Data"), code(source("data")),
        md("## The four recipes\n\nK-means minimises $J = \\sum_k \\sum_{x \\in C_k} \\lVert x - \\mu_k \\rVert^2$. Multiplying a feature by $s$ multiplies its share of every squared distance by $s^2$."),
        code(source("pipelines")),
        md("## Choosing k without labels"), code(source("selection")),
        md("## Scoring on the held-out beans"), code(source("evaluation")),
        md("## Stability"), code(source("stability")),
        md("## Run everything"),
        code(source("run_study") + "\nfrom pathlib import Path\nmain('results')"),
        md("## Figures"), code(source("figures") + "\nmake_all('results')"),
        md("## Results"),
        code(SHOW + """
info = json.load(open('results/run_info.json'))
print(info['data']); print(info['rotation_check']); print('seconds', info['seconds'])
for f in ['chosen_k', 'scores', 'contrasts', 'stability', 'null_silhouette', 'feature_scale']:
    print(f); display(pd.read_csv(f'results/{f}.csv').round(4))
for f in ['fig1_silhouette_by_k', 'fig2_scores_at_k7', 'fig3_stability']:
    display(Image(f'results/figures/{f}.png', width=640))
"""),
    ]


checkpoint = [
    md("# Feasibility checkpoint\n\nKhizar Mohammad Hashmi (26I-8009), Alishba Abbas (26I-8032), MDS-1A.\n\n"
       "Shows that the data loads and is usable, the baseline runs, and the planned evaluation can be computed."),
    code("import warnings\nwarnings.filterwarnings('ignore')\nimport os, time\nos.environ.setdefault('OMP_NUM_THREADS', '4')"),
    code(source("config") + "\nimport types\ncfg = types.SimpleNamespace(**{k: v for k, v in globals().items() if k.isupper()})"),
    md("## 1. The data loads\n\nUCI Dry Bean, DOI 10.24432/C50S4B, licence CC BY 4.0."),
    code(source("data") + "\ntable = read_table()\nprint(describe(table))\nX, labels, names = features_and_labels(table)\nX_fit, X_hold, y_fit, y_hold = split(X, labels)\nprint(X_fit.shape, X_hold.shape)\ntable.head()"),
    md("## 2. Why scaling should matter"),
    code("print(pd.Series(X_fit.std(axis=0, ddof=1), index=names).sort_values(ascending=False).round(4))"),
    md("## 3. The baseline runs: k-means on raw features"),
    code(source("pipelines") + "\nA, B, _ = transform_pair(RAW, X_fit, X_hold)\nt = time.time()\nbaseline = kmeans(cfg.K_VARIETIES).fit(A)\nprint('fit seconds', round(time.time() - t, 2), '| J', baseline.inertia_)"),
    md("## 4. The planned evaluation can be computed\n\nThe silhouette is computed on the held-out 20%. Labels are used only for ARI, after clustering."),
    code(source("evaluation") + "\nZ_fit, Z_hold = common_space(X_fit, X_hold)\nresult, pred = score(RAW, A, B, Z_hold, y_hold, cfg.K_VARIETIES)\nprint(result)\nprint('ARI 95% interval', ari_interval(y_hold, pred).round(3))"),
    md("## 5. One stability number"),
    code(source("stability") + "\nprint(restart_stability('Raw', A, B))"),
]

save(checkpoint, "checkpoint.ipynb")
if (SRC / "run_study.py").exists() and (SRC / "figures.py").exists():
    save(full_notebook(), "reproduce_study.ipynb")
