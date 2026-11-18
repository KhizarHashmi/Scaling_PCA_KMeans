"""Everything we fixed before running the study."""

SPLIT_SEED = 2026
HOLDOUT_SHARE = 0.20

K_RANGE = list(range(2, 13))
K_VARIETIES = 7  # number of bean varieties, used only to compare pipelines at the same k

N_INIT = 10  # restarts for every "proper" k-means fit
N_RESTARTS = 30  # single-start runs for the initialization check
N_BOOTSTRAP = 30  # refits on resampled data
FIT_SEED = 0

PCA_VARIANCE = 0.95

GAP_REFERENCES = 20
GAP_N_INIT = 3

CI_RESAMPLES = 2000
CI_SEED = 7
NULL_SEED = 11

SENSITIVITY_SPLIT_SEEDS = list(range(100, 110))

DATA_URL = "https://archive.ics.uci.edu/static/public/602/dry+bean+dataset.zip"
XLSX_SHA256 = None  # filled in data.py from the file we ship
