"""Choosing k without looking at the labels: silhouette and the gap statistic."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from config import FIT_SEED, GAP_N_INIT, GAP_REFERENCES, K_RANGE
from pipelines import kmeans


def k_curve(A, recipe_name, seed=FIT_SEED):
    """J, silhouette and gap for every k, on the fit part only."""
    rng = np.random.default_rng(seed)
    low, high = A.min(axis=0), A.max(axis=0)
    rows = []
    for k in K_RANGE:
        model = kmeans(k, seed).fit(A)
        reference = [
            np.log(KMeans(k, n_init=GAP_N_INIT, random_state=r).fit(rng.uniform(low, high, A.shape)).inertia_)
            for r in range(GAP_REFERENCES)
        ]
        rows.append({
            "recipe": recipe_name,
            "k": k,
            "J": model.inertia_,
            "silhouette": silhouette_score(A, model.labels_),
            "gap": np.mean(reference) - np.log(model.inertia_),
            "gap_se": np.std(reference) * np.sqrt(1 + 1 / GAP_REFERENCES),
        })
    return pd.DataFrame(rows)


def choose_by_silhouette(curve):
    return int(curve.loc[curve["silhouette"].idxmax(), "k"])


def choose_by_gap(curve):
    """Tibshirani et al.: smallest k with Gap(k) >= Gap(k+1) - s(k+1).

    Returns (k, rule_met). If the rule never fires inside our range we
    return the largest k and say so, rather than pretend it was chosen.
    """
    c = curve.sort_values("k").reset_index(drop=True)
    for i in range(len(c) - 1):
        if c.loc[i, "gap"] >= c.loc[i + 1, "gap"] - c.loc[i + 1, "gap_se"]:
            return int(c.loc[i, "k"]), True
    return int(c.loc[len(c) - 1, "k"]), False
