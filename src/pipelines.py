"""The four recipes of the 2 x 2 design, and the k-means fit they all share."""
from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from config import FIT_SEED, N_INIT, PCA_VARIANCE


@dataclass(frozen=True)
class Recipe:
    name: str
    scaler: str = "none"  # none, zscore or minmax
    pca: float | None = None  # share of variance to keep, or None


RAW = Recipe("Raw")
RAW_PCA = Recipe("Raw+PCA", pca=PCA_VARIANCE)
Z = Recipe("Z", scaler="zscore")
Z_PCA = Recipe("Z+PCA", scaler="zscore", pca=PCA_VARIANCE)
RECIPES = [RAW, RAW_PCA, Z, Z_PCA]

# Each contrast changes exactly one factor.
CONTRASTS = [("Z", "Raw"), ("Z+PCA", "Raw+PCA"), ("Raw+PCA", "Raw"), ("Z+PCA", "Z")]


def make_preprocessor(recipe):
    steps = []
    if recipe.scaler == "zscore":
        steps.append(("scale", StandardScaler()))
    elif recipe.scaler == "minmax":
        steps.append(("scale", MinMaxScaler()))
    if recipe.pca is not None:
        steps.append(("pca", PCA(n_components=recipe.pca, svd_solver="full")))
    if not steps:
        steps.append(("identity", "passthrough"))
    return Pipeline(steps)


def transform_pair(recipe, X_fit, X_hold):
    """Fit on the 80% only, then apply to both parts. This is the leakage guard."""
    prep = make_preprocessor(recipe).fit(X_fit)
    return prep.transform(X_fit), prep.transform(X_hold), prep


def kmeans(k, seed=FIT_SEED, n_init=N_INIT):
    return KMeans(n_clusters=k, init="k-means++", n_init=n_init, random_state=seed)


def rotation_check(X_fit, k=7, seed=1):
    """Full-rank PCA only centres and rotates, so J must not move.

    Both runs start from the same centroids, one set rotated, so any gap
    in J is numerical error and not a different local optimum.
    """
    pca = PCA(svd_solver="full").fit(X_fit)
    rotated = pca.transform(X_fit)
    start = X_fit[np.random.default_rng(seed).choice(len(X_fit), k, replace=False)]
    a = KMeans(k, init=start, n_init=1, max_iter=300, tol=0, algorithm="lloyd").fit(X_fit)
    b = KMeans(k, init=pca.transform(start), n_init=1, max_iter=300, tol=0, algorithm="lloyd").fit(rotated)
    return a.inertia_, b.inertia_, a.labels_, b.labels_
