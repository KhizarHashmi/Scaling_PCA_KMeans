"""Scoring on the held-out 20%. This is the only place the labels are used."""
import numpy as np
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from config import CI_RESAMPLES, CI_SEED, FIT_SEED, NULL_SEED
from pipelines import kmeans, make_preprocessor


def common_space(X_fit, X_hold):
    """One shared z-scored space, so silhouettes can be compared across recipes."""
    scaler = StandardScaler().fit(X_fit)
    return scaler.transform(X_fit), scaler.transform(X_hold)


def score(recipe, A_fit, A_hold, Z_hold, labels_hold, k, seed=FIT_SEED):
    """Fit k-means in the recipe's own space and score the held-out beans."""
    model = kmeans(k, seed).fit(A_fit)
    pred = model.predict(A_hold)
    return {
        "recipe": recipe.name,
        "k": k,
        "J_fit": model.inertia_,
        "silhouette_own": silhouette_score(A_hold, pred),
        "silhouette_common": silhouette_score(Z_hold, pred),
        "ari": adjusted_rand_score(labels_hold, pred),
    }, pred


def ari_interval(labels, pred, seed=CI_SEED):
    """Percentile interval for ARI from resampling the held-out beans."""
    rng = np.random.default_rng(seed)
    n = len(labels)
    values = []
    for _ in range(CI_RESAMPLES):
        idx = rng.integers(0, n, n)
        values.append(adjusted_rand_score(labels[idx], pred[idx]))
    return np.percentile(values, [2.5, 97.5])


def ari_difference_interval(labels, pred_a, pred_b, seed=CI_SEED):
    """Paired interval for ARI(a) - ARI(b), resampling the same beans for both."""
    rng = np.random.default_rng(seed)
    n = len(labels)
    diffs = []
    for _ in range(CI_RESAMPLES):
        idx = rng.integers(0, n, n)
        diffs.append(adjusted_rand_score(labels[idx], pred_a[idx]) - adjusted_rand_score(labels[idx], pred_b[idx]))
    return np.percentile(diffs, [2.5, 97.5])


def shuffle_columns(X, seed):
    """Shuffle every feature on its own: same marginals, no relation between features."""
    return np.column_stack([np.random.default_rng(seed + j).permutation(X[:, j]) for j in range(X.shape[1])])


def null_silhouette(recipe, X_fit, X_hold, k, seed=NULL_SEED):
    """Run the recipe on data with no real structure and score it the same way."""
    fit, hold = shuffle_columns(X_fit, seed), shuffle_columns(X_hold, seed + 1000)
    prep = make_preprocessor(recipe).fit(fit)
    A_fit, A_hold = prep.transform(fit), prep.transform(hold)
    pred = kmeans(k).fit(A_fit).predict(A_hold)
    return silhouette_score(A_hold, pred)
