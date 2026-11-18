"""How much do the clusters move when only the luck changes?"""
from itertools import combinations

import numpy as np
from sklearn.metrics import adjusted_rand_score

from config import K_VARIETIES, N_BOOTSTRAP, N_RESTARTS
from pipelines import kmeans, make_preprocessor


def _pairwise_ari(predictions):
    return np.array([adjusted_rand_score(a, b) for a, b in combinations(predictions, 2)])


def _summary(recipe_name, check, values, extra=None):
    row = {
        "recipe": recipe_name,
        "check": check,
        "mean_ari": values.mean(),
        "sd_ari": values.std(ddof=1),
        "min_ari": values.min(),
        "share_pairs_above_0_9": float((values >= 0.9).mean()),
    }
    row.update(extra or {})
    return row


def restart_stability(recipe_name, A_fit, A_hold, k=K_VARIETIES):
    """Single-start k-means, different seed each time, same data."""
    models = [kmeans(k, seed=s, n_init=1).fit(A_fit) for s in range(N_RESTARTS)]
    predictions = [m.predict(A_hold) for m in models]
    J = np.array([m.inertia_ for m in models])
    return _summary(recipe_name, "restarts", _pairwise_ari(predictions),
                    {"worst_J_over_best_J": J.max() / J.min()})


def bootstrap_stability(recipe, X_fit, X_hold, k=K_VARIETIES):
    """Resample the fit part and refit the whole recipe, scaler and PCA included."""
    predictions = []
    components = []
    for s in range(N_BOOTSTRAP):
        idx = np.random.default_rng(s).integers(0, len(X_fit), len(X_fit))
        prep = make_preprocessor(recipe).fit(X_fit[idx])
        model = kmeans(k, seed=s).fit(prep.transform(X_fit[idx]))
        predictions.append(model.predict(prep.transform(X_hold)))
        if recipe.pca is not None:
            components.append(prep.named_steps["pca"].n_components_)
    extra = {"components_seen": sorted(set(int(c) for c in components))} if components else {}
    return _summary(recipe.name, "bootstrap", _pairwise_ari(predictions), extra)
