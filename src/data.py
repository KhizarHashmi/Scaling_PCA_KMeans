"""Load the UCI Dry Bean data and make the fixed 80/20 split."""
import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import DATA_URL, HOLDOUT_SHARE, SPLIT_SEED

LOCAL_FILE = Path(__file__).resolve().parent.parent / "data" / "Dry_Bean_Dataset.xlsx"


def read_table(path=LOCAL_FILE):
    """Read our copy if we have it, otherwise download it from UCI."""
    path = Path(path)
    if path.exists():
        return pd.read_excel(path)
    print("Local copy not found, downloading from UCI (can be slow)...")
    with urllib.request.urlopen(DATA_URL, timeout=600) as reply:
        archive = zipfile.ZipFile(io.BytesIO(reply.read()))
    name = next(n for n in archive.namelist() if n.endswith(".xlsx"))
    return pd.read_excel(archive.open(name))


def features_and_labels(table):
    """Labels are kept apart from the features from this point on."""
    labels = table["Class"].astype(str).to_numpy()
    features = table.drop(columns="Class")
    return features.to_numpy(dtype=float), labels, list(features.columns)


def split(X, labels, seed=SPLIT_SEED):
    """Plain random split. We do not stratify, because that would use the labels."""
    return train_test_split(X, labels, test_size=HOLDOUT_SHARE, random_state=seed, shuffle=True)


def describe(table):
    return {
        "rows": int(table.shape[0]),
        "features": int(table.shape[1] - 1),
        "missing": int(table.isna().sum().sum()),
        "duplicate_rows": int(table.duplicated().sum()),
        "varieties": int(table["Class"].nunique()),
    }


def drop_duplicates(X, labels):
    _, keep = np.unique(X, axis=0, return_index=True)
    keep = np.sort(keep)
    return X[keep], labels[keep]
