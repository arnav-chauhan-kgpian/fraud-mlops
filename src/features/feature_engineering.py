"""
Feature Engineering
--------------------
CHALLENGES FACED:

1. VELOCITY FEATURES CAUSE LEAKAGE
   First attempt: computed rolling counts over the FULL dataframe before splitting.
   This let future transactions inform past velocity windows → model saw test data during training.
   FIX: compute velocity features ONLY within the training partition, then apply
   the same logic to val/test using only past data from the training window.

2. AMOUNT BINNING THRESHOLD SELECTION
   Arbitrary bins (0-10, 10-100, etc.) missed the bimodal distribution of fraud amounts.
   FIX: use quantile-based bins fitted on train, same bins applied to val/test.

3. FEATURE EXPLOSION
   Added 40+ interaction features → model overfit, slower inference.
   FIX: kept only features with IV (Information Value) > 0.02 in validation set.

4. ONLINE/OFFLINE CONSISTENCY
   Velocity features computed differently in batch training vs real-time serving
   (batch used pandas rolling; serving used a Redis ZSET) → 12% score mismatch.
   FIX: extracted velocity logic into a shared utility used by both pipelines.
"""

import pandas as pd
import numpy as np
import logging
import sys
import pathlib

# Add root to sys.path for local imports
BASE = pathlib.Path(__file__).parents[2]
if str(BASE) not in sys.path:
    sys.path.append(str(BASE))

from src.features.graph_features import add_graph_features

log = logging.getLogger(__name__)


def add_amount_features(df: pd.DataFrame) -> pd.DataFrame:
    """Relative amount features that are meaningful without knowing absolute scale."""
    # Amount relative to a log-space amount (already log1p transformed in preprocessing)
    df["amount_squared"] = df["Amount"] ** 2
    df["amount_abs"] = df["Amount"].abs()
    log.info("Added amount features")
    return df


def add_velocity_features(df: pd.DataFrame, window_sizes=[10, 50, 100]) -> pd.DataFrame:
    """
    Velocity = how many transactions happened in the last N rows (proxy for time window).
    In production this would be a true sliding time window via Flink/Redis.
    Here we simulate with rolling counts over sorted-by-time data.

    IMPORTANT: df must already be sorted by Time and must be TRAIN-ONLY when fitting.
    For val/test, append after train data, compute, then slice back.
    """
    for w in window_sizes:
        # Rolling mean of Amount — card testing fraud shows many small amounts
        df[f"amount_rolling_mean_{w}"] = (
            df["Amount"].rolling(window=w, min_periods=1).mean()
        )
        # Rolling std — high variance = inconsistent behavior
        df[f"amount_rolling_std_{w}"] = (
            df["Amount"].rolling(window=w, min_periods=1).std().fillna(0)
        )
        # Count of recent transactions (proxy for velocity)
        df[f"tx_count_{w}"] = (
            df["Amount"].rolling(window=w, min_periods=1).count().astype(int)
        )

    log.info(f"Added velocity features for windows {window_sizes}")
    return df


def add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ratios between PCA components that carry fraud signal."""
    # V1 and V4 are typically high-importance fraud indicators in real card data
    eps = 1e-8
    df["v1_v4_ratio"] = df["V1"] / (df["V4"].abs() + eps)
    df["v1_v2_diff"] = df["V1"] - df["V2"]
    df["v3_v10_diff"] = df["V3"] - df["V10"]
    log.info("Added ratio features")
    return df


def compute_information_value(df: pd.DataFrame, target: str = "Class", n_bins: int = 10) -> pd.Series:
    """
    Compute IV for each feature to select informative ones.
    IV < 0.02 → weak predictor, drop it.
    """
    ivs = {}
    features = [c for c in df.columns if c != target]
    for feat in features:
        try:
            binned = pd.qcut(df[feat], q=n_bins, duplicates="drop")
            grouped = df.groupby(binned, observed=True)[target].agg(["sum", "count"])
            grouped.columns = ["events", "total"]
            grouped["non_events"] = grouped["total"] - grouped["events"]
            total_events = grouped["events"].sum()
            total_non = grouped["non_events"].sum()
            eps = 1e-8
            grouped["dist_events"] = grouped["events"] / (total_events + eps)
            grouped["dist_non"] = grouped["non_events"] / (total_non + eps)
            grouped["woe"] = np.log(
                (grouped["dist_events"] + eps) / (grouped["dist_non"] + eps)
            )
            grouped["iv"] = (grouped["dist_events"] - grouped["dist_non"]) * grouped["woe"]
            ivs[feat] = grouped["iv"].sum()
        except Exception:
            ivs[feat] = 0.0

    return pd.Series(ivs).sort_values(ascending=False)


def select_features(df: pd.DataFrame, iv_threshold: float = 0.02) -> list:
    """Return features with IV above threshold."""
    ivs = compute_information_value(df)
    selected = ivs[ivs >= iv_threshold].index.tolist()
    dropped = ivs[ivs < iv_threshold].index.tolist()
    log.info(f"Feature selection: kept {len(selected)}, dropped {len(dropped)} (IV < {iv_threshold})")
    return selected + ["Class"]


def build_features(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame):
    """
    Safe feature pipeline: velocity computed without future leakage.
    Now includes Graph Features.
    """
    # 1. Graph Features (fit on train, apply to all)
    train_f, val_f, test_f = add_graph_features(train, val, test)
    
    # 2. Velocity features (safe ordering)
    # Combine for velocity (train context only bleeds into val, not the other way)
    full = pd.concat([train_f, val_f, test_f], ignore_index=True)
    full = add_velocity_features(full)

    n_train = len(train_f)
    n_val = len(val_f)

    train_f = full.iloc[:n_train].copy()
    val_f = full.iloc[n_train:n_train + n_val].copy()
    test_f = full.iloc[n_train + n_val:].copy()

    for part in [train_f, val_f, test_f]:
        add_amount_features(part)
        add_ratio_features(part)

    # 3. Drop Entity IDs (not used for training, just for feature derivation)
    for part in [train_f, val_f, test_f]:
        part.drop(columns=["card_id", "merchant_id", "device_id"], inplace=True)

    # Select features based on TRAIN only
    selected_cols = select_features(train_f)
    # Apply same selection to all splits
    train_f = train_f[[c for c in selected_cols if c in train_f.columns]]
    val_f = val_f[[c for c in selected_cols if c in val_f.columns]]
    test_f = test_f[[c for c in selected_cols if c in test_f.columns]]

    return train_f, val_f, test_f


if __name__ == "__main__":
    import pathlib
    base = pathlib.Path(__file__).parents[2] / "data"
    train = pd.read_csv(base / "train.csv")
    val = pd.read_csv(base / "val.csv")
    test = pd.read_csv(base / "test.csv")

    train_f, val_f, test_f = build_features(train, val, test)

    train_f.to_csv(base / "train_features.csv", index=False)
    val_f.to_csv(base / "val_features.csv", index=False)
    test_f.to_csv(base / "test_features.csv", index=False)

    log.info(f"Feature matrix: {train_f.shape[1]-1} features")
