"""
Preprocessing Pipeline
-----------------------
CHALLENGES FACED:
1. Null Amount values — median imputation per class leaks label info if done globally.
   FIX: fit imputer on TRAIN only, transform train+val+test with same fitted value.

2. Negative amounts — not errors in refund context, but noise for fraud detection.
   FIX: clip to 0 (refund is not fraud signal we want), log-transform for scaling.

3. Amount scale vs V-features — V1-V28 are already PCA-transformed and roughly
   standardised. Amount is raw and skewed, causing gradient issues in tree models.
   FIX: log1p + StandardScaler fitted only on train.

4. Time feature — raw seconds since first transaction. Cyclical pattern matters
   (nighttime fraud peaks). Converted to hour-of-day sin/cos encoding.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import pickle
import logging
from pathlib import Path

log = logging.getLogger(__name__)


def time_to_cyclical(df: pd.DataFrame) -> pd.DataFrame:
    """Convert raw seconds → hour-of-day → sin/cos to capture daily cycle."""
    hour = (df["Time"] % 86400) / 3600  # 0–24
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df = df.drop(columns=["Time"])
    log.info("Converted Time → hour_sin, hour_cos (cyclical encoding)")
    return df


def fix_amount(df: pd.DataFrame) -> pd.DataFrame:
    """Clip negatives, apply log1p to reduce skew."""
    df["Amount"] = df["Amount"].clip(lower=0)
    df["Amount"] = np.log1p(df["Amount"])
    log.info("Clipped negative Amount values, applied log1p transform")
    return df


class Preprocessor:
    """
    Stateful preprocessor — fit on train, transform everything else.
    Saves artifacts so serving uses identical transforms.
    """

    def __init__(self):
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self._fitted = False

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = time_to_cyclical(df.copy())
        df = fix_amount(df)

        feature_cols = [c for c in df.columns if c not in ["Class", "card_id", "merchant_id", "device_id"]]
        df[feature_cols] = self.imputer.fit_transform(df[feature_cols])
        df[feature_cols] = self.scaler.fit_transform(df[feature_cols])
        self._fitted = True
        log.info("Preprocessor fitted and transformed train set")
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        assert self._fitted, "Call fit_transform on train first"
        df = time_to_cyclical(df.copy())
        df = fix_amount(df)
        feature_cols = [c for c in df.columns if c not in ["Class", "card_id", "merchant_id", "device_id"]]
        df[feature_cols] = self.imputer.transform(df[feature_cols])
        df[feature_cols] = self.scaler.transform(df[feature_cols])
        return df

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)
        log.info(f"Preprocessor saved → {path}")

    @staticmethod
    def load(path: str) -> "Preprocessor":
        with open(path, "rb") as f:
            return pickle.load(f)


def time_aware_split(df: pd.DataFrame, val_frac=0.15, test_frac=0.15):
    """
    CRITICAL: Always split by time, never randomly.
    Random splits let future features leak into training.
    """
    df = df.sort_values("Time") if "Time" in df.columns else df
    n = len(df)
    train_end = int(n * (1 - val_frac - test_frac))
    val_end = int(n * (1 - test_frac))

    train = df.iloc[:train_end].reset_index(drop=True)
    val = df.iloc[train_end:val_end].reset_index(drop=True)
    test = df.iloc[val_end:].reset_index(drop=True)

    log.info(f"Time split → train: {len(train)}, val: {len(val)}, test: {len(test)}")
    return train, val, test


if __name__ == "__main__":
    base = Path(__file__).parents[2]
    df = pd.read_csv(base / "data" / "creditcard_ingested.csv")

    train, val, test = time_aware_split(df)

    prep = Preprocessor()
    train_p = prep.fit_transform(train)
    val_p = prep.transform(val)
    test_p = prep.transform(test)

    train_p.to_csv(base / "data" / "train.csv", index=False)
    val_p.to_csv(base / "data" / "val.csv", index=False)
    test_p.to_csv(base / "data" / "test.csv", index=False)
    prep.save(str(base / "data" / "preprocessor.pkl"))

    log.info("Preprocessing complete.")
