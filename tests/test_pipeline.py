"""
tests/test_pipeline.py
-----------------------
Unit tests for each stage of the fraud detection pipeline.
Run with: pytest tests/ -v
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))


# ── Ingestion tests ──────────────────────────────────────────────────────────

class TestIngestion:
    def _make_df(self):
        n = 100
        df = pd.DataFrame({f"V{i}": np.random.randn(n) for i in range(1, 29)})
        df["Time"] = np.sort(np.random.uniform(0, 172800, n))
        df["Amount"] = np.random.lognormal(3, 1.5, n)
        df["Class"] = np.random.choice([0, 1], n, p=[0.998, 0.002])
        return df

    def test_duplicate_removal(self):
        from src.ingestion.ingest import remove_duplicates
        df = self._make_df()
        # Inject 5 duplicates
        dups = df.iloc[:5].copy()
        df_with_dups = pd.concat([df, dups]).reset_index(drop=True)
        cleaned = remove_duplicates(df_with_dups)
        assert len(cleaned) == len(df)

    def test_validate_detects_nulls(self):
        from src.ingestion.ingest import validate
        df = self._make_df()
        df.loc[0:5, "Amount"] = np.nan
        report = validate(df)
        assert report["null_counts"]["Amount"] == 6

    def test_validate_detects_negatives(self):
        from src.ingestion.ingest import validate
        df = self._make_df()
        df.loc[0:2, "Amount"] = -10.0
        report = validate(df)
        assert report["negative_amount_rows"] == 3

    def test_validate_no_missing_columns(self):
        from src.ingestion.ingest import validate
        df = self._make_df()
        report = validate(df)
        assert report["missing_columns"] == []


# ── Preprocessing tests ───────────────────────────────────────────────────────

class TestPreprocessing:
    def _make_split_df(self, n=500):
        df = pd.DataFrame({f"V{i}": np.random.randn(n) for i in range(1, 29)})
        df["Time"] = np.sort(np.random.uniform(0, 172800, n))
        df["Amount"] = np.random.lognormal(3, 1.5, n)
        df["Class"] = np.random.choice([0, 1], n, p=[0.998, 0.002])
        return df

    def test_time_split_is_ordered(self):
        from src.preprocessing.preprocess import time_aware_split
        df = self._make_split_df()
        train, val, test = time_aware_split(df)
        assert train["Time"].max() <= val["Time"].min(), "Train/val time boundary violated"
        assert val["Time"].max() <= test["Time"].min(), "Val/test time boundary violated"

    def test_time_split_sizes(self):
        from src.preprocessing.preprocess import time_aware_split
        df = self._make_split_df(n=1000)
        train, val, test = time_aware_split(df, val_frac=0.15, test_frac=0.15)
        total = len(train) + len(val) + len(test)
        assert total == len(df)
        assert abs(len(train) / len(df) - 0.70) < 0.02

    def test_preprocessor_no_refitting_on_val(self):
        from src.preprocessing.preprocess import Preprocessor, time_aware_split
        df = self._make_split_df(n=600)
        train, val, _ = time_aware_split(df)

        prep = Preprocessor()
        train_p = prep.fit_transform(train)
        val_p1 = prep.transform(val)
        val_p2 = prep.transform(val)

        # Transform is deterministic (no refitting happens)
        pd.testing.assert_frame_equal(val_p1, val_p2)

    def test_negative_amount_clipped(self):
        from src.preprocessing.preprocess import fix_amount
        df = pd.DataFrame({"Amount": [-5.0, 0.0, 10.0, 100.0]})
        result = fix_amount(df)
        assert (result["Amount"] >= 0).all(), "Negative amounts not clipped"

    def test_cyclical_time_encoding(self):
        from src.preprocessing.preprocess import time_to_cyclical
        df = pd.DataFrame({"Time": [0, 43200, 86400], "Amount": [10, 20, 30]})
        result = time_to_cyclical(df)
        assert "hour_sin" in result.columns
        assert "hour_cos" in result.columns
        assert "Time" not in result.columns
        # sin²+cos²=1 for any angle
        for _, row in result.iterrows():
            assert abs(row["hour_sin"]**2 + row["hour_cos"]**2 - 1.0) < 1e-6

    def test_preprocessor_no_nan_output(self):
        from src.preprocessing.preprocess import Preprocessor, time_aware_split
        df = self._make_split_df(n=300)
        # Inject some nulls
        df.loc[0:10, "Amount"] = np.nan
        train, val, _ = time_aware_split(df)
        prep = Preprocessor()
        train_p = prep.fit_transform(train)
        val_p = prep.transform(val)
        assert not train_p.isnull().any().any(), "NaNs remain in train after preprocessing"
        assert not val_p.isnull().any().any(), "NaNs remain in val after preprocessing"


# ── Feature Engineering tests ────────────────────────────────────────────────

class TestFeatureEngineering:
    def _make_preprocessed(self, n=300):
        df = pd.DataFrame({f"V{i}": np.random.randn(n) for i in range(1, 29)})
        df["hour_sin"] = np.sin(2 * np.pi * np.random.uniform(0, 24, n) / 24)
        df["hour_cos"] = np.cos(2 * np.pi * np.random.uniform(0, 24, n) / 24)
        df["Amount"] = np.random.randn(n)
        df["Class"] = np.random.choice([0, 1], n, p=[0.998, 0.002])
        return df

    def test_velocity_features_added(self):
        from src.features.feature_engineering import add_velocity_features
        df = self._make_preprocessed()
        result = add_velocity_features(df, window_sizes=[10])
        assert "amount_rolling_mean_10" in result.columns
        assert "amount_rolling_std_10" in result.columns
        assert "tx_count_10" in result.columns

    def test_ratio_features_added(self):
        from src.features.feature_engineering import add_ratio_features
        df = self._make_preprocessed()
        result = add_ratio_features(df)
        assert "v1_v4_ratio" in result.columns
        assert "v1_v2_diff" in result.columns

    def test_no_inf_in_ratio_features(self):
        from src.features.feature_engineering import add_ratio_features
        df = self._make_preprocessed()
        # Make V4 zero to test the epsilon guard
        df["V4"] = 0.0
        result = add_ratio_features(df)
        assert not np.isinf(result["v1_v4_ratio"]).any(), "Inf values in ratio feature"

    def test_feature_selection_reduces_columns(self):
        from src.features.feature_engineering import select_features
        df = self._make_preprocessed()
        selected = select_features(df, iv_threshold=0.0)
        # At threshold 0.0, should keep at least the V features
        assert "Class" in selected
        assert len(selected) >= 10


# ── Monitoring tests ──────────────────────────────────────────────────────────

class TestMonitoring:
    def test_psi_identical_distributions(self):
        from src.monitoring.drift_monitor import compute_psi
        data = np.random.normal(0, 1, 1000)
        psi = compute_psi(data, data.copy())
        assert psi < 0.05, f"PSI should be near 0 for identical distributions, got {psi}"

    def test_psi_different_distributions(self):
        from src.monitoring.drift_monitor import compute_psi
        data_a = np.random.normal(0, 1, 1000)
        data_b = np.random.normal(5, 1, 1000)  # large shift
        psi = compute_psi(data_a, data_b)
        assert psi > 0.2, f"PSI should be > 0.2 for very different distributions, got {psi}"

    def test_drift_report_has_summary(self):
        from src.monitoring.drift_monitor import monitor_features
        n = 200
        train = pd.DataFrame({"feat_a": np.random.normal(0, 1, n), "Class": 0})
        current = pd.DataFrame({"feat_a": np.random.normal(0, 1, n), "Class": 0})
        report = monitor_features(train, current)
        assert "summary" in report
        assert "features" in report
        assert "feat_a" in report["features"]

    def test_alert_triggered_on_high_psi(self):
        from src.monitoring.drift_monitor import monitor_features
        n = 500
        train = pd.DataFrame({"feat_a": np.random.normal(0, 1, n), "Class": 0})
        # Massive distribution shift
        current = pd.DataFrame({"feat_a": np.random.normal(10, 1, n), "Class": 0})
        report = monitor_features(train, current, threshold_alert=0.2)
        assert report["summary"]["retrain_triggered"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
