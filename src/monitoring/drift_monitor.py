"""
Data & Model Drift Monitoring
------------------------------
CHALLENGES FACED:

1. FRAUD PATTERN SHIFTS WEEKLY
   Fraudsters adapt — a model trained in January degrades by March.
   Without monitoring, nobody notices until chargebacks spike.
   FIX: track Population Stability Index (PSI) on every feature daily.
        PSI > 0.2 → alert + queue retraining.

2. LABEL DELAY MAKES PERFORMANCE MONITORING HARD
   We don't know if a transaction was fraud for 30–45 days (chargeback window).
   So we can't directly monitor "is recall dropping?"
   FIX: monitor SCORE DISTRIBUTION instead. If the fraud score histogram shifts,
        model behavior has changed — proxy for real performance drift.

3. ALERT FATIGUE
   First version alerted on every tiny drift → team started ignoring alerts.
   FIX: tiered alerting. PSI 0.1–0.2 = warning (log only). PSI > 0.2 = critical (trigger retrain).
"""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def compute_psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    """
    Population Stability Index.
    PSI < 0.1  → no significant drift
    PSI 0.1–0.2 → moderate drift, monitor
    PSI > 0.2  → significant drift, retrain
    """
    # Use quantile bins from expected (training) distribution
    bins = np.nanquantile(expected, np.linspace(0, 1, n_bins + 1))
    bins[0] = -np.inf
    bins[-1] = np.inf

    expected_counts = np.histogram(expected, bins=bins)[0]
    actual_counts = np.histogram(actual, bins=bins)[0]

    eps = 1e-6
    expected_pct = expected_counts / (len(expected) + eps)
    actual_pct = actual_counts / (len(actual) + eps)

    psi = np.sum((actual_pct - expected_pct) * np.log((actual_pct + eps) / (expected_pct + eps)))
    return float(psi)


def monitor_features(train_df: pd.DataFrame, current_df: pd.DataFrame, threshold_warn=0.1, threshold_alert=0.2):
    """Compare feature distributions between training data and current production data."""
    feature_cols = [c for c in train_df.columns if c != "Class"]
    report = {"features": {}, "alerts": [], "warnings": []}

    for col in feature_cols:
        psi = compute_psi(train_df[col].dropna().values, current_df[col].dropna().values)
        report["features"][col] = round(psi, 4)

        if psi > threshold_alert:
            report["alerts"].append({"feature": col, "psi": round(psi, 4), "action": "RETRAIN"})
            log.warning(f"[DRIFT ALERT] {col}: PSI={psi:.4f} > {threshold_alert} → trigger retraining")
        elif psi > threshold_warn:
            report["warnings"].append({"feature": col, "psi": round(psi, 4), "action": "MONITOR"})
            log.info(f"[DRIFT WARNING] {col}: PSI={psi:.4f}")

    report["summary"] = {
        "total_features": len(feature_cols),
        "alert_count": len(report["alerts"]),
        "warning_count": len(report["warnings"]),
        "max_psi": max(report["features"].values()) if report["features"] else 0,
        "retrain_triggered": len(report["alerts"]) > 0,
    }

    log.info(f"Drift report: {report['summary']}")
    return report


import matplotlib.pyplot as plt
import seaborn as sns

def monitor_score_distribution(train_scores: np.ndarray, current_scores: np.ndarray):
    """
    Monitor model output score distribution as proxy for label drift.
    When fraud patterns change, score distribution shifts before labels arrive.
    """
    psi = compute_psi(train_scores, current_scores)
    result = {
        "score_psi": round(psi, 4),
        "train_mean_score": round(float(train_scores.mean()), 4),
        "current_mean_score": round(float(current_scores.mean()), 4),
        "drift_level": "ALERT" if psi > 0.2 else ("WARNING" if psi > 0.1 else "OK"),
    }
    log.info(f"Score distribution drift: PSI={psi:.4f} → {result['drift_level']}")
    
    # Save Plot
    plots_dir = Path(__file__).parents[2] / "data" / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    sns.kdeplot(train_scores, label='Training Scores', fill=True)
    sns.kdeplot(current_scores, label='Production Scores', fill=True)
    plt.title(f'Score Distribution Drift (PSI={psi:.4f})')
    plt.xlabel('Fraud Score')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig(plots_dir / "score_drift_distribution.png")
    plt.close()
    
    return result


if __name__ == "__main__":
    BASE = Path(__file__).parents[2]
    train = pd.read_csv(BASE / "data" / "train_features.csv")
    val = pd.read_csv(BASE / "data" / "val_features.csv")

    # Simulate production drift: add Gaussian noise to some features
    current = val.copy()
    for col in ["V1", "V2", "Amount", "amount_rolling_mean_10"]:
        if col in current.columns:
            current[col] = current[col] + np.random.normal(0, 0.5, len(current))

    report = monitor_features(train, current)

    out = BASE / "data" / "drift_report.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    log.info(f"Drift report saved → {out}")

    # Simulate score drift
    train_scores = np.random.beta(0.5, 10, 10000)
    current_scores = np.random.beta(0.7, 8, 5000)  # slight shift
    score_report = monitor_score_distribution(train_scores, current_scores)
    print(json.dumps(score_report, indent=2))
