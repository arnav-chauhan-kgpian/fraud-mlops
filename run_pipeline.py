"""
run_pipeline.py
---------------
Single entry point to run the entire fraud detection MLOps pipeline.
Executes all 5 stages in order with clear logging between each step.

Usage:
    python run_pipeline.py
    python run_pipeline.py --skip-data   # skip dataset generation if already exists
"""

import argparse
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("pipeline.log"),
    ],
)
log = logging.getLogger(__name__)

BASE = Path(__file__).parent


def banner(title: str):
    width = 60
    log.info("=" * width)
    log.info(f"  {title}")
    log.info("=" * width)


def run_stage(name: str, func, *args, **kwargs):
    banner(f"STAGE: {name}")
    t0 = time.time()
    try:
        result = func(*args, **kwargs)
        elapsed = time.time() - t0
        log.info(f"✓ {name} completed in {elapsed:.1f}s")
        return result
    except Exception as e:
        log.error(f"✗ {name} FAILED: {e}")
        raise


def generate_dataset():
    """Generate synthetic dataset if raw CSV doesn't exist."""
    raw_path = BASE / "data" / "creditcard_raw.csv"
    if raw_path.exists():
        log.info(f"Dataset already exists at {raw_path} — skipping generation")
        return

    log.info("Generating synthetic credit card fraud dataset...")
    import pandas as pd
    import numpy as np
    from sklearn.datasets import make_classification

    (BASE / "data").mkdir(exist_ok=True)
    np.random.seed(42)
    n = 284807

    time_vals = np.sort(np.random.uniform(0, 172800, n))
    X, y = make_classification(
        n_samples=n, n_features=28, n_informative=15,
        n_redundant=5, n_clusters_per_class=2,
        weights=[0.9983, 0.0017], flip_y=0, random_state=42,
    )
    amount = np.where(
        y == 0, np.random.lognormal(4.0, 1.5, n),
        np.random.lognormal(2.5, 1.2, n),
    )
    amount = np.clip(amount, 0.01, 25000)

    cols = {f"V{i+1}": X[:, i] for i in range(28)}
    cols["Time"] = time_vals
    cols["Amount"] = amount
    cols["Class"] = y
    df = pd.DataFrame(cols)

    # Inject real-world data quality issues
    null_mask = np.random.random(n) < 0.003
    df.loc[null_mask, "Amount"] = np.nan

    dup_idx = np.random.choice(n, int(n * 0.001), replace=False)
    df = pd.concat([df, df.iloc[dup_idx].copy()]).reset_index(drop=True)

    neg_mask = np.random.random(len(df)) < 0.0005
    df.loc[neg_mask, "Amount"] = -df.loc[neg_mask, "Amount"]

    df.to_csv(raw_path, index=False)
    log.info(f"Saved {len(df)} rows, {int(df['Class'].sum())} fraud cases → {raw_path}")


def stage_ingest():
    import sys
    sys.path.insert(0, str(BASE))
    from src.ingestion.ingest import load_raw, validate, remove_duplicates, simulate_ids

    raw_path = BASE / "data" / "creditcard_raw.csv"
    df = load_raw(str(raw_path))
    report = validate(df)
    df = remove_duplicates(df)
    
    # Generate ID columns for graph features
    df = simulate_ids(df)
    
    out = BASE / "data" / "creditcard_ingested.csv"
    df.to_csv(out, index=False)
    log.info(f"Ingested data saved → {out}  ({len(df)} rows)")
    return report


def stage_preprocess():
    import sys
    sys.path.insert(0, str(BASE))
    import pandas as pd
    from src.preprocessing.preprocess import Preprocessor, time_aware_split

    df = pd.read_csv(BASE / "data" / "creditcard_ingested.csv")
    train, val, test = time_aware_split(df)

    prep = Preprocessor()
    train_p = prep.fit_transform(train)
    val_p = prep.transform(val)
    test_p = prep.transform(test)

    train_p.to_csv(BASE / "data" / "train.csv", index=False)
    val_p.to_csv(BASE / "data" / "val.csv", index=False)
    test_p.to_csv(BASE / "data" / "test.csv", index=False)
    prep.save(str(BASE / "data" / "preprocessor.pkl"))
    log.info(f"Splits saved. Train={len(train_p)}, Val={len(val_p)}, Test={len(test_p)}")


def stage_features():
    import sys
    sys.path.insert(0, str(BASE))
    import pandas as pd
    from src.features.feature_engineering import build_features

    train = pd.read_csv(BASE / "data" / "train.csv")
    val = pd.read_csv(BASE / "data" / "val.csv")
    test = pd.read_csv(BASE / "data" / "test.csv")

    train_f, val_f, test_f = build_features(train, val, test)

    train_f.to_csv(BASE / "data" / "train_features.csv", index=False)
    val_f.to_csv(BASE / "data" / "val_features.csv", index=False)
    test_f.to_csv(BASE / "data" / "test_features.csv", index=False)
    log.info(f"Feature matrix: {train_f.shape[1]-1} features (excl. label)")


def stage_train():
    import sys
    sys.path.insert(0, str(BASE))
    from src.training.train import train_models
    return train_models()


def stage_monitor():
    import sys
    sys.path.insert(0, str(BASE))
    import json
    import numpy as np
    import pandas as pd
    from src.monitoring.drift_monitor import monitor_features, monitor_score_distribution

    train = pd.read_csv(BASE / "data" / "train_features.csv")
    val = pd.read_csv(BASE / "data" / "val_features.csv")

    # Simulate production drift for demo purposes
    current = val.copy()
    for col in ["V1", "V2", "Amount", "amount_rolling_mean_10"]:
        if col in current.columns:
            current[col] = current[col] + np.random.normal(0, 0.5, len(current))

    report = monitor_features(train, current)

    out = BASE / "data" / "drift_report.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    log.info(f"Drift report → {out}")

    train_scores = np.random.beta(0.5, 10, 10000)
    current_scores = np.random.beta(0.7, 8, 5000)
    score_report = monitor_score_distribution(train_scores, current_scores)
    log.info(f"Score drift PSI={score_report['score_psi']} → {score_report['drift_level']}")
    return report


def main():
    parser = argparse.ArgumentParser(description="Fraud Detection MLOps Pipeline")
    parser.add_argument("--skip-data", action="store_true", help="Skip dataset generation")
    args = parser.parse_args()

    banner("FRAUD DETECTION MLOPS PIPELINE")
    log.info("Starting full pipeline run...\n")

    t_total = time.time()

    if not args.skip_data:
        run_stage("Dataset Generation", generate_dataset)

    run_stage("1 — Data Ingestion & Validation", stage_ingest)
    run_stage("2 — Preprocessing (time-split, scale, encode)", stage_preprocess)
    run_stage("3 — Feature Engineering (velocity, IV selection)", stage_features)
    results = run_stage("4 — Model Training (SMOTE + threshold tuning)", stage_train)
    run_stage("5 — Drift Monitoring (PSI per feature)", stage_monitor)

    elapsed = time.time() - t_total

    banner("PIPELINE COMPLETE")
    log.info(f"Total runtime: {elapsed:.1f}s")

    if results:
        log.info("\n=== FINAL MODEL RESULTS ===")
        for model_name, metrics in results.items():
            log.info(
                f"{model_name:25s}  "
                f"val PR-AUC={metrics['val']['pr_auc']:.4f}  "
                f"test PR-AUC={metrics['test']['pr_auc']:.4f}  "
                f"threshold={metrics['threshold']}"
            )

    log.info("\nNext steps:")
    log.info("  mlflow ui --backend-store-uri mlruns   → view experiment runs")
    log.info("  cat data/drift_report.json             → view drift alerts")
    log.info("  cat data/training_results.json         → view all model metrics")


if __name__ == "__main__":
    main()
