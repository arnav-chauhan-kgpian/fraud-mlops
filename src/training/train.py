"""
Model Training with MLflow Experiment Tracking
-----------------------------------------------
CHALLENGES FACED:

1. CLASS IMBALANCE (0.17% fraud rate)
   Accuracy was 99.83% predicting all-negative. Completely useless.
   FIX: SMOTE oversampling on train only (never on val/test — that's leakage).
        Also set scale_pos_weight in XGBoost = n_negative / n_positive.

2. WRONG METRIC OPTIMISATION
   Initially tuned for ROC-AUC → model was decent but caught only 60% of fraud.
   ROC-AUC is misleading on imbalanced data (TN swamps the metric).
   FIX: switched primary metric to PR-AUC (Precision-Recall AUC).
        KS statistic added as secondary business metric.

3. THRESHOLD: NOT 0.5
   Default 0.5 threshold gave 94% precision but 45% recall.
   Business wanted: catch 80% of fraud, minimise false positives.
   FIX: swept threshold from 0.01 to 0.99 on val set, picked threshold
        that maximises F-beta (beta=2, recall weighted 2x more than precision).

4. MODEL REPRODUCIBILITY
   Results differed across runs — random seeds not fully set.
   FIX: explicit random_state in all estimators + SMOTE + train/val split.
        MLflow logs all params so runs are reproducible.

5. MLFLOW EXPERIMENT NOT FOUND
   First run crashed because experiment wasn't created before logging.
   FIX: mlflow.set_experiment() creates it if missing.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score, roc_auc_score,
    precision_recall_curve, f1_score, classification_report
)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
import json
import logging
import pathlib
import sys

BASE = pathlib.Path(__file__).parents[2]
if str(BASE) not in sys.path:
    sys.path.append(str(BASE))

from src.features.graph_features import GraphFeatureExtractor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)


def load_data():
    train = pd.read_csv(BASE / "data" / "train_features.csv")
    val = pd.read_csv(BASE / "data" / "val_features.csv")
    test = pd.read_csv(BASE / "data" / "test_features.csv")
    return train, val, test


def split_xy(df):
    return df.drop(columns=["Class"]), df["Class"]


def apply_smote(X_train, y_train):
    """Oversample minority class on TRAIN ONLY."""
    log.info(f"Before SMOTE: {y_train.value_counts().to_dict()}")
    sm = SMOTE(random_state=42, k_neighbors=5)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    log.info(f"After SMOTE: {pd.Series(y_res).value_counts().to_dict()}")
    return X_res, y_res


def find_best_threshold(y_true, y_prob, beta=2.0):
    """
    Sweep thresholds and find one maximising F-beta.
    beta=2 weights recall twice as much as precision.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    fbeta = ((1 + beta**2) * precision * recall) / (beta**2 * precision + recall + 1e-8)
    best_idx = np.argmax(fbeta[:-1])
    return float(thresholds[best_idx]), float(fbeta[best_idx])


def compute_ks(y_true, y_prob):
    """KS statistic: max separation between fraud/legit score distributions."""
    fraud_scores = y_prob[y_true == 1]
    legit_scores = y_prob[y_true == 0]
    thresholds = np.linspace(0, 1, 200)
    ks_values = [abs(
        (fraud_scores >= t).mean() - (legit_scores >= t).mean()
    ) for t in thresholds]
    return float(max(ks_values))


def evaluate(model, X, y, threshold=0.5, model_type="sklearn"):
    if model_type == "xgb":
        y_prob = model.predict_proba(X)[:, 1]
    elif model_type == "lgb":
        y_prob = model.predict_proba(X)[:, 1]
    else:
        y_prob = model.predict_proba(X)[:, 1]

    y_pred = (y_prob >= threshold).astype(int)
    return {
        "pr_auc": round(average_precision_score(y, y_prob), 4),
        "roc_auc": round(roc_auc_score(y, y_prob), 4),
        "f1": round(f1_score(y, y_pred), 4),
        "ks": round(compute_ks(y.values, y_prob), 4),
        "recall_at_threshold": round((y_pred[y == 1] == 1).mean(), 4),
        "precision_at_threshold": round(
            y_pred[y == 1].sum() / max(y_pred.sum(), 1), 4
        ),
    }


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import joblib
import os
from src.explainability.shap_analysis import generate_shap_explanations

def save_plots(model_name, y_true, y_prob, threshold, dataset_type="test"):
    """Save PR curve, ROC curve, and Confusion Matrix for the model."""
    plots_dir = BASE / "data" / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    # 1. Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'{model_name} (PR-AUC={average_precision_score(y_true, y_prob):.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {model_name} ({dataset_type.upper()} set)')
    plt.legend()
    plt.grid(True)
    plt.savefig(plots_dir / f"{model_name}_{dataset_type}_pr_curve.png")
    plt.close()
    
    # 2. ROC Curve
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc_val = roc_auc_score(y_true, y_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'{model_name} (ROC-AUC={roc_auc_val:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name} ({dataset_type.upper()} set)')
    plt.legend()
    plt.grid(True)
    plt.savefig(plots_dir / f"{model_name}_{dataset_type}_roc_curve.png")
    plt.close()
    
    # 3. Confusion Matrix
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {model_name} ({dataset_type.upper()} set)\n(Threshold={threshold:.3f})')
    plt.savefig(plots_dir / f"{model_name}_{dataset_type}_confusion_matrix.png")
    plt.close()

def get_feature_importance(model, model_name, feature_names):
    """Extract feature importance from the model."""
    if model_name == "logistic_regression":
        # Logistic Regression uses coefficients as importance
        importance = np.abs(model.coef_[0])
    elif model_name == "random_forest":
        importance = model.feature_importances_
    elif model_name == "xgboost":
        importance = model.feature_importances_
    elif model_name == "lightgbm":
        importance = model.feature_importances_
    else:
        return None
    
    feat_imp = pd.DataFrame({
        "feature": feature_names,
        "importance": importance
    }).sort_values("importance", ascending=False)
    
    return feat_imp

def train_models():
    # Fit and save GraphFeatureExtractor on raw training data
    raw_train = pd.read_csv(BASE / "data" / "train.csv")
    graph_extractor = GraphFeatureExtractor()
    graph_extractor.fit(raw_train)
    
    # Create models directory if it doesn't exist
    (BASE / "models").mkdir(exist_ok=True)
    
    joblib.dump(graph_extractor, BASE / "models" / "graph_extractor.pkl")
    log.info("Graph Feature Extractor saved to models/graph_extractor.pkl")

    train, val, test = load_data()
    X_train, y_train = split_xy(train)
    X_val, y_val = split_xy(val)
    X_test, y_test = split_xy(test)

    feature_names = X_train.columns.tolist()
    X_train_sm, y_train_sm = apply_smote(X_train, y_train)

    scale_pos = int((y_train == 0).sum() / max((y_train == 1).sum(), 1))

    models = {
        "logistic_regression": LogisticRegression(
            C=0.1, max_iter=1000, class_weight="balanced", random_state=42
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200, max_depth=12, class_weight="balanced",
            n_jobs=-1, random_state=42
        ),
        "xgboost": xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            scale_pos_weight=scale_pos, eval_metric="aucpr",
            use_label_encoder=False, random_state=42, n_jobs=-1
        ),
        "lightgbm": lgb.LGBMClassifier(
            n_estimators=300, max_depth=8, learning_rate=0.05,
            is_unbalance=True, random_state=42, n_jobs=-1,
            importance_type="gain", verbosity=-1
        ),
    }

    mlflow.set_tracking_uri("file:///" + str(BASE / "mlruns").replace("\\", "/"))
    mlflow.set_experiment("fraud_detection")

    results = {}
    best_models_objects = {}

    for name, model in models.items():
        log.info(f"\n{'='*50}\nTraining: {name}\n{'='*50}")

        with mlflow.start_run(run_name=name):
            mlflow.log_params({
                "model": name,
                "smote": True if name != "lightgbm" else False, # LGBM uses is_unbalance
                "imbalance_ratio": scale_pos,
                "train_size": len(X_train_sm) if name != "lightgbm" else len(X_train),
                "val_size": len(X_val),
            })

            # Train
            if name == "lightgbm":
                # Use raw train (not SMOTE) for LGBM as it has built-in is_unbalance
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)])
                y_prob_val = model.predict_proba(X_val)[:, 1]
                y_prob_test = model.predict_proba(X_test)[:, 1]
                y_prob_train = model.predict_proba(X_train)[:, 1]
                current_X_train = X_train
                current_y_train = y_train
            else:
                if isinstance(model, xgb.XGBClassifier):
                    model.fit(
                        X_train_sm, y_train_sm,
                        eval_set=[(X_val, y_val)],
                        verbose=False
                    )
                else:
                    model.fit(X_train_sm, y_train_sm)
                
                y_prob_val = model.predict_proba(X_val)[:, 1]
                y_prob_test = model.predict_proba(X_test)[:, 1]
                y_prob_train = model.predict_proba(X_train_sm)[:, 1]
                current_X_train = X_train_sm
                current_y_train = y_train_sm

            # Find best threshold on validation
            best_thresh, best_fbeta = find_best_threshold(y_val, y_prob_val)
            log.info(f"Best threshold: {best_thresh:.3f} (F2={best_fbeta:.4f})")

            # Save plots for both training and testing sets
            save_plots(name, y_val, y_prob_val, best_thresh, "validation")
            save_plots(name, y_test, y_prob_test, best_thresh, "test")
            save_plots(name, current_y_train, y_prob_train, best_thresh, "training")
            
            # 1. Feature Importance
            feat_imp = get_feature_importance(model, name, feature_names)
            if feat_imp is not None:
                feat_imp_path = BASE / "data" / f"feature_importance_{name}.json"
                feat_imp.to_json(feat_imp_path, orient="records")
                mlflow.log_artifact(str(feat_imp_path))
                log.info(f"Feature importance saved for {name}")

            # 2. SHAP Explanations
            generate_shap_explanations(model, X_test, model_name=name)
            shap_plots_dir = BASE / "data" / "plots" / "shap"
            for plot_file in os.listdir(shap_plots_dir):
                if plot_file.startswith(name):
                    mlflow.log_artifact(str(shap_plots_dir / plot_file))

            # Evaluate
            model_type_code = "lgb" if name == "lightgbm" else ("xgb" if name == "xgboost" else "sklearn")
            val_metrics = evaluate(model, X_val, y_val, threshold=best_thresh, model_type=model_type_code)
            test_metrics = evaluate(model, X_test, y_test, threshold=best_thresh, model_type=model_type_code)

            mlflow.log_param("best_threshold", round(best_thresh, 4))
            mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})
            mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

            if name == "lightgbm":
                mlflow.lightgbm.log_model(model, artifact_path="model")
            elif name == "xgboost":
                mlflow.xgboost.log_model(model, artifact_path="model")
            else:
                mlflow.sklearn.log_model(model, artifact_path="model")

            results[name] = {
                "val": val_metrics,
                "test": test_metrics,
                "threshold": round(best_thresh, 4)
            }
            best_models_objects[name] = model

            log.info(f"Val PR-AUC: {val_metrics['pr_auc']} | Test PR-AUC: {test_metrics['pr_auc']}")

    # Save results summary
    results_path = BASE / "data" / "training_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    best_model_name = max(results, key=lambda k: results[k]["val"]["pr_auc"])
    log.info(f"\nBest model: {best_model_name} (val PR-AUC: {results[best_model_name]['val']['pr_auc']})")
    
    # Save the best model and preprocessor for serving
    models_dir = BASE / "models"
    models_dir.mkdir(exist_ok=True)
    best_model = best_models_objects[best_model_name]
    joblib.dump(best_model, models_dir / "model.pkl")
    
    # Save consolidated feature importance for the best model
    best_feat_imp = get_feature_importance(best_model, best_model_name, feature_names)
    if best_feat_imp is not None:
        best_feat_imp.to_json(BASE / "data" / "feature_importance.json", orient="records")
        log.info("Consolidated feature importance for best model saved to data/feature_importance.json")
    
    # Copy preprocessor if it exists
    preprocessor_path = BASE / "data" / "preprocessor.pkl"
    if preprocessor_path.exists():
        import shutil
        shutil.copy(preprocessor_path, models_dir / "preprocessor.pkl")
        log.info(f"Best model ({best_model_name}) and preprocessor saved to {models_dir}")
        
    return results


if __name__ == "__main__":
    results = train_models()
    print("\n=== FINAL RESULTS ===")
    for model, metrics in results.items():
        print(f"\n{model}:")
        print(f"  Val  PR-AUC={metrics['val']['pr_auc']}  KS={metrics['val']['ks']}  Recall={metrics['val']['recall_at_threshold']}")
        print(f"  Test PR-AUC={metrics['test']['pr_auc']}  KS={metrics['test']['ks']}  Recall={metrics['test']['recall_at_threshold']}")
        print(f"  Threshold: {metrics['threshold']}")
