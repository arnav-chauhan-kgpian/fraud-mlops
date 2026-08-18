"""
Model Explainability with SHAP
-------------------------------
Uses SHAP (SHapley Additive exPlanations) to explain model predictions.
This makes the fraud detection system auditable and interpretable.
"""

import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import pathlib
import joblib
import logging

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BASE = pathlib.Path(__file__).parents[2]

def generate_shap_explanations(model, X_test, model_name="xgboost"):
    """
    Generate and save SHAP summary plots and feature importance.
    """
    plots_dir = BASE / "data" / "plots" / "shap"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    log.info(f"Generating SHAP explanations for {model_name}...")
    
    # SHAP TreeExplainer for Tree-based models (XGBoost, LightGBM, RandomForest)
    try:
        if model_name in ["xgboost", "lightgbm", "random_forest"]:
            explainer = shap.TreeExplainer(model)
            # Use a subset of X_test if it's too large to speed up calculation
            X_sample = X_test.sample(min(1000, len(X_test)), random_state=42)
            shap_values = explainer.shap_values(X_sample)
            
            # 1. SHAP Summary Plot
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_sample, show=False)
            plt.title(f"SHAP Summary Plot - {model_name}")
            plt.tight_layout()
            plt.savefig(plots_dir / f"{model_name}_shap_summary.png")
            plt.close()
            
            # 2. SHAP Bar Plot (Feature Importance)
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
            plt.title(f"SHAP Feature Importance - {model_name}")
            plt.tight_layout()
            plt.savefig(plots_dir / f"{model_name}_shap_bar.png")
            plt.close()
            
            log.info(f"SHAP plots saved to {plots_dir}")
            return shap_values, X_sample
        else:
            log.warning(f"SHAP TreeExplainer not supported for {model_name}. Skipping.")
            return None, None
    except Exception as e:
        log.error(f"Error generating SHAP explanations: {e}")
        return None, None

def main():
    # Example usage (can be integrated into training pipeline)
    try:
        # Load test data
        test_df = pd.read_csv(BASE / "data" / "test_features.csv")
        X_test = test_df.drop(columns=["Class"])
        
        # This script expects a trained model. 
        # In a real pipeline, we'd pass the model object directly.
        # For now, we'll assume we can load the best model if it was saved.
        # But since train.py logs to MLflow, we might need to load from there or just run it after training.
        log.info("SHAP analysis script ready. Integrate with training pipeline for best results.")
        
    except FileNotFoundError:
        log.error("Test data not found. Run the ingestion and preprocessing pipelines first.")

if __name__ == "__main__":
    main()
