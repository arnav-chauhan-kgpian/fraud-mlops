"""
FastAPI Inference Service
--------------------------
Serves fraud detection predictions via a REST API.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import joblib
import pathlib
import sys
import logging

# Add src to path to allow importing Preprocessor
BASE = pathlib.Path(__file__).parents[2]
sys.path.append(str(BASE))

import json

from src.preprocessing.preprocess import Preprocessor

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

app = FastAPI(title="Fraud Detection API", version="1.0.0")

# Global variables for model and preprocessor
model = None
preprocessor = None
serving_meta = None

class Transaction(BaseModel):
    Time: float = Field(..., example=0.0)
    card_id: str = Field(..., example="C_123")
    merchant_id: str = Field(..., example="M_456")
    device_id: str = Field(..., example="D_789")
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float = Field(..., example=100.0)

@app.on_event("startup")
def load_artifacts():
    global model, preprocessor, serving_meta
    try:
        models_dir = BASE / "models"
        model = joblib.load(models_dir / "model.pkl")
        preprocessor = Preprocessor.load(str(models_dir / "preprocessor.pkl"))

        # serving_meta.json holds the exact (ordered) serving feature list and the
        # trained optimal decision threshold produced by train_serving.py.
        with open(models_dir / "serving_meta.json", "r") as f:
            serving_meta = json.load(f)

        log.info(
            "Artifacts loaded: %d serving features, threshold=%.4f",
            len(serving_meta["features"]), serving_meta["threshold"],
        )
    except Exception as e:
        log.error(f"Error loading artifacts: {e}")
        # Note: In production, you might want to exit if artifacts fail to load
        pass

@app.post("/predict")
def predict(transaction: Transaction):
    if model is None or preprocessor is None or serving_meta is None:
        raise HTTPException(status_code=503, detail="Model artifacts not loaded")

    try:
        # 1. Convert to DataFrame
        df = pd.DataFrame([transaction.dict()])

        # 2. Serving-safe preprocessing only (Time->cyclical, log1p+scale, impute).
        #    No rolling-window or entity-graph features: they require transaction
        #    history / the full graph and cannot be computed for a single
        #    transaction. Zero-filling them caused train/serve skew and made
        #    genuine transactions score as fraud.
        df_p = preprocessor.transform(df)

        # 3. Select the exact feature set (and order) the model was trained on.
        df_final = df_p[serving_meta["features"]]

        # 4. Predict using the trained optimal threshold (not a hardcoded 0.5).
        prob = float(model.predict_proba(df_final)[0][1])
        threshold = float(serving_meta["threshold"])

        return {
            "fraud_probability": prob,
            "is_fraud": bool(prob >= threshold),
            "threshold_used": threshold,
        }
    except Exception as e:
        log.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
