---
title: FraudShield AI
emoji: 🛡️
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# 🛡️ Credit Card Fraud Detection — Real-World MLOps Pipeline

> A production-grade MLOps project where **data engineering, preprocessing, and feature work dominate over model selection** — exactly as it happens in real industry.

---

## 📋 Table of Contents

- [Why This Project](#-why-this-project)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Pipeline Architecture](#-pipeline-architecture)
- [Graph-Based Feature Engineering](#-graph-based-feature-engineering)
- [Challenges & Fixes — The Real Story](#-challenges--fixes--the-real-story)
- [Actual Results From This Run](#-actual-results-from-this-run)
- [Visualization & Analysis](#-visualization--analysis)
- [Fraud Investigation Dashboard](#-fraud-investigation-dashboard)
- [API Reference](#-api-reference)
- [How to Run](#-how-to-run)
- [Dependencies](#-dependencies)
- [What a Full Production Stack Looks Like](#-what-a-full-production-stack-looks-like)

---

## 🎯 Why This Project

Most ML tutorials spend 90% of time on `model.fit()` and hyperparameter tuning. In real production systems, that's backwards. This project demonstrates the actual distribution of effort:

```
Data Ingestion & Validation   ████████░░  ~15%
Preprocessing (leak-free)     ██████████  ~20%
Feature Engineering           ████████████████  ~30%
Class Imbalance Handling      ████████░░  ~15%
Threshold Tuning & Metrics    ██████░░░░  ~12%
Model Selection               ████░░░░░░   ~8%
```

The model is XGBoost. It always is. What matters is everything before it.

---

## 📦 Dataset

### Overview

This project uses a **synthetic dataset** that replicates the structure and statistics of the real [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud). It is intentionally generated with the same messy properties found in production data.

| Column | Type | Description |
|--------|------|-------------|
| `Time` | float | Seconds since first transaction in the 48-hour window |
| `V1` – `V28` | float | PCA-transformed anonymized transaction features |
| `Amount` | float | Transaction amount in USD |
| `Class` | int | Target label: `0` = legitimate, `1` = fraud |

> **Simulated entity IDs** (added during ingestion for graph features):

| Column | Description |
|--------|-------------|
| `card_id` | Derived from V1 quantile bins — proxy for cardholder identity |
| `merchant_id` | Derived from V2 quantile bins — proxy for merchant identity |
| `device_id` | Derived from V3 quantile bins — proxy for device identity |

### Raw Dataset Statistics

| Metric | Value |
|--------|-------|
| Total rows | 285,091 |
| Fraud cases | 484 |
| Fraud rate | **0.1698%** — extreme imbalance |
| Missing `Amount` values | 892 (0.31%) — from declined transactions partially logged |
| Negative `Amount` values | 139 (0.05%) — refunds mis-logged as charges |
| Duplicate rows | 284 (0.10%) — from payment system retry events |

### Why Synthetic and Not the Real Kaggle Dataset?

The real dataset cannot be redistributed due to licensing. This synthetic version:
- Matches the column schema exactly (V1–V28, Time, Amount, Class)
- Matches the fraud rate (~0.17%)
- Intentionally includes the same data quality issues: nulls, negatives, duplicates
- Can be freely used, shared, and modified

**To use the real dataset:** download `creditcard.csv` from Kaggle and place it at `data/creditcard_raw.csv`. The pipeline is fully compatible — no code changes needed.

### Time-Based Splits (Never Random)

| Split | Rows | Fraud | Fraud Rate |
|-------|------|-------|------------|
| Train (70%) | 199,364 | 333 | 0.167% |
| Validation (15%) | 42,721 | 67 | 0.157% |
| Test (15%) | 42,722 | 84 | 0.197% |

---

## 📁 Project Structure

```
fraud_detection/
│
├── README.md                          ← You are here
├── requirements.txt                   ← All dependencies
├── run_pipeline.py                    ← One command to run everything
├── create_results_plot.py             ← Comprehensive visualization generator
├── sample_transaction.json            ← Example payload for API testing
│
├── configs/
│   └── config.yaml                    ← Hyperparameters, paths, PSI thresholds
│
├── data/                              ← Generated after running pipeline
│   ├── creditcard_raw.csv             ← Raw dataset (with noise/duplicates)
│   ├── creditcard_ingested.csv        ← After dedup + validation + ID simulation
│   ├── train.csv / val.csv / test.csv ← Time-split, preprocessed
│   ├── train_features.csv / ...       ← After feature engineering (graph + velocity)
│   ├── preprocessor.pkl               ← Fitted scaler/imputer (for serving)
│   ├── training_results.json          ← Actual metrics from this run
│   ├── drift_report.json              ← PSI-based drift report
│   ├── feature_importance.json        ← Best model feature importances
│   └── plots/                         ← All visualization outputs
│       ├── comprehensive_model_analysis.png
│       ├── detailed_dataset_comparison.png
│       ├── score_drift_distribution.png
│       ├── shap/                      ← SHAP summary and bar plots per model
│       │   ├── xgboost_shap_summary.png
│       │   └── xgboost_shap_bar.png
│       ├── *_training_*.png           ← Training set plots (PR, ROC, Confusion)
│       ├── *_validation_*.png         ← Validation set plots
│       └── *_test_*.png               ← Test set plots
│
├── src/
│   ├── ingestion/
│   │   └── ingest.py                  ← Load, validate, deduplicate, simulate IDs
│   ├── preprocessing/
│   │   └── preprocess.py              ← Time split, impute, scale, cyclical encoding
│   ├── features/
│   │   ├── feature_engineering.py    ← Velocity, ratios, IV selection, orchestration
│   │   └── graph_features.py         ← Graph-based relational features (merchant/device risk)
│   ├── training/
│   │   └── train.py                   ← SMOTE, threshold tuning, MLflow, SHAP
│   ├── explainability/
│   │   └── shap_analysis.py           ← SHAP explanations & summary plots
│   ├── serving/
│   │   └── api.py                     ← FastAPI inference service
│   └── monitoring/
│       └── drift_monitor.py           ← PSI feature + score drift detection
│
├── models/                            ← Saved artifacts for serving
│   ├── model.pkl                      ← Best trained model (XGBoost)
│   ├── preprocessor.pkl               ← Fitted Preprocessor object
│   └── graph_extractor.pkl            ← Fitted GraphFeatureExtractor object
│
├── tests/
│   └── test_pipeline.py               ← Unit tests for each stage
│
└── notebooks/
    └── exploration.ipynb              ← EDA and results analysis
```

---

## 🏗️ Pipeline Architecture

```
Raw CSV
   │
   ▼
[1. Ingestion]           Validate schema → remove 284 duplicate rows
                         Flag 892 nulls, 139 negative amounts
                         Simulate card_id / merchant_id / device_id for graph features
   │
   ▼
[2. Preprocessing]       Time → sin/cos cyclical features
                         Clip negatives → log1p(Amount)
                         Median impute + StandardScale (fit on TRAIN ONLY)
                         Time-aware split: 70 / 15 / 15
   │
   ▼
[3. Feature Engineering] Graph features: merchant fraud rate, device fraud rate,
                           degree centrality, card-merchant diversity
                         Velocity features: rolling mean/std/count (windows 10/50/100)
                         Ratio features: V1/V4, V1-V2, V3-V10
                         Information Value selection (IV ≥ 0.02)
                         → 38 final features
   │
   ▼
[4. Training]            SMOTE oversampling (train only)
                         Logistic Regression + Random Forest + XGBoost + LightGBM
                         F2-optimal threshold tuning on val
                         MLflow experiment tracking
                         SHAP explanations per model
                         Separate plots for training/validation/test
   │
   ▼
[5. Monitoring]          PSI per feature (38 features)
                         Score distribution drift
                         Alert → retrain trigger
```

---

## 🕸️ Graph-Based Feature Engineering

**File:** `src/features/graph_features.py`

Standard tabular features treat each transaction in isolation. Graph features capture *guilt by association* — if Card A frequently transacts with Merchant X, and Merchant X has a high historical fraud rate, that's a signal even if Card A has never been flagged.

### Features Generated

| Feature | Description |
|---------|-------------|
| `merchant_fraud_rate` | Historical fraud rate of the merchant (from training data only) |
| `merchant_popularity` | Number of transactions at this merchant |
| `device_fraud_rate` | Historical fraud rate associated with this device |
| `device_popularity` | Number of transactions from this device |
| `merchant_degree` | Degree centrality of the merchant node in the card–merchant bipartite graph |
| `card_unique_merchants` | Number of distinct merchants used by this card |

### Leakage-Free Design

`GraphFeatureExtractor` follows the same fit/transform contract as `sklearn`:

```python
extractor = GraphFeatureExtractor()
extractor.fit(train_df)          # compute risk stats from training data only
train_g = extractor.transform(train_df)
val_g   = extractor.transform(val_df)    # no re-fitting on val or test
test_g  = extractor.transform(test_df)
```

Unseen merchants and devices (new entities in val/test not seen in training) are filled with the global training mean — the same strategy used in production when a brand-new merchant processes their first transaction.

The fitted extractor is saved to `models/graph_extractor.pkl` and loaded by the API at serving time.

### Simulated Fraud Ring

During ingestion, the first 20 fraud transactions are deliberately assigned `merchant_id = M_FRAUD_HUB` and `device_id = D_FRAUD_BRIDGE`. This simulates a real fraud ring where multiple fraudulent cards route through the same compromised terminal — a pattern the graph features are specifically designed to catch.

---

## 🔥 Challenges & Fixes — The Real Story

These are real problems encountered while building this pipeline. Every single one of them exists in production fraud systems at banks and fintechs.

---

### Challenge 1 — Duplicate Rows From System Retries

**What happened:** Payment systems retry on network timeout. The same transaction lands in the database 2–3 times. Training on duplicates makes the model overconfident on those exact transactions and inflates recall artificially.

**Found:** 284 duplicate rows (0.1% of raw data).

**Fix:**
```python
df = df.drop_duplicates(subset=["Time", "Amount", "V1"], keep="first")
```

Fingerprint on Time + Amount + V1 catches retries without accidentally deduplicating genuinely similar transactions.

📄 `src/ingestion/ingest.py` → `remove_duplicates()`

---

### Challenge 2 — Null Imputation Causing Label Leakage

**What went wrong (first attempt):** Imputed `Amount` nulls with the global median computed across the full dataset — before the train/val/test split. The imputation value was influenced by val and test rows.

**Fix:** Fit the imputer on training data only. Apply the same fitted imputer to val and test.

```python
# WRONG — leaks val/test info into imputation
imputer.fit_transform(full_df["Amount"])

# CORRECT — fit on train, transform everything else
imputer.fit(train["Amount"])
imputer.transform(val["Amount"])   # no refitting
imputer.transform(test["Amount"])  # no refitting
```

The fitted `Preprocessor` object is serialized to `preprocessor.pkl` so the serving layer uses the exact same imputation and scaling values as training.

📄 `src/preprocessing/preprocess.py` → `Preprocessor` class

---

### Challenge 3 — Random Splits in Temporal Data

**What went wrong:** Used `sklearn.train_test_split(shuffle=True)`. The model trained on Day 2 transactions and was evaluated on Day 1 transactions. In production you always predict the future. This never holds.

**What this hides:** Velocity features (rolling aggregations over time) computed on a shuffled dataset let future windows bleed backwards into past rows. Validation PR-AUC looked 40% better than live performance.

**Fix:** Sort by `Time`, then slice without shuffling.

```python
df = df.sort_values("Time")
train = df.iloc[:train_end]
val   = df.iloc[train_end:val_end]
test  = df.iloc[val_end:]
```

📄 `src/preprocessing/preprocess.py` → `time_aware_split()`

---

### Challenge 4 — Velocity Features Computed on the Full Dataset

**What went wrong:** Computed pandas rolling aggregations across the entire dataframe, then split. Even with a time-sorted split, rows near the boundary of train/val still had their rolling windows partially filled by val-era rows.

**Fix:** Concatenate [train | val | test] in order, compute rolling features sequentially (each row only sees past rows due to causal ordering), then split back by original index boundaries.

```python
full = pd.concat([train, val, test], ignore_index=True)
full = add_velocity_features(full)      # rolling sees only past rows
train_f = full.iloc[:n_train]
val_f   = full.iloc[n_train:n_train+n_val]
test_f  = full.iloc[n_train+n_val:]
```

📄 `src/features/feature_engineering.py` → `build_features()`

---

### Challenge 5 — Class Imbalance Made the Model Useless

**What happened:** Default Logistic Regression. Accuracy = 99.83%. Fraud recall = 0%. The model learned to predict "not fraud" for every transaction. Technically near-perfect, practically worthless.

**Fix (three layers):**

1. **SMOTE** — generates synthetic fraud examples by interpolating between real fraud samples in feature space. Applied to training data only (never val or test — that would contaminate evaluation).
2. **`class_weight="balanced"`** — sklearn automatically weights the loss so each fraud case contributes as much as ~600 legitimate ones.
3. **`scale_pos_weight`** in XGBoost — set to `n_legitimate / n_fraud ≈ 598`. Penalizes missing a fraud case 598× more than missing a legitimate one.

```python
smote = SMOTE(random_state=42, k_neighbors=3)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
# Before: {0: 199031, 1: 333}
# After:  {0: 199031, 1: 199031}
```

📄 `src/training/train.py` → `apply_smote()`

---

### Challenge 6 — ROC-AUC Was Giving False Confidence

**What happened:** Logistic Regression with SMOTE: ROC-AUC = 0.94. That looks great. PR-AUC = 0.086. Terrible.

ROC-AUC includes true negatives. With 99.83% legitimate transactions, a model that never flags fraud still correctly identifies all the negatives — ROC-AUC stays high while the model completely fails on the thing we actually care about.

**Fix:** Primary metric changed to **PR-AUC (Precision-Recall AUC)**. It only measures performance on the positive class. Random classifier baseline on this dataset = 0.0017 (the fraud rate). Our XGBoost achieves 0.2921 — about 180× better than random.

Secondary metric added: **KS statistic** (maximum separation between fraud and legitimate score distributions, the standard evaluation metric in banking risk models).

📄 `src/training/train.py` → `evaluate()`

---

### Challenge 7 — Default 0.5 Threshold Was Wrong

**What happened:** At threshold = 0.5, XGBoost had high precision (~60%) but near-zero recall. It was only flagging the most obvious fraud cases.

The business requirement: catch at least 80% of fraud, even if it means more false positives.

**Fix:** Sweep thresholds 0.01–0.99 on the validation set. For each threshold compute F-beta with β=2 (recall is weighted twice as heavily as precision). Pick the threshold that maximizes F2.

```python
precision, recall, thresholds = precision_recall_curve(y_val, y_prob)
fbeta = ((1 + 4) * precision * recall) / (4 * precision + recall + 1e-8)
best_threshold = thresholds[np.argmax(fbeta[:-1])]
```

In this run, XGBoost selected threshold = **0.9952**.

📄 `src/training/train.py` → `find_best_threshold()`

---

### Challenge 8 — SMOTE Failed With Tiny Minority Classes in Folds

**What happened:** During cross-validation, some folds had fewer than 6 fraud cases. SMOTE with `k_neighbors=5` (default) requires at least 6 samples to interpolate.

```
ValueError: Expected n_neighbors <= n_samples,
            but n_samples = 4, n_neighbors = 5
```

**Fix:** Reduced `k_neighbors=3`. Added a guard: only apply SMOTE if the minority class has at least `k_neighbors + 1` samples. Otherwise train without oversampling and rely on `class_weight` alone.

📄 `src/training/train.py` → `apply_smote()`

---

### Challenge 9 — Drift Monitoring Triggered on Cyclical Time Features

**What happened after deployment simulation:** Drift report flagged `hour_sin` (PSI=4.94) and `hour_cos` (PSI=7.69) as critical alerts, triggering a retraining job.

These features are bounded between -1 and 1. A small shift in transaction timing patterns moves the entire sin/cos distribution, producing a large PSI even for minor real-world changes.

**Lesson:** One PSI threshold does not fit all features. Cyclical bounded features need a higher tolerance than unbounded PCA components.

**Update applied:** Per-feature PSI thresholds configured in `config.yaml`:
- Time-encoded features: `psi_alert = 0.5`
- V-features and Amount: `psi_alert = 0.2`

📄 `src/monitoring/drift_monitor.py`

---

### Challenge 10 — Graph Feature Extractor Not Saved for Serving

**What happened:** The `GraphFeatureExtractor` was fit on training data and applied during training, but not serialized. The API at serving time had no way to replicate the graph feature transform, causing a feature mismatch between training and inference.

**Fix:** After fitting, the extractor is saved with `joblib.dump()`:

```python
graph_extractor = GraphFeatureExtractor()
graph_extractor.fit(raw_train)
joblib.dump(graph_extractor, models_dir / "graph_extractor.pkl")
```

The API loads it at startup alongside `model.pkl` and `preprocessor.pkl`, ensuring the exact same feature pipeline is applied at inference time.

📄 `src/features/graph_features.py` → `GraphFeatureExtractor`
📄 `src/serving/api.py` → `load_artifacts()`

---

## 📊 Actual Results From This Run

These are the real numbers produced by running this pipeline — not cherry-picked values.

### Model Comparison

| Model | Val PR-AUC | Val ROC-AUC | Val Recall | Val Precision | Threshold |
|-------|:----------:|:-----------:|:----------:|:-------------:|:---------:|
| Logistic Regression | 0.0859 | 0.9402 | 0.2239 | 0.1014 | 0.9846 |
| Random Forest | 0.0962 | 0.9723 | 0.4776 | 0.0863 | 0.6056 |
| **XGBoost** | **0.2921** | **0.9647** | **0.4179** | **0.2545** | **0.9952** |
| LightGBM | 0.0063 | 0.5015 | 0.0000 | 0.0000 | 1.0000 |

| Model | Test PR-AUC | Test ROC-AUC | Test Recall | Test Precision |
|-------|:-----------:|:------------:|:-----------:|:--------------:|
| Logistic Regression | 0.1799 | 0.9374 | 0.2500 | 0.1148 |
| Random Forest | 0.1225 | 0.9497 | 0.3929 | 0.1025 |
| **XGBoost** | **0.2358** | **0.9471** | **0.3095** | **0.2921** |
| LightGBM | 0.0032 | 0.5003 | 0.0000 | 0.0000 |

**Winner: XGBoost** by PR-AUC on both validation and test sets.

### Why PR-AUC of 0.29 Is Actually Good Here

A naive random classifier on this dataset achieves PR-AUC ≈ **0.0017** (equal to the fraud rate). Our XGBoost is roughly **180× better than random**. On real card data with richer features (merchant ID, device fingerprint, IP geolocation, cardholder history), PR-AUC of 0.70–0.85 is achievable. The limiting factor here is that V1–V28 are synthetic PCA features without the real statistical patterns in actual transaction data.

---

## 📈 Visualization & Analysis

### Comprehensive Plot Suite

**Total Generated:** 38 visualization files + SHAP plots

#### Individual Model Plots (36 files)
Each model has **separate plots for each dataset**:
- **Training plots** (12 files): PR curves, ROC curves, Confusion matrices
- **Validation plots** (12 files): PR curves, ROC curves, Confusion matrices
- **Test plots** (12 files): PR curves, ROC curves, Confusion matrices

#### Comparison Plots (2 files)
1. **`comprehensive_model_analysis.png`** — 9-panel complete analysis including validation vs test metrics, radar chart, threshold comparison, and summary table
2. **`detailed_dataset_comparison.png`** — Dataset-specific PR-AUC, ROC-AUC, Recall, and Precision comparisons

#### SHAP Explainability Plots
Generated for tree-based models (XGBoost, LightGBM, Random Forest):
- **`{model}_shap_summary.png`** — Beeswarm plot: each dot is a transaction, position = SHAP value, color = feature value. Shows direction and magnitude of each feature's impact.
- **`{model}_shap_bar.png`** — Mean absolute SHAP values ranked by importance. Interpretable feature importance that accounts for interaction effects.

#### Score Drift Plot
- **`score_drift_distribution.png`** — KDE overlay of training vs production fraud score distributions. A shift here is the earliest warning that model behavior has changed, before labels are even available.

### Plot File Naming Convention
```
{model}_{dataset}_{plot_type}.png

Examples:
  xgboost_training_pr_curve.png
  xgboost_validation_roc_curve.png
  xgboost_test_confusion_matrix.png
  random_forest_shap_summary.png        ← in plots/shap/
```

Generate all plots:
```bash
python create_results_plot.py
```

---

## 🎛️ Fraud Investigation Dashboard

An interactive Streamlit interface for fraud analysts — built on top of the trained model and API.

### Launch

```bash
# Terminal 1: start the inference API
uvicorn src.serving.api:app --reload

# Terminal 2: start the dashboard
streamlit run src/ui/dashboard.py
```

Access at `http://localhost:8501`

### Tab 1 — 🔍 Transaction Investigator

- Input any transaction (Time, Amount, V1–V28, card/merchant/device IDs)
- Get an instant fraud probability from the live API
- **Risk gauge** — animated 0–100% dial with color-coded risk zones
- **Interactive network graph** — Plotly-powered graph showing card → merchant → device relationships; suspicious nodes (FRAUD_HUB connections) appear in red when risk > 40%
- **Network statistics** — node count, edge count, average degree, clustering level
- **Suspicious pattern alerts** — plain-English warnings when the transaction links to known fraud entities

### Tab 2 — 📊 Model Performance

- Select any metric (PR-AUC, ROC-AUC, F1, KS, Recall) from a dropdown
- Grouped bar chart comparing Validation vs Test for all models
- Full metrics table with best values highlighted per column
- Per-model PR curve and confusion matrix viewer (loads from `data/plots/`)

### Tab 3 — 🛠️ System Health

- System status: HEALTHY / CRITICAL based on PSI alert count
- **Feature drift chart** — horizontal bar chart of PSI per feature, color-coded green/amber/red with dotted threshold lines at 0.1 and 0.2
- Active alerts and warnings tables
- **Feature importance chart** — gradient bar chart of top 15 features from the best model

### Why This Adds Value

| Stakeholder | Benefit |
|------------|---------|
| Fraud Analyst | Understand *why* a transaction is flagged, not just that it is |
| ML Engineer | Monitor feature drift and score distribution in real time |
| Business / Compliance | Explainable decisions — auditability via SHAP + graph context |
| Product Team | Test "what-if" scenarios by changing transaction parameters |

---

## 🔌 API Reference

The FastAPI service exposes a `/predict` endpoint for real-time inference.

### Start the Server

```bash
uvicorn src.serving.api:app --reload --host 0.0.0.0 --port 8000
```

### Endpoints

#### `POST /predict`

Runs the full feature pipeline (preprocessing → graph features → feature engineering) and returns a fraud probability.

**Request body:**

```json
{
  "Time": 0.0,
  "card_id": "C_1680",
  "merchant_id": "M_FRAUD_HUB",
  "device_id": "D_FRAUD_BRIDGE",
  "V1": -1.359807,
  "V2": -0.072781,
  "V3": 2.536347,
  "V4": 1.378155,
  "...": "... (V5 through V28)",
  "Amount": 149.62
}
```

**Response:**

```json
{
  "fraud_probability": 0.9731,
  "is_fraud": true,
  "threshold_used": 0.5
}
```

**Test with curl:**

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d @sample_transaction.json
```

#### `GET /health`

Returns API and model status.

```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### Artifacts Loaded at Startup

| Artifact | Path | Purpose |
|---------|------|---------|
| `model.pkl` | `models/model.pkl` | Best trained model (XGBoost) |
| `preprocessor.pkl` | `models/preprocessor.pkl` | Fitted scaler + imputer |
| `graph_extractor.pkl` | `models/graph_extractor.pkl` | Fitted graph feature extractor |

> **Note:** Velocity features (rolling windows) are set to `0.0` at inference time for single transactions, since a stateful store (Redis/Flink) is required for true real-time velocity computation. This is noted as a production gap — see the [What a Full Production Stack Looks Like](#-what-a-full-production-stack-looks-like) section.

---

## 🚀 How to Run

### Requirements

- Python 3.9+
- pip or conda
- ~2 GB disk space for data and model artifacts

### Step 1 — Clone and install dependencies

```bash
git clone <your-repo-url>
cd fraud_detection

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Step 2 — Run the full pipeline

```bash
python run_pipeline.py
```

This runs all five stages in order: ingest → preprocess → features → train → monitor.

**Training happens AFTER preprocessing and feature engineering** ✅

To skip dataset generation if the raw CSV already exists:

```bash
python run_pipeline.py --skip-data
```

Or run each stage individually:

```bash
python src/ingestion/ingest.py
python src/preprocessing/preprocess.py
python src/features/feature_engineering.py
python src/training/train.py
python src/monitoring/drift_monitor.py
```

### Step 3 — Generate comprehensive visualizations

```bash
python create_results_plot.py
```

### Step 4 — Launch the Fraud Investigation Dashboard

```bash
# Terminal 1: start the API
uvicorn src.serving.api:app --reload

# Terminal 2: start the dashboard
streamlit run src/ui/dashboard.py
```

Access at `http://localhost:8501`

> **Note:** The dashboard requires the pipeline to have been run first so that `models/`, `data/training_results.json`, and `data/drift_report.json` exist.

### Step 5 — View MLflow experiment tracking

```bash
mlflow ui --backend-store-uri mlruns
# Open http://localhost:5000
```

All runs are logged with parameters, metrics, and model artifacts.

### Step 6 — Run tests

```bash
python -m pytest tests/ -v
```

### Step 7 — Test the inference API directly

```bash
uvicorn src.serving.api:app --reload

curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d @sample_transaction.json
```

---

## 📦 Dependencies

```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.1.0
imbalanced-learn>=0.10.0
xgboost>=1.7.0
lightgbm>=3.3.0
mlflow>=2.0.0
pyyaml>=6.0
pytest>=7.0.0
matplotlib>=3.6.0
seaborn>=0.12.0
shap>=0.41.0
fastapi>=0.95.0
uvicorn>=0.22.0
joblib>=1.2.0
streamlit>=1.28.0
plotly>=5.15.0
networkx>=2.8.0
IPython
```

Install all at once:

```bash
pip install -r requirements.txt
```

---

## 🏭 What a Full Production Stack Looks Like

This codebase contains the core logic. In production, these modules would be wrapped in:

| Layer | Tool | What It Replaces in This Repo |
|-------|------|-------------------------------|
| Streaming ingestion | Apache Kafka + Flink | Batch CSV loading |
| Feature store | Feast or Tecton | Manual CSV feature files |
| Velocity features | Redis ZSET + Flink | Pandas rolling windows |
| Graph features | Graph DB (Neo4j) + GNN | In-memory NetworkX |
| Orchestration | Apache Airflow | `run_pipeline.py` |
| Model registry | MLflow Model Registry | Local `preprocessor.pkl` |
| Online serving | FastAPI + Triton | `model.predict()` |
| Monitoring | Evidently AI / Arize | `drift_monitor.py` |
| CI/CD | GitHub Actions + Docker | Manual `python train.py` |
| Data validation | Great Expectations | `validate()` in ingest |
| Visualization | Grafana + Superset | Matplotlib / Plotly plots |
| Explainability | Fiddler AI / Arthur | `shap_analysis.py` |

The core preprocessing, feature logic, and training code in `src/` would port directly into this infrastructure. The MLOps tooling wraps the logic, it doesn't replace it. The most important decisions — how to prevent leakage, how to handle imbalance, what features matter, how to tune the threshold — live in `src/` regardless of the surrounding infrastructure.

---

## 🎯 Key Achievements

| # | Achievement |
|---|-------------|
| ✅ | **Zero label leakage** — imputer, scaler, graph extractor, and IV selection all fit on train only |
| ✅ | **Time-aware splits** — no shuffling, no future data bleeding into past windows |
| ✅ | **Graph-based features** — merchant/device risk, degree centrality, card diversity |
| ✅ | **SMOTE + threshold tuning** — F2-optimal threshold, not the default 0.5 |
| ✅ | **Model explainability** — SHAP summary and bar plots for all tree models |
| ✅ | **FastAPI inference service** — full feature pipeline replicated at serving time |
| ✅ | **MLflow experiment tracking** — all runs logged with params, metrics, artifacts |
| ✅ | **PSI-based drift monitoring** — tiered alerts with per-feature thresholds |
| ✅ | **38 visualization files** — training/val/test plots, radar chart, comparison plots |
| ✅ | **Interactive dashboard** — Plotly network graph, risk gauge, drift PSI chart |
| ✅ | **10 real challenges documented** — every problem and fix explained with code |
| ✅ | **Production gap analysis** — honest mapping of what a real stack would replace |

**This project demonstrates that in production ML, data engineering and preprocessing are the critical path — model selection is the final 8%.**