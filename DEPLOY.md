# Deploying FraudShield AI to Hugging Face Spaces (Docker) — Free

This runs **both** services in one free container:
- FastAPI inference API — internal `http://localhost:8000`
- Streamlit dashboard — public on port **7860** (what HF exposes)

Files that make this work (already in the repo):
`Dockerfile`, `start.sh`, `requirements-deploy.txt`, `.dockerignore`, `.gitattributes`,
and the `sdk: docker` / `app_port: 7860` front matter at the top of `README.md`.

---

## 0) Refresh the serving artifacts (one local run)

Make sure the committed artifacts reflect the latest fixes (serving model + HEALTHY drift report):

```bash
python -m src.training.train_serving
```

This (re)writes: `models/model.pkl`, `models/preprocessor.pkl`, `models/serving_meta.json`,
`data/training_results.json`, `data/feature_importance.json`, `data/drift_report.json`,
and `data/plots/xgboost_*.png`.

> The 150 MB `data/creditcard_raw.csv` is **not** needed at runtime and is excluded from the
> image by `.dockerignore`. Don't ship it.

---

## 1) Create the Space

1. Sign up / log in at https://huggingface.co (free).
2. **New → Space**. Name it (e.g. `fraudshield-ai`).
3. **SDK: Docker** → **Blank**. Visibility: Public (free). Create.

---

## 2) Stage the artifacts the dashboard needs

Most artifacts are already tracked. The four `data/` files are git-ignored, so force-add them **once**
(after which they stay tracked and will push normally):

```bash
git add Dockerfile start.sh requirements-deploy.txt .dockerignore .gitattributes README.md
git add models/serving_meta.json
git add -f data/training_results.json data/feature_importance.json data/drift_report.json
git add -f data/plots/xgboost_test_pr_curve.png data/plots/xgboost_test_confusion_matrix.png
git commit -m "Add Hugging Face Spaces Docker deployment"
```

---

## 3) Push to the Space

```bash
# Use the remote URL shown on your Space page:
git remote add space https://huggingface.co/spaces/<your-username>/fraudshield-ai
git push space main        # if your branch is 'master', use:  git push space master:main
```

Authentication: when prompted for a password, paste a Hugging Face **access token**
(Settings → Access Tokens → create a `write` token).

---

## 4) Watch it build

HF auto-builds the Docker image (a few minutes). When the log shows Streamlit starting,
the app is live at:

```
https://huggingface.co/spaces/<your-username>/fraudshield-ai
```

Open the **🔍 Investigator** tab and submit a transaction.

---

## Notes / expectations
- **Free CPU tier**: ~2 vCPU / 16 GB RAM — plenty for this app. No GPU needed.
- **Idle sleep**: free Spaces sleep after inactivity and wake on the next visit (cold start ~30s).
- **Updates**: `git push space …` again → rebuilds automatically.
- **Logs**: the Space's "Logs" tab shows both uvicorn and Streamlit output; `start.sh` waits
  for the API's `/health` before launching the dashboard.
- **Local Docker test (optional, needs Docker Desktop):**
  ```bash
  docker build -t fraudshield .
  docker run -p 7860:7860 fraudshield
  # open http://localhost:7860
  ```
