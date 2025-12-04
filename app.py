import streamlit as st
import pandas as pd
import pickle, json, numpy as np, sys, subprocess
from sklearn.metrics import roc_auc_score, brier_score_loss
from utils import calculate_psi

st.set_page_config(layout="wide")
st.title("📉 Credit Risk – Enterprise Drift & Performance Monitoring")

baseline = pd.read_csv("data/baseline.csv")
batch = pd.read_csv("data/batch_data.csv")

model = pickle.load(open("model/model.pkl", "rb"))
scaler = pickle.load(open("model/scaler.pkl", "rb"))
logs = json.load(open("model/training_log.json"))

# ---- PSI Calculation ----
psi_scores = {
    col: calculate_psi(baseline[col], batch[col])
    for col in ["income", "utilization", "age"]
}

psi_df = pd.DataFrame.from_dict(psi_scores, orient="index", columns=["PSI"])
st.subheader("Feature Drift (PSI)")
st.dataframe(psi_df)

st.bar_chart(psi_df)

# ---- Batch Performance ----
X_batch = scaler.transform(batch.drop(["default", "batch_date"], axis=1))
y_batch = batch["default"]
preds = model.predict_proba(X_batch)[:, 1]

auc = roc_auc_score(y_batch, preds)
brier = brier_score_loss(y_batch, preds)

st.subheader("Real-Time Model Performance")
c1, c2 = st.columns(2)
c1.metric("AUC", round(auc, 4))
c2.metric("Brier", round(brier, 4))

# ---- Auto Alert ----
baseline_auc = logs["baseline_auc"]

if auc < baseline_auc - 0.05:
    st.error("⚠️ Significant performance degradation detected")
else:
    st.success("✅ Model performance is stable")

# ---- Training History ----
hist = pd.DataFrame(logs["history"])
st.subheader("Model Training & Retraining History")
st.dataframe(hist)
st.line_chart(hist[["auc"]])

# ---- Manual Retrain ----
if st.button("🔁 Retrain Model"):
    try:
        result = subprocess.run(
            [sys.executable, "train_model.py"],
            check=True,
            capture_output=True,
            text=True
        )
        st.success("✅ Model retrained successfully")
        st.code(result.stdout)
    except subprocess.CalledProcessError as e:
        st.error("❌ Retraining failed")
        st.code(e.stderr)
