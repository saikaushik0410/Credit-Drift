import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
import pickle, json, datetime, os

df = pd.read_csv("data/baseline.csv")

X = df.drop("default", axis=1)
y = df["default"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

preds = model.predict_proba(X_test)[:, 1]

auc = roc_auc_score(y_test, preds)
brier = brier_score_loss(y_test, preds)

os.makedirs("model", exist_ok=True)

pickle.dump(model, open("model/model.pkl", "wb"))
pickle.dump(scaler, open("model/scaler.pkl", "wb"))

log_path = "model/training_log.json"

entry = {
    "timestamp": datetime.datetime.now().isoformat(),
    "auc": float(auc),
    "brier": float(brier)
}

if os.path.exists(log_path):
    logs = json.load(open(log_path))

    if "history" not in logs:
        logs["history"] = []

    logs["history"].append(entry)

else:
    logs = {
        "baseline_auc": auc,
        "history": [entry]
    }

json.dump(logs, open(log_path, "w"), indent=2)


print(f"✅ Baseline model trained | AUC={auc:.4f} | Brier={brier:.4f}")
