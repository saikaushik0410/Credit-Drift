import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

os.makedirs("data", exist_ok=True)

np.random.seed(42)

N = 50000

income = np.random.normal(65000, 15000, N)
utilization = np.random.beta(2, 5, N)
age = np.random.randint(21, 65, N)

logit = (
    -4
    + 0.00004 * income
    + 2.5 * utilization
    - 0.01 * age
    + np.random.normal(0, 0.4, N)
)

prob_default = 1 / (1 + np.exp(-logit))
default = np.random.binomial(1, prob_default)

baseline = pd.DataFrame({
    "income": income,
    "utilization": utilization,
    "age": age,
    "default": default
})

baseline.to_csv("data/baseline.csv", index=False)

# ---- Generate Drifted Monthly Batches ----
batches = []
current = baseline.copy()
date = datetime.today()

for i in range(6):
    current["income"] *= np.random.normal(0.97, 0.01)
    current["utilization"] *= np.random.normal(1.05, 0.02)

    current["default"] = np.random.binomial(
        1, 1 / (1 + np.exp(-(-4 + 2.8 * current["utilization"])))
    )

    current["batch_date"] = date.strftime("%Y-%m-%d")
    batches.append(current.sample(5000))

    date += timedelta(days=30)

batch_df = pd.concat(batches)
batch_df.to_csv("data/batch_data.csv", index=False)

print("✅ Enterprise baseline + drift batches generated")
