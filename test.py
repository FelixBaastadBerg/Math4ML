from pathlib import Path
import re, ast, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

SEED = 44
METHOD = "random"

# 1) Rebuild results_all.csv from per-ns (avoids stale file)
res_dir = Path("./MLP_optimization") / METHOD
per_n = sorted(res_dir.glob("results_n*.csv"))
df_all = pd.concat((pd.read_csv(p) for p in per_n), ignore_index=True).sort_values("n")
(df_all).to_csv(res_dir / "results_all.csv", index=False)

# 2) Load best params for n=20
row = df_all.query("n==20").iloc[0]
best_params = ast.literal_eval(re.sub(r"np\.float64\(([^)]+)\)", r"\1", row["best_params"]))
best_params.pop("clf__random_state", None)  # we control seed below

# 3) Load train/test
Xtr = np.load("Datasets/Train_Data/kryptonite-20-X-train.npy")
ytr = np.load("Datasets/Train_Data/kryptonite-20-y-train.npy")
Xte = np.load("Datasets/Test_Data/kryptonite-20-X-test.npy")
yte = np.load("Datasets/Test_Data/kryptonite-20-y-test.npy")

# 4) Build the exact eval pipeline and measure:
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", MLPClassifier(max_iter=1000, early_stopping=True,
                          n_iter_no_change=15, random_state=SEED))
])
pipe.set_params(**best_params)

# CV on Train_Data with the SAME splitter used in tuning:
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED+1)
cv_scores = cross_val_score(pipe, Xtr, ytr, cv=cv, scoring="accuracy", n_jobs=10)
print(f"CV (n=20, seed=44) mean={cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Refit on full Train_Data and test:
pipe.fit(Xtr, ytr)
print("Number of training iterations:", pipe.named_steps["clf"].n_iter_)
test_acc = (pipe.score(Xte, yte))
print(f"Test ACC (same model): {test_acc:.4f}")
