# Evaluation: Train on Train_Data, Test on Test_Data for each n
import os
import re
import ast
import json
import time
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, RocCurveDisplay
)

TRAIN_DIR = Path("./Datasets/Train_Data")
TEST_DIR  = Path("./Datasets/Test_Data")

# which tuning results to use: "random", "grid", or "bayes"
METHOD = "random"

res_dir = Path("./MLP_optimization") / METHOD
per_n = sorted(res_dir.glob("results_n*.csv"))
if per_n:
    df_all = pd.concat((pd.read_csv(p) for p in per_n), ignore_index=True).sort_values("n")
    df_all.to_csv(res_dir / "results_all.csv", index=False)

RESULTS_CSV = Path(f"./MLP_optimization/{METHOD}/results_all.csv")

# where to save evaluation artifacts
EVAL_DIR = Path("./Evaluation") / METHOD
EVAL_DIR.mkdir(parents=True, exist_ok=True)

SEED = 45

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def clean_np_floats(s: str) -> str:
    """Replace np.float64(1.23e-5) with 1.23e-5 in a string representation."""
    return re.sub(r"np\.float64\(([^)]+)\)", r"\1", s)

def load_best_params_by_n(results_csv: Path) -> Dict[int, Dict[str, Any]]:
    """
    Reads results_all.csv and returns { n: best_params_dict }.
    Handles stringified dicts and cleans np.float64(...) wrappers.
    """
    assert results_csv.exists(), f"Missing results CSV: {results_csv}"
    df = pd.read_csv(results_csv)
    out = {}
    for _, row in df.iterrows():
        n = int(row["n"])
        raw = row["best_params"]
        if isinstance(raw, str):
            cleaned = clean_np_floats(raw)
            try:
                params = ast.literal_eval(cleaned)
            except Exception as e:
                raise ValueError(f"Failed to parse best_params for n={n}:\n{raw}\nError: {e}")
        elif isinstance(raw, dict):
            params = raw
        else:
            params = dict(raw)

        # Be safe: convert hidden_layer_sizes list -> tuple if needed
        hls_key = "clf__hidden_layer_sizes"
        if hls_key in params and isinstance(params[hls_key], list):
            params[hls_key] = tuple(params[hls_key])

        out[n] = params
    return out

# Accept files like: kryptonite-<n>-(X|x|Y|y)-(train|test).npy
_PAT = re.compile(r"^kryptonite-(\d+)-([xy])-(train|test)\.npy$", re.IGNORECASE)

def _index_split(dir_path: Path, expected_split: str) -> Dict[int, Dict[str, Path]]:
    """
    Scan dir_path for kryptonite-n-(x|y)-<expected_split>.npy (case-insensitive).
    Return { n: {'X': Path or None, 'y': Path or None} }.
    """
    idx: Dict[int, Dict[str, Path]] = {}
    for p in dir_path.rglob("*.npy"):
        m = _PAT.match(p.name)
        if not m:
            continue
        n = int(m.group(1))
        xy = m.group(2).lower() # 'x' or 'y'
        split = m.group(3).lower() # 'train' or 'test'
        if split != expected_split:
            continue
        d = idx.setdefault(n, {"X": None, "y": None})
        if xy == "x":
            d["X"] = p
        else:
            d["y"] = p
    return idx

def discover_dataset_pairs(train_dir: Path, test_dir: Path) -> Dict[int, Dict[str, Path]]:
    """
    Find matching train/test pairs by n.
    Returns { n: {"Xtr","ytr","Xte","yte"} } only when all four exist.
    """
    tr = _index_split(train_dir, "train")
    te = _index_split(test_dir,  "test")

    pairs = {}
    for n in sorted(set(tr.keys()) & set(te.keys())):
        if tr[n]["X"] and tr[n]["y"] and te[n]["X"] and te[n]["y"]:
            pairs[n] = {
                "Xtr": tr[n]["X"], "ytr": tr[n]["y"],
                "Xte": te[n]["X"], "yte": te[n]["y"],
            }
    return pairs

def build_pipeline_from_params(best_params: Dict[str, Any]) -> Pipeline:
    """
    Returns Pipeline(StandardScaler -> MLPClassifier) with best_params applied.
    Keeps robust defaults (max_iter, early_stopping).
    """
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(
            max_iter=500,
            tol=1e-3,
            early_stopping=False,
            n_iter_no_change=15,
            random_state=SEED
        ))
    ])
    pipe.set_params(**best_params)
    pipe.set_params(clf__early_stopping=False, clf__max_iter=1000)
    return pipe

# Load the best parameters
best_by_n = load_best_params_by_n(RESULTS_CSV)

pairs = discover_dataset_pairs(TRAIN_DIR, TEST_DIR)
assert pairs, f"No matching train/test pairs found under {TRAIN_DIR} and {TEST_DIR}"

print(f"Found best params for n: {sorted(best_by_n.keys())}")
print(f"Found dataset pairs for n: {sorted(pairs.keys())}")

summary_rows = []

for n, paths in pairs.items():
    if n not in best_by_n:
        print(f"Skipping n={n}: no best params in {RESULTS_CSV}")
        continue

    print(f"\n=== Evaluating n = {n} (method={METHOD}) ===")
    Xtr = np.load(paths["Xtr"])
    ytr = np.load(paths["ytr"])
    Xte = np.load(paths["Xte"])
    yte = np.load(paths["yte"])

    print(" Train:", Xtr.shape, ytr.shape, " Test:", Xte.shape, yte.shape)

    best_params = best_by_n[n]
    model = build_pipeline_from_params(best_params)

    # Train
    t0 = time.time()
    model.fit(Xtr, ytr)
    train_time = time.time() - t0

    # Preds + probabilities
    y_pred = model.predict(Xte)
    proba = None
    try:
        proba = model.predict_proba(Xte)[:, 1]
    except Exception:
        pass

    # Metrics
    acc = accuracy_score(yte, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(yte, y_pred, average="binary", zero_division=0)
    auc = None
    if proba is not None and len(np.unique(yte)) == 2:
        try:
            auc = roc_auc_score(yte, proba)
        except Exception:
            auc = None

    print(f"  Test Acc={acc:.4f}  F1={f1:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  AUC={auc if auc is not None else 'NA'}")
    print("\nClassification report:\n", classification_report(yte, y_pred, digits=4, zero_division=0))

    # Per-n output dir
    out_dir = EVAL_DIR / f"n{n}"
    ensure_dir(out_dir)

    # Confusion Matrix
    try:
        fig_cm, ax_cm = plt.subplots(figsize=(4.5, 4))
        ConfusionMatrixDisplay(confusion_matrix(yte, y_pred)).plot(ax=ax_cm, colorbar=False)
        ax_cm.set_title(f"Kryptonite-{n} : Confusion Matrix")
        fig_cm.tight_layout()
        fig_cm.savefig(out_dir / "confusion_matrix.png", dpi=150)
        plt.close(fig_cm)
    except Exception as e:
        print(f"  (confusion matrix plot skipped: {e})")

    # ROC
    if proba is not None and len(np.unique(yte)) == 2:
        try:
            fig_roc, ax_roc = plt.subplots(figsize=(5.5, 4))
            RocCurveDisplay.from_predictions(yte, proba, ax=ax_roc)
            ax_roc.set_title(f"Kryptonite-{n} : ROC Curve (AUC={auc:.4f})")
            fig_roc.tight_layout()
            fig_roc.savefig(out_dir / "roc_curve.png", dpi=150)
            plt.close(fig_roc)
        except Exception as e:
            print(f"  (ROC plot skipped: {e})")

    # Training / validation curves (from early_stopping internal split)
    try:
        clf = model.named_steps["clf"]
        fig_curves, ax_curves = plt.subplots(figsize=(6, 4))
        if hasattr(clf, "loss_curve_") and clf.loss_curve_:
            ax_curves.plot(clf.loss_curve_, label="Training loss")
        if hasattr(clf, "validation_scores_") and clf.validation_scores_ is not None:
            ax_curves.plot(clf.validation_scores_, label="Validation score")
        ax_curves.set_title(f"Kryptonite-{n} : Training curves")
        ax_curves.set_xlabel("Iterations (epochs)")
        ax_curves.legend()
        fig_curves.tight_layout()
        fig_curves.savefig(out_dir / "training_curves.png", dpi=150)
        plt.close(fig_curves)

        # report how many epochs actually ran
        epochs_run = len(getattr(clf, "loss_curve_", []))
        print(f"  Epochs run (until early stopping or max_iter): {epochs_run}")
    except Exception as e:
        print(f"  (training curves skipped: {e})")

    # Save per-n details as JSON
    details = {
        "n": n,
        "method": METHOD,
        "best_params": best_params,
        "train_time_sec": train_time,
        "metrics": {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "auc": None if auc is None else float(auc)
        },
        "shapes": {
            "Xtr": list(Xtr.shape),
            "ytr": int(len(ytr)),
            "Xte": list(Xte.shape),
            "yte": int(len(yte)),
        }
    }
    with open(out_dir / "evaluation_summary.json", "w") as f:
        json.dump(details, f, indent=2)

    summary_rows.append({
        "n": n,
        "method": METHOD,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": auc,
        "train_time_sec": train_time
    })

print("\nDone evaluating all datasets.")

# Summary
df_summary = pd.DataFrame(summary_rows).sort_values("n")
print(df_summary)

summary_csv = EVAL_DIR / "summary_test_metrics.csv"
df_summary.to_csv(summary_csv, index=False)
print(f"\nSaved overall summary CSV → {summary_csv}")

TEST_SUMMARY = (Path("./Evaluation") / METHOD / "summary_test_metrics.csv")
SAVE_DIR     = (Path("./Evaluation") / METHOD)
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Load data
df_val = pd.read_csv(RESULTS_CSV)
df_test = pd.read_csv(TEST_SUMMARY)

# Clean up and align columns
df_val = df_val[["n", "acc_mean"]].rename(columns={"acc_mean": "val_acc"})
df_test = df_test[["n", "accuracy"]].rename(columns={"accuracy": "test_acc"})

# Merge on n
df = pd.merge(df_val, df_test, on="n", how="inner").sort_values("n")

# Thresholds dictionary
thresholds = {10: 0.94, 12: 0.93, 14: 0.92, 16: 0.91, 18: 0.80, 20: 0.75}
df["threshold"] = df["n"].map(thresholds)

# --- Plot ---
plt.figure(figsize=(8, 5))
plt.plot(df["n"], df["val_acc"], marker="o", label="Validation Accuracy", color="tab:blue", linewidth=2)
plt.plot(df["n"], df["test_acc"], marker="s", label="Test Accuracy", color="tab:green", linewidth=2)
plt.plot(df["n"], df["threshold"], marker="^", linestyle="-", color="red", linewidth=2, label="Threshold")

plt.title("Validation vs Test Accuracy across Kryptonite-n datasets")
plt.xlabel("n dimensional features")
plt.ylabel("Accuracy")
plt.ylim(0.7, 1.0)
plt.xticks(df["n"])
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()

# Save the plot
save_path = SAVE_DIR / "validation_vs_test_accuracy.png"
plt.savefig(save_path, dpi=300)

print(f"Saved plot to {save_path}")
print("\nComparison table:")
print(df)

