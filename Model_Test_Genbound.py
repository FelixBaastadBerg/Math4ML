
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Model_Test.py
# Epoch-wise training + norm-based bound logging for n = 10
import os
import re
import ast
import json
import time
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import warnings
from sklearn.exceptions import ConvergenceWarning

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Silence the "did not converge" warning since we deliberately use max_iter=1 with warm_start
warnings.filterwarnings("ignore", category=ConvergenceWarning)

TRAIN_DIR = Path("./Datasets/Train_Data")
TEST_DIR  = Path("./Datasets/Test_Data")

# which tuning results to use: "random", "grid", or "bayes"
METHOD = "random"

res_dir = Path("./MLP_optimization") / METHOD
per_n = sorted(res_dir.glob("results_n*.csv"))
if per_n:
    df_all = (
        pd.concat((pd.read_csv(p) for p in per_n), ignore_index=True)
        .sort_values("n")
    )
    df_all.to_csv(res_dir / "results_all.csv", index=False)

RESULTS_CSV = Path(f"./MLP_optimization/{METHOD}/results_all.csv")

# where to save evaluation artifacts
EVAL_DIR = Path("./Evaluation") / METHOD
EVAL_DIR.mkdir(parents=True, exist_ok=True)

SEED = 45
TARGET_N = 20          # we only do n = 10 for now
N_EPOCHS = 100         # how many epochs to train/log

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

"""
Replace np.float64(1.23e-5) with 1.23e-5 in a string representation
"""
def clean_np_floats(s: str) -> str:
    
    return re.sub(r"np\.float64\(([^)]+)\)", r"\1", s)


"""
Reads results_all.csv and returns { n: best_params_dict }
Handles stringified dicts and cleans np.float64(...) wrappers
"""
def load_best_params_by_n(results_csv: Path) -> Dict[int, Dict[str, Any]]:
    
    assert results_csv.exists(), f"Missing results CSV: {results_csv}"
    df = pd.read_csv(results_csv)
    out: Dict[int, Dict[str, Any]] = {}
    for _, row in df.iterrows():
        n = int(row["n"])
        raw = row["best_params"]
        if isinstance(raw, str):
            cleaned = clean_np_floats(raw)
            try:
                params = ast.literal_eval(cleaned)
            except Exception as e:
                raise ValueError(
                    f"Failed to parse best_params for n={n}:\n{raw}\nError: {e}"
                )
        elif isinstance(raw, dict):
            params = raw
        else:
            params = dict(raw)

        # Be safe... convert hidden_layer_sizes list -> tuple if needed
        hls_key = "clf__hidden_layer_sizes"
        if hls_key in params and isinstance(params[hls_key], list):
            params[hls_key] = tuple(params[hls_key])

        out[n] = params
    return out

# Accept files like the kryptonite.npy files
_PAT = re.compile(r"^kryptonite-(\d+)-([xy])-(train|test)\.npy$", re.IGNORECASE)

"""
Scan dir_path for kryptonite-n-(x|y)-<expected_split>.npy 
Return { n: {'X': Path or None, 'y': Path or None} }
"""
def _index_split(dir_path: Path, expected_split: str) -> Dict[int, Dict[str, Path]]:
    
    idx: Dict[int, Dict[str, Path]] = {}
    for p in dir_path.rglob("*.npy"):
        m = _PAT.match(p.name)
        if not m:
            continue
        n = int(m.group(1))
        xy = m.group(2).lower()  # 'x' or 'y'
        split = m.group(3).lower()  # 'train' or 'test'
        if split != expected_split:
            continue
        d = idx.setdefault(n, {"X": None, "y": None})
        if xy == "x":
            d["X"] = p
        else:
            d["y"] = p
    return idx

"""
Find matching train/test pairs by n
Returns { n: {"Xtr","ytr","Xte","yte"} } only when all four exist
"""
def discover_dataset_pairs(train_dir: Path, test_dir: Path) -> Dict[int, Dict[str, Path]]:
    
    tr = _index_split(train_dir, "train")
    te = _index_split(test_dir, "test")

    pairs: Dict[int, Dict[str, Path]] = {}
    for n in sorted(set(tr.keys()) & set(te.keys())):
        if tr[n]["X"] and tr[n]["y"] and te[n]["X"] and te[n]["y"]:
            pairs[n] = {
                "Xtr": tr[n]["X"], "ytr": tr[n]["y"],
                "Xte": te[n]["X"], "yte": te[n]["y"],
            }
    return pairs

"""
Returns Pipeline(StandardScaler -> MLPClassifier) with best_params applied

Override to:
    - warm_start=True
    - max_iter=1
so that each call to fit() performs ONE epoch, and we can log per-epoch metrics
"""
def build_pipeline_from_params(best_params: Dict[str, Any]) -> Pipeline:
    
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(
            max_iter=1,              # exactly 1 epoch per fit call
            tol=0.0,                 # disable tol-based early stopping here
            early_stopping=False,
            n_iter_no_change=1000,
            random_state=SEED,
            warm_start=True,         # keep weights between successive fit() calls
            shuffle=True
        ))
    ])
    # Apply tuned params
    pipe.set_params(**best_params)

    # Force our per-epoch training setup
    pipe.set_params(
        clf__early_stopping=False,
        clf__max_iter=1,
        clf__warm_start=True,
        clf__shuffle=True,
    )
    return pipe

"""
Heuristic implementation of a spectrally-normalized, margin-based bound term.

Structure inspired by Bartlett et al. (2017):
    bound_term ~ ( R * sqrt(S) ) / (gamma_hat * sqrt(m))

    where:
    R  = product of spectral norms ||W_l||_2
    S  = sum_l ||W_l||_F^2 / ||W_l||_2^2
    m  = # of training examples
    gamma_hat = empirical margin, here approximated from predict_proba:
        gamma_hat = min_i [ p_true(i) - max_{y!=true} p_y(i) ]

This is a *qualitative* SOTA-style norm-based bound component,
meant for logging how the theoretical complexity evolves.
"""
def compute_spectral_norm_bound_term(clf: MLPClassifier,
                                     X_train: np.ndarray,
                                     y_train: np.ndarray) -> float | None:
    

    # Must have probabilities and classes
    if not hasattr(clf, "predict_proba") or not hasattr(clf, "classes_"):
        return None

    try:
        proba = clf.predict_proba(X_train)  # shape (m, C)
    except Exception:
        return None

    if proba.ndim != 2:
        return None

    m, C = proba.shape
    if m == 0 or C < 2:
        return None

    classes = clf.classes_
    if classes.shape[0] != C:
        return None

    # Map label -> index in proba
    label_to_idx = {label: idx for idx, label in enumerate(classes)}

    # Build margin from probabilities:
    # margin_i = p_true(i) - max_{y!=true} p_y(i)
    margins: List[float] = []
    for yi, probs in zip(y_train, proba):
        if yi not in label_to_idx:
            return None
        idx_true = label_to_idx[yi]
        p_true = float(probs[idx_true])
        # max probability among incorrect classes
        p_other = float(np.max(np.delete(probs, idx_true)))
        margins.append(p_true - p_other)

    margins = np.array(margins, dtype=float)
    gamma_hat = float(np.min(margins))
    if gamma_hat <= 0:
        # classifier not margin-separable yet... bound will be huge / meaningless
        gamma_hat = 1e-8

    # layer-wise weights
    coefs: List[np.ndarray] = getattr(clf, "coefs_", None)
    if coefs is None or len(coefs) == 0:
        return None

    spectral_prod = 1.0
    sum_frob_over_spec2 = 0.0

    for W in coefs:
        spec = float(np.linalg.norm(W, 2))
        frob = float(np.linalg.norm(W, "fro"))
        if spec <= 0:
            spec = 1e-8
        spectral_prod *= spec
        sum_frob_over_spec2 += (frob ** 2) / (spec ** 2 + 1e-12)

    complexity = spectral_prod * np.sqrt(sum_frob_over_spec2)
    bound_term = complexity / (gamma_hat * np.sqrt(m) + 1e-8)

    return float(bound_term)

# ---------------------------------------------------------------------
# Main: load data, build model for n=10, epoch-wise training + logging

if __name__ == "__main__":
    # Load tuned hyperparameters
    best_by_n = load_best_params_by_n(RESULTS_CSV)

    # Discover datasets
    pairs = discover_dataset_pairs(TRAIN_DIR, TEST_DIR)
    assert pairs, f"No matching train/test pairs found under {TRAIN_DIR} and {TEST_DIR}"

    print(f"Found best params for n: {sorted(best_by_n.keys())}")
    print(f"Found dataset pairs for n: {sorted(pairs.keys())}")

    if TARGET_N not in pairs:
        raise RuntimeError(f"Dataset pair for n={TARGET_N} not found.")
    if TARGET_N not in best_by_n:
        raise RuntimeError(f"Best params for n={TARGET_N} not found in {RESULTS_CSV}.")

    n = TARGET_N
    paths = pairs[n]

    print(f"\n=== Epoch-wise evaluation for n = {n} (method={METHOD}) ===")
    Xtr = np.load(paths["Xtr"])
    ytr = np.load(paths["ytr"])
    Xte = np.load(paths["Xte"])
    yte = np.load(paths["yte"])

    print(" Train:", Xtr.shape, ytr.shape, " Test:", Xte.shape, yte.shape)

    best_params = best_by_n[n]
    model = build_pipeline_from_params(best_params)

    out_dir = EVAL_DIR / f"n{n}" / "epoch_logs"
    ensure_dir(out_dir)

    # Prepare logging structure
    epoch_logs = []

    # Epoch-wise training
    for epoch in range(1, N_EPOCHS + 1):
        t0 = time.time()
        model.fit(Xtr, ytr)   # 1 epoch because clf__max_iter=1 and warm_start=True
        train_time = time.time() - t0

        # Predictions for accuracy
        ytr_pred = model.predict(Xtr)
        yte_pred = model.predict(Xte)

        train_acc = accuracy_score(ytr, ytr_pred)
        test_acc = accuracy_score(yte, yte_pred)

        train_err = 1.0 - float(train_acc)
        test_err  = 1.0 - float(test_acc)

        # Extract underlying classifier
        clf: MLPClassifier = model.named_steps["clf"]

        # Compute norm-based bound term and implied error bound
        bound_term = compute_spectral_norm_bound_term(clf, Xtr, ytr)
        if bound_term is None or np.isnan(bound_term):
            bound_term_print = "NA"
            bound_error = None
        else:
            bound_error = float(np.clip(train_err + bound_term, 0.0, 1.0))
            bound_term_print = f"{bound_term:.4e}"

        log_row = {
            "epoch": epoch,
            "train_accuracy": float(train_acc),
            "test_accuracy": float(test_acc),
            "train_error": train_err,
            "test_error": test_err,
            "bound_term": None if bound_term is None else float(bound_term),
            "bound_error": bound_error,
            "train_time_sec": train_time,
        }
        epoch_logs.append(log_row)

        print(
            f"Epoch {epoch:03d} | "
            f"train_acc={train_acc:.4f}, test_acc={test_acc:.4f}, "
            f"bound_term={bound_term_print}"
        )

    # Save epoch-wise logs to CSV
    df_epochs = pd.DataFrame(epoch_logs)
    csv_path = out_dir / f"epoch_bounds_n{n}.csv"
    df_epochs.to_csv(csv_path, index=False)
    print(f"\nSaved epoch-wise accuracies, errors, and bound terms to {csv_path}")

    # Optional quick plot of accuracies + bound term magnitude
    try:
        fig, ax1 = plt.subplots(figsize=(7, 4))

        ax1.plot(df_epochs["epoch"], df_epochs["train_accuracy"],
                 label="Train Accuracy", linewidth=2)
        ax1.plot(df_epochs["epoch"], df_epochs["test_accuracy"],
                 label="Test Accuracy", linewidth=2, linestyle="--")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Accuracy")
        ax1.set_ylim(0.0, 1.0)
        ax1.grid(True, linestyle="--", alpha=0.5)

        ax2 = ax1.twinx()
        if df_epochs["bound_term"].notna().any():
            ax2.plot(
                df_epochs["epoch"],
                df_epochs["bound_term"],
                label="Norm-based bound term",
                linewidth=1.5,
                linestyle=":",
            )
            ax2.set_ylabel("Bound term (scale arbitrary)")

        lines, labels = ax1.get_legend_handles_labels()
        if df_epochs["bound_term"].notna().any():
            lines2, labels2 = ax2.get_legend_handles_labels()
            lines += lines2
            labels += labels2
        ax1.legend(lines, labels, loc="lower right")

        fig.tight_layout()
        plot_path = out_dir / f"epoch_bounds_n{n}.png"
        fig.savefig(plot_path, dpi=200)
        plt.close(fig)
        print(f"Saved epoch-wise accuracy/bound plot to {plot_path}")
    except Exception as e:
        print(f"(Plotting epoch-wise curves failed: {e})")
