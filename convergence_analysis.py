import os
import re
import ast
import time
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, accuracy_score

import warnings
from sklearn.exceptions import ConvergenceWarning

# Silence sklearn convergence warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Directory with kryptonite-n-x-train.npy / kryptonite-n-y-train.npy
TRAIN_DIR = Path("./Datasets/Train_Data")

# which tuning results to use: "random", "grid", or "bayes"
METHOD = "random"

# Where hyperparameter tuning results live
res_dir = Path("./MLP_optimization") / METHOD
per_n = sorted(res_dir.glob("results_n*.csv"))
if per_n:
    df_all = (
        pd.concat((pd.read_csv(p) for p in per_n), ignore_index=True)
        .sort_values("n")
    )
    df_all.to_csv(res_dir / "results_all.csv", index=False)

RESULTS_CSV = res_dir / "results_all.csv"

# where to save evaluation artifacts
CONV_DIR = Path("./Convergence_Analysis") / METHOD
CONV_DIR.mkdir(parents=True, exist_ok=True)

SEED = 45
cv_obj = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)


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

        # Be safe: convert hidden_layer_sizes list -> tuple if needed
        hls_key = "clf__hidden_layer_sizes"
        if hls_key in params and isinstance(params[hls_key], list):
            params[hls_key] = tuple(params[hls_key])

        out[n] = params
    return out


# Accept files like: kryptonite-<n>-(X|x|Y|y)-train.npy
_PAT_TRAIN = re.compile(r"^kryptonite-(\d+)-([xy])-(train)\.npy$", re.IGNORECASE)


def discover_train_sets(train_dir: Path) -> Dict[int, Dict[str, Path]]:
    """
    Scan train_dir for kryptonite-n-(x|y)-train.npy (case-insensitive).
    Return { n: {'X': Path, 'y': Path} } only when both exist.
    """
    idx: Dict[int, Dict[str, Path]] = {}
    for p in train_dir.rglob("*.npy"):
        m = _PAT_TRAIN.match(p.name)
        if not m:
            continue
        n = int(m.group(1))
        xy = m.group(2).lower()  # 'x' or 'y'
        d = idx.setdefault(n, {"X": None, "y": None})
        if xy == "x":
            d["X"] = p
        else:
            d["y"] = p

    # Keep only complete pairs
    complete: Dict[int, Dict[str, Path]] = {}
    for n, d in idx.items():
        if d["X"] is not None and d["y"] is not None:
            complete[n] = d
    return complete


def build_base_pipeline(best_params: Dict[str, Any]) -> Pipeline:
    """
    Returns Pipeline(StandardScaler -> MLPClassifier) with best_params applied.
    We'll override max_iter and warm_start when doing epoch-wise CV training.
    """
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                MLPClassifier(
                    max_iter=1000,  # will be overridden later
                    random_state=SEED,
                ),
            ),
        ]
    )
    # Apply tuned hyperparameters (they are in the sklearn "set_params" format)
    pipe.set_params(**best_params)

    # Make sure solver works with warm_start:
    solver = pipe.get_params().get("clf__solver", "adam")
    if solver not in ("adam", "sgd"):
        print(
            f"Warning: solver={solver} not ideal for warm_start epoch-wise "
            f"training. Overriding to 'adam'."
        )
        pipe.set_params(clf__solver="adam")

    return pipe


def compute_cv_val_curves(
    X: np.ndarray,
    y: np.ndarray,
    best_params: Dict[str, Any],
    n_splits: int = 5,
    max_epochs: int = 100,
) -> (List[float], List[float], List[float]):
    """
    Perform Stratified K-Fold cross-validation on (X, y) and
    compute the mean TRAIN loss, VALIDATION loss (log loss),
    and VALIDATION accuracy across folds for each epoch.

    Returns:
        mean_train_loss: list of mean training log losses per epoch
        mean_val_loss:   list of mean validation log losses per epoch
        mean_val_acc:    list of mean validation accuracies per epoch
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    all_fold_train_losses: List[List[float]] = []
    all_fold_val_losses: List[List[float]] = []
    all_fold_accs: List[List[float]] = []

    unique_labels = np.unique(y)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        print(f"  Fold {fold_idx}/{n_splits}")
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        # Build a fresh pipeline for this fold
        pipe = build_base_pipeline(best_params)

        # Ensure warm_start and one-epoch-at-a-time training
        pipe.set_params(clf__warm_start=True, clf__max_iter=1)

        fold_train_losses: List[float] = []
        fold_val_losses: List[float] = []
        fold_accs: List[float] = []

        for epoch in range(1, max_epochs + 1):
            pipe.fit(X_tr, y_tr)  # continues training due to warm_start

            # Training predictions (for training loss)
            y_tr_proba = pipe.predict_proba(X_tr)
            train_loss = log_loss(y_tr, y_tr_proba, labels=unique_labels)
            fold_train_losses.append(train_loss)

            # Validation predictions
            y_val_proba = pipe.predict_proba(X_val)
            y_val_pred = pipe.predict(X_val)

            # Validation log loss
            val_loss = log_loss(y_val, y_val_proba, labels=unique_labels)
            fold_val_losses.append(val_loss)

            # Validation accuracy
            acc = accuracy_score(y_val, y_val_pred)
            fold_accs.append(acc)

        all_fold_train_losses.append(fold_train_losses)
        all_fold_val_losses.append(fold_val_losses)
        all_fold_accs.append(fold_accs)

    # Convert to arrays and average across folds
    train_losses_arr = np.array(all_fold_train_losses)
    val_losses_arr = np.array(all_fold_val_losses)
    accs_arr = np.array(all_fold_accs)

    mean_train_loss = train_losses_arr.mean(axis=0)
    mean_val_loss = val_losses_arr.mean(axis=0)
    mean_val_acc = accs_arr.mean(axis=0)

    return mean_train_loss.tolist(), mean_val_loss.tolist(), mean_val_acc.tolist()

def compare_optimizers_for_n16(
    X, y, best_params, n_splits=5, max_epochs=60
):
    solvers = ["adam", "sgd"]
    results = {}

    for solver in solvers:
        print(f"\n=== Evaluating solver: {solver.upper()} ===")

        params = best_params.copy()
        params["clf__solver"] = solver

        if solver == "sgd":
            params["clf__learning_rate_init"] = 0.01
            params["clf__momentum"] = 0.9


        mean_train_loss, mean_val_loss, mean_val_acc = compute_cv_val_curves(
            X, y, params, n_splits=n_splits, max_epochs=max_epochs
        )

        results[solver] = {
            "train_loss": mean_train_loss,
            "val_loss": mean_val_loss,
            "val_acc": mean_val_acc,
        }

    epochs = np.arange(1, max_epochs + 1)

    # --- Validation loss: Adam vs SGD ---
    plt.figure(figsize=(8, 5))
    for solver in solvers:
        plt.plot(epochs, results[solver]["val_loss"], label=f"{solver.upper()}")
    plt.title("Validation Log Loss vs Epochs (Adam vs SGD, n=16)")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Log Loss")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    save_path = CONV_DIR / "adam_vs_sgd_val_loss_n16.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")

    # --- Validation accuracy: Adam vs SGD ---
    plt.figure(figsize=(8, 5))
    for solver in solvers:
        plt.plot(epochs, results[solver]["val_acc"], label=f"{solver.upper()}")
    plt.title("Validation Accuracy vs Epochs (Adam vs SGD, n=16)")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    save_path = CONV_DIR / "adam_vs_sgd_val_accuracy_n16.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")

    return results


def run_single_n16(best_by_n, train_sets):
    """Re-runs convergence analysis and Adam-vs-SGD comparison for n=16 only."""
    n = 16
    if n not in best_by_n:
        print("ERROR: No best params for n=16.")
        return
    if n not in train_sets:
        print("ERROR: No training set for n=16.")
        return

    print("\n=== Re-running experiments for n = 16 ===")
    X = np.load(train_sets[n]["X"])
    y = np.load(train_sets[n]["y"])

    best_params = best_by_n[n]

    # Note: compute_cv_val_curves now returns 3 values
    mean_train_loss, mean_val_loss, mean_val_acc = compute_cv_val_curves(
        X, y, best_params, n_splits=5, max_epochs=100
    )

    epochs = np.arange(1, len(mean_val_loss) + 1)

    # ---- Validation loss plot ----
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, mean_val_loss, marker="o")
    plt.title("Validation Log Loss vs Epochs (n=16)")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Log Loss")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    path1 = CONV_DIR / "val_loss_vs_epoch_kryptonite-16_retry.png"
    plt.savefig(path1, dpi=300)
    plt.close()
    print(f"Saved: {path1}")

    # ---- Validation accuracy plot ----
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, mean_val_acc, marker="o")
    plt.title("Validation Accuracy vs Epochs (n=16)")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    path2 = CONV_DIR / "val_acc_vs_epoch_kryptonite-16_retry.png"
    plt.savefig(path2, dpi=300)
    plt.close()
    print(f"Saved: {path2}")

    # (Optional) Adam vs SGD comparison, if you still want it:
    print("\nRunning Adam vs SGD comparison again...")
    compare_optimizers_for_n16(X, y, best_params)

    print("\nFinished n=16-only run.")


def main():
    # Load the best parameters for each n
    best_by_n = load_best_params_by_n(RESULTS_CSV)
    

    # Discover available training sets
    train_sets = discover_train_sets(TRAIN_DIR)
    assert train_sets, f"No training sets found under {TRAIN_DIR}"

    run_single_n16(best_by_n, train_sets)
    return

    print(f"Found best params for n: {sorted(best_by_n.keys())}")
    print(f"Found training sets for n: {sorted(train_sets.keys())}")

    for n, paths in sorted(train_sets.items()):
        if n not in best_by_n:
            print(f"Skipping n={n}: no best params in {RESULTS_CSV}")
            continue

        print(f"\n=== Cross-validated training for n = {n} (method={METHOD}) ===")
        Xtr = np.load(paths["X"])
        ytr = np.load(paths["y"])
        print(" Train:", Xtr.shape, ytr.shape)

        best_params = best_by_n[n]

        # Compute cross-validated curves
        t0 = time.time()
        mean_train_loss, mean_val_loss, mean_val_acc = compute_cv_val_curves(
            Xtr, ytr, best_params, n_splits=5, max_epochs=100
        )
        elapsed = time.time() - t0
        print(f"  Finished CV for n={n} in {elapsed:.1f} s")

        epochs = np.arange(1, len(mean_val_loss) + 1)

        # ---- Plot TRAIN + VALIDATION loss vs epochs ----
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, mean_train_loss, marker="o", label="Train Loss", color="orange")
        plt.plot(epochs, mean_val_loss, marker="s", label="Validation Loss", color="blue")
        plt.title(f"Mean CV Train & Validation Loss vs Epochs (kryptonite-{n})")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()
        plt.tight_layout()

        save_path_loss = CONV_DIR / f"train_val_loss_vs_epoch_kryptonite-{n}.png"
        plt.savefig(save_path_loss, dpi=300)
        plt.close()
        print(f"  Saved train+val loss plot to {save_path_loss}")

        # ---- Plot validation accuracy vs epochs ----
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, mean_val_acc, marker="o")
        plt.title(f"Mean CV Validation Accuracy vs Epochs (kryptonite-{n})")
        plt.xlabel("Epoch")
        plt.ylabel("Validation Accuracy")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()

        save_path_acc = CONV_DIR / f"val_acc_vs_epoch_kryptonite-{n}.png"
        plt.savefig(save_path_acc, dpi=300)
        plt.close()
        print(f"  Saved accuracy plot to {save_path_acc}")

    print("\nDone: train+val loss and val accuracy plots for all n.")


if __name__ == "__main__":
    main()
