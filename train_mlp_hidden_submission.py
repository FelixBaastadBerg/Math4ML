#!/usr/bin/env python3
"""
train_mlp_for_hidden.py

Train a single MLP per Kryptonite-n dataset using the best hyperparameters
from results_all.csv on the full labelled datasets

This script then predicts labels for the hidden sets

Outputs are written to:
    ./Hidden_Kryptonite_Submission/
        n10/
            mlp_model.joblib
            hidden_predictions.npy
            hidden_predictions.csv
            manifest.json
        ...
        hidden_predictions_all.csv
"""

import re
import ast
import json
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

# ---------------- Paths / config ----------------

DATA_ROOT = Path("./Datasets")
SUBMIT_DIR = Path("./Hidden_Kryptonite_Submission")
SUBMIT_DIR.mkdir(parents=True, exist_ok=True)

METHOD = "random" # which tuning results to use: "random", "grid", or "bayes"
RESULTS_CSV = Path(f"./MLP_ECE/MLP_optimization/{METHOD}/results_all.csv")

SEED = 45


# ---------------- Utilities for loading best hyperparameters ----------------

def clean_np_floats(s: str) -> str:
    """Replace np.float64(1.23e-5) with 1.23e-5 in a string representation."""
    return re.sub(r"np\.float64\(([^)]+)\)", r"\1", s)



def load_best_params_by_n(results_csv: Path) -> Dict[int, Dict[str, Any]]:
    """
    Reads results_all.csv and returns { n: best_params_dict }
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
                raise ValueError(f"Failed to parse best_params for n={n}:\n{raw}\nError: {e}")
        elif isinstance(raw, dict):
            params = raw
        else:
            params = dict(raw)

        hls_key = "clf__hidden_layer_sizes"
        if hls_key in params and isinstance(params[hls_key], list):
            params[hls_key] = tuple(params[hls_key])

        out[n] = params
    return out


def build_pipeline_from_params(best_params: Dict[str, Any]) -> Pipeline:
    """
    Returns Pipeline(StandardScaler -> MLPClassifier) with best_params applied
    """
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(
            max_iter=1000,
            tol=1e-3,
            early_stopping=False,
            n_iter_no_change=15,
            random_state=SEED,
        )),
    ])
    pipe.set_params(**best_params)
    return pipe



# ---------------- Dataset discovery ----------------

# Full labelled data: kryptonite-n-X.npy, kryptonite-n-y.npy
_FULL_PAT = re.compile(r"^kryptonite-(\d+)-([xy])\.npy$", re.IGNORECASE)

# Hidden data: hidden-kryptonite-n-X.npy
_HIDDEN_PAT = re.compile(r"^hidden-kryptonite-(\d+)-x\.npy$", re.IGNORECASE)



def discover_full_labelled(data_root: Path) -> Dict[int, Dict[str, Path]]:
    """
    Discover full labelled datasets
        kryptonite-n-X.npy
        kryptonite-n-y.npy

    Returns { n: {"X": Path, "y": Path} } for n where both exist
    """
    idx: Dict[int, Dict[str, Path]] = {}
    for p in data_root.rglob("kryptonite-*-*.npy"):
        m = _FULL_PAT.match(p.name)
        if not m:
            continue
        n = int(m.group(1))
        xy = m.group(2).lower()  # 'x' or 'y'
        d = idx.setdefault(n, {"X": None, "y": None})
        if xy == "x":
            d["X"] = p
        else:
            d["y"] = p

    out: Dict[int, Dict[str, Path]] = {}
    for n, d in idx.items():
        if d["X"] is not None and d["y"] is not None:
            out[n] = d
    return out


def discover_hidden_inputs(data_root: Path) -> Dict[int, Path]:
    """
    Discover hidden datasets:
        hidden-kryptonite-n-X.npy

    Returns { n: X_path }.
    """
    hidden: Dict[int, Path] = {}
    for p in data_root.rglob("hidden-kryptonite-*-*.npy"):
        m = _HIDDEN_PAT.match(p.name)
        if not m:
            continue
        n = int(m.group(1))
        hidden[n] = p
    return hidden




# ---------------- Main ----------------

def main():
    # Load best hyperparameters per n
    best_by_n = load_best_params_by_n(RESULTS_CSV)

    # Discover full labelled datasets and hidden inputs
    full_sets = discover_full_labelled(DATA_ROOT)
    hidden_sets = discover_hidden_inputs(DATA_ROOT)

    ns = sorted(set(best_by_n.keys()) & set(full_sets.keys()) & set(hidden_sets.keys()))
    if not ns:
        raise RuntimeError("No matching n found across best_params, full datasets, and hidden datasets.")

    print("Using n values:", ns)

    all_pred_rows = []

    for n in ns:
        print(f"\n——— Training final MLP for n={n} ———")

        full_paths = full_sets[n]
        hidden_X_path = hidden_sets[n]
        best_params = best_by_n[n]

        # Load full labelled data
        X_full = np.load(full_paths["X"])
        y_full = np.load(full_paths["y"])
        print(f"  Full data: X{X_full.shape}, y{y_full.shape}")
        print(f"  Hidden X: {hidden_X_path.name}")

        # Build and train model
        model = build_pipeline_from_params(best_params)
        model.fit(X_full, y_full)

        # Predict on hidden data
        X_hidden = np.load(hidden_X_path)
        y_hidden_pred = model.predict(X_hidden)

        # Optionally compute probabilities
        try:
            y_hidden_prob = model.predict_proba(X_hidden)[:, 1]
        except Exception:
            y_hidden_prob = None

        # Per-n output directory
        n_dir = SUBMIT_DIR / f"n{n}"
        n_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        model_path = n_dir / "mlp_model.joblib"
        dump(model, model_path)
        print(f"  Saved model -> {model_path}")

        # Save predictions (npy + csv)
        npy_path = n_dir / "hidden_predictions.npy"
        np.save(npy_path, y_hidden_pred.astype(int))

        if y_hidden_prob is not None:
            df_pred = pd.DataFrame({
                "index": np.arange(len(y_hidden_pred)),
                "y_pred": y_hidden_pred.astype(int),
                "y_prob": y_hidden_prob.astype(float),
            })
        else:
            df_pred = pd.DataFrame({
                "index": np.arange(len(y_hidden_pred)),
                "y_pred": y_hidden_pred.astype(int),
            })

        csv_path = n_dir / "hidden_predictions.csv"
        df_pred.to_csv(csv_path, index=False)
        print(f"  Saved hidden predictions -> {csv_path}")

        manifest = {
            "n": n,
            "model_path": str(model_path),
            "hidden_X_path": str(hidden_X_path),
            "hidden_pred_npy": str(npy_path),
            "hidden_pred_csv": str(csv_path),
            "num_hidden_samples": int(len(y_hidden_pred)),
            "best_params": best_params,
        }
        with open(n_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        # Collect for global summary
        df_pred_global = df_pred.copy()
        df_pred_global.insert(0, "n", n)
        all_pred_rows.append(df_pred_global)

    # Global CSV with predictions for all n
    if all_pred_rows:
        df_all = pd.concat(all_pred_rows, ignore_index=True)
        all_csv_path = SUBMIT_DIR / "hidden_predictions_all.csv"
        df_all.to_csv(all_csv_path, index=False)
        print(f"\nSaved combined predictions -> {all_csv_path}")

    print("\nDone training and generating hidden predictions")


if __name__ == "__main__":
    main()
