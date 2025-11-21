#!/usr/bin/env python3
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import argparse

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from tqdm.auto import tqdm

SEED = 42


def discover_variants(data_dir: Path) -> Dict[int, Dict[str, Path]]:
    variants = {}
    for p in data_dir.rglob("*.npy"):
        name = p.name.lower()
        # try to extract n from the filename (works for most naming schemes)
        n = None
        for tok in name.replace("-", "_").split("_"):
            if tok.isdigit():
                n = int(tok)
                break
        if n is None:
            continue
        d = variants.setdefault(n, {"X": None, "y": None, "X_hidden": None})
        # categorize the file based on its name pattern
        if name.startswith("kryptonite") and name.endswith("-x.npy"):
            d["X"] = p
        elif name.startswith("kryptonite") and name.endswith("-y.npy"):
            d["y"] = p
        elif name.startswith("hidden-kryptonite") and name.endswith("-x.npy"):
            d["X_hidden"] = p
    # only keep n values where we have at minimum X and y
    return {n: d for n, d in variants.items() if d["X"] and d["y"]}


def cv_grid(model, param_grid, X, y, cv=5, n_jobs=-1, verbose=0):
    """
    Wrapper around GridSearchCV
    """
    gs = GridSearchCV(model, param_grid=param_grid, cv=cv, scoring="accuracy",
                      n_jobs=n_jobs, verbose=verbose)
    gs.fit(X, y)
    return gs



def build_all_searches() -> Dict[str, Tuple[str, object, dict]]:
    """
    Define all baseline models we want to benchmark
    """
    searches = {}

    # Logistic Regression with L2 regularization - simple, fast baseline
    searches["logreg"] = (
        "LogReg(L2)",
        Pipeline([("scaler", StandardScaler()),
                  ("clf", LogisticRegression(max_iter=2000, random_state=SEED))]),
        {"clf__C": [0.1, 1.0, 10.0]}
    )

    # polynomial features + logistic Regression can capture some non-linearity
    searches["poly"] = (
        "Poly2+LogReg",
        Pipeline([("poly", PolynomialFeatures(degree=2, include_bias=False)),
                  ("scaler", StandardScaler(with_mean=False)),
                  ("clf", LogisticRegression(max_iter=2000, random_state=SEED))]),
        {"clf__C": [0.1, 1.0, 10.0]}
    )

    # SVM with RBF kernel = good non-linear separator, but can be slow on large datasets
    searches["svm"] = (
        "SVM(RBF)",
        Pipeline([("scaler", StandardScaler()),
                  ("clf", SVC(kernel="rbf", random_state=SEED))]),
        {"clf__C": [4, 5, 6],
         "clf__gamma": [0.15, 0.2, 0.25]}
    )

    # Random Forest 
    searches["rf"] = (
        "RandomForest",
        RandomForestClassifier(random_state=SEED),
        {"n_estimators": [200, 400], "max_depth": [None, 10, 20], "min_samples_leaf": [1, 2]}
    )

    # MLP with 2 hidden layers
    searches["mlp"] = (
        "MLP(2x)",
        Pipeline([("scaler", StandardScaler()),
                  ("clf", MLPClassifier(hidden_layer_sizes=(128, 128),
                                        activation="relu",
                                        alpha=1e-4,
                                        learning_rate_init=1e-3,
                                        batch_size=128,
                                        max_iter=1000,
                                        random_state=SEED))]),
        {"clf__hidden_layer_sizes": [(256,128), (256,192), (256,256)],
         "clf__alpha": [1e-5, 3e-5, 1e-4, 3e-4],
         "clf__learning_rate_init": [7e-4, 1e-3, 1.5e-3],
         "clf__early_stopping": [True]}
    )

    return searches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("./Datasets"))
    ap.add_argument("--results-dir", type=Path, default=Path("./results"))
    ap.add_argument("--export-hidden", action="store_true", help="Save predictions on hidden test data")
    ap.add_argument("--models", type=str, default="logreg,poly,svm,rf,mlp",
                    help="Comma-separated subset of {logreg,poly,svm,rf,mlp}")
    ap.add_argument("--n-list", type=str, default="",
                    help="Comma-separated list of n values to include (e.g., 10,12,14). Empty = all found.")
    ap.add_argument("--folds", type=int, default=5, help="Number of CV folds")
    ap.add_argument("--gs-verbose", type=int, default=0, help="Verbosity for GridSearchCV")
    ap.add_argument("--gs-jobs", type=int, default=-1, help="n_jobs for GridSearchCV (-1 uses all cores)")
    args = ap.parse_args()

    args.results_dir.mkdir(parents=True, exist_ok=True)
    Path("./hiddenlabels").mkdir(parents=True, exist_ok=True)

    # Find all available datasets
    variants = discover_variants(args.data_dir)
    if not variants:
        print(f"No datasets found under {args.data_dir}.")
        return

    # Filter by n-list if provided (useful for debugging or running a subset)
    if args.n_list.strip():
        keep = set(int(x.strip()) for x in args.n_list.split(",") if x.strip())
        variants = {n: d for n, d in variants.items() if n in keep}
        if not variants:
            print("No matching n values after filtering with --n-list.")
            return

    # select which models to run
    all_searches = build_all_searches()
    wanted_keys = [k.strip() for k in args.models.split(",") if k.strip()]
    invalid = [k for k in wanted_keys if k not in all_searches]
    if invalid:
        raise SystemExit(f"Unknown model keys: {invalid}. Choose from {{logreg,poly,svm,rf,mlp}}.")
    selected = {k: all_searches[k] for k in wanted_keys}

    all_rows = []

    # Outer progress bar over datasets
    for n, paths in tqdm(sorted(variants.items()), desc="Datasets", unit="set"):
        print(f"\nn = {n}")
        X = np.load(paths["X"])
        y = np.load(paths["y"])
        print("X shape:", X.shape, " y mean:", y.mean())

        best_score = -1.0
        best_search = None
        best_name = None

        # Inner progress bar over models 
        for key in tqdm(selected, desc=f"Models (n={n})", leave=False):
            name, est, grid = selected[key]
            print(f" > Grid-searching {name} ...")
            gs = cv_grid(est, grid, X, y, cv=args.folds, n_jobs=args.gs_jobs, verbose=args.gs_verbose)
            mean_acc = gs.best_score_
            print(f"best_acc={mean_acc:.4f} best_params={gs.best_params_}")
            all_rows.append({"n": n, "model": name, "acc_mean": mean_acc, "best_params": gs.best_params_})
            # keep track of the best performing model for this n
            if mean_acc > best_score:
                best_score, best_search, best_name = mean_acc, gs, name

        # Save per-n CSV
        df_n = pd.DataFrame([r for r in all_rows if r["n"] == n]).sort_values("acc_mean", ascending=False)
        out_csv = args.results_dir / f"results_n{n}.csv"
        df_n.to_csv(out_csv, index=False)
        print(f"Saved {out_csv}")

        # Export hidden predictions if indicated
        # Fit the best model on the full training set and predict on hidden test data
        if args.export_hidden and paths.get("X_hidden"):
            print(f"Fitting best model '{best_name}' on full train and exporting hidden predictions...")
            best_est = best_search.best_estimator_
            best_est.fit(X, y)
            Xh = np.load(paths["X_hidden"])
            yh = best_est.predict(Xh).astype(int)
            out_path = Path("./hiddenlabels") / f"y_predicted_{n}.npy"
            np.save(out_path, yh)
            print(f"  Wrote {out_path} (shape={yh.shape})")

    # Save combined CSV across all models and n values
    df_all = pd.DataFrame(all_rows).sort_values(["n", "acc_mean"], ascending=[True, False])
    df_all.to_csv(args.results_dir / "results_all.csv", index=False)
    print("\nSummary")
    print(df_all.to_string(index=False))


if __name__ == "__main__":
    main()
