"""
Optimise MLP on Kryptonite-n datasets

search grid     : GridSearchCV 
search random   : RandomizedSearchCV
search bayes    : Optuna Bayesian optimization
"""
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import argparse

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold

from tqdm.auto import tqdm
from scipy.stats import loguniform 
import optuna

from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

SEED = 45

cv_obj = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)


def discover_variants(data_dir: Path) -> Dict[int, Dict[str, Path]]:
    """
    Maps n -> {'X': path, 'y': path}
    """
    variants: Dict[int, Dict[str, Path]] = {} 
    for p in data_dir.rglob("*.npy"):
        name = p.name.lower()
        if not name.startswith("kryptonite-"):
            continue
        if name.endswith("-x-train.npy") or name.endswith("-y-train.npy"):
            parts = name.split("-")
            if len(parts) < 4:
                continue
            try:
                n = int(parts[1])
            except ValueError:
                continue
            is_x = name.endswith("-x-train.npy")
            d = variants.setdefault(n, {"X": None, "y": None})
            if is_x:
                d["X"] = p
            else:
                d["y"] = p
    return {n: d for n, d in variants.items() if d["X"] and d["y"]}


def base_mlp_pipeline() -> Pipeline:
    """
    Current Optimal MLP based on results from run_baselines_selectable.py
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(256, 256),
            activation="relu",
            alpha=3e-5,
            tol=1e-3,
            learning_rate_init=7e-4,
            batch_size=128,
            max_iter=500,
            early_stopping=False,
            n_iter_no_change=15,
            random_state=SEED
        ))
    ])

# Search Spaces
def grid_params() -> dict:
    return {
        "clf__hidden_layer_sizes": [(256,128), (256,192), (256,256), (320,256), (384,256)],
        "clf__alpha":              [1e-5, 3e-5, 1e-4],
        "clf__learning_rate_init": [5e-4, 7e-4, 1e-3, 1.5e-3],
        "clf__learning_rate":      ["constant", "adaptive"],
        "clf__batch_size":         [64, 128, 256],
        "clf__activation":         ["relu", "tanh"],
    }


def random_distributions() -> dict:
    """
    Parameter distributions for RandomizedSearchCV
    """
    hls_choices = [
        (256,128), (256,192), (256,256), (320,256), (384,256),
        (256,), (384,), (256,256,128)
    ]
    return {
        "clf__hidden_layer_sizes": hls_choices,
        "clf__activation": ["relu", "tanh"],
        "clf__alpha": loguniform(5e-6, 5e-4), # around 3e-5
        "clf__learning_rate_init": loguniform(5e-4, 2e-3), # around 7e-4
        "clf__learning_rate": ["constant", "adaptive"],
        "clf__batch_size": [64, 128, 256],
    }


def run_grid(X, y, cv, n_jobs, verbose):
    est = base_mlp_pipeline()
    params = grid_params()
    gs = GridSearchCV(est, params, cv=cv_obj, scoring="accuracy", n_jobs=n_jobs, verbose=verbose)
    gs.fit(X, y)
    return gs.best_score_, gs.best_params_, gs.best_estimator_


def run_random(X, y, cv, n_jobs, verbose, n_iter):
    est = base_mlp_pipeline()
    params = random_distributions()
    rs = RandomizedSearchCV(
        estimator=est,
        param_distributions=params,
        n_iter=n_iter,
        scoring="accuracy",
        cv=cv_obj,
        random_state=SEED,
        n_jobs=n_jobs,
        verbose=verbose
    )
    rs.fit(X, y)
    return rs.best_score_, rs.best_params_, rs.best_estimator_


def run_bayes(X, y, cv_splits, n_trials, n_jobs):
    est = base_mlp_pipeline()

    def objective(trial):
        hls_choices = [
            (256,128), (256,192), (256,256), (320,256), (384,256),
            (256,), (384,), (256,256,128)
        ]
        params = {
            "clf__hidden_layer_sizes": trial.suggest_categorical("clf__hidden_layer_sizes", hls_choices),
            "clf__activation": trial.suggest_categorical("clf__activation", ["relu", "tanh"]),
            "clf__alpha": trial.suggest_float("clf__alpha", 5e-6, 5e-4, log=True),
            "clf__learning_rate_init": trial.suggest_float("clf__learning_rate_init", 5e-4, 2e-3, log=True),
            "clf__learning_rate": trial.suggest_categorical("clf__learning_rate", ["constant", "adaptive"]),
            "clf__batch_size": trial.suggest_categorical("clf__batch_size", [64, 128, 256]),
        }
        model = est.set_params(**params)
        scores = cross_val_score(model, X, y, cv=cv_splits, scoring="accuracy", n_jobs=n_jobs)
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best_params = study.best_params
    best_est = est.set_params(**best_params).fit(X, y)
    return study.best_value, best_params, best_est

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("./Datasets/Train_Data"))
    ap.add_argument("--results-root", type=Path, default=Path("./MLP_optimization"))
    ap.add_argument("--n-list", type=str, default="")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--gs-verbose", type=int, default=0)
    ap.add_argument("--gs-jobs", type=int, default=-1)
    ap.add_argument("--search", type=str, default="random", choices=["grid", "random", "bayes"])
    ap.add_argument("--n-trials", type=int, default=40, help="Random search iterations or Optuna trials")
    args = ap.parse_args()
    print("Searching in:", args.data_dir.resolve(), " | search:", args.search, " | n-trials:", args.n_trials, flush=True)


    # Separate subfolder per method so runs don't overwrite each other
    results_dir = Path(args.results_root) / args.search
    results_dir.mkdir(parents=True, exist_ok=True)

    variants = discover_variants(args.data_dir)
    if not variants:
        print(f"No train datasets found under {args.data_dir}.")
        return

    # Optional filter by n-list
    if args.n_list.strip():
        keep = set(int(x.strip()) for x in args.n_list.split(",") if x.strip())
        variants = {n: d for n, d in variants.items() if n in keep}
        if not variants:
            print("No matching n values after filtering with --n-list.")
            return

    all_rows: List[dict] = []

    for n, paths in tqdm(sorted(variants.items()), desc="Datasets", unit="set"):
        print(f"\n=== n = {n} ===")
        X = np.load(paths["X"])
        y = np.load(paths["y"])
        try:
            print("X shape:", X.shape, " y mean:", f"{float(np.mean(y)):.4f}")
        except Exception:
            classes, counts = np.unique(y, return_counts=True)
            print("X shape:", X.shape, " classes:", dict(zip(classes.tolist(), counts.tolist())))

        if args.search == "grid":
            print("  > Grid search near best ...")
            best_score, best_params, best_est = run_grid(X, y, cv=args.folds, n_jobs=args.gs_jobs, verbose=args.gs_verbose)
        elif args.search == "random":
            print(f"  > RandomizedSearchCV (n_iter={args.n_trials}) near best ...")
            best_score, best_params, best_est = run_random(X, y, cv=args.folds, n_jobs=args.gs_jobs, verbose=args.gs_verbose, n_iter=args.n_trials)
        else: 
            print(f"  > Optuna Bayesian optimization (n_trials={args.n_trials}) near best ...")
            best_score, best_params, best_est = run_bayes(X, y, cv_splits=args.folds, n_trials=args.n_trials, n_jobs=args.gs_jobs)

        print(f"    best_acc={best_score:.4f}  best_params={best_params}")
        all_rows.append({"n": n, "model": "MLP", "acc_mean": best_score, "best_params": best_params})

        # Per-n CSV (saved inside method subfolder)
        df_n = pd.DataFrame([r for r in all_rows if r["n"] == n]).sort_values("acc_mean", ascending=False)
        out_csv = results_dir / f"results_n{n}.csv"
        df_n.to_csv(out_csv, index=False)
        print(f"  Saved {out_csv}")

    # Combined CSV for the run
    df_all = pd.DataFrame(all_rows).sort_values(["n", "acc_mean"], ascending=[True, False])
    df_all.to_csv(results_dir / "results_all.csv", index=False)
    print("\n=== Summary ===")
    print(df_all.to_string(index=False))


if __name__ == "__main__":
    main()
