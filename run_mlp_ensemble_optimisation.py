"""
Optimise an ensemble of MLPs on Kryptonite-n datasets

search grid     : GridSearchCV 
search random   : RandomizedSearchCV
search bayes    : Optuna Bayesian optimization
"""

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import argparse
import warnings

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.exceptions import ConvergenceWarning

from tqdm.auto import tqdm
from scipy.stats import loguniform
import optuna

warnings.filterwarnings("ignore", category=ConvergenceWarning)

SEED = 42


def discover_variants(data_dir: Path) -> Dict[int, Dict[str, Path]]:
    """
    Maps n to {'X': path, 'y': path}
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
            max_iter=500,
            random_state=SEED,
            early_stopping=True
        ))
    ])

# Search Spaces
def grid_params() -> dict:
    """
    Exhaustive
    """
    return {
        "clf__hidden_layer_sizes": [(128,), (256,), (256,128), (512,256)],
        "clf__activation": ["relu", "tanh"],
        "clf__alpha": [1e-5, 1e-4, 1e-3],
        "clf__learning_rate_init": [1e-3, 5e-4, 2e-3],
        "clf__learning_rate": ["constant", "adaptive"],
    }


def random_distributions() -> dict:
    """
    Parameter distributions for RandomizedSearchCV
    """
    return {
        "clf__hidden_layer_sizes": [(128,), (256,), (256,128), (512,256)],
        "clf__activation": ["relu", "tanh"],
        "clf__alpha": loguniform(1e-6, 1e-3),
        "clf__learning_rate_init": loguniform(5e-4, 5e-3),
        "clf__learning_rate": ["constant", "adaptive"],
    }

def run_grid(X, y, cv, n_jobs, verbose):
    """
    Ties all combinations
    """
    est = base_mlp_pipeline()
    params = grid_params()
    gs = GridSearchCV(est, params, cv=cv, scoring="accuracy", n_jobs=n_jobs, verbose=verbose)
    gs.fit(X, y)
    return gs.best_score_, gs.best_params_, gs.best_estimator_


def run_random(X, y, cv, n_jobs, verbose, n_iter):
    """
    Randomized search
    """
    est = base_mlp_pipeline()
    params = random_distributions()
    rs = RandomizedSearchCV(
        estimator=est,
        param_distributions=params,
        n_iter=n_iter,
        scoring="accuracy",
        cv=cv,
        random_state=SEED,
        n_jobs=n_jobs,
        verbose=verbose
    )
    rs.fit(X, y)
    return rs.best_score_, rs.best_params_, rs.best_estimator_


def run_bayes(X, y, cv_splits, n_trials, n_jobs):
    """
    Bayesian optimization with Optuna
    """
    est = base_mlp_pipeline()

    def objective(trial):
        params = {
            "clf__hidden_layer_sizes": trial.suggest_categorical("clf__hidden_layer_sizes", [(128,), (256,), (256,128), (512,256)]),
            "clf__activation": trial.suggest_categorical("clf__activation", ["relu", "tanh"]),
            "clf__alpha": trial.suggest_float("clf__alpha", 1e-6, 1e-3, log=True),
            "clf__learning_rate_init": trial.suggest_float("clf__learning_rate_init", 5e-4, 5e-3, log=True),
            "clf__learning_rate": trial.suggest_categorical("clf__learning_rate", ["constant", "adaptive"]),
        }
        model = est.set_params(**params)
        scores = cross_val_score(model, X, y, cv=cv_splits, scoring="accuracy", n_jobs=n_jobs)
        return float(np.mean(scores))

    # TPE sampler (Tree-structured Parzen Estimator)
    # Optuna's default 
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best_params = study.best_params
    best_est = est.set_params(**best_params).fit(X, y)
    return study.best_value, best_params, best_est


# ---------------- Ensemble training after hyperparameter search ----------------

def build_ensemble(best_params: dict, n_estimators: int = 5) -> VotingClassifier:
    """
    Build an ensemble of MLPs with the same hyperparameters but different seeds.
    Each member gets a unique seed so they learn slightly different patterns.
    Uses soft voting (average probabilities) which usually works better than hard voting.
    """
    estimators = []
    for i in range(n_estimators):

        params_clean = {k.replace("clf__", ""): v for k, v in best_params.items()}
        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(
                **params_clean,
                max_iter=500,
                random_state=SEED + i,  # different seed for diversity
                early_stopping=True
            ))
        ])
        estimators.append((f"mlp_{i}", clf))
    ensemble = VotingClassifier(estimators=estimators, voting="soft", n_jobs=-1)
    return ensemble


# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("./Datasets/Train_Data"))
    ap.add_argument("--results-root", type=Path, default=Path("./MLP_ensemble_optimization"))
    ap.add_argument("--n-list", type=str, default="", help="Comma-separated list of n values to process")
    ap.add_argument("--folds", type=int, default=5, help="CV folds for hyperparameter tuning")
    ap.add_argument("--gs-verbose", type=int, default=0, help="Verbosity for GridSearchCV/RandomizedSearchCV")
    ap.add_argument("--gs-jobs", type=int, default=-1, help="Parallel workers (-1 = all cores)")
    ap.add_argument("--search", type=str, default="random", choices=["grid", "random", "bayes"], help="Hyperparameter search method")
    ap.add_argument("--n-trials", type=int, default=40, help="Number of iterations for random/bayes search")
    ap.add_argument("--ensemble-size", type=int, default=5, help="Number of models in the voting ensemble")
    args = ap.parse_args()

    print("Searching in:", args.data_dir.resolve(), " search:", args.search, " n-trials:", args.n_trials, flush=True)

    # separate subfolder per method to avoid mixing results
    results_dir = Path(args.results_root) / args.search
    results_dir.mkdir(parents=True, exist_ok=True)

    variants = discover_variants(args.data_dir)
    if not variants:
        print(f"No train datasets found under {args.data_dir}.")
        return

    # optional filtering by n values
    if args.n_list.strip():
        keep = set(int(x.strip()) for x in args.n_list.split(",") if x.strip())
        variants = {n: d for n, d in variants.items() if n in keep}
        if not variants:
            print("No matching n values after filtering with --n-list.")
            return

    all_rows: List[dict] = []

    for n, paths in tqdm(sorted(variants.items()), desc="Datasets", unit="set"):
        print(f"\nn = {n}")
        X = np.load(paths["X"])
        y = np.load(paths["y"])
        print("X shape:", X.shape, " y mean:", f"{float(np.mean(y)):.4f}")

        if args.search == "grid":
            print(" > Grid search for best single MLP ...")
            best_score, best_params, _ = run_grid(X, y, cv=args.folds, n_jobs=args.gs_jobs, verbose=args.gs_verbose)
        elif args.search == "random":
            print(f" > Random search (n_iter={args.n_trials}) for best single MLP ...")
            best_score, best_params, _ = run_random(X, y, cv=args.folds, n_jobs=args.gs_jobs, verbose=args.gs_verbose, n_iter=args.n_trials)
        else:
            print(f" > Bayesian search (n_trials={args.n_trials}) for best single MLP ...")
            best_score, best_params, _ = run_bayes(X, y, cv_splits=args.folds, n_trials=args.n_trials, n_jobs=args.gs_jobs)

        print(f" > Single MLP best acc={best_score:.4f}")
        print(f" > Best params={best_params}")

        print(f"  > Building ensemble with {args.ensemble_size} members ...")
        ensemble = build_ensemble(best_params, n_estimators=args.ensemble_size)
        ensemble.fit(X, y)

        scores = cross_val_score(ensemble, X, y, cv=args.folds, scoring="accuracy", n_jobs=args.gs_jobs)
        print(f"  > Ensemble CV: mean acc={scores.mean():.4f} ± {scores.std():.4f}")

        all_rows.append({
            "n": n,
            "model": f"MLP_Ensemble_{args.ensemble_size}",
            "single_mlp_acc": best_score,
            "ensemble_acc_mean": scores.mean(),
            "ensemble_acc_std": scores.std(),
            "best_params": best_params
        })

        df_n = pd.DataFrame([r for r in all_rows if r["n"] == n])
        out_csv = results_dir / f"results_n{n}.csv"
        df_n.to_csv(out_csv, index=False)
        print(f"  Saved {out_csv}")

    df_all = pd.DataFrame(all_rows).sort_values(["n", "ensemble_acc_mean"], ascending=[True, False])
    df_all.to_csv(results_dir / "results_all.csv", index=False)
    print("\nSummary")
    print(df_all.to_string(index=False))


if __name__ == "__main__":
    main()
