#!/usr/bin/env python3
"""
train_mlp_ensemble.py — Train and evaluate an ensemble of MLPs on Kryptonite-n datasets.

- Trains per-n ensembles using best hyperparameters you discovered.
- Saves the trained ensemble members.
- Validates robustness on ./Datasets/Test_Data (X-test, y-test).
"""

from pathlib import Path
from turtle import speed
from typing import Dict, List, Tuple
import argparse
import json
import time


import numpy as np
import pandas as pd
from joblib import dump, load, Parallel, delayed, cpu_count

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, log_loss, brier_score_loss,
    confusion_matrix
)
from sklearn.model_selection import cross_val_score

SEED = 42

# ---------- BEST PARAMS ----------
BEST_PARAMS: Dict[int, dict] = {
    10: {'clf__activation': 'relu', 'clf__alpha': 1.9762189340280066e-05,
         'clf__hidden_layer_sizes': (128,), 'clf__learning_rate': 'constant',
         'clf__learning_rate_init': 0.002045610287221893},
    12: {'clf__activation': 'relu', 'clf__alpha': 5.232216089948754e-06,
         'clf__hidden_layer_sizes': (256,), 'clf__learning_rate': 'constant',
         'clf__learning_rate_init': 0.0015432019542643118},
    14: {'clf__activation': 'relu', 'clf__alpha': 7.411299781083242e-05,
         'clf__hidden_layer_sizes': (256,), 'clf__learning_rate': 'constant',
         'clf__learning_rate_init': 0.0005787997763336387},
    16: {'clf__activation': 'tanh', 'clf__alpha': 2.0797422068521745e-05,
         'clf__hidden_layer_sizes': (256, 128), 'clf__learning_rate': 'constant',
         'clf__learning_rate_init': 0.0039330874246197986},
    18: {'clf__activation': 'tanh', 'clf__alpha': 5.503363365456096e-06,
         'clf__hidden_layer_sizes': (256, 128), 'clf__learning_rate': 'adaptive',
         'clf__learning_rate_init': 0.002577748805373576},
    20: {'clf__activation': 'relu', 'clf__alpha': 0.0006010389754773889,
         'clf__hidden_layer_sizes': (256,), 'clf__learning_rate': 'constant',
         'clf__learning_rate_init': 0.00411083153928461},
}

# ---------- Calibration helpers (ECE / MCE) ----------

def _bin_edges(strategy: str, y_prob: np.ndarray, n_bins: int) -> np.ndarray:
    """
    Compute bin edges for calibration:
    - 'uniform': fixed-width bins over [0,1]
    - 'quantile': equal-mass (adaptive) bins based on y_prob distribution
    """
    if strategy == "uniform":
        return np.linspace(0.0, 1.0, n_bins + 1)
    elif strategy == "quantile":
        qs = np.linspace(0, 1, n_bins + 1)
        edges = np.unique(np.quantile(y_prob, qs))
        # ensure full coverage
        edges[0], edges[-1] = 0.0, 1.0
        return edges
    else:
        raise ValueError("strategy must be 'uniform' or 'quantile'")


def calibration_table(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15,
    strategy: str = "uniform",
) -> Tuple[pd.DataFrame, float, float]:
    """
    Compute per-bin calibration stats and ECE/MCE.

    Returns (df_bins, ECE, MCE), where df_bins has columns:
      ['bin', 'left', 'right', 'count', 'accuracy', 'confidence', 'gap'].

    ECE = sum_b (n_b / N) * |acc_b - conf_b|
    MCE = max_b |acc_b - conf_b|
    """
    assert y_prob.ndim == 1, "y_prob must be 1D probabilities for the positive class"
    assert len(y_true) == len(y_prob)

    edges = _bin_edges(strategy, y_prob, n_bins)
    # digitize into bins; 0..n_bins-1
    bin_idx = np.digitize(y_prob, edges[1:-1], right=True)

    rows = []
    n = len(y_true)
    ece = 0.0
    mce = 0.0

    for b in range(len(edges) - 1):
        mask = (bin_idx == b)
        count = int(mask.sum())
        if count == 0:
            acc = np.nan
            conf = np.nan
            gap = 0.0
        else:
            acc = float(np.mean(y_true[mask]))
            conf = float(np.mean(y_prob[mask]))
            gap = abs(acc - conf)
            w = count / n
            ece += w * gap
            mce = max(mce, gap)

        rows.append({
            "bin": b,
            "left": float(edges[b]),
            "right": float(edges[b+1]),
            "count": count,
            "accuracy": acc,
            "confidence": conf,
            "gap": gap,
        })

    df_bins = pd.DataFrame(rows)
    return df_bins, float(ece), float(mce)

# ---------- UTIL ----------
def load_split(root: Path, n: int, split: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    split ∈ {"Train_Data", "Test_Data"}
    """
    base = root / split
    Xp = base / f"kryptonite-{n}-X-{'train' if split=='Train_Data' else 'test'}.npy"
    yp = base / f"kryptonite-{n}-y-{'train' if split=='Train_Data' else 'test'}.npy"
    if not Xp.exists() or not yp.exists():
        raise FileNotFoundError(f"Missing files for n={n} in {split}: {Xp.name}, {yp.name}")
    X = np.load(Xp)
    y = np.load(yp)
    return X, y


def make_member(seed: int, best_params: dict) -> Pipeline:
    """One MLP pipeline with scaling, seeded for diversity"""
    # Strip the "clf__" prefix because we set params at the Pipeline level later.
    params_clean = {k.replace("clf__", ""): v for k, v in best_params.items()}
    clf = MLPClassifier(
        **params_clean,
        max_iter=1000,
        early_stopping=True,
        n_iter_no_change=20,
        tol=1e-4,
        random_state=seed
    )
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", clf)
    ])
    return pipe

# """
# Original, untimed ensemble fitting emthod
# """
# def fit_ensemble(X: np.ndarray, y: np.ndarray, best_params: dict, k: int) -> List[Pipeline]:
#     """Train k members with different seeds"""
#     members: List[Pipeline] = []
#     for i in range(k):
#         member = make_member(SEED + i, best_params)
#         member.fit(X, y)
#         members.append(member)
#     return members

def _fit_one_member(i: int, X: np.ndarray, y: np.ndarray, best_params: dict) -> Tuple[Pipeline, float, int]:
    """Train a single member and return (model, train_seconds, index)"""
    member = make_member(SEED + i, best_params)
    t0 = time.perf_counter()
    member.fit(X, y)
    dt = time.perf_counter() - t0
    return member, dt, i

"""
Timed ensemble fitting emthod
"""
def fit_ensemble(
    X: np.ndarray,
    y: np.ndarray,
    best_params: dict,
    k: int,
    jobs: int
) -> Tuple[List[Pipeline], List[float], float]:
    """
    Train k members (optionally in parallel).
    Returns (members, per_member_times, wall_clock_seconds).
    """
    wall_start = time.perf_counter()
    if jobs == 1:
        members, times = [], []
        for i in range(k):
            m, dt, _ = _fit_one_member(i, X, y, best_params)
            members.append(m)
            times.append(dt)
    else:
        # joblib uses 'loky' backend by default (separate processes)
        results = Parallel(n_jobs=jobs, verbose=0)(
            delayed(_fit_one_member)(i, X, y, best_params) for i in range(k)
        )
        # restore original order by index
        results.sort(key=lambda t: t[2])
        members = [t[0] for t in results]
        times   = [t[1] for t in results]
    wall_elapsed = time.perf_counter() - wall_start
    return members, times, wall_elapsed


def ensemble_predict_proba(members: List[Pipeline], X: np.ndarray) -> np.ndarray:
    """Average member probabilities; returns shape (n_samples, 2)"""
    probs = [m.predict_proba(X) for m in members]
    return np.mean(probs, axis=0)


def ensemble_disagreement(members: List[Pipeline], X: np.ndarray) -> float:
    """
    Fraction of samples where not all members agree on the predicted class
    Values closer to 0 = more consensus; higher = less robust
    """
    preds = np.column_stack([m.predict(X) for m in members])  # (n_samples, k)
    disagreed = np.any(preds != preds[:, [0]], axis=1).mean()
    return float(disagreed)


def prob_variance(members: List[Pipeline], X: np.ndarray) -> float:
    """
    Mean variance of the positive-class probabilities across ensemble members
    Higher variance => higher predictive uncertainty
    """
    pos = np.column_stack([m.predict_proba(X)[:, 1] for m in members])  # (n_samples, k)
    return float(np.var(pos, axis=1).mean())


def evaluate(members: List[Pipeline], X: np.ndarray, y: np.ndarray) -> dict:
    """Compute accuracy, F1, ROC-AUC, log loss, Brier, disagreement, prob variance"""
    proba = ensemble_predict_proba(members, X)
    y_pred = (proba[:, 1] >= 0.5).astype(int)

    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "f1": f1_score(y, y_pred),
        "log_loss": log_loss(y, proba, labels=[0,1]),
        "brier": brier_score_loss(y, proba[:, 1]),
        "disagreement": ensemble_disagreement(members, X),
        "prob_variance": prob_variance(members, X),
        "confusion_matrix": confusion_matrix(y, y_pred).tolist()
    }
    # ROC-AUC is only defined if both classes are present
    try:
        metrics["roc_auc"] = roc_auc_score(y, proba[:, 1])
    except ValueError:
        metrics["roc_auc"] = None
    return metrics


def discover_ns(train_root: Path) -> List[int]:
    """Find all n values available in Train_Data"""
    ns = set()
    for p in (train_root / "Train_Data").rglob("kryptonite-*-X-train.npy"):
        name = p.name.lower()
        try:
            n = int(name.split("-")[1])
            ns.add(n)
        except Exception:
            pass
    return sorted(ns)


# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path("./Datasets"))
    ap.add_argument("--save-dir", type=Path, default=Path("./trained_mlp_ensembles"))
    ap.add_argument("--ensemble-size", type=int, default=5)
    ap.add_argument("--n-list", type=str, default="", help="Comma-separated list, e.g. 10,12,14")
    ap.add_argument("--jobs", type=int, default=-1, help="Parallel workers for training members (-1=all cores, 1=serial)")
    ap.add_argument("--skip-save", action="store_true", help="Do not save trained members/manifests")
    args = ap.parse_args()

    args.save_dir.mkdir(parents=True, exist_ok=True)
    used_jobs = cpu_count() if args.jobs == -1 else args.jobs
    print(f"Using jobs={used_jobs} (cpu_count={cpu_count()})")

    # Which n's to process??
    if args.n_list.strip():
        n_values = [int(x.strip()) for x in args.n_list.split(",") if x.strip()]
    else:
        n_values = discover_ns(args.data_root)

    all_rows = []
    for n in n_values:
        print(f"\n=== Training ensemble for n = {n} ===")
        if n not in BEST_PARAMS:
            print(f"  ! No best_params available for n={n}; skipping.")
            continue

        # Load data
        Xtr, ytr = load_split(args.data_root, n, "Train_Data")
        Xte, yte = load_split(args.data_root, n, "Test_Data")

        # Fit ensemble (timed)
        members, member_times, wall_time = fit_ensemble(
            Xtr, ytr, BEST_PARAMS[n], args.ensemble_size, jobs=used_jobs
        )
        sum_member_time = float(np.sum(member_times))
        mean_member_time = float(np.mean(member_times))
        std_member_time = float(np.std(member_times))
        speedup = sum_member_time / wall_time if wall_time > 0 else np.nan

        print(f"  Timing: wall={wall_time:.2f}s | "
              f"sum(member)={sum_member_time:.2f}s | mean={mean_member_time:.2f}s ± {std_member_time:.2f}s | "
              f"eff. speedup≈{speedup:.2f}×")

        # Save members and manifest
        if not args.skip_save:
            n_dir = args.save_dir / f"n{n}"
            n_dir.mkdir(parents=True, exist_ok=True)
            member_paths = []
            for i, m in enumerate(members):
                path = n_dir / f"mlp_member_{i}.joblib"
                dump(m, path)
                member_paths.append(str(path))
            # Save manifest with params & metadata
            manifest = {
                "n": n,
                "ensemble_size": args.ensemble_size,
                "member_paths": member_paths,
                "best_params": BEST_PARAMS[n],
                "seed_base": SEED,
                "timing": {
                    "wall_sec": wall_time,
                    "member_times_sec": member_times,
                    "sum_member_sec": sum_member_time,
                    "mean_member_sec": mean_member_time,
                    "std_member_sec": std_member_time,
                    "effective_speedup": speedup,
                    "jobs": used_jobs
                }
            }
            with open(n_dir / "manifest.json", "w") as f:
                json.dump(manifest, f, indent=2)

        # Evaluate on TRAIN (sanity) and TEST (robustness)
        train_metrics = evaluate(members, Xtr, ytr)
        test_metrics = evaluate(members, Xte, yte)

        # --- Calibration (ECE/MCE on TEST) ---
        # Use ensemble average probability for positive class
        proba_test = ensemble_predict_proba(members, Xte)[:, 1]

        df_bins_u, ece_uniform, mce_uniform = calibration_table(
            yte, proba_test, n_bins=15, strategy="uniform"
        )
        df_bins_q, ece_adapt, mce_adapt = calibration_table(
            yte, proba_test, n_bins=15, strategy="quantile"
        )

        print(f"  Test ECE (uniform)  = {ece_uniform:.4f}, MCE = {mce_uniform:.4f}")
        print(f"  Test ECE (adaptive) = {ece_adapt:.4f}, MCE = {mce_adapt:.4f}")
        # --------------------------------------

        # Log
        print(f"  Train: acc={train_metrics['accuracy']:.4f}, "
              f"f1={train_metrics['f1']:.4f}, "
              f"auc={train_metrics['roc_auc'] if train_metrics['roc_auc'] is not None else 'NA'}, "
              f"loss={train_metrics['log_loss']:.4f}, "
              f"brier={train_metrics['brier']:.4f}, "
              f"disagree={train_metrics['disagreement']:.4f}, "
              f"pvar={train_metrics['prob_variance']:.6f}")

        print(f"  Test : acc={test_metrics['accuracy']:.4f}, "
              f"f1={test_metrics['f1']:.4f}, "
              f"auc={test_metrics['roc_auc'] if test_metrics['roc_auc'] is not None else 'NA'}, "
              f"loss={test_metrics['log_loss']:.4f}, "
              f"brier={test_metrics['brier']:.4f}, "
              f"disagree={test_metrics['disagreement']:.4f}, "
              f"pvar={test_metrics['prob_variance']:.6f}")

        all_rows.append({
            "n": n,
            "ensemble_size": args.ensemble_size,
            "jobs": used_jobs,
            #timing
            "wall_train_sec": wall_time,
            "member_time_sum_sec": sum_member_time,
            "member_time_mean_sec": mean_member_time,
            "member_time_std_sec": std_member_time,
            "effective_speedup": speedup,
            # metrics
            "train_acc": train_metrics["accuracy"],
            "test_acc": test_metrics["accuracy"],
            "test_f1": test_metrics["f1"],
            "test_auc": test_metrics["roc_auc"],
            "test_log_loss": test_metrics["log_loss"],
            "test_brier": test_metrics["brier"],
            "test_disagreement": test_metrics["disagreement"],
            "test_prob_variance": test_metrics["prob_variance"],
            # calibration metrics (test)
            "test_ece_uniform": ece_uniform,
            "test_mce_uniform": mce_uniform,
            "test_ece_adaptive": ece_adapt,
            "test_mce_adaptive": mce_adapt,
        })

        # Optional per-n CSV
        n_out_dir = args.save_dir / f"n{n}" if not args.skip_save else args.save_dir
        per_n_df = pd.DataFrame([all_rows[-1]])
        per_n_df.to_csv(n_out_dir / f"timing_and_metrics_n{n}.csv", index=False)

    # Combined CSV
    if all_rows:
        df = pd.DataFrame(all_rows).sort_values("n")
        df.to_csv(args.save_dir / "summary_timing_and_metrics.csv", index=False)
        print("\n=== Summary (test) ===")
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
