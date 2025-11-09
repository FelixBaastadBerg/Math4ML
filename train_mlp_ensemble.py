#!/usr/bin/env python3
"""
train_mlp_ensemble.py — Train and evaluate an ensemble of MLPs on Kryptonite-n datasets.

- Trains per-n ensembles using best hyperparameters you discovered.
- Saves the trained ensemble members.
- Validates robustness on ./Datasets/Test_Data (X-test, y-test).
"""

from pathlib import Path
from typing import Dict, List, Tuple
import argparse
import json

import numpy as np
import pandas as pd
from joblib import dump, load

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


def fit_ensemble(X: np.ndarray, y: np.ndarray, best_params: dict, k: int) -> List[Pipeline]:
    """Train k members with different seeds"""
    members: List[Pipeline] = []
    for i in range(k):
        member = make_member(SEED + i, best_params)
        member.fit(X, y)
        members.append(member)
    return members


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
    """Find all n values available in Train_Data."""
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
    args = ap.parse_args()

    args.save_dir.mkdir(parents=True, exist_ok=True)

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

        # Fit ensemble
        members = fit_ensemble(Xtr, ytr, BEST_PARAMS[n], args.ensemble_size)

        # Save members and manifest
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
            "seed_base": SEED
        }
        with open(n_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        # Evaluate on TRAIN (sanity) and TEST (robustness)
        train_metrics = evaluate(members, Xtr, ytr)
        test_metrics = evaluate(members, Xte, yte)

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
            "train_acc": train_metrics["accuracy"],
            "test_acc": test_metrics["accuracy"],
            "test_f1": test_metrics["f1"],
            "test_auc": test_metrics["roc_auc"],
            "test_log_loss": test_metrics["log_loss"],
            "test_brier": test_metrics["brier"],
            "test_disagreement": test_metrics["disagreement"],
            "test_prob_variance": test_metrics["prob_variance"]
        })

        # Save per-n report
        metrics_list = []
        for k in ["accuracy", "f1", "roc_auc", "log_loss", "brier", "disagreement", "prob_variance"]:
            metrics_list.append({
                "metric": k,
                "train": train_metrics.get(k),
                "test": test_metrics.get(k)
            })
        pd.DataFrame(metrics_list).to_csv(n_dir / "metrics.csv", index=False)

    # Combined CSV
    if all_rows:
        df = pd.DataFrame(all_rows).sort_values("n")
        df.to_csv(args.save_dir / "summary_metrics.csv", index=False)
        print("\n=== Summary (test) ===")
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
