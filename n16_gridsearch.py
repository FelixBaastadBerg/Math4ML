import os

# Try to suppress Intel MKL / OpenMP warnings before importing numpy/torch
os.environ.setdefault("KMP_WARNINGS", "0")
os.environ.setdefault("MKL_SERVICE_FORCE_INTEL", "1")

import itertools
import re
import csv
import json                        # NEW: for saving best hyperparameters
import argparse
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split

from sklearn.preprocessing import StandardScaler

# ------------------- config -------------------

TRAIN_DIR = Path("Datasets/Train_Data")
TEST_DIR  = Path("Datasets/Test_Data")

# where to dump all debugging/diagnostic artefacts
OUT_ROOT = Path("Temporary")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ------------------- hyperparameter grid -------------------
# Expanded search:
#  - more/wider/deeper hidden layer settings
#  - activation ∈ {relu, tanh}
#  - optimizer ∈ {adam (adaptive), sgd (constant LR + momentum)}

PARAM_GRID = {
    "hidden_layers": [
        (256,),
        (384,),
        (512,),                 # NEW
        (256, 128),
        (256, 192),
        (256, 256),
        (320, 256),
        (384, 256),
        (384, 384),             # NEW
        (256, 256, 128),
        (256, 256, 256),        # NEW
    ],
    "learning_rate": [5e-4, 1e-3, 2e-3],
    "weight_decay":  [5e-6, 5e-5, 5e-4],
    "batch_size":    [64, 128],
    "activation":    ["relu", "tanh"],     # NEW
    "optimizer":     ["adam", "sgd"],      # NEW (adam = adaptive, sgd = constant)
}

MAX_EPOCHS_GRID = 50   # epochs during grid search
MAX_EPOCHS_FINAL = 80  # epochs when retraining with best hyperparameters
VAL_FRACTION = 0.2

# ------------------- data utilities -------------------

_PAT = re.compile(r"^kryptonite-(\d+)-([xyXY])-(train|test)\.npy$")


def discover_kryptonite_pairs(train_dir: Path, test_dir: Path) -> Dict[int, Dict[str, Path]]:
    """
    Find matching train/test X,y files for each n.
    Returns { n: {"Xtr","ytr","Xte","yte"} }
    """
    def index_dir(d: Path, split: str) -> Dict[int, Dict[str, Path]]:
        out = {}
        for p in d.glob("*.npy"):
            m = _PAT.match(p.name)
            if not m:
                continue
            n = int(m.group(1))
            xy = m.group(2).lower()  # 'x' or 'y'
            s  = m.group(3).lower()  # 'train' or 'test'
            if s != split:
                continue
            d_n = out.setdefault(n, {"X": None, "y": None})
            d_n["X" if xy == "x" else "y"] = p
        return out

    tr = index_dir(train_dir, "train")
    te = index_dir(test_dir, "test")

    pairs = {}
    for n in sorted(set(tr.keys()) & set(te.keys())):
        if tr[n]["X"] and tr[n]["y"] and te[n]["X"] and te[n]["y"]:
            pairs[n] = {
                "Xtr": tr[n]["X"],
                "ytr": tr[n]["y"],
                "Xte": te[n]["X"],
                "yte": te[n]["y"],
            }
    return pairs

# ------------------- model -------------------

class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_layers: Tuple[int, ...], activation: str = "relu"):
        super().__init__()

        act_name = activation.lower()
        if act_name == "relu":
            Act = nn.ReLU
        elif act_name == "tanh":
            Act = nn.Tanh
        else:
            raise ValueError(f"Unknown activation '{activation}'")

        layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(prev, h))
            layers.append(Act())
            prev = h
        layers.append(nn.Linear(prev, 1))  # binary logits
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)  # (batch,)

# ------------------- training / evaluation -------------------

def make_loaders(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    val_fraction: float,
) -> Tuple[DataLoader, DataLoader, StandardScaler]:
    """
    Standardise X, make TensorDatasets, and split into train/val loaders.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_tensor = torch.from_numpy(X_scaled.astype(np.float32))
    y_tensor = torch.from_numpy(y.astype(np.float32))

    dataset = TensorDataset(X_tensor, y_tensor)
    val_size = int(len(dataset) * val_fraction)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, scaler


def _make_optimizer(params, optimizer_name: str, lr: float, weight_decay: float):
    """
    Helper to create optimizer given name and hyperparameters.
    """
    name = optimizer_name.lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif name == "sgd":
        # "constant" learning rate SGD (+ momentum) as a contrast to adaptive Adam
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer '{optimizer_name}'")

def train_one_model(
    input_dim: int,
    hidden_layers: Tuple[int, ...],
    lr: float,
    weight_decay: float,
    batch_size: int,
    activation: str,
    optimizer_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    max_epochs: int,
    cfg_name: str = "",
    verbose: bool = False,
) -> Tuple[float, float, float, List[float], List[float], List[float]]:
    """
    Train for max_epochs and return:
      (best_val_loss, best_val_acc, final_train_loss,
       train_losses, val_losses, val_accs)
    using an internal train/val split.

    If verbose=False, no per-epoch printing.
    """
    train_loader, val_loader, _ = make_loaders(X_train, y_train, batch_size, VAL_FRACTION)

    model = MLP(input_dim, hidden_layers, activation=activation).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = _make_optimizer(model.parameters(), optimizer_name, lr, weight_decay)

    best_val_loss = float("inf")
    best_val_acc = 0.0
    last_train_loss = None

    train_losses: List[float] = []
    val_losses: List[float] = []
    val_accs: List[float] = []

    for epoch in range(max_epochs):
        model.train()
        running_loss = 0.0
        n_train = 0
        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * yb.size(0)
            n_train += yb.size(0)

        last_train_loss = running_loss / n_train

        # validation
        model.eval()
        val_loss = 0.0
        correct = 0
        n_val = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(DEVICE)
                yb = yb.to(DEVICE)

                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss += loss.item() * yb.size(0)

                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).float()
                correct += (preds == yb).sum().item()
                n_val += yb.size(0)

        val_loss /= n_val
        val_acc = correct / n_val

        train_losses.append(last_train_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        if verbose:
            prefix = f"    [{cfg_name}] " if cfg_name else "    "
            print(
                f"{prefix}Epoch {epoch+1:3d}/{max_epochs}: "
                f"train_loss={last_train_loss:.4f}, "
                f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
            )

        # track best (primarily by accuracy, then by loss)
        if (val_acc > best_val_acc) or (val_acc == best_val_acc and val_loss < best_val_loss):
            best_val_acc = val_acc
            best_val_loss = val_loss

    return best_val_loss, best_val_acc, last_train_loss, train_losses, val_losses, val_accs


def grid_search_for_n(
    X_train: np.ndarray,
    y_train: np.ndarray,
    out_dir: Path,
) -> Dict[str, object]:
    """
    Run grid search over PARAM_GRID, return dict with best hyperparameters and
    corresponding val metrics and training/validation curves for the best config.

    During grid search we DO NOT print per-epoch metrics.
    """
    input_dim = X_train.shape[1]
    best_cfg = None
    best_val_acc = -1.0
    best_val_loss = float("inf")
    best_train_loss = None
    best_history = None  # (train_ls, val_ls, val_as)

    keys = list(PARAM_GRID.keys())
    for i, values in enumerate(itertools.product(*[PARAM_GRID[k] for k in keys]), start=1):
        cfg = dict(zip(keys, values))
        cfg_name = f"cfg{i}"
        print(f"\n  Trying config {cfg_name}: {cfg}")

        val_loss, val_acc, train_loss, train_ls, val_ls, val_as = train_one_model(
            input_dim=input_dim,
            hidden_layers=cfg["hidden_layers"],
            lr=cfg["learning_rate"],
            weight_decay=cfg["weight_decay"],
            batch_size=cfg["batch_size"],
            activation=cfg["activation"],
            optimizer_name=cfg["optimizer"],
            X_train=X_train,
            y_train=y_train,
            max_epochs=MAX_EPOCHS_GRID,
            cfg_name=cfg_name,
            verbose=False,   # <<< NO per-epoch printing during grid search
        )
        print(f"    => final val_acc={val_acc:.4f}, val_loss={val_loss:.4f}")

        if (val_acc > best_val_acc) or (val_acc == best_val_acc and val_loss < best_val_loss):
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_train_loss = train_loss
            best_cfg = cfg
            best_history = (train_ls, val_ls, val_as)

    assert best_cfg is not None
    assert best_history is not None

    # save per-epoch curves for the best configuration
    train_ls, val_ls, val_as = best_history
    history_path = out_dir / "best_train_val_history.csv"
    with open(history_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_acc"])
        for epoch, (tr, vl, va) in enumerate(zip(train_ls, val_ls, val_as), start=1):
            writer.writerow([epoch, tr, vl, va])

    print(f"  Saved best train/val history to {history_path}")

    # --- NEW: save best hyperparameters to JSON as well ---
    best_params_payload = {
        "best_params": best_cfg,
        "best_val_acc": best_val_acc,
        "best_val_loss": best_val_loss,
        "last_train_loss": best_train_loss,
    }
    best_params_path = out_dir / "best_params.json"
    with open(best_params_path, "w") as f:
        json.dump(best_params_payload, f, indent=2)
    print(f"  Saved best hyperparameters to {best_params_path}")

    return {
        "best_params": best_cfg,
        "best_val_acc": best_val_acc,
        "best_val_loss": best_val_loss,
        "last_train_loss": best_train_loss,
        "best_history": best_history,
    }


def train_final_and_test(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    best_params: Dict[str, object],
    out_dir: Path,
) -> Tuple[float, float]:
    """
    Retrain with best hyperparameters on full training data (train+val),
    then evaluate on test set. Returns (test_loss, test_acc).
    Also saves final training-loss curve in out_dir.

    This stage DOES print per-epoch training loss (but no val metrics).
    """
    # Standardise using full training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    Xtr_t = torch.from_numpy(X_train_scaled.astype(np.float32))
    ytr_t = torch.from_numpy(y_train.astype(np.float32))
    Xte_t = torch.from_numpy(X_test_scaled.astype(np.float32))
    yte_t = torch.from_numpy(y_test.astype(np.float32))

    train_ds = TensorDataset(Xtr_t, ytr_t)
    train_loader = DataLoader(train_ds, batch_size=best_params["batch_size"], shuffle=True)

    model = MLP(
        input_dim=X_train.shape[1],
        hidden_layers=best_params["hidden_layers"],
        activation=best_params.get("activation", "relu"),  # default to relu if missing
    ).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = _make_optimizer(
        model.parameters(),
        optimizer_name=best_params.get("optimizer", "adam"),
        lr=best_params["learning_rate"],
        weight_decay=best_params["weight_decay"],
    )

    epoch_train_losses: List[float] = []

    # train
    for epoch in range(MAX_EPOCHS_FINAL):
        model.train()
        running = 0.0
        n_train = 0
        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            running += loss.item() * yb.size(0)
            n_train += yb.size(0)

        epoch_loss = running / n_train
        epoch_train_losses.append(epoch_loss)
        # This is the only per-epoch logging now
        print(f"    [final] Epoch {epoch+1:3d}/{MAX_EPOCHS_FINAL}: train_loss={epoch_loss:.4f}")

    # save training curve for final fit
    final_curve_path = out_dir / "final_train_loss_curve.csv"
    with open(final_curve_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss"])
        for epoch, tr in enumerate(epoch_train_losses, start=1):
            writer.writerow([epoch, tr])
    print(f"    Saved final training-loss curve to {final_curve_path}")

    # test eval
    model.eval()
    with torch.no_grad():
        logits = model(Xte_t.to(DEVICE))
        loss = criterion(logits, yte_t.to(DEVICE)).item()
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()
        correct = (preds.cpu() == yte_t).sum().item()
        acc = correct / yte_t.size(0)

    return loss, acc

# ------------------- main -------------------

def main(selected_ns: List[int] | None = None):
    pairs = discover_kryptonite_pairs(TRAIN_DIR, TEST_DIR)
    if not pairs:
        raise RuntimeError(
            f"No kryptonite-* train/test pairs found under {TRAIN_DIR} and {TEST_DIR}"
        )

    all_ns = sorted(pairs.keys())
    print("Found n values on disk:", all_ns)

    # If user requested a subset, filter here
    if selected_ns is not None:
        selected_ns = sorted(set(selected_ns))
        print("Requested n values:", selected_ns)
        pairs = {n: paths for n, paths in pairs.items() if n in selected_ns}
        if not pairs:
            raise RuntimeError(
                f"None of the requested n values {selected_ns} exist in the data."
            )

    results = []

    for n, paths in sorted(pairs.items()):
        print(f"\n===== n = {n} =====")
        Xtr = np.load(paths["Xtr"])
        ytr = np.load(paths["ytr"])
        Xte = np.load(paths["Xte"])
        yte = np.load(paths["yte"])

        # ensure 1D labels in {0,1}
        ytr = ytr.reshape(-1).astype(np.float32)
        yte = yte.reshape(-1).astype(np.float32)

        print(f"Train: {Xtr.shape}, Test: {Xte.shape}")

        # per-n output directory under Temporary
        out_dir = OUT_ROOT / f"n{n}"
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1) grid search on train data (with train/val split), no per-epoch printing
        gs = grid_search_for_n(Xtr, ytr, out_dir)
        best_params = gs["best_params"]
        print(f"\nBest params for n={n}: {best_params}")
        print(f"Best val_acc={gs['best_val_acc']:.4f}, best_val_loss={gs['best_val_loss']:.4f}")

        # 2) retrain on full train data and evaluate on test (this prints per-epoch train loss)
        test_loss, test_acc = train_final_and_test(Xtr, ytr, Xte, yte, best_params, out_dir)
        print(f"Test_acc={test_acc:.4f}, Test_loss={test_loss:.4f}")

        results.append({
            "n": n,
            "best_params": best_params,
            "val_accuracy": gs["best_val_acc"],
            "val_loss": gs["best_val_loss"],
            "test_accuracy": test_acc,
            "test_loss": test_loss,
        })

    print("\n===== Summary =====")
    for r in sorted(results, key=lambda d: d["n"]):
        print(
            f"n={r['n']:>2d} | "
            f"val_acc={r['val_accuracy']:.4f}  val_loss={r['val_loss']:.4f} | "
            f"test_acc={r['test_accuracy']:.4f}  test_loss={r['test_loss']:.4f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Grid-search + train/test MLP on Kryptonite-n datasets."
    )
    parser.add_argument(
        "-n", "--n",
        type=int,
        nargs="+",
        help="Which Kryptonite-n datasets to run (e.g. -n 18 20). "
             "If omitted, all available n are used.",
    )
    args = parser.parse_args()

    # args.n is either a list of ints or None
    main(selected_ns=args.n)
