import numpy as np
import pandas as pd
import ast
import re
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

plt.rcParams.update({'font.size': 20})

N_EPOCHS = 100
PATIENCE = 50  
MIN_DELTA = 1e-3 

N_VALUES = [10,12,14,16,18,20]
CSV_HPARAMS = "./MLP_ECE/MLP_optimization/random/results_all.csv"
TRAIN_DATA_PATH = "./Datasets/Train_Data"
OUTPUT_DIR = "PyTorch_Convergence" 
VAL_SUMMARY_CSV = "pytorch_val_summary.csv" 
# Just to commit again


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_sizes, activation, dropout, num_classes=1):
        super().__init__()

        layers = []
        prev = input_dim

        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            else:
                raise ValueError(f"Unsupported activation: {activation}")

            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h

        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)

        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * xb.size(0)

    return running_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0

    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)

            total_loss += loss.item() * xb.size(0)

            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            correct += (preds == yb).sum().item()

    avg_loss = total_loss / len(loader.dataset)
    accuracy = correct / len(loader.dataset)
    return avg_loss, accuracy


def compute_accuracy(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            correct += (preds == yb).sum().item()
            total += yb.size(0)
    return correct / total



def parse_params(row):
    """
    Get hyperparameters from results_all.csv - added dropout after convergence analysis
    """
    raw = row["best_params"]
    dropout = float(row["dropout"])

    clean = re.sub(r"np\.float64\(([^)]+)\)", r"\1", raw)
    d = ast.literal_eval(clean)

    activation = d["clf__activation"]
    batch_size = int(d["clf__batch_size"])
    alpha = float(d["clf__alpha"])
    lr = float(d["clf__learning_rate_init"])
    hidden_sizes = tuple(int(h) for h in d["clf__hidden_layer_sizes"])

    return {
        "activation": activation,
        "batch_size": batch_size,
        "alpha": alpha,
        "lr": lr,
        "hidden_sizes": hidden_sizes,
        "dropout": dropout,
    }


def run_5fold_cv(X, y, params, device,
                 epochs=N_EPOCHS, patience=PATIENCE, min_delta=MIN_DELTA):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    train_losses_all = []
    val_losses_all = []
    val_acc_all = []
    train_acc_all = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"Running Fold {fold_idx+1}...")

        X_train_fold, y_train_fold = X[train_idx], y[train_idx]
        X_val_fold, y_val_fold     = X[val_idx], y[val_idx]

        scaler = StandardScaler()
        X_train_fold = scaler.fit_transform(X_train_fold)
        X_val_fold   = scaler.transform(X_val_fold)

        X_train_t = torch.tensor(X_train_fold, dtype=torch.float32)
        y_train_t = torch.tensor(y_train_fold, dtype=torch.float32).unsqueeze(1)
        X_val_t   = torch.tensor(X_val_fold,   dtype=torch.float32)
        y_val_t   = torch.tensor(y_val_fold,   dtype=torch.float32).unsqueeze(1)


        train_loader = DataLoader(
            TensorDataset(X_train_t, y_train_t),
            batch_size=params["batch_size"],
            shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(X_val_t, y_val_t),
            batch_size=params["batch_size"],
            shuffle=False
        )

        model = MLP(
            input_dim=X.shape[1],
            hidden_sizes=params["hidden_sizes"],
            activation=params["activation"],
            dropout=params["dropout"]
        ).to(device)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(
            model.parameters(),
            lr=params["lr"],
            weight_decay=params["alpha"]
        )

        best_val_loss = float("inf")
        epochs_no_improve = 0

        train_loss_curve = []
        val_loss_curve   = []
        val_acc_curve    = []
        train_acc_curve  = []

        for epoch in range(epochs):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            train_acc = compute_accuracy(model, train_loader, device)

            train_loss_curve.append(train_loss)
            val_loss_curve.append(val_loss)
            val_acc_curve.append(val_acc)
            train_acc_curve.append(train_acc)

            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print(f"  Early stopping on fold {fold_idx+1} at epoch {epoch+1}")
                break

        last_tl = train_loss_curve[-1]
        last_vl = val_loss_curve[-1]
        last_va = val_acc_curve[-1]
        last_ta = train_acc_curve[-1]
        while len(train_loss_curve) < epochs:
            train_loss_curve.append(last_tl)
            val_loss_curve.append(last_vl)
            val_acc_curve.append(last_va)
            train_acc_curve.append(last_ta)

        train_losses_all.append(train_loss_curve)
        val_losses_all.append(val_loss_curve)
        val_acc_all.append(val_acc_curve)
        train_acc_all.append(train_acc_curve)

    return (
        np.mean(train_losses_all, axis=0),
        np.mean(val_losses_all,  axis=0),
        np.mean(val_acc_all,     axis=0),
        np.mean(train_acc_all,   axis=0),
    )


def run_convergence_for_n(n, device):
    df = pd.read_csv(CSV_HPARAMS)
    row = df[df.n == n].iloc[0]
    params = parse_params(row)

    print(f"\n===== Convergence analysis for n={n} =====")
    print("Parameters:", params)

    X_train = np.load(f"{TRAIN_DATA_PATH}/kryptonite-{n}-x-train.npy")
    y_train = np.load(f"{TRAIN_DATA_PATH}/kryptonite-{n}-y-train.npy")

    train_curve_cv, val_curve_cv, val_acc_curve_cv, train_acc_curve_cv = run_5fold_cv(
        X_train, y_train, params, device, epochs=N_EPOCHS
    )

    return train_curve_cv, val_curve_cv, val_acc_curve_cv, train_acc_curve_cv


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    summary_rows = []

    for n in N_VALUES:
        train_curve, val_curve, val_acc_curve, train_acc_curve = run_convergence_for_n(n, device)

        plt.figure(figsize=(10, 6))
        plt.plot(train_curve, label="Train Loss (CV)", linewidth=2)
        plt.plot(val_curve, label="Validation Loss (CV)", linewidth=2)
        plt.title(f"Loss Curves for n={n}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.legend()
        path_loss = os.path.join(OUTPUT_DIR, f"loss_curves_n_{n}.png")
        plt.savefig(path_loss, dpi=300)
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.plot(train_acc_curve, label="Train Accuracy (CV)", linewidth=2)
        plt.plot(val_acc_curve, label="Validation Accuracy (CV)", linewidth=2)
        plt.title(f"Accuracy Curves for n={n}")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.grid(True)
        plt.legend()
        path_acc = os.path.join(OUTPUT_DIR, f"accuracy_curves_n_{n}.png")
        plt.savefig(path_acc, dpi=300)
        plt.close()

        final_val_acc = float(val_acc_curve[-1])
        best_val_acc  = float(val_acc_curve.max())

        print(f"n={n}  Final Val Acc={final_val_acc:.4f}  Best Val Acc={best_val_acc:.4f}")

        summary_rows.append({
            "n": n,
            "val_acc_final": final_val_acc,
            "val_acc_best": best_val_acc
        })

    # save validation summary for later use
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(VAL_SUMMARY_CSV, index=False)
    print(f"\nSaved validation summary to {VAL_SUMMARY_CSV}")


if __name__ == "__main__":
    main()
