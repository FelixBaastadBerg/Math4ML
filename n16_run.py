import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler

# =========================
# Config
# =========================

N = 16  # kryptonite-16
DATA_ROOT_TRAIN = Path("./Datasets/Train_Data")
DATA_ROOT_TEST  = Path("./Datasets/Test_Data")

OUT_DIR = Path("temp_3")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# Best hyperparameters provided
BEST_PARAMS = {
    "hidden_layers": (384,),
    "learning_rate": 0.001,
    "weight_decay": 0.0005,
    "batch_size": 64,
    "activation": "relu",
    "optimizer": "adam",
}

N_EPOCHS = 100
PATIENCE = 20      # early stopping patience (epochs)
MIN_DELTA = 1e-3   # minimum improvement in val loss to reset patience
VAL_FRACTION = 0.2


# =========================
# Model
# =========================

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_layers, activation="relu", num_classes=2):
        super().__init__()

        if activation.lower() == "relu":
            Act = nn.ReLU
        elif activation.lower() == "tanh":
            Act = nn.Tanh
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        layers = []
        prev = input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(prev, h))
            layers.append(Act())
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# =========================
# Helpers
# =========================

def make_loaders(X, y, batch_size, val_fraction):
    """
    Standardize X, then create train/val DataLoaders.
    Returns: train_loader, val_loader, scaler
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_t = torch.tensor(X_scaled, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)

    dataset = TensorDataset(X_t, y_t)
    n_total = len(dataset)
    n_val = int(n_total * val_fraction)
    n_train = n_total - n_val

    train_ds, val_ds = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(SEED),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, scaler


def evaluate(model, loader, criterion=None, device=DEVICE):
    """
    Evaluate on loader. If criterion is provided, also compute loss.
    Returns: (avg_loss or None, accuracy)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    n_samples = 0

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            n_samples += yb.size(0)

            if criterion is not None:
                loss = criterion(logits, yb)
                total_loss += loss.item() * yb.size(0)

    acc = correct / n_samples
    if criterion is None:
        return None, acc
    else:
        avg_loss = total_loss / n_samples
        return avg_loss, acc


# =========================
# Training routine
# =========================

def train_and_evaluate_n16():
    # ---------- Load data ----------
    X_train = np.load(DATA_ROOT_TRAIN / f"kryptonite-{N}-x-train.npy")
    y_train = np.load(DATA_ROOT_TRAIN / f"kryptonite-{N}-y-train.npy")
    X_test  = np.load(DATA_ROOT_TEST  / f"kryptonite-{N}-x-test.npy")
    y_test  = np.load(DATA_ROOT_TEST  / f"kryptonite-{N}-y-test.npy")

    y_train = y_train.reshape(-1).astype(np.int64)
    y_test  = y_test.reshape(-1).astype(np.int64)

    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # ---------- Build loaders ----------
    batch_size = BEST_PARAMS["batch_size"]
    train_loader, val_loader, scaler = make_loaders(
        X_train, y_train, batch_size=batch_size, val_fraction=VAL_FRACTION
    )

    # Standardize test with the SAME scaler
    X_test_scaled = scaler.transform(X_test)
    X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)
    test_ds = TensorDataset(X_test_t, y_test_t)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # ---------- Model, loss, optimizer ----------
    model = MLP(
        input_dim=X_train.shape[1],
        hidden_layers=BEST_PARAMS["hidden_layers"],
        activation=BEST_PARAMS["activation"],
        num_classes=2,
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    if BEST_PARAMS["optimizer"].lower() == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=BEST_PARAMS["learning_rate"],
            weight_decay=BEST_PARAMS["weight_decay"],
        )
    else:
        optimizer = optim.SGD(
            model.parameters(),
            lr=BEST_PARAMS["learning_rate"],
            momentum=0.9,
            weight_decay=BEST_PARAMS["weight_decay"],
        )

    # ---------- Training loop with early stopping ----------
    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    train_losses = []
    val_losses   = []
    val_accs     = []
    test_accs    = []

    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        n_train_samples = 0

        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * yb.size(0)
            n_train_samples += yb.size(0)

        train_loss = running_loss / n_train_samples

        # validation metrics
        val_loss, val_acc = evaluate(model, val_loader, criterion, DEVICE)
        # test accuracy (for curve)
        _, test_acc = evaluate(model, test_loader, None, DEVICE)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        test_accs.append(test_acc)

        print(
            f"Epoch {epoch:3d}/{N_EPOCHS} | "
            f"train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
            f"test_acc={test_acc:.4f}"
        )

        # early stopping on val_loss
        if val_loss < best_val_loss - MIN_DELTA:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= PATIENCE:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    # load best model weights (by val_loss)
    if best_state is not None:
        model.load_state_dict(best_state)

    # ---------- Final test evaluation ----------
    final_test_loss, final_test_acc = evaluate(model, test_loader, criterion, DEVICE)
    print(f"\nFinal test metrics (best val-loss checkpoint): "
          f"test_loss={final_test_loss:.4f}, test_acc={final_test_acc:.4f}")

    # ---------- Save metrics ----------
    epochs = np.arange(1, len(train_losses) + 1)
    metrics_path = OUT_DIR / "metrics_n16.csv"
    import csv
    with open(metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_acc", "test_acc"])
        for e, tr, vl, va, ta in zip(epochs, train_losses, val_losses, val_accs, test_accs):
            writer.writerow([e, tr, vl, va, ta])
    print(f"Saved metrics to {metrics_path}")

    # Also save final test metrics in a text file
    with open(OUT_DIR / "final_test_metrics_n16.txt", "w") as f:
        f.write(f"Final test loss: {final_test_loss:.6f}\n")
        f.write(f"Final test accuracy: {final_test_acc:.6f}\n")

    # ---------- Plot curves ----------
    # Loss curves
    plt.figure(figsize=(7, 5))
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses,   label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss (n=16)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    loss_plot_path = OUT_DIR / "loss_curves_n16.png"
    plt.tight_layout()
    plt.savefig(loss_plot_path, dpi=200)
    plt.close()
    print(f"Saved loss curves to {loss_plot_path}")

    # Accuracy curves (val + test)
    plt.figure(figsize=(7, 5))
    plt.plot(epochs, val_accs,  label="Validation Accuracy")
    plt.plot(epochs, test_accs, label="Test Accuracy (per epoch)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation & Test Accuracy (n=16)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    acc_plot_path = OUT_DIR / "accuracy_curves_n16.png"
    plt.tight_layout()
    plt.savefig(acc_plot_path, dpi=200)
    plt.close()
    print(f"Saved accuracy curves to {acc_plot_path}")


if __name__ == "__main__":
    train_and_evaluate_n16()
