import numpy as np
import pandas as pd
import ast
import re
import os
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

N_EPOCHS = 100
PATIENCE = 50
MIN_DELTA = 1e-3

N_VALUES = [10, 12, 14, 16, 18, 20]

CSV_HPARAMS = "./MLP_ECE/MLP_optimization/random/results_all.csv"
TRAIN_DATA_PATH = "./Datasets/Train_Data"
TEST_DATA_PATH = "./Datasets/Test_Data"
HIDDEN_DATA_PATH = "./Datasets"               
VAL_SUMMARY_CSV = "pytorch_val_summary.csv"
OUTPUT_DIR = "PyTorch_Test_Results"
SUBMISSION_DIR = "Kryptonite_Label_Submission"  


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
    """
    Returns avg_loss and accuracy using BCE logits
    """
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


def parse_params(row):
    """
    Convert params from CSV (sklearn random search output) to pytorch format
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


def train_full_and_test(X_train, y_train, X_test, y_test, params, device,
                        epochs=N_EPOCHS, patience=PATIENCE, min_delta=MIN_DELTA):

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test_t  = torch.tensor(X_test, dtype=torch.float32)
    y_test_t  = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    train_ds = TensorDataset(X_train_t, y_train_t)
    test_ds  = TensorDataset(X_test_t, y_test_t)

    n_train = X_train_t.shape[0]
    val_size = max(int(0.1 * n_train), 1)
    train_size = n_train - val_size
    train_subset, val_subset = torch.utils.data.random_split(
        train_ds, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    train_loader_es = DataLoader(train_subset, batch_size=params["batch_size"], shuffle=True)
    val_loader_es   = DataLoader(val_subset,   batch_size=params["batch_size"], shuffle=False)
    test_loader     = DataLoader(test_ds,      batch_size=params["batch_size"], shuffle=False)

    model = MLP(
        input_dim=X_train.shape[1],
        hidden_sizes=params["hidden_sizes"],
        activation=params["activation"],
        dropout=params["dropout"]
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=params["lr"], weight_decay=params["alpha"])

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(epochs):
        _ = train_one_epoch(model, train_loader_es, criterion, optimizer, device)
        val_loss, _ = evaluate(model, val_loader_es, criterion, device)

        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    # return model + scaler to use for hidden data
    return test_loss, test_acc, model, scaler


def predict_hidden_labels(model, scaler, hidden_X):
    hidden_scaled = scaler.transform(hidden_X)
    Xh_t = torch.tensor(hidden_scaled, dtype=torch.float32)

    with torch.no_grad():
        logits = model(Xh_t)
        probs = torch.sigmoid(logits).cpu().numpy().ravel()
        preds = (probs > 0.5).astype(int)

    return preds, probs



def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df_val = pd.read_csv(VAL_SUMMARY_CSV).set_index("n")

    results = []

    for n in N_VALUES:
        df = pd.read_csv(CSV_HPARAMS)
        row = df[df.n == n].iloc[0]
        params = parse_params(row)

        print(f"\nFinal train + test for n={n}")
        print("Parameters:", params)

        X_train = np.load(f"{TRAIN_DATA_PATH}/kryptonite-{n}-x-train.npy")
        y_train = np.load(f"{TRAIN_DATA_PATH}/kryptonite-{n}-y-train.npy")
        X_test  = np.load(f"{TEST_DATA_PATH}/kryptonite-{n}-x-test.npy")
        y_test  = np.load(f"{TEST_DATA_PATH}/kryptonite-{n}-y-test.npy")

        # train model 
        test_loss, test_acc, model, scaler = train_full_and_test(
            X_train, y_train, X_test, y_test, params, device
        )

        print(f"Test accuracy: {test_acc:.4f}")

        # generate hidden labels
        hidden_path = os.path.join(HIDDEN_DATA_PATH, f"hidden-kryptonite-{n}-X.npy")
        if os.path.exists(hidden_path):
            hidden_X = np.load(hidden_path)
            preds, probs = predict_hidden_labels(model, scaler, hidden_X)

            save_path = os.path.join(SUBMISSION_DIR, f"hidden-kryptonite-{n}-Y.npy")
            np.save(save_path, preds)
            print(f"Saved hidden labels for n={n} → {save_path}")
        else:
            print(f"(No hidden dataset for n={n}, skipping label generation.)")

        results.append({
            "n": n,
            "test_acc": float(test_acc),
        })

    # save summary
    pd.DataFrame(results).to_csv(os.path.join(OUTPUT_DIR, "pytorch_test_results.csv"), index=False)
    print("\nSaved summary to pytorch_test_results.csv")


if __name__ == "__main__":
    main()
