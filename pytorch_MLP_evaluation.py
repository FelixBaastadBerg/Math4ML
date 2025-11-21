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
from sklearn.model_selection import train_test_split

# Configurations

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
    Returns avg_loss, accuracy
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
    Convert params from results_all.csv (sklearn random search output) to pytorch format
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


def compute_ece(y_true, y_prob, n_bins=15, strategy="uniform"):
    """
    Computes expected calibration error (ECE) with either uniform or adaptive bins
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()

    assert y_true.shape == y_prob.shape, "y_true and y_prob must have same shape"

    if strategy == "uniform":
        # Uniform binning
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    else:
        # Adaptive (quantile-based) binning
        quantiles = np.linspace(0.0, 1.0, n_bins + 1)
        bin_edges = np.unique(np.quantile(y_prob, quantiles))
        if bin_edges[0] > 0.0:
            bin_edges[0] = 0.0
        if bin_edges[-1] < 1.0:
            bin_edges[-1] = 1.0

    bin_ids = np.digitize(y_prob, bin_edges[1:-1])

    ece = 0.0
    rows = []
    n = len(y_true)

    for b in range(len(bin_edges) - 1):
        mask = (bin_ids == b)
        count = int(mask.sum())
        if count == 0:
            rows.append([b, bin_edges[b], bin_edges[b + 1], 0, np.nan, np.nan, 0.0])
            continue

        acc = float(y_true[mask].mean())
        conf = float(y_prob[mask].mean())
        gap = abs(acc - conf)
        ece += (count / n) * gap

        rows.append([b, bin_edges[b], bin_edges[b + 1], count, acc, conf, gap])

    df = pd.DataFrame(
        rows, columns=["bin", "left", "right", "count", "acc", "conf", "gap"]
    )
    return float(ece), df


def plot_reliability_diagram(df, ece, n, strategy, out_dir):
    """
    Plot a reliability diagram from the ECE bin dataframe
    """

    df_nonempty = df[df["count"] > 0].copy()
    if df_nonempty.empty:
        print(f"Warning: no non-empty bins for n={n}, strategy={strategy}; skipping plot.")
        return

    acc = df_nonempty["acc"].values
    conf = df_nonempty["conf"].values

    plt.figure(figsize=(6, 5))

    plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfect Calibration")

    plt.plot(conf, acc, marker="o", linestyle="-", label="Empirical Accuracy")

    plt.plot(conf, conf, marker="s", linestyle="--", label="Mean Confidence")

    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.xlabel("Predicted Probability")
    plt.ylabel("Accuracy")
    plt.title(f"Kryptonite-{n} {strategy.capitalize()} ECE\nECE={ece:.4f}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    fname = f"kryptonite_{n}_ece_{strategy}.png"
    path = os.path.join(out_dir, fname)
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved {strategy} reliability diagram for n={n} to {path}")


def train_full_and_test(
    X_train,
    y_train,
    X_test,
    y_test,
    params,
    device,
    epochs=N_EPOCHS,
    patience=PATIENCE,
    min_delta=MIN_DELTA,
):
    """
    Train with early stopping on a STRATIFIED validation split,
    using a StandardScaler fit ONLY on the training split,
    then evaluate on test.
    """
    # 1) Stratified split on RAW (unscaled) training data
    X_tr_np, X_val_np, y_tr_np, y_val_np = train_test_split(
        X_train,
        y_train,
        test_size=0.1,
        random_state=42,
        stratify=y_train,
    )

    # 2) Fit scaler ONLY on the training split, then transform val + test
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr_np)
    X_val_scaled = scaler.transform(X_val_np)
    X_test_scaled = scaler.transform(X_test)

    # 3) Convert to tensors
    X_tr_t = torch.tensor(X_tr_scaled, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr_np, dtype=torch.float32).unsqueeze(1)

    X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32)
    y_val_t = torch.tensor(y_val_np, dtype=torch.float32).unsqueeze(1)

    X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    # 4) Build datasets and loaders
    train_ds = TensorDataset(X_tr_t, y_tr_t)
    val_ds = TensorDataset(X_val_t, y_val_t)
    test_ds = TensorDataset(X_test_t, y_test_t)

    train_loader_es = DataLoader(
        train_ds, batch_size=params["batch_size"], shuffle=True
    )
    val_loader_es = DataLoader(
        val_ds, batch_size=params["batch_size"], shuffle=False
    )
    test_loader = DataLoader(
        test_ds, batch_size=params["batch_size"], shuffle=False
    )

    # 5) Model, loss, optimizer
    model = MLP(
        input_dim=X_train.shape[1],
        hidden_sizes=params["hidden_sizes"],
        activation=params["activation"],
        dropout=params["dropout"],
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=params["lr"], weight_decay=params["alpha"]
    )

    # 6) Early stopping on validation loss
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
            print(f"Early stopping full-train at epoch {epoch+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # 7) Final evaluation on TEST set
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    # 8) Collect probs + labels for ECE (on test)
    model.eval()
    all_probs = []
    all_y = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_y.append(yb.numpy())

    all_probs = np.vstack(all_probs).ravel()
    all_y = np.vstack(all_y).ravel()

    return test_loss, test_acc, all_y, all_probs, model, scaler



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

    # load validation summary from convergence script
    val_summary = pd.read_csv(VAL_SUMMARY_CSV).set_index("n")

    test_rows = []

    for n in N_VALUES:
        df = pd.read_csv(CSV_HPARAMS)
        row = df[df.n == n].iloc[0]
        params = parse_params(row)

        print(f"\nFinal train + test for n={n}")
        print("Parameters:", params)

        X_train = np.load(f"{TRAIN_DATA_PATH}/kryptonite-{n}-x-train.npy")
        y_train = np.load(f"{TRAIN_DATA_PATH}/kryptonite-{n}-y-train.npy")
        X_test = np.load(f"{TEST_DATA_PATH}/kryptonite-{n}-x-test.npy")
        y_test = np.load(f"{TEST_DATA_PATH}/kryptonite-{n}-y-test.npy")

        (
            test_loss,
            test_acc,
            y_true_test,
            y_prob_test,
            model,
            scaler,
        ) = train_full_and_test(
            X_train, y_train, X_test, y_test, params, device, epochs=N_EPOCHS
        )

        # ECE on test set 
        ece_uniform, df_uniform = compute_ece(
            y_true_test, y_prob_test, n_bins=15, strategy="uniform"
        )
        ece_adapt, df_adapt = compute_ece(
            y_true_test, y_prob_test, n_bins=15, strategy="adaptive"
        )

        # reliability diagrams 
        plot_reliability_diagram(
            df_uniform, ece_uniform, n, strategy="uniform", out_dir=OUTPUT_DIR
        )
        plot_reliability_diagram(
            df_adapt, ece_adapt, n, strategy="adaptive", out_dir=OUTPUT_DIR
        )

        val_best = float(val_summary.loc[n, "val_acc_best"])
        val_final = float(val_summary.loc[n, "val_acc_final"])

        print(
            f"n={n} Test Acc={test_acc:.4f} "
            f"Val Best={val_best:.4f} Val Final={val_final:.4f} "
            f"ECE_u={ece_uniform:.4f} ECE_a={ece_adapt:.4f} "
        )

        # hidden labels
        hidden_path = os.path.join(HIDDEN_DATA_PATH, f"hidden-kryptonite-{n}-X.npy")
        if os.path.exists(hidden_path):
            hidden_X = np.load(hidden_path)
            preds, probs = predict_hidden_labels(model, scaler, hidden_X)
            save_path = os.path.join(
                SUBMISSION_DIR, f"hidden-kryptonite-{n}-Y.npy"
            )
            np.save(save_path, preds)
            print(f"Saved hidden labels for n={n} to {save_path}")
        else:
            print(f"(No hidden dataset for n={n}, skipping label generation.)")

        test_rows.append(
            {
                "n": n,
                "val_acc_best": val_best,
                "val_acc_final": val_final,
                "test_acc": float(test_acc),
                "ece_uniform": float(ece_uniform),
                "ece_adaptive": float(ece_adapt),
            }
        )

    # save overall test/ECE summary
    test_df = pd.DataFrame(test_rows)
    results_csv = os.path.join(OUTPUT_DIR, "pytorch_test_results.csv")
    test_df.to_csv(results_csv, index=False)
    print(f"\nSaved test results to {results_csv}")

    # plot val vs test accuracy
    plt.figure(figsize=(8, 5))
    xs = [str(r["n"]) for r in test_rows]
    val_best = [r["val_acc_best"] for r in test_rows]
    test_accs = [r["test_acc"] for r in test_rows]

    plt.plot(xs, val_best, marker="o", label="Best Validation Accuracy")
    plt.plot(xs, test_accs, marker="s", label="Test Accuracy")
    plt.xlabel("n (number of features)")
    plt.ylabel("Accuracy")
    plt.title("Validation vs Test Accuracy per n")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "val_vs_test_accuracy.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print(f"Saved val vs test accuracy plot to {plot_path}")


if __name__ == "__main__":
    main()
