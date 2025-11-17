#!/usr/bin/env python3
"""
run_cc_kryptonite10_genbound.py

Train a 2-layer 1D CNN (with a full last classification layer) on Kryptonite-10
using the *best* hyperparameters found earlier, and log:
- training accuracy
- test accuracy
- generalization bound (Galanti & Xu-style norm-based bound for CNNs)
  for each epoch.

The script:
1. Loads kryptonite-10-X.npy and kryptonite-10-Y.npy
2. Splits into train/test
3. Scales inputs with StandardScaler
4. Trains the CNN and logs metrics per epoch
5. Writes logs to a CSV file

Usage:
    python run_cc_kryptonite10_genbound.py --data-dir ./Datasets \
        --epochs 40 --batch-size 64 --log-file genbound_log.csv
"""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers


SEED = 42


# ---------------------------------------------------------------------------
# Generalization bound utilities
# ---------------------------------------------------------------------------

def compute_rho_product(model: keras.Model) -> float:
    """
    Compute ρ(w) = product of Frobenius norms of the weight matrices across
    all Conv1D and Dense layers.

    ρ(w) = Π_l ||W_l||_F
    """
    rho = 1.0
    for layer in model.layers:
        if isinstance(layer, (layers.Conv1D, layers.Dense)):
            # layer.kernel is a tf.Variable; convert to numpy and take Frobenius norm
            W = layer.kernel.numpy()
            rho *= np.linalg.norm(W)  # Frobenius norm
    return float(rho)


def compute_cnn_gen_bound(
    model: keras.Model,
    X_train: np.ndarray,
    num_classes: int,
    gamma: float = 1.0,
    delta: float = 1e-3,
) -> float:
    """
    Compute the Galanti & Xu–style generalization bound for CNNs with a full
    last classification layer, adapted to this 1D CNN.

    Bound (architecture-aware Rademacher-style) in simplified form:

    err_P(f_w) - err_S^γ(f_w)
        ≤
    2√2 (ρ(w) + 1) / (γ m)
      * (1 + sqrt(2 ( log(2)L + Σ_{l=1}^{L-1} log(k_l) + log(C) )))
      * sqrt( Π_{l=1}^{L-1} k_l * max_j Σ_i |x_{ij}|^2 )
      + 3 * sqrt( log(2 (ρ(w) + 2)^2 / δ) / (2 m) )

    where:
      - m is the number of training samples
      - C is the number of classes
      - k_l is the kernel size of conv layer l
      - ρ(w) = Π_l ||W_l||_F over all weight layers (conv + dense)
      - X_train is the (scaled) input to the network

    We treat:
      - L = (#conv layers) + 1 (final fully connected classification stage)
      - conv layers are the ones contributing k_l
    """
    m, d0 = X_train.shape
    if m == 0:
        return float("nan")

    # Collect conv kernel sizes (k_l)
    conv_kernel_sizes = []
    for layer in model.layers:
        if isinstance(layer, layers.Conv1D):
            conv_kernel_sizes.append(int(layer.kernel_size[0]))

    if len(conv_kernel_sizes) == 0:
        # Degenerate case: no conv – fallback to something trivial
        conv_kernel_sizes = [1]

    # L: conv layers + final full classification layer
    L = len(conv_kernel_sizes) + 1

    # ρ(w): product of Frobenius norms of all Conv1D and Dense weights
    rho = compute_rho_product(model)

    # Σ log(k_l)
    sum_log_k = np.sum(np.log(np.array(conv_kernel_sizes, dtype=np.float64)))

    # Constant factor inside the sqrt:
    # log(2)L + Σ log(k_l) + log(C)
    const_inner = np.log(2.0) * L + sum_log_k + np.log(float(num_classes))
    const_factor = 1.0 + np.sqrt(2.0 * const_inner)

    # Π k_l
    product_k = float(np.prod(conv_kernel_sizes))

    # max_j Σ_i |x_{ij}|^2
    # X_train: shape (m, d0)
    # Σ_i |x_{ij}|^2 over rows, then max over j (features)
    z_term = float(np.max(np.sum(np.square(X_train), axis=0)))

    term1 = (
        2.0
        * np.sqrt(2.0)
        * (rho + 1.0)
        / (gamma * m)
        * const_factor
        * np.sqrt(product_k * z_term)
    )

    term2 = 3.0 * np.sqrt(
        np.log(2.0 * (rho + 2.0) ** 2 / delta) / (2.0 * m)
    )

    return float(term1 + term2)


# ---------------------------------------------------------------------------
# Model definition (best hyperparameters)
# ---------------------------------------------------------------------------

def make_best_cnn_1d(
    input_dim: int,
    filters1: int = 200,
    filters2: int = 200,
    dense_units: int = 128, 
    dropout: float = 0.0, 
    lr: float = 3e-4,      
    kernel_size: int = 2, 
    lambda_l2: float = 5e-3,  
) -> keras.Model:
    """
    Build the 2-layer 1D CNN with one hidden Dense layer and a single-unit
    sigmoid output, using the best hyperparameters you found.
    """
    inputs = keras.Input(shape=(input_dim,))
    x = layers.Reshape((input_dim, 1))(inputs)

    x = layers.Conv1D(
        filters1,
        kernel_size=kernel_size,
        padding="same",
        use_bias=False,  # <- disable bias since BN will learn a shift
        kernel_regularizer=regularizers.l2(lambda_l2),
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv1D(
        filters1,
        kernel_size=kernel_size,
        padding="same",
        use_bias=False,  # <- disable bias since BN will learn a shift
        kernel_regularizer=regularizers.l2(lambda_l2),
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    # Second conv block
    x = layers.Conv1D(
        filters2,
        kernel_size=kernel_size,
        padding="same",
        use_bias=False,
        kernel_regularizer=regularizers.l2(lambda_l2),
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Flatten()(x)

    if dropout and dropout > 0.0:
        x = layers.Dropout(dropout)(x)

    x = layers.Dense(
        dense_units,
        activation="relu",
        kernel_regularizer=regularizers.l2(lambda_l2),
    )(x)

    outputs = layers.Dense(
        1,
        activation="sigmoid",
        kernel_regularizer=regularizers.l2(lambda_l2),
    )(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        # optimizer=keras.optimizers.Adam(learning_rate=lr),
        optimizer=keras.optimizers.SGD(learning_rate=lr, momentum=0.9),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------

def find_kryptonite10(data_dir: Path):
    """Find kryptonite-10-X.npy and kryptonite-10-Y.npy under data_dir."""
    X_path = None
    y_path = None

    for p in data_dir.rglob("*.npy"):
        name = p.name.lower()

        # Extract integer token
        n = None
        for tok in name.replace("-", "_").split("_"):
            if tok.isdigit():
                n = int(tok)
                break

        if n != 10:
            continue

        if name.startswith("kryptonite") and name.endswith("-x.npy"):
            X_path = p
        elif name.startswith("kryptonite") and name.endswith("-y.npy"):
            y_path = p

    return X_path, y_path


# ---------------------------------------------------------------------------
# Callback for logging per epoch
# ---------------------------------------------------------------------------

class GenBoundLogger(keras.callbacks.Callback):
    """
    Keras callback that logs, at the end of each epoch:
      - train accuracy
      - test (validation) accuracy
      - generalization bound for the current weights
    and stores them in self.records.
    """

    def __init__(self, X_train_scaled, num_classes, gamma=1.0, delta=1e-3):
        super().__init__()
        self.X_train_scaled = X_train_scaled
        self.num_classes = num_classes
        self.gamma = gamma
        self.delta = delta
        self.records = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        train_acc = logs.get("accuracy") or logs.get("acc")
        val_acc = logs.get("val_accuracy") or logs.get("val_acc")

        gen_bound = compute_cnn_gen_bound(
            self.model,
            self.X_train_scaled,
            num_classes=self.num_classes,
            gamma=self.gamma,
            delta=self.delta,
        )

        self.records.append(
            {
                "epoch": epoch + 1,
                "train_accuracy": float(train_acc) if train_acc is not None else np.nan,
                "test_accuracy": float(val_acc) if val_acc is not None else np.nan,
                "gen_bound": float(gen_bound),
            }
        )

        print(
            f"[Epoch {epoch+1:02d}] "
            f"train_acc={train_acc:.4f} "
            f"test_acc={val_acc:.4f} "
            f"gen_bound={gen_bound:.4f}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("./Datasets"))
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=32)  # 64
    ap.add_argument(
        "--log-file",
        type=Path,
        default=Path("kryptonite10_genbound_log.csv"),
    )
    args = ap.parse_args()

    # Set seeds for reproducibility
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    def lr_schedule(epoch, lr):
        if epoch < 40:
            return 0.01
        elif epoch < 70:
            return 0.003
        else:
            return 0.001


    # ---------------------------------------------------------------------
    # Load data
    # ---------------------------------------------------------------------
    X_path, y_path = find_kryptonite10(args.data_dir)
    if X_path is None or y_path is None:
        raise SystemExit(
            f"Could not find kryptonite-10-X.npy and kryptonite-10-Y.npy under {args.data_dir}"
        )

    print(f"Using X file: {X_path}")
    print(f"Using y file: {y_path}")

    X = np.load(X_path)
    y = np.load(y_path)

    print("Data loaded.")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("y mean:", y.mean(), "(class balance proxy)")

    # Ensure y is 0/1 float
    y = y.astype(np.float32)
    num_classes = len(np.unique(y))
    if num_classes != 2:
        print(
            f"WARNING: expected binary classification, but found {num_classes} classes."
        )

    # ---------------------------------------------------------------------
    # Train/test split
    # ---------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=SEED,
        stratify=y,
    )

    # ---------------------------------------------------------------------
    # Standardize features
    # ---------------------------------------------------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ---------------------------------------------------------------------
    # Build model with best hyperparameters
    # ---------------------------------------------------------------------
    input_dim = X_train_scaled.shape[1]
    model = make_best_cnn_1d(
        input_dim=input_dim,
        filters1=128,
        filters2=128,
        dense_units=128,
        dropout=0.0,
        lr=0.01,        # 3e-4
        kernel_size=2,  # 3
        lambda_l2=1e-3,
    )

    model.summary()

    # ---------------------------------------------------------------------
    # Train with callback for logging generalization bound
    # ---------------------------------------------------------------------
    gen_logger = GenBoundLogger(
        X_train_scaled=X_train_scaled,
        num_classes=num_classes,
        gamma=1.0,
        delta=1e-3,
    )


    lr_cb = keras.callbacks.LearningRateScheduler(lr_schedule, verbose=1)
    history = model.fit(
        X_train_scaled,
        y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(X_test_scaled, y_test),
        callbacks=[gen_logger, lr_cb],
        verbose=1,
    )

    # ---------------------------------------------------------------------
    # Save logs to CSV
    # ---------------------------------------------------------------------
    df_log = pd.DataFrame(gen_logger.records)
    df_log.to_csv(args.log_file, index=False)
    print(
        f"\nSaved per-epoch logs (train_acc, test_acc, gen_bound) to: {args.log_file}"
    )

    # Final test accuracy for convenience
    test_loss, test_acc = model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"\nFinal test accuracy: {test_acc:.4f}")

    train_loss, train_acc_full = model.evaluate(X_train_scaled, y_train, verbose=0)
    test_loss, test_acc_full = model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"Final full-train accuracy (no dropout): {train_acc_full:.4f}")
    print(f"Final test accuracy (no dropout): {test_acc_full:.4f}")


if __name__ == "__main__":
    main()
