from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from pathlib import Path
import argparse

args = argparse.ArgumentParser()
args.add_argument("--n", type=int, default=10)
args = args.parse_args()

def main():

    gamma_map ={
        10: 0.25,
        12: 0.9,
        14: 1.55,
        16: 2.3,
        18: 4.3,
        20: 5.0,
    }

    # Path to your dataset folder
    DATA_DIR = Path("Datasets")

    # Load smallest dataset
    X = np.load(DATA_DIR / f"kryptonite-{args.n}-X.npy")
    y = np.load(DATA_DIR / f"kryptonite-{args.n}-y.npy")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Pipeline with scaling + SVM
    svm = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", C=2, gamma=gamma_map[args.n], random_state=42)
    )

    # Fit on training data
    svm.fit(X_train[:], y_train[:])

    # Evaluate on test split
    y_pred = svm.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"RBF SVM test accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
