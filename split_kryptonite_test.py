import os, re, json
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split


DATA_DIR   = Path("/Math4ML/Datasets")
SCRIPT_DIR = Path(__file__).resolve().parent # Path to the folder containing this script
DATA_DIR   = SCRIPT_DIR / "Datasets" # Folder containing kryptonite-N-X.npy/-y.npy
TRAIN_DIR  = DATA_DIR / "Train_Data"
TEST_DIR   = DATA_DIR / "Test_Data"
SPLITS_DIR = DATA_DIR / "Splits"

TEST_SIZE  = 0.20 # 20% test
SEED       = 42
OVERWRITE  = False # Set True to re-generate splits if they already exist

for d in (TRAIN_DIR, TEST_DIR, SPLITS_DIR):
    d.mkdir(parents=True, exist_ok=True)

x_pat = re.compile(r"^kryptonite-(\d+)-X\.npy$", re.IGNORECASE) # To detect/match filenames

def find_pairs(base: Path):
    """
    Find corresponding X and Y files
    """
    pairs = []
    for name in os.listdir(base):
        m = x_pat.match(name)
        if not m: 
            continue
        n = m.group(1) # Extracts the digit (no. of features)
        x_path = base / name
        y_path = base / f"kryptonite-{n}-y.npy"
        if not y_path.exists():
            print(f"Missing y for kryptonite-{n}; skipping.")
            continue
        pairs.append((n, x_path, y_path))
    pairs.sort(key=lambda t: int(t[0]))
    return pairs

def split_and_save(n, x_path, y_path):
    X = np.load(x_path, allow_pickle=False)
    y = np.load(y_path, allow_pickle=False)
    if len(X) != len(y):
        raise ValueError(f"Len mismatch kryptonite-{n}: X={len(X)} vs y={len(y)}")
    # Defining the output paths
    trX = TRAIN_DIR / f"kryptonite-{n}-X-train.npy"
    trY = TRAIN_DIR / f"kryptonite-{n}-y-train.npy"
    teX = TEST_DIR  / f"kryptonite-{n}-X-test.npy"
    teY = TEST_DIR  / f"kryptonite-{n}-y-test.npy"
    idx_tr = SPLITS_DIR / f"kryptonite-{n}_train_indices.npy"
    idx_te = SPLITS_DIR / f"kryptonite-{n}_test_indices.npy"

    already = all(p.exists() for p in [trX, trY, teX, teY, idx_tr, idx_te])
    if already and not OVERWRITE:
        return {"name": n, "status": "exists", "num_samples": len(X)}

    indices = np.arange(len(X))
    idx_train, idx_test = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=SEED,
        shuffle=True,
        stratify=y
    )

    # Save actual array copies to train/test folders
    np.save(trX, X[idx_train], allow_pickle=False)
    np.save(trY, y[idx_train], allow_pickle=False)
    np.save(teX, X[idx_test],  allow_pickle=False)
    np.save(teY, y[idx_test],  allow_pickle=False)

    # Save indices for provenance
    np.save(idx_tr, idx_train, allow_pickle=False)
    np.save(idx_te, idx_test,  allow_pickle=False)

    return {
        "name": n, "status": "created", "num_samples": len(X),
        "train_samples": len(idx_train), "test_samples": len(idx_test)
    }

def main():
    results = []
    for n, x_path, y_path in find_pairs(DATA_DIR):
        try:
            results.append(split_and_save(n, x_path, y_path))
        except Exception as e:
            results.append({"name": n, "status": "error", "error": str(e)})
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
