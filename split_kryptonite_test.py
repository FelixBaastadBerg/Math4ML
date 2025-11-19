import os, re, json
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split


DATA_DIR   = Path("/Math4ML/Datasets")
SCRIPT_DIR = Path(__file__).resolve().parent # Path to the folder containing this script
DATA_DIR   = SCRIPT_DIR / "Datasets" # Folder containing datasets
TRAIN_DIR  = DATA_DIR / "Train_Data"
TEST_DIR   = DATA_DIR / "Test_Data"
SPLITS_DIR = DATA_DIR / "Splits"

TEST_SIZE  = 0.20 # 20% test
SEED       = 42
OVERWRITE  = False # Set to True if you need to regenerate splits (careful, this deletes old ones)

for d in (TRAIN_DIR, TEST_DIR, SPLITS_DIR):
    d.mkdir(parents=True, exist_ok=True)

detect_x = re.compile(r"^kryptonite-(\d+)-X\.npy$", re.IGNORECASE)


def find_pairs(base: Path):
    """
    Scan directory for X files and match them with corresponding y files.
    Returns sorted list of (n, X_path, y_path) tuples.
    """
    pairs = []
    for name in os.listdir(base):
        m = detect_x.match(name)
        if not m: 
            continue
        n = m.group(1) # extract the feature count from filename
        x_path = base / name
        y_path = base / f"kryptonite-{n}-y.npy"
        if not y_path.exists():
            print(f"Missing y for kryptonite-{n}. Skipping")
            continue
        pairs.append((n, x_path, y_path))
    pairs.sort(key=lambda t: int(t[0]))
    return pairs

def split_and_save(n, x_path, y_path):
    """
    Load data, stratified train/test split, save everything to designated folders.
    Also saves the split indices for reproducibility.
    """
    X = np.load(x_path, allow_pickle=False)
    y = np.load(y_path, allow_pickle=False)
    if len(X) != len(y):
        raise ValueError(f"Len mismatch kryptonite-{n}: X={len(X)} vs y={len(y)}")
    
    # Define output paths for train/test data and indices
    trX = TRAIN_DIR / f"kryptonite-{n}-X-train.npy"
    trY = TRAIN_DIR / f"kryptonite-{n}-y-train.npy"
    teX = TEST_DIR  / f"kryptonite-{n}-X-test.npy"
    teY = TEST_DIR  / f"kryptonite-{n}-y-test.npy"
    idx_tr = SPLITS_DIR / f"kryptonite-{n}_train_indices.npy"
    idx_te = SPLITS_DIR / f"kryptonite-{n}_test_indices.npy"

    # Quick check: if everything exists and we're not forcing a rebuild, skip it
    already = all(p.exists() for p in [trX, trY, teX, teY, idx_tr, idx_te])
    if already and not OVERWRITE:
        return {"name": n, "status": "exists", "num_samples": len(X)}

    # Perform stratified split to preserve class balance
    indices = np.arange(len(X))
    idx_train, idx_test = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=SEED,
        shuffle=True,
        stratify=y  # ensure class distribution is same in train and test
    )

    # Save the actual array copies to train/test folders
    np.save(trX, X[idx_train], allow_pickle=False)
    np.save(trY, y[idx_train], allow_pickle=False)
    np.save(teX, X[idx_test],  allow_pickle=False)
    np.save(teY, y[idx_test],  allow_pickle=False)

    # Also save the indices in case we need to cross-reference later
    np.save(idx_tr, idx_train, allow_pickle=False)
    np.save(idx_te, idx_test,  allow_pickle=False)

    return {
        "name": n, "status": "created", "num_samples": len(X),
        "train_samples": len(idx_train), "test_samples": len(idx_test)
    }

def main():
    """
    Find all kryptonite pairs and split them. Report results as JSON.
    """
    results = []
    for n, x_path, y_path in find_pairs(DATA_DIR):
        try:
            results.append(split_and_save(n, x_path, y_path))
        except Exception as e:
            # If something goes wrong, log it and keep going
            results.append({"name": n, "status": "error", "error": str(e)})
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
