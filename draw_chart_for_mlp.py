import numpy as np
import matplotlib.pyplot as plt
import os

# Output directory
OUTPUT_DIR = "Misc Artifacts"
os.makedirs(OUTPUT_DIR, exist_ok=True) 

# data
# MLP test accuracies
mlp_acc = np.array([0.957, 0.967, 0.965, 0.964, 0.960, 0.971])



# Hoeffding confidence intervals: [lower, upper]
hoeffding_ci = np.array([
    [0.931, 0.983],
    [0.944, 0.990],
    [0.943, 0.987],
    [0.944, 0.984],
    [0.941, 0.979],
    [0.953, 0.989],
])
ci_lower = hoeffding_ci[:, 0]
ci_upper = hoeffding_ci[:, 1]

# Threshold values
threshold = np.array([0.94, 0.93, 0.92, 0.91, 0.80, 0.75])

# x-axis: experiment index 1..6
x = np.arange(1, len(mlp_acc) + 1)

# Set font size to 10
plt.rcParams.update({'font.size': 10})

# plot
plt.figure(figsize=(6, 4))

# MLP accuracy
plt.plot(x, mlp_acc, marker="o", linestyle="-", label="MLP accuracy")

# Threshold
plt.plot(x, threshold, marker="o", linestyle="-", label="Threshold", color="red")

# Hoeffding CI band
plt.fill_between(x, ci_lower, ci_upper, alpha=0.15, label="Hoeffding CI")

# Optional lower/upper CI outlines
plt.plot(x, ci_lower, linestyle="--", linewidth=1)
plt.plot(x, ci_upper, linestyle="--", linewidth=1)

# Labels / formatting
plt.xticks(x)
plt.xlabel("Experiment index")
plt.ylabel("Accuracy")
plt.ylim(0.7, 1.0)   # adjust if you want
plt.title("MLP accuracy, Hoeffding CI, and Threshold")
plt.legend()
plt.tight_layout()

# save
save_path = os.path.join(OUTPUT_DIR, "mlp_hoeffding_threshold_plot.png")
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved figure to {save_path}")
