import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection
if True:  # Include project path
    import sys
    import os
    ROOT = os.path.dirname(os.path.abspath(__file__))+"/../"
    CURR_PATH = os.path.dirname(os.path.abspath(__file__))+"/"
    sys.path.append(ROOT)

    import utils.lib_plot as lib_plot
    import utils.lib_commons as lib_commons
    from utils.lib_classifier import ClassifierOfflineTrain
def par(path):
    """Prepend ROOT to path if not absolute."""
    return ROOT + path if (path and path[0] != "/") else path

# -- Load configuration
cfg_all = lib_commons.read_yaml(ROOT + "config/config.yaml")
cfg = cfg_all["s4_train.py"]

# Load dataset paths
SRC_PROCESSED_FEATURES = par(cfg["input"]["processed_features"])
SRC_PROCESSED_FEATURES_LABELS = par(cfg["input"]["processed_features_labels"])

# Load features and labels
X = np.loadtxt(SRC_PROCESSED_FEATURES, dtype=float)
Y = np.loadtxt(SRC_PROCESSED_FEATURES_LABELS, dtype=int)

# Get class names
CLASSES = np.array(cfg_all["classes"])

# Split dataset (70% train, 30% test)
tr_X, te_X, tr_Y, te_Y = sklearn.model_selection.train_test_split(
    X, Y, test_size=0.3, random_state=1, stratify=Y
)

# Split train into train and validation (80-20 split)
tr_X, val_X, tr_Y, val_Y = sklearn.model_selection.train_test_split(
    tr_X, tr_Y, test_size=0.2, random_state=1, stratify=tr_Y
)

# Count samples per class
train_counts = np.bincount(tr_Y, minlength=len(CLASSES))
val_counts = np.bincount(val_Y, minlength=len(CLASSES))
test_counts = np.bincount(te_Y, minlength=len(CLASSES))

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
bar_width = 0.3
indices = np.arange(len(CLASSES))

plt.bar(indices - bar_width, train_counts, width=bar_width, label="Train", color="royalblue")
plt.bar(indices, val_counts, width=bar_width, label="Validation", color="forestgreen")
plt.bar(indices + bar_width, test_counts, width=bar_width, label="Test", color="crimson")

# Formatting
plt.xlabel("Classes", fontsize=12)
plt.ylabel("Number of Samples", fontsize=12)
plt.title("Train-Test-Validation Split per Class", fontsize=14, fontweight="bold")
plt.xticks(indices, CLASSES, rotation=45, ha="right")
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()

# Save and Show
SAVE_PATH = par("results/train_test_val_distribution.png")
plt.savefig(SAVE_PATH, dpi=300, bbox_inches="tight", facecolor="white")
plt.show()

print(f"Plot saved at: {SAVE_PATH}")
