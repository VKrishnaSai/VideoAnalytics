import os
import matplotlib.pyplot as plt
import pandas as pd

# Create results directory if it doesn't exist
os.makedirs("results", exist_ok=True)

# Read CSV file directly (update 'data.csv' to your CSV file path)
csv_file = '28feb_logs.csv'
df = pd.read_csv(csv_file)

# -------------------------
# Plot 1: Train and Validation Loss
# -------------------------
plt.figure(figsize=(10, 6))
plt.plot(df['Epoch'], df['Train Loss'], marker='o', color="royalblue", label='Train Loss')
plt.plot(df['Epoch'], df['Validation Loss'], marker='o', color="crimson", label='Validation Loss')
plt.title('Train and Validation Loss', fontsize=14)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.grid(True)
plt.legend()
plt.tight_layout()
loss_path = "results/train_val_loss.png"
plt.savefig(loss_path, dpi=300, bbox_inches="tight", facecolor="white")
plt.show()

# -------------------------
# Plot 2: Train and Validation Accuracy
# -------------------------
plt.figure(figsize=(10, 6))
plt.plot(df['Epoch'], df['Train Acc'], marker='o', color="royalblue", label='Train Accuracy')
plt.plot(df['Epoch'], df['Val Acc'], marker='o', color="crimson", label='Validation Accuracy')
plt.title('Train and Validation Accuracy', fontsize=14)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.grid(True)
plt.legend()
plt.tight_layout()
acc_path = "results/train_val_accuracy.png"
plt.savefig(acc_path, dpi=300, bbox_inches="tight", facecolor="white")
plt.show()

# -------------------------
# Plot 3: Precision, Recall, and F1 Score
# -------------------------
plt.figure(figsize=(10, 6))
plt.plot(df['Epoch'], df['Precision'], marker='o', color="forestgreen", label='Precision')
plt.plot(df['Epoch'], df['Recall'], marker='o', color="mediumorchid", label='Recall')
plt.plot(df['Epoch'], df['F1'], marker='o', color="cyan", label='F1 Score')
plt.title('Precision, Recall, and F1 Score', fontsize=14)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.grid(True)
plt.legend()
plt.tight_layout()
metrics_path = "results/metrics.png"
plt.savefig(metrics_path, dpi=300, bbox_inches="tight", facecolor="white")
plt.show()

print(f"Loss plot saved at: {loss_path}")
print(f"Accuracy plot saved at: {acc_path}")
print(f"Metrics plot saved at: {metrics_path}")
