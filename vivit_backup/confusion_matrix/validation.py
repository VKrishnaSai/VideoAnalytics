import os
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from torchvision import transforms
import cv2

# Import your ViViT model and load_video function
from vivit import ViViT, load_video

# ================================
# 1. Read Classes from File
# ================================
data_subset = []
with open("truncated_classes.txt", "r") as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("//"):
            data_subset.append(line)

print("Classes from file:", data_subset)
print("Total classes from file:", len(data_subset))  # Expected: 46

# Create label mapping: class name -> index
label_map = {cls: idx for idx, cls in enumerate(data_subset)}

# ================================
# 2. Build Dictionary of Videos Per Class (Only .mp4 files)
# ================================
data_root = './kinetics400_5per/train/'
videos_by_class = {}
for cls in data_subset:
    cls_path = os.path.join(data_root, cls)
    if not os.path.isdir(cls_path):
        print(f"Warning: Class directory {cls_path} not found!")
        videos_by_class[cls] = []
        continue
    # Only include files ending with .mp4 (case-insensitive)
    video_files = [os.path.join(cls_path, v) for v in os.listdir(cls_path) if v.lower().endswith(".mp4")]
    videos_by_class[cls] = video_files

# ================================
# 3. Sample 20% Videos per Class (Ensure at least one sample per class)
# ================================
val_videos = []
val_labels = []
for cls in data_subset:
    vids = videos_by_class.get(cls, [])
    n = len(vids)
    if n == 0:
        print(f"Warning: No videos available for class '{cls}'")
        continue
    n_val = max(1, int(0.2 * n))
    sampled = random.sample(vids, n_val)
    for vid in sampled:
        val_videos.append(vid)
        val_labels.append(label_map[cls])

# ================================
# 4. Ensure Every Class is Represented in Validation Set
# ================================
present_labels = set(val_labels)
all_labels_set = set(range(len(data_subset)))  # 0 ... 45 for 46 classes
missing_class_indices = all_labels_set - present_labels
if missing_class_indices:
    print("The following classes were missing from the validation sample:")
    for idx in sorted(missing_class_indices):
        cls = data_subset[idx]
        print(f" - {cls} (index {idx})")
        if cls in videos_by_class and len(videos_by_class[cls]) > 0:
            sample = random.choice(videos_by_class[cls])
            val_videos.append(sample)
            val_labels.append(label_map[cls])
        else:
            print(f"   Warning: No videos available for class '{cls}'.")
else:
    print("All classes are present in the validation set.")

unique_val_labels = sorted(list(set(val_labels)))
print("Unique classes present in validation set (indices):", unique_val_labels)
print("Total unique classes in validation set:", len(unique_val_labels))

# ================================
# 5. Define Validation Dataset
# ================================
class VideoDataset(Dataset):
    def __init__(self, video_paths, labels, num_frames=16, image_size=224):
        self.video_paths = video_paths
        self.labels = labels
        self.num_frames = num_frames
        self.image_size = image_size

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        # load_video returns tensor with shape [1, T, C, H, W]
        video = load_video(self.video_paths[idx], num_frames=self.num_frames, image_size=self.image_size)
        if video.ndim == 5 and video.shape[0] == 1:
            video = video.squeeze(0)  # Now shape: [T, C, H, W]
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return video, label

val_dataset = VideoDataset(val_videos, val_labels, num_frames=16, image_size=224)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
print(f"Validation set: {len(val_dataset)} videos.")

# ================================
# 6. Load Best Model Checkpoint and Prepare Model
# ================================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_classes = len(data_subset)  # Should be 46
model = ViViT(image_size=224, patch_size=16, num_classes=num_classes, num_frames=16)
model.to(device)

checkpoint_path = os.path.join("checkpoints", "resume_best_checkpoint.pth")
if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

checkpoint = torch.load(checkpoint_path, map_location=device)
if "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)
model.eval()

# ================================
# 7. Run Inference on Validation Set
# ================================
all_preds = []
all_labels = []
with torch.no_grad():
    for videos_batch, labels_batch in val_loader:
        videos_batch = videos_batch.to(device)  # Expected shape: [B, T, C, H, W]
        labels_batch = labels_batch.to(device)
        outputs = model(videos_batch)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels_batch.cpu().numpy())

# ================================
# 8. Compute Metrics and Generate Analysis
# ================================
# We use the unique labels actually present for generating the report.
unique_val_labels = sorted(list(set(all_labels) | set(all_preds)))
target_names = [data_subset[label] for label in unique_val_labels]

report = classification_report(all_labels, all_preds, labels=unique_val_labels,
                               target_names=target_names, digits=4)
print("Classification Report:\n", report)
with open("classification_report.txt", "w") as f:
    f.write(report)

cm = confusion_matrix(all_labels, all_preds, labels=unique_val_labels)
print("Confusion Matrix:\n", cm)

plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=[data_subset[i] for i in unique_val_labels],
            yticklabels=[data_subset[i] for i in unique_val_labels])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()

# Compute per-class precision, recall, F1-score
precisions, recalls, f1s, supports = precision_recall_fscore_support(
    all_labels, all_preds, labels=unique_val_labels, zero_division=0
)

x = np.arange(len(unique_val_labels))
width = 0.25

plt.figure(figsize=(14, 6))
plt.bar(x - width, precisions, width, label='Precision')
plt.bar(x, recalls, width, label='Recall')
plt.bar(x + width, f1s, width, label='F1-score')
plt.xticks(x, [data_subset[i] for i in unique_val_labels], rotation=45, ha="right")
plt.xlabel("Class")
plt.ylabel("Score")
plt.title("Per-Class Metrics")
plt.legend()
plt.tight_layout()
plt.savefig("per_class_metrics.png")
plt.close()

overall_accuracy = np.mean(np.array(all_labels) == np.array(all_preds))
analysis_summary = f"""
Validation Analysis Summary
===========================
Total validation videos: {len(val_dataset)}
Overall Accuracy: {overall_accuracy:.4f}

Classification Report:
{report}

Confusion Matrix:
{cm}
"""

with open("validation_analysis_summary.txt", "w") as f:
    f.write(analysis_summary)

print("Validation analysis complete. Figures and summary saved.")
