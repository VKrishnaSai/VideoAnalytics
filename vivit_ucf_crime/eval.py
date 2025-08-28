import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score

# Import your ViViT model and load_video function
from vivit import ViViT, load_video

# ----------------------- Load Classes and Build Data Subset -----------------------
class_file = "truncated_classes.txt"
data_subset = []
categories = {}
with open(class_file, "r") as f:
    current_category = None
    for line in f:
        line = line.strip()
        if not line:
            continue
        if line.startswith("//"):
            current_category = line.strip("//").strip()
            if current_category not in categories:
                categories[current_category] = []
        else:
            data_subset.append(line)
            if current_category is not None:
                categories.setdefault(current_category, []).append(line)

print("Classes:", data_subset)
print("Categories:", categories)

# ----------------------- Build Video Paths and Labels -----------------------
data_root = './kinetics400_5per/train/'
videos = []
labels = []
label_map = {cls: idx for idx, cls in enumerate(data_subset)}
for cls in data_subset:
    cls_path = os.path.join(data_root, cls)
    if not os.path.isdir(cls_path):
        print(f"Warning: Class directory {cls_path} does not exist!")
        continue
    for video in os.listdir(cls_path):
        if video.endswith('.mp4'):
            videos.append(os.path.join(cls_path, video))
            labels.append(label_map[cls])
print(f"Loaded {len(videos)} videos.")

# ----------------------- Custom Dataset -----------------------
class VideoDataset(Dataset):
    def __init__(self, video_paths, labels, num_frames=16, image_size=224):
        self.video_paths = video_paths
        self.labels = labels
        self.num_frames = num_frames
        self.image_size = image_size

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video = load_video(self.video_paths[idx], num_frames=self.num_frames, image_size=self.image_size)
        # Remove extra batch dimension if present (expected shape: [1, T, C, H, W])
        if video.ndim == 5 and video.shape[0] == 1:
            video = video.squeeze(0)  # now shape: [T, C, H, W]
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return video, label

# Create full dataset
full_dataset = VideoDataset(videos, labels)

# ----------------------- Split Data -----------------------
# Create indices for the full dataset
all_indices = list(range(len(full_dataset)))
# We use stratification on labels to split data; here we use 30% for test.
from sklearn.model_selection import train_test_split
train_indices, test_indices = train_test_split(
    all_indices, test_size=0.30, random_state=42, stratify=labels
)

# For evaluation, we only need the test split.
test_dataset = Subset(full_dataset, test_indices)
testloader = DataLoader(test_dataset, batch_size=4, shuffle=False)
print(f"Test samples: {len(test_dataset)}")

# ----------------------- Evaluation Function -----------------------
def evaluate_per_class(model, test_dataloader, classes, device='cuda'):
    """
    Evaluate the model on the test set and compute per-class metrics.
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for videos_batch, labels_batch in test_dataloader:
            videos_batch = videos_batch.to(device)
            labels_batch = labels_batch.to(device)
            outputs = model(videos_batch)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels_batch.cpu().numpy())
    
    # Compute per-class accuracy using confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    per_class_accuracy = cm.diagonal() / np.sum(cm, axis=1)
    
    # Compute precision, recall, F1 score per class
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, zero_division=0
    )
    
    print("Per-class Evaluation Metrics:")
    for idx, cls in enumerate(classes):
        print(f"Class: {cls}")
        print(f"  Accuracy : {per_class_accuracy[idx]:.4f}")
        print(f"  Precision: {precision[idx]:.4f}")
        print(f"  Recall   : {recall[idx]:.4f}")
        print(f"  F1 Score : {f1[idx]:.4f}")
        print(f"  Support  : {support[idx]}")
        print("-" * 40)

# ----------------------- Load Pretrained Model -----------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = ViViT(image_size=224, patch_size=16, num_classes=len(data_subset), num_frames=16).to(device)

pretrained_model_path = "vivit_model_resume.pth"
if not os.path.exists(pretrained_model_path):
    print(f"Pretrained model not found at: {pretrained_model_path}")
    exit(1)
else:
    model.load_state_dict(torch.load(pretrained_model_path, map_location=device))
    print("Pretrained model loaded successfully.")

# ----------------------- Run Evaluation -----------------------
evaluate_per_class(model, testloader, data_subset, device=device)
