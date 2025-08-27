import torch
import numpy as np
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from decord import VideoReader, cpu
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
# Paths (Update these!)
UCF101_PATH = "../UCF101/UCF-101"  # Path to UCF101 dataset
UCF_TESTLIST_PATH = "../ucfTrainTestlist"  # Path to test list files
SAVE_DIR = "benchmark_results"  # Where results will be saved
os.makedirs(SAVE_DIR, exist_ok=True)

# Load UCF101 class labels
def load_ucf101_labels(filepath):
    labels = {}
    with open(filepath, "r") as f:
        for line in f:
            index, name = line.strip().split()
            labels[name] = int(index) - 1  # Convert to 0-based index
    return labels

ucf101_labels = load_ucf101_labels(os.path.join(UCF_TESTLIST_PATH, "classInd.txt"))
idx_to_class = {v: k for k, v in ucf101_labels.items()}  # Reverse mapping

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
feature_extractor = VideoMAEImageProcessor.from_pretrained("nateraw/videomae-base-finetuned-ucf101")
model = VideoMAEForVideoClassification.from_pretrained("nateraw/videomae-base-finetuned-ucf101").to(device)
model.eval()

# Load video using Decord only
def load_video_decord(video_path, num_frames=16, image_size=224):
    try:
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)

        if total_frames == 0:
            print(f"Warning: Video {video_path} has no frames. Returning zeros.")
            return torch.zeros(1, num_frames, 3, image_size, image_size)

        # Sample frames evenly across the video
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

        # Get frames
        frames = vr.get_batch(indices).asnumpy()  # Shape: (num_frames, H, W, C)

        # Ensure the frames are in (num_frames, H, W, 3)
        if frames.shape[-1] == 1:  # If grayscale, convert to RGB
            frames = np.repeat(frames, 3, axis=-1)

        # Preprocess using feature extractor
        encoding = feature_extractor(list(frames), return_tensors="pt")
        pixel_values = encoding.pixel_values.to(device)

        return pixel_values

    except Exception as e:
        print(f"Error loading video {video_path}: {e}")
        return torch.zeros(1, num_frames, 3, image_size, image_size)

# Helper function to process a video and get predictions
def process_video(video_path):
    pixel_values = load_video_decord(video_path)

    # Run inference
    with torch.no_grad():
        outputs = model(pixel_values)
        logits = outputs.logits
        top5_probs, top5_indices = torch.topk(logits, 5, dim=-1)

    return top5_indices.cpu().numpy(), top5_probs.cpu().numpy()

# Evaluate the model on a test split and store results
def evaluate_on_test_split(testlist_file, split_name):
    test_videos = []
    with open(testlist_file, "r") as f:
        test_videos = [line.strip() for line in f]

    correct_top1, correct_top5 = 0, 0
    total = len(test_videos)
    true_labels, pred_labels = [], []
    results = []

    for vid_path in tqdm(test_videos, desc=f"Evaluating {split_name}"):
        full_path = os.path.join(UCF101_PATH, vid_path)
        if not os.path.exists(full_path):
            print(f"Skipping {vid_path} (file not found)")
            continue

        true_class = vid_path.split('/')[0]
        true_label = ucf101_labels[true_class]

        top5_indices, _ = process_video(full_path)
        top5_preds = top5_indices[0].tolist()

        # Store predictions
        results.append([vid_path, true_class, idx_to_class[top5_preds[0]], top5_preds])

        # Compute accuracy
        if true_label == top5_preds[0]:
            correct_top1 += 1
        if true_label in top5_preds:
            correct_top5 += 1

        true_labels.append(true_label)
        pred_labels.append(top5_preds[0])

    # Calculate accuracy
    top1_acc = 100.0 * correct_top1 / total
    top5_acc = 100.0 * correct_top5 / total

    # Save results
    results_df = pd.DataFrame(results, columns=["Video", "True Class", "Predicted Class (Top-1)", "Top-5 Predictions"])
    results_df.to_csv(f"{SAVE_DIR}/{split_name}_predictions.csv", index=False)

    # Log accuracy
    with open(f"{SAVE_DIR}/results.txt", "a") as f:
        f.write(f"{split_name} - Top-1 Accuracy: {top1_acc:.2f}% | Top-5 Accuracy: {top5_acc:.2f}%\n")

    # Generate and save confusion matrix
    cm = confusion_matrix(true_labels, pred_labels, labels=list(ucf101_labels.values()))
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=idx_to_class.values(), yticklabels=idx_to_class.values())
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix - {split_name}")
    plt.savefig(f"{SAVE_DIR}/{split_name}_confusion_matrix.png")
    plt.close()

    return top1_acc, top5_acc

# Run evaluation
for split in tqdm(["testlist01.txt", "testlist02.txt", "testlist03.txt"]):
    print(f"Evaluating on {split}...")
    top1, top5 = evaluate_on_test_split(os.path.join(UCF_TESTLIST_PATH, split), split.replace(".txt", ""))
    print(f"{split} - Top-1 Accuracy: {top1:.2f}% | Top-5 Accuracy: {top5:.2f}%\n")

print(f"All results saved in {SAVE_DIR}/")
