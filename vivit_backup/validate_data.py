import os
import torch
from tqdm import tqdm
from vivit import load_video  # Make sure this uses the updated load_video with error handling

# Set your data root and expected parameters
data_root = './kinetics400_5per/train/'
num_frames = 16
image_size = 224

# Load class names from your truncated_classes.txt
class_file = "truncated_classes.txt"
data_subset = []
categories = {}
current_category = None

with open(class_file, "r") as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("//"):
            if line.startswith("//"):
                current_category = line.strip("//").strip()
                categories[current_category] = []
        else:
            categories[current_category].append(line)
            data_subset.append(line)
print('we have', len(data_subset), 'classes')
# Build list of video paths
videos = []
for cls in data_subset:
    cls_dir = os.path.join(data_root, cls)
    if not os.path.isdir(cls_dir):
        print(f"Warning: Class directory {cls_dir} does not exist!")
        continue
    for video in os.listdir(cls_dir):
        video_path = os.path.join(cls_dir, video)
        videos.append(video_path)

print(f"Total videos found: {len(videos)}")

# Validate each video by loading it
issue_count = 0
for video_path in tqdm(videos, desc="Validating Videos"):
    # Load video using your load_video function.
    video_tensor = load_video(video_path, num_frames=num_frames, image_size=image_size)
    # Expected shape: [1, num_frames, 3, image_size, image_size]
    expected_shape = (1, num_frames, 3, image_size, image_size)
    if video_tensor.shape != expected_shape:
        print(f"Warning: Video {video_path} has unexpected shape {video_tensor.shape} (expected {expected_shape})")
        issue_count += 1
    # Optionally, check if the tensor is all zeros (which indicates a problem)
    if torch.sum(video_tensor) == 0:
        print(f"Warning: Video {video_path} appears to be empty (all zeros)")
        issue_count += 1

print(f"Validation complete. {issue_count} issues found out of {len(videos)} videos.")
