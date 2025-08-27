# -*- coding: utf-8 -*-
"""Quick inference demo with VideoMAE model finetuned on UCF101.

This demo uses the Hugging Face Transformers interface.
"""

# Download a sample video. (Replace with a UCF101 video if available.)

# Display the video (optional)
from ipywidgets import Video
video_path = "../cluster_results/checkpoints/drumming.mp4"
Video.from_file(video_path, width=500)

# Prepare video for the model
from transformers import VideoMAEImageProcessor
from decord import VideoReader, cpu
import numpy as np

# Load the feature extractor for the UCF101 finetuned model.
# (Note: Update the model name if yours differs.)
feature_extractor = VideoMAEImageProcessor.from_pretrained("nateraw/videomae-base-finetuned-ucf101")

# Open the video using decord
vr = VideoReader(video_path, num_threads=1, ctx=cpu(0))

# Function to sample frame indices from the video
def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    converted_len = min(int(clip_len * frame_sample_rate), seg_len)
    
    if seg_len < clip_len:
        index = np.linspace(0, seg_len - 1, num=seg_len, dtype=int)
    else:
        end_idx = np.random.randint(converted_len, seg_len) if seg_len > converted_len else seg_len
        str_idx = max(0, end_idx - converted_len)
        index = np.linspace(str_idx, end_idx - 1, num=clip_len, dtype=int)

    return np.clip(index, 0, seg_len - 1)


# For example, sample 16 frames with a sampling rate of 4.
vr.seek(0)
index = sample_frame_indices(clip_len=16, frame_sample_rate=4, seg_len=len(vr))
buffer = vr.get_batch(index).asnumpy()

# Create a list of frames (each as a NumPy array)
video = [buffer[i] for i in range(buffer.shape[0])]

# Preprocess the video frames with the feature extractor.
encoding = feature_extractor(video, return_tensors="pt")
print("Pixel values shape:", encoding.pixel_values.shape)

# Load the VideoMAE model finetuned on UCF101
from transformers import VideoMAEForVideoClassification
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model = VideoMAEForVideoClassification.from_pretrained("nateraw/videomae-base-finetuned-ucf101")
model.to(device)

# Forward pass
pixel_values = encoding.pixel_values.to(device)

with torch.no_grad():
    outputs = model(pixel_values)
    logits = outputs.logits

predicted_class_idx = logits.argmax(-1).item()

# Print the predicted class label (using the model config mapping)
print("Predicted class:", model.config.id2label[predicted_class_idx])
