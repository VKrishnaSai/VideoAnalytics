import os
import numpy as np
import av
import torch
from tqdm import tqdm

CACHE_FILE = "video_dataset.pt"  # Path to save/load dataset

def read_video_pyav(container, indices):
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    
    for i, frame in enumerate(tqdm(container.decode(video=0), desc="Extracting frames", leave=False)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            reformatted_frame = frame.reformat(width=224, height=224)
            frames.append(reformatted_frame)
    
    new = np.stack([x.to_ndarray(format="rgb24") for x in frames])
    return new

def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    converted_len = int(clip_len * frame_sample_rate)
    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices

def frames_convert_and_create_dataset_dictionary(directory):
    # Check if cached dataset exists
    if os.path.exists(CACHE_FILE):
        print("Loading cached dataset...")
        return torch.load(CACHE_FILE)

    print("Processing dataset...")
    all_videos = []
    class_folders = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]

    for class_name in tqdm(class_folders, desc="Processing classes"):
        class_path = os.path.join(directory, class_name)
        video_files = [os.path.join(class_path, file) for file in os.listdir(class_path) if file.lower().endswith('.mp4')]
        
        for video_path in tqdm(video_files, desc=f"Processing {class_name}", leave=False):
            try:
                container = av.open(video_path)
                indices = sample_frame_indices(clip_len=10, frame_sample_rate=2, seg_len=container.streams.video[0].frames)
                video = read_video_pyav(container=container, indices=indices)
                all_videos.append({'video': video, 'labels': class_name})
            except Exception as e:
                print(f"Skipping {video_path} due to error: {e}")
    
    # Save dataset for future use
    torch.save(all_videos, CACHE_FILE)
    print(f"Dataset saved to {CACHE_FILE}")
    
    return all_videos
