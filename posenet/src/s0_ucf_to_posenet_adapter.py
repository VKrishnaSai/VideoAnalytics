#!/usr/bin/env python
# coding: utf-8

'''
UCF-101 to PoseNet Pipeline Adapter (Step 0)
===========================================

This script adapts UCF-101 videos to work with the existing PoseNet pipeline (s1-s5).

What it does:
1. Reads UCF-101 videos from trainlist/testlist files
2. Extracts frames from videos on-the-fly (memory efficient)
3. Creates a valid_images.txt file compatible with s1_get_skeletons_from_training_imgs.py
4. Organizes frames in the expected folder structure for the pipeline

Pipeline Integration:
s0 (this script) → s1 → s2 → s3 → s4 → s5

Usage:
    python src/s0_ucf_to_posenet_adapter.py --ucf_root /path/to/UCF-101 --subset_classes 10
'''

import os
import cv2
import argparse
import tempfile
import shutil
from pathlib import Path
import yaml

if True:  # Include project path
    import sys
    ROOT = os.path.dirname(os.path.abspath(__file__))+"/../"
    CURR_PATH = os.path.dirname(os.path.abspath(__file__))+"/"
    sys.path.append(ROOT)
    
    from tools.video2images import ReadFromVideo
    import utils.lib_commons as lib_commons

def par(path):  # Pre-Append ROOT to the path if it's not absolute
    return ROOT + path if (path and path[0] != "/") else path

# -- Configuration
cfg_all = lib_commons.read_yaml(ROOT + "config/config.yaml")
IMG_FILENAME_FORMAT = cfg_all["image_filename_format"]

def parse_args():
    parser = argparse.ArgumentParser(description="UCF-101 to PoseNet Pipeline Adapter")
    parser.add_argument("--ucf_root", type=str, required=True,
                        help="Path to UCF-101 dataset root folder")
    parser.add_argument("--ucf_traintest_root", type=str, default=None,
                        help="Path to UCF train/test split files (default: auto-detect as ../ucfTrainTestlist relative to UCF-101 root)")
    parser.add_argument("--subset_classes", type=int, default=None,
                        help="Use only first N classes for faster testing (None = all 101 classes)")
    parser.add_argument("--frames_per_video", type=int, default=50,
                        help="Max frames to extract per video")
    parser.add_argument("--sample_interval", type=int, default=2,
                        help="Sample every Nth frame from video")
    parser.add_argument("--output_images_folder", type=str, default="data/source_images_ucf",
                        help="Output folder for extracted frames")
    parser.add_argument("--split", type=str, default="01", choices=["01", "02", "03"],
                        help="Which train/test split to use")
    return parser.parse_args()

def load_ucf_classes(ucf_traintest_root):
    """Load UCF-101 class indices"""
    class_index_file = os.path.join(ucf_traintest_root, "classInd.txt")
    class_indices = {}
    with open(class_index_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                idx, class_name = parts
                class_indices[int(idx)] = class_name
    return class_indices

def load_ucf_split_data(ucf_traintest_root, ucf_root, split="01", subset_classes=None):
    """Load train/test video paths from UCF split files"""
    
    # Load class mapping
    class_indices = load_ucf_classes(ucf_traintest_root)
    
    # Filter classes if subset requested
    if subset_classes:
        class_indices = {k: v for k, v in class_indices.items() if k <= subset_classes}
        print(f"Using subset of {subset_classes} classes: {list(class_indices.values())}")
    
    # Load training videos
    trainlist_file = os.path.join(ucf_traintest_root, f"trainlist{split}.txt")
    train_videos = []
    train_labels = []
    
    print(f"Loading training videos from {trainlist_file}")
    with open(trainlist_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            video_rel_path = parts[0]
            label_idx = int(parts[1])
            
            # Skip if not in subset
            if subset_classes and label_idx > subset_classes:
                continue
                
            video_path = os.path.join(ucf_root, video_rel_path)
            if os.path.isfile(video_path):
                train_videos.append(video_path)
                train_labels.append(class_indices[label_idx])
    
    # Load test videos  
    testlist_file = os.path.join(ucf_traintest_root, f"testlist{split}.txt")
    test_videos = []
    test_labels = []
    
    print(f"Loading test videos from {testlist_file}")
    with open(testlist_file, "r") as f:
        for line in f:
            video_rel_path = line.strip()
            if not video_rel_path:
                continue
                
            # Extract class name from path
            class_name = video_rel_path.split('/')[0]
            
            # Find class index
            label_idx = None
            for idx, name in class_indices.items():
                if name == class_name:
                    label_idx = idx
                    break
            
            # Skip if class not found or not in subset
            if label_idx is None or (subset_classes and label_idx > subset_classes):
                continue
                
            video_path = os.path.join(ucf_root, video_rel_path)
            if os.path.isfile(video_path):
                test_videos.append(video_path)
                test_labels.append(class_name)
    
    return train_videos, train_labels, test_videos, test_labels, class_indices

def extract_frames_from_video(video_path, output_folder, max_frames=50, sample_interval=2):
    """Extract frames from a single video (memory optimized)"""
    try:
        # Use OpenCV directly for better memory management
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Could not open video: {video_path}")
            return []
        
        os.makedirs(output_folder, exist_ok=True)
        
        frame_count = 0
        extracted_frames = []
        frame_idx = 0
        
        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Sample frames according to interval
            if frame_idx % sample_interval == 0:
                # Resize frame to save space (optional)
                # frame = cv2.resize(frame, (640, 480))  # Uncomment if storage is critical
                
                # Save frame
                frame_filename = IMG_FILENAME_FORMAT.format(frame_count + 1)
                frame_path = os.path.join(output_folder, frame_filename)
                cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])  # Compressed JPEG
                
                extracted_frames.append(frame_count + 1)
                frame_count += 1
            
            frame_idx += 1
            
        cap.release()
        return extracted_frames
        
    except Exception as e:
        print(f"Error processing {video_path}: {e}")
        return []

def process_ucf_videos_to_frames(videos, labels, output_base_folder, max_frames=50, sample_interval=2, split_name="train"):
    """Process all videos and extract frames"""
    
    valid_images_entries = []
    total_videos = len(videos)
    
    print(f"\nProcessing {total_videos} {split_name} videos...")
    
    for i, (video_path, label) in enumerate(zip(videos, labels)):
        print(f"  [{i+1}/{total_videos}] Processing {os.path.basename(video_path)} (class: {label})")
        
        # Create output folder for this video's frames
        video_name = Path(video_path).stem
        video_folder_name = f"{label}_{video_name}"
        video_output_folder = os.path.join(output_base_folder, video_folder_name)
        
        # Extract frames
        extracted_frames = extract_frames_from_video(
            video_path, video_output_folder, max_frames, sample_interval
        )
        
        if extracted_frames:
            # Add entry for valid_images.txt
            # Format: folder_name \n start_frame end_frame
            start_frame = min(extracted_frames)
            end_frame = max(extracted_frames)
            valid_images_entries.append(f"{video_folder_name}\n{start_frame} {end_frame}")
        
    return valid_images_entries

def create_valid_images_txt(valid_images_entries, output_path):
    """Create valid_images.txt file compatible with s1 pipeline"""
    with open(output_path, 'w') as f:
        for entry in valid_images_entries:
            f.write(entry + '\n\n')  # Double newline between entries
    print(f"Created valid_images.txt with {len(valid_images_entries)} video entries at: {output_path}")

def update_config_for_ucf(class_indices, subset_classes=None):
    """Update config.yaml to use UCF classes instead of default ones"""
    
    # Get class names (filter if subset requested)
    if subset_classes:
        ucf_classes = [class_indices[i] for i in sorted(class_indices.keys()) if i <= subset_classes]
    else:
        ucf_classes = [class_indices[i] for i in sorted(class_indices.keys())]
    
    # Read current config
    config_path = ROOT + "config/config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update classes
    config['classes'] = ucf_classes
    
    # Update paths to use UCF data
    config['s1_get_skeletons_from_training_imgs.py']['input']['images_description_txt'] = 'data/source_images_ucf/valid_images.txt'
    config['s1_get_skeletons_from_training_imgs.py']['input']['images_folder'] = 'data/source_images_ucf/'
    
    # Backup original config
    backup_path = config_path + ".backup"
    if not os.path.exists(backup_path):
        shutil.copy2(config_path, backup_path)
        print(f"Backed up original config to: {backup_path}")
    
    # Save updated config
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"Updated config.yaml with {len(ucf_classes)} UCF classes")
    return ucf_classes

def main():
    args = parse_args()
    
    # Auto-detect ucfTrainTestlist path if not provided
    if args.ucf_traintest_root is None:
        # Standard UCF-101 structure: UCF-101/ and ucfTrainTestlist/ are siblings
        ucf_parent = os.path.dirname(os.path.abspath(args.ucf_root))
        args.ucf_traintest_root = os.path.join(ucf_parent, "ucfTrainTestlist")
        print(f"Auto-detected train/test list path: {args.ucf_traintest_root}")
    
    # Verify that the train/test list directory exists
    if not os.path.exists(args.ucf_traintest_root):
        print(f"ERROR: Train/test list directory not found: {args.ucf_traintest_root}")
        print("Please ensure the ucfTrainTestlist folder is in the correct location or specify --ucf_traintest_root")
        sys.exit(1)
    
    print("UCF-101 to PoseNet Pipeline Adapter")
    print("="*50)
    print(f"UCF Root: {args.ucf_root}")
    print(f"Train/Test Root: {args.ucf_traintest_root}")
    print(f"Subset Classes: {args.subset_classes}")
    print(f"Frames per Video: {args.frames_per_video}")
    print(f"Sample Interval: {args.sample_interval}")
    print(f"Split: {args.split}")
    
    # Load UCF data
    train_videos, train_labels, test_videos, test_labels, class_indices = load_ucf_split_data(
        args.ucf_traintest_root, args.ucf_root, args.split, args.subset_classes
    )
    
    print(f"\nLoaded {len(train_videos)} training videos and {len(test_videos)} test videos")
    
    # Combine train and test for frame extraction (PoseNet pipeline will do its own split)
    all_videos = train_videos + test_videos
    all_labels = train_labels + test_labels
    
    # Create output folder
    output_folder = par(args.output_images_folder)
    os.makedirs(output_folder, exist_ok=True)
    
    # Process videos and extract frames
    valid_images_entries = process_ucf_videos_to_frames(
        all_videos, all_labels, output_folder, 
        args.frames_per_video, args.sample_interval, "combined"
    )
    
    # Create valid_images.txt
    valid_images_path = os.path.join(output_folder, "valid_images.txt")
    create_valid_images_txt(valid_images_entries, valid_images_path)
    
    # Update config.yaml
    ucf_classes = update_config_for_ucf(class_indices, args.subset_classes)
    
    print(f"\n✅ UCF-101 to PoseNet conversion completed!")
    print(f"   📁 Frames saved to: {output_folder}")
    print(f"   📄 Valid images: {valid_images_path}")
    print(f"   ⚙️  Config updated with {len(ucf_classes)} classes")
    print(f"\nNext steps:")
    print(f"   1. Run: python src/s1_get_skeletons_from_training_imgs.py")
    print(f"   2. Run: python src/s2_put_skeleton_txts_to_a_single_txt.py")
    print(f"   3. Run: python src/s3_preprocess_features.py")
    print(f"   4. Run: python src/s4_train.py")
    print(f"   5. Run: python src/s5_test.py")

if __name__ == "__main__":
    main()
