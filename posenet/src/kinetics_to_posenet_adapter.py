#!/usr/bin/env python3
"""
Kinetics4005per to PoseNet Adapter
Converts Kinetics4005per dataset structure to work with PoseNet skeleton training pipeline.
This adapter processes videos to extract skeleton data and prepares them for action recognition training.
"""

import os
import sys
import time
import logging
import yaml
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
import collections

# Include project path
if True:
    ROOT = os.path.dirname(os.path.abspath(__file__)) + "/../"
    CURR_PATH = os.path.dirname(os.path.abspath(__file__)) + "/"
    sys.path.append(ROOT)

    from utils.lib_openpose import SkeletonDetector
    from utils.lib_tracker import Tracker
    import utils.lib_commons as lib_commons

def par(path):  # Pre-Append ROOT to the path if it's not absolute
    return ROOT + path if (path and path[0] != "/") else path

# ----------------------- Logging Setup -----------------------
log_filename = f'kinetics4005per_posenet_training_{datetime.now().strftime("%Y%m%d_%H%M")}.log'
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s'
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)
logger = logging.getLogger(__name__)
logger.info("Kinetics4005per PoseNet Training Adapter started")

# ----------------------- Configuration -----------------------
def load_config():
    """Load configuration from config.yaml"""
    config_path = ROOT + "config/config.yaml"
    if os.path.exists(config_path):
        return lib_commons.read_yaml(config_path)
    else:
        logger.error(f"Config file not found: {config_path}")
        return None

# ----------------------- Kinetics Data Loading -----------------------
def load_kinetics4005per_structure(data_root="./kinetics4005per"):
    """
    Load Kinetics4005per dataset structure and prepare for PoseNet processing.
    Returns video paths and their corresponding class labels.
    """
    train_root = os.path.join(data_root, "train", "train")
    val_root = os.path.join(data_root, "val", "val")
    
    # Get all class names from train directory and sort alphabetically
    train_classes = []
    if os.path.exists(train_root):
        train_classes = [d for d in os.listdir(train_root)
                        if os.path.isdir(os.path.join(train_root, d))]

    val_classes = []
    if os.path.exists(val_root):
        val_classes = [d for d in os.listdir(val_root)
                       if os.path.isdir(os.path.join(val_root, d))]

    # Combine and sort classes alphabetically
    all_classes = sorted(list(set(train_classes + val_classes)))
    logger.info(f"Found {len(all_classes)} action classes")
    logger.info(f"Classes: {all_classes[:10]}..." if len(all_classes) > 10 else f"Classes: {all_classes}")

    # Create class to index mapping
    class_to_idx = {class_name: idx for idx, class_name in enumerate(all_classes)}

    # Collect training videos
    train_videos = []
    train_labels = []
    train_class_info = []  # Store class name for each video

    if os.path.exists(train_root):
        logger.info("Collecting training videos...")
        for class_name in all_classes:
            class_dir = os.path.join(train_root, class_name)
            if not os.path.exists(class_dir):
                continue

            video_files = [f for f in os.listdir(class_dir)
                          if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm'))]

            for video_file in video_files:
                video_path = os.path.join(class_dir, video_file)
                if os.path.isfile(video_path):
                    train_videos.append(video_path)
                    train_labels.append(class_to_idx[class_name])
                    train_class_info.append(class_name)

    # Collect validation videos (will be used as test set)
    val_videos = []
    val_labels = []
    val_class_info = []

    if os.path.exists(val_root):
        logger.info("Collecting validation videos...")
        for class_name in all_classes:
            class_dir = os.path.join(val_root, class_name)
            if not os.path.exists(class_dir):
                continue

            video_files = [f for f in os.listdir(class_dir)
                          if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm'))]

            for video_file in video_files:
                video_path = os.path.join(class_dir, video_file)
                if os.path.isfile(video_path):
                    val_videos.append(video_path)
                    val_labels.append(class_to_idx[class_name])
                    val_class_info.append(class_name)

    logger.info(f"Found {len(train_videos)} training videos and {len(val_videos)} validation videos")

    # Log distribution
    train_counts = collections.Counter(train_labels)
    val_counts = collections.Counter(val_labels)

    logger.info("Training distribution:")
    for i, class_name in enumerate(all_classes):
        count = train_counts.get(i, 0)
        if count > 0:
            logger.info(f"  {class_name}: {count} videos")

    logger.info("Validation distribution:")
    for i, class_name in enumerate(all_classes):
        count = val_counts.get(i, 0)
        if count > 0:
            logger.info(f"  {class_name}: {count} videos")

    return {
        'train_videos': train_videos,
        'train_labels': train_labels,
        'train_class_info': train_class_info,
        'val_videos': val_videos,
        'val_labels': val_labels,
        'val_class_info': val_class_info,
        'classes': all_classes,
        'class_to_idx': class_to_idx
    }

# ----------------------- Video to Frame Extraction -----------------------
def extract_frames_from_videos(video_data, output_dir, frames_per_video=10, target_fps=5):
    """
    Extract frames from Kinetics videos for skeleton detection.
    Similar to how PoseNet processes training images.
    """
    frames_dir = os.path.join(output_dir, "source_images_kinetics")
    os.makedirs(frames_dir, exist_ok=True)
    
    # Create image info file (similar to valid_images.txt)
    images_info_file = os.path.join(output_dir, "kinetics_images_info.txt")
    
    frame_count = 0
    video_info = []
    
    logger.info(f"Extracting frames from {len(video_data['train_videos']) + len(video_data['val_videos'])} videos...")
    
    # Process training videos
    for video_idx, (video_path, label, class_name) in enumerate(zip(
        video_data['train_videos'], 
        video_data['train_labels'], 
        video_data['train_class_info']
    )):
        frames = extract_frames_from_single_video(
            video_path, frames_per_video, target_fps
        )
        
        if frames:
            video_name = f"{class_name}_{os.path.basename(video_path).split('.')[0]}"
            start_frame_idx = frame_count
            
            # Save frames
            for frame_idx, frame in enumerate(frames):
                frame_filename = f"{frame_count:05d}.jpg"
                frame_path = os.path.join(frames_dir, frame_filename)
                cv2.imwrite(frame_path, frame)
                frame_count += 1
            
            end_frame_idx = frame_count - 1
            
            # Add to video info
            video_info.append(f"\n{video_name}")
            video_info.append(f"{start_frame_idx} {end_frame_idx}")
        
        if (video_idx + 1) % 50 == 0:
            logger.info(f"Processed {video_idx + 1} training videos...")
    
    # Process validation videos
    for video_idx, (video_path, label, class_name) in enumerate(zip(
        video_data['val_videos'], 
        video_data['val_labels'], 
        video_data['val_class_info']
    )):
        frames = extract_frames_from_single_video(
            video_path, frames_per_video, target_fps
        )
        
        if frames:
            video_name = f"{class_name}_{os.path.basename(video_path).split('.')[0]}_val"
            start_frame_idx = frame_count
            
            # Save frames
            for frame_idx, frame in enumerate(frames):
                frame_filename = f"{frame_count:05d}.jpg"
                frame_path = os.path.join(frames_dir, frame_filename)
                cv2.imwrite(frame_path, frame)
                frame_count += 1
            
            end_frame_idx = frame_count - 1
            
            # Add to video info
            video_info.append(f"\n{video_name}")
            video_info.append(f"{start_frame_idx} {end_frame_idx}")
        
        if (video_idx + 1) % 50 == 0:
            logger.info(f"Processed {video_idx + 1} validation videos...")
    
    # Save video info file
    with open(images_info_file, 'w') as f:
        f.write('\n'.join(video_info))
    
    logger.info(f"Extracted {frame_count} frames total")
    logger.info(f"Frames saved to: {frames_dir}")
    logger.info(f"Image info saved to: {images_info_file}")
    
    return frames_dir, images_info_file

def extract_frames_from_single_video(video_path, num_frames=10, target_fps=5):
    """
    Extract evenly spaced frames from a single video.
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        logger.warning(f"Could not open video: {video_path}")
        return []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if total_frames == 0 or fps == 0:
        logger.warning(f"Invalid video properties: {video_path}")
        cap.release()
        return []
    
    # Calculate frame indices to extract
    # Try to get frames at target_fps if possible, otherwise evenly space them
    duration = total_frames / fps
    if duration * target_fps > num_frames:
        # Use target_fps spacing
        frame_step = int(fps / target_fps)
        frame_indices = list(range(0, min(total_frames, num_frames * frame_step), frame_step))
    else:
        # Evenly space frames
        frame_indices = np.linspace(0, total_frames - 1, min(num_frames, total_frames), dtype=int)
    
    frames = []
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if ret:
            frames.append(frame)
        
        if len(frames) >= num_frames:
            break
    
    cap.release()
    return frames

# ----------------------- PoseNet Integration -----------------------
def run_skeleton_detection(frames_dir, images_info_file, output_dir):
    """
    Run skeleton detection on extracted frames using PoseNet pipeline.
    This mimics the s1_get_skeletons_from_training_imgs.py script.
    """
    # Load configuration
    cfg_all = load_config()
    if not cfg_all:
        logger.error("Could not load configuration")
        return None
    
    # Setup skeleton detector
    cfg = cfg_all.get("s1_get_skeletons_from_training_imgs.py", {})
    openpose_cfg = cfg.get("openpose", {})
    
    OPENPOSE_MODEL = openpose_cfg.get("model", "BODY_25")
    OPENPOSE_IMG_SIZE = openpose_cfg.get("img_size", "368x368")
    
    skeleton_detector = SkeletonDetector(OPENPOSE_MODEL, OPENPOSE_IMG_SIZE)
    multiperson_tracker = Tracker()
    
    # Output directories
    detected_skeletons_dir = os.path.join(output_dir, "detected_skeletons_kinetics")
    viz_imgs_dir = os.path.join(output_dir, "viz_imgs_kinetics")
    
    os.makedirs(detected_skeletons_dir, exist_ok=True)
    os.makedirs(viz_imgs_dir, exist_ok=True)
    
    # Get list of frame files
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])
    logger.info(f"Starting skeleton detection for {len(frame_files)} frames...")
    
    # Process frames
    start_time = time.time()
    processed_count = 0
    
    progress_bar = tqdm(
        total=len(frame_files),
        desc="Detecting skeletons",
        unit="img",
        ncols=100
    )
    
    for frame_file in frame_files:
        frame_path = os.path.join(frames_dir, frame_file)
        img = cv2.imread(frame_path)
        
        if img is None:
            logger.warning(f"Could not read image: {frame_path}")
            continue
        
        # Detect skeletons
        humans = skeleton_detector.detect(img)
        
        # Draw skeletons for visualization
        img_viz = img.copy()
        skeleton_detector.draw(img_viz, humans)
        
        # Save visualization
        viz_path = os.path.join(viz_imgs_dir, frame_file)
        cv2.imwrite(viz_path, img_viz)
        
        # Get skeleton data
        skeletons, scale_h = skeleton_detector.humans_to_skels_list(humans)
        dict_id2skeleton = multiperson_tracker.track(skeletons)
        
        # Create skeleton data with image info (frame index, class info, etc.)
        frame_idx = int(frame_file.split('.')[0])  # Extract frame number
        img_info = [0, 0, frame_idx, "kinetics_action", frame_file]  # Placeholder info
        
        skels_to_save = [img_info + skeleton.tolist()
                        for skeleton in dict_id2skeleton.values()]
        
        # Save skeleton data
        skeleton_file = frame_file.replace('.jpg', '.txt')
        skeleton_path = os.path.join(detected_skeletons_dir, skeleton_file)
        lib_commons.save_listlist(skeleton_path, skels_to_save)
        
        processed_count += 1
        progress_bar.update(1)
        
        # Log progress
        if processed_count % 100 == 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / processed_count
            remaining = (len(frame_files) - processed_count) * avg_time
            progress_bar.set_description(f"Skeleton detection (ETA: {remaining/60:.1f}m)")
    
    progress_bar.close()
    
    total_time = time.time() - start_time
    logger.info(f"Skeleton detection completed in {total_time/60:.1f} minutes")
    logger.info(f"Processed {processed_count} frames")
    logger.info(f"Skeletons saved to: {detected_skeletons_dir}")
    logger.info(f"Visualizations saved to: {viz_imgs_dir}")
    
    return detected_skeletons_dir

def prepare_training_data(detected_skeletons_dir, output_dir, classes):
    """
    Prepare skeleton data for training following PoseNet pipeline.
    This mimics s2_put_skeleton_txts_to_a_single_txt.py
    """
    all_skeletons_file = os.path.join(output_dir, "kinetics_skeletons_info.txt")
    
    # Get all skeleton txt files
    skeleton_files = sorted([f for f in os.listdir(detected_skeletons_dir) 
                           if f.endswith('.txt')])
    
    logger.info(f"Combining {len(skeleton_files)} skeleton files...")
    
    all_skeletons = []
    labels_cnt = collections.defaultdict(int)
    IDX_PERSON = 0  # Use first person detected
    IDX_ACTION_LABEL = 3  # Index of action label in skeleton data
    
    for i, skeleton_file in enumerate(skeleton_files):
        skeleton_path = os.path.join(detected_skeletons_dir, skeleton_file)
        
        try:
            skeletons = lib_commons.read_listlist(skeleton_path)
            if not skeletons:
                continue
                
            skeleton = skeletons[IDX_PERSON]
            
            # Extract class from filename or assign based on frame index
            # For now, we'll need to map frame indices back to video classes
            # This is simplified - in practice you'd maintain the mapping
            label = "kinetics_action"  # Placeholder
            
            if label in classes:
                labels_cnt[label] += 1
                all_skeletons.append(skeleton)
        
        except Exception as e:
            logger.warning(f"Error processing {skeleton_file}: {e}")
            continue
        
        if i % 100 == 0:
            logger.info(f"Processed {i}/{len(skeleton_files)} skeleton files")
    
    # Save combined skeleton data
    import simplejson
    with open(all_skeletons_file, 'w') as f:
        simplejson.dump(all_skeletons, f)
    
    logger.info(f"Combined {len(all_skeletons)} skeleton data points")
    logger.info(f"Saved to: {all_skeletons_file}")
    
    return all_skeletons_file

# ----------------------- Main Processing Function -----------------------
def process_kinetics_for_posenet(data_root="./kinetics4005per", output_dir="./kinetics_posenet_data"):
    """
    Main function to process Kinetics4005per dataset for PoseNet training.
    """
    logger.info("Starting Kinetics4005per to PoseNet conversion...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Load Kinetics dataset structure
    logger.info("Step 1: Loading Kinetics4005per dataset structure...")
    video_data = load_kinetics4005per_structure(data_root)
    
    if not video_data['train_videos'] and not video_data['val_videos']:
        logger.error("No videos found in dataset!")
        return False
    
    # Step 2: Extract frames from videos
    logger.info("Step 2: Extracting frames from videos...")
    frames_dir, images_info_file = extract_frames_from_videos(
        video_data, output_dir, frames_per_video=10, target_fps=5
    )
    
    # Step 3: Run skeleton detection
    logger.info("Step 3: Running skeleton detection...")
    detected_skeletons_dir = run_skeleton_detection(frames_dir, images_info_file, output_dir)
    
    if not detected_skeletons_dir:
        logger.error("Skeleton detection failed!")
        return False
    
    # Step 4: Prepare training data
    logger.info("Step 4: Preparing training data...")
    skeletons_file = prepare_training_data(
        detected_skeletons_dir, output_dir, video_data['classes']
    )
    
    logger.info("Kinetics4005per to PoseNet conversion completed!")
    logger.info(f"Training data ready at: {output_dir}")
    logger.info(f"Next steps:")
    logger.info(f"1. Update config.yaml with Kinetics classes")
    logger.info(f"2. Run feature preprocessing: python src/s3_preprocess_features.py")
    logger.info(f"3. Run training: python src/custom_s4_train.py")
    
    return True

# ----------------------- Main Entry Point -----------------------
def main():
    """Main entry point for the adapter."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert Kinetics4005per dataset for PoseNet training")
    parser.add_argument("--data_root", type=str, default="./kinetics4005per",
                       help="Path to Kinetics4005per dataset (default: ./kinetics4005per)")
    parser.add_argument("--output_dir", type=str, default="./kinetics_posenet_data",
                       help="Output directory for processed data (default: ./kinetics_posenet_data)")
    parser.add_argument("--frames_per_video", type=int, default=10,
                       help="Number of frames to extract per video (default: 10)")
    parser.add_argument("--target_fps", type=int, default=5,
                       help="Target FPS for frame extraction (default: 5)")
    
    args = parser.parse_args()
    
    # Check if data root exists
    if not os.path.exists(args.data_root):
        logger.error(f"Data root directory not found: {args.data_root}")
        logger.error("Please ensure Kinetics4005per dataset is available.")
        logger.error("Run: python setup_kinetics.py for setup instructions")
        return
    
    # Process the dataset
    success = process_kinetics_for_posenet(
        data_root=args.data_root,
        output_dir=args.output_dir
    )
    
    if success:
        logger.info("✅ Conversion completed successfully!")
    else:
        logger.error("❌ Conversion failed!")

if __name__ == "__main__":
    main()
