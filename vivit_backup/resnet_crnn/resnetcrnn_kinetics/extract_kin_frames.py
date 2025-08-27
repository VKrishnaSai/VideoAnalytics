import os
import cv2
from pathlib import Path
import argparse
from tqdm import tqdm
import numpy as np

def extract_frames_from_video(video_path, output_dir, target_frames=13):
    """
    Extract exactly target_frames from a video file uniformly distributed across the video duration.
    
    Args:
        video_path (str): Path to the video file
        output_dir (str): Directory to save extracted frames
        target_frames (int): Exact number of frames to extract (default: 13)
    
    Returns:
        int: Number of frames extracted
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return 0
    
    # Get total frame count
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        print(f"Error: Video {video_path} has 0 frames")
        cap.release()
        return 0
    
    # Calculate which frames to extract (uniformly distributed)
    if total_frames <= target_frames:
        # If video has fewer or equal frames than target, extract all available frames
        frame_indices = list(range(total_frames))
        print(f"Video {video_path} has only {total_frames} frames, extracting all")
    else:
        # Sample uniformly across the video
        frame_indices = np.linspace(0, total_frames - 1, target_frames, dtype=int)
        # Remove duplicates while preserving order (shouldn't happen with 13 frames, but safety check)
        frame_indices = sorted(list(set(frame_indices)))
    
    extracted_count = 0
    
    # Extract only the selected frames
    for i, frame_idx in enumerate(frame_indices):
        # Set video position to the desired frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            print(f"Warning: Could not read frame {frame_idx} from {video_path}")
            continue
        
        # Save frame as JPEG with sequential naming
        frame_filename = f"frame_{i+1:03d}.jpg"
        frame_path = os.path.join(output_dir, frame_filename)
        
        # Resize frame to 256x256
        frame_resized = cv2.resize(frame, (256, 256))
        
        # Save frame
        cv2.imwrite(frame_path, frame_resized)
        extracted_count += 1
    
    cap.release()
    return extracted_count

def convert_train_videos_to_frames(train_dir, jpegs_dir, target_frames_per_video=13, video_extensions=None):
    """
    Convert all videos in train directory to frames in jpegs_256 structure.
    Each video will have exactly target_frames_per_video frames extracted uniformly.
    
    Args:
        train_dir (str): Path to train directory containing class folders with videos
        jpegs_dir (str): Path to jpegs_256 directory to create frame folders
        target_frames_per_video (int): Exact number of frames to extract per video (default: 13)
        video_extensions (list): List of video file extensions to process
    """
    
    if video_extensions is None:
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']
    
    train_path = Path(train_dir)
    jpegs_path = Path(jpegs_dir)
    
    if not train_path.exists():
        print(f"Error: Train directory {train_dir} does not exist!")
        return
    
    # Create jpegs directory if it doesn't exist
    jpegs_path.mkdir(exist_ok=True)
    
    # Get all class directories
    class_dirs = [d for d in train_path.iterdir() if d.is_dir()]
    
    total_videos = 0
    total_frames = 0
    videos_with_fewer_frames = 0
    frame_counts = []
    
    print(f"Found {len(class_dirs)} classes in {train_dir}")
    print(f"Converting videos to frames in {jpegs_dir}")
    print(f"Target frames per video: {target_frames_per_video}")
    print("=" * 60)
    
    for class_dir in class_dirs:
        class_name = class_dir.name
        print(f"\nProcessing class: {class_name}")
        
        # Create class directory in jpegs folder
        class_jpegs_dir = jpegs_path / class_name
        class_jpegs_dir.mkdir(exist_ok=True)
        
        # Get all video files in this class directory
        video_files = []
        for ext in video_extensions:
            video_files.extend(list(class_dir.glob(f"*{ext}")))
            video_files.extend(list(class_dir.glob(f"*{ext.upper()}")))
        
        if not video_files:
            print(f"  No video files found in {class_dir}")
            continue
        
        print(f"  Found {len(video_files)} videos")
        
        # Process each video file
        for video_file in tqdm(video_files, desc=f"  {class_name}"):
            video_name = video_file.stem  # filename without extension
            
            # Create directory for this video's frames
            video_frames_dir = class_jpegs_dir / video_name
            
            # Skip if already processed
            if video_frames_dir.exists() and len(list(video_frames_dir.glob("*.jpg"))) > 0:
                existing_frames = len(list(video_frames_dir.glob("*.jpg")))
                if existing_frames == target_frames_per_video:
                    continue
                else:
                    print(f"    Re-processing {video_name} (has {existing_frames}, need {target_frames_per_video})")
                    # Remove existing frames to re-extract
                    for frame_file in video_frames_dir.glob("*.jpg"):
                        frame_file.unlink()
            
            # Check video frame count first
            cap = cv2.VideoCapture(str(video_file))
            if cap.isOpened():
                video_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                frame_counts.append(video_total_frames)
                
                if video_total_frames < target_frames_per_video:
                    videos_with_fewer_frames += 1
            
            # Extract frames
            frames_extracted = extract_frames_from_video(
                str(video_file), 
                str(video_frames_dir), 
                target_frames_per_video
            )
            
            if frames_extracted > 0:
                total_videos += 1
                total_frames += frames_extracted
            else:
                print(f"    Failed to process {video_name}")
                # Remove empty directory
                if video_frames_dir.exists():
                    video_frames_dir.rmdir()
    
    print("=" * 60)
    print(f"Conversion completed!")
    print(f"Total videos processed: {total_videos}")
    print(f"Total frames extracted: {total_frames}")
    print(f"Videos with fewer than {target_frames_per_video} frames: {videos_with_fewer_frames}")
    if frame_counts:
        print(f"Frame count statistics:")
        print(f"  Min frames in dataset: {min(frame_counts)}")
        print(f"  Max frames in dataset: {max(frame_counts)}")
        print(f"  Average frames: {np.mean(frame_counts):.1f}")
    print(f"Frames saved to: {jpegs_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert training videos to uniformly sampled frames")
    parser.add_argument("--train_dir", type=str, default="../kinetics4005per/train/train", 
                       help="Path to train directory containing class folders with videos")
    parser.add_argument("--jpegs_dir", type=str, default="./jpegs_256", 
                       help="Path to output jpegs directory")
    parser.add_argument("--target_frames", type=int, default=13, 
                       help="Exact number of frames to extract per video (default: 13)")
    parser.add_argument("--video_ext", nargs="+", default=['.mp4', '.avi', '.mov', '.mkv'], 
                       help="Video file extensions to process")
    
    args = parser.parse_args()
    
    # Run conversion
    convert_train_videos_to_frames(
        train_dir=args.train_dir,
        jpegs_dir=args.jpegs_dir,
        target_frames_per_video=args.target_frames,
        video_extensions=args.video_ext
    )
