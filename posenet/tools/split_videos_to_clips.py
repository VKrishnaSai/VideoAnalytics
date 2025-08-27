"""
Split videos into 10-second clips for ViViT training.
This tool takes videos from /output/videos and creates 10-second clips,
which helps increase dataset size and provides more manageable video lengths for training.
"""

import os
import cv2
import argparse
from pathlib import Path
import numpy as np

def get_video_info(video_path):
    """Get video information including duration, fps, and frame count."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    
    cap.release()
    return {
        'fps': fps,
        'frame_count': frame_count,
        'duration': duration
    }

def split_video_into_clips(input_video_path, output_dir, clip_duration=10, overlap=0):
    """
    Split a video into clips of specified duration.
    
    Args:
        input_video_path: Path to input video
        output_dir: Directory to save clips
        clip_duration: Duration of each clip in seconds (default: 10)
        overlap: Overlap between clips in seconds (default: 0)
    """
    video_info = get_video_info(input_video_path)
    if not video_info:
        print(f"Error: Could not read video {input_video_path}")
        return []
    
    fps = video_info['fps']
    total_duration = video_info['duration']
    
    print(f"Processing {input_video_path}")
    print(f"  Duration: {total_duration:.2f}s, FPS: {fps:.2f}")
    
    # If video is shorter than clip_duration, just copy it
    if total_duration <= clip_duration:
        print(f"  Video is shorter than {clip_duration}s, copying as single clip")
        return [input_video_path]
    
    cap = cv2.VideoCapture(input_video_path)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    
    clip_paths = []
    clip_num = 0
    
    # Calculate step size (frames to advance for next clip)
    step_frames = int((clip_duration - overlap) * fps)
    clip_frames = int(clip_duration * fps)
    
    start_frame = 0
    
    while start_frame + clip_frames <= video_info['frame_count']:
        # Create output filename
        video_name = Path(input_video_path).stem
        clip_filename = f"{video_name}_clip_{clip_num:03d}.avi"
        clip_path = os.path.join(output_dir, clip_filename)
        
        # Create video writer
        out = cv2.VideoWriter(clip_path, fourcc, fps, (width, height))
        
        # Set starting position
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        frames_written = 0
        while frames_written < clip_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            out.write(frame)
            frames_written += 1
        
        out.release()
        
        if frames_written > 0:
            clip_paths.append(clip_path)
            print(f"  Created clip {clip_num}: {clip_filename} ({frames_written} frames)")
        
        clip_num += 1
        start_frame += step_frames
    
    cap.release()
    return clip_paths

def process_all_videos(input_dir, output_dir, clip_duration=10, overlap=0, flat_structure=False):
    """
    Process all videos in the input directory and create clips.
    
    Args:
        input_dir: Directory containing input videos
        output_dir: Directory to save clips
        clip_duration: Duration of each clip in seconds
        overlap: Overlap between clips in seconds
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all video files
    video_extensions = ['.avi', '.mp4', '.mov', '.mkv', '.wmv']
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(input_path.glob(f"*{ext}"))
        video_files.extend(input_path.glob(f"*{ext.upper()}"))
    
    if not video_files:
        print(f"No video files found in {input_dir}")
        return
    
    print(f"Found {len(video_files)} video files")
    print(f"Output directory: {output_dir}")
    print(f"Clip duration: {clip_duration}s, Overlap: {overlap}s")
    print("-" * 50)
    
    all_clips = []
    action_stats = {}
    
    for video_file in video_files:
        # Extract action label from filename (assuming format: action_timestamp.avi)
        video_name = video_file.stem
        action = video_name.split('_')[0] if '_' in video_name else 'unknown'
        
        # Choose output directory based on structure preference
        if flat_structure:
            # Save all clips in the main output directory
            action_output_dir = output_path
        else:
            # Create action-specific subdirectory
            action_output_dir = output_path / action
            action_output_dir.mkdir(exist_ok=True)
        
        clips = split_video_into_clips(
            str(video_file), 
            str(action_output_dir), 
            clip_duration, 
            overlap
        )
        
        all_clips.extend(clips)
        
        if action not in action_stats:
            action_stats[action] = 0
        action_stats[action] += len(clips)
    
    print("-" * 50)
    print("Summary:")
    print(f"Total clips created: {len(all_clips)}")
    print("\nClips per action:")
    for action, count in sorted(action_stats.items()):
        print(f"  {action}: {count} clips")
    
    return all_clips

def main():
    parser = argparse.ArgumentParser(description="Split videos into clips for ViViT training")
    parser.add_argument(
        "--input_dir", 
        type=str, 
        default="../output/videos",
        help="Directory containing input videos (default: ../output/videos)"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="../output/video_clips",
        help="Directory to save clips (default: ../output/video_clips)"
    )
    parser.add_argument(
        "--clip_duration", 
        type=int, 
        default=10,
        help="Duration of each clip in seconds (default: 10)"
    )
    parser.add_argument(
        "--overlap", 
        type=float, 
        default=0,
        help="Overlap between clips in seconds (default: 0)"
    )
    parser.add_argument(
        "--flat_structure", 
        action="store_true",
        help="Save all clips in a flat directory structure instead of action subdirectories"
    )
    
    args = parser.parse_args()
    
    # Convert relative paths to absolute paths
    script_dir = Path(__file__).parent
    input_dir = script_dir / args.input_dir
    output_dir = script_dir / args.output_dir
    
    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist")
        return
    
    process_all_videos(
        str(input_dir), 
        str(output_dir), 
        args.clip_duration, 
        args.overlap,
        args.flat_structure
    )

if __name__ == "__main__":
    main()
