'''
Description:
    Convert each folder in data/source_images3 to video files using subprocess,
    but only using the frame ranges specified in valid_images.txt.
    
Example of usage:
    python convert_all_data_to_video.py

'''
import os
import subprocess
import sys
import shutil
import tempfile

ROOT = os.path.dirname(os.path.abspath(__file__)) + "/../"
TOOLS_PATH = os.path.dirname(os.path.abspath(__file__)) + "/"

def parse_valid_images_txt(valid_images_path):
    """
    Parse valid_images.txt to get frame ranges for each action folder.
    
    Returns:
        dict: {folder_name: [(start, end), (start, end), ...]}
    """
    folder_ranges = {}
    current_folder = None
    
    with open(valid_images_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:  # Empty line
                continue
            
            if '_' in line and len(line.split()) == 1:  # Folder name line
                current_folder = line
                if current_folder not in folder_ranges:
                    folder_ranges[current_folder] = []
            else:  # Frame range line
                if current_folder and len(line.split()) == 2:
                    try:
                        start, end = map(int, line.split())
                        folder_ranges[current_folder].append((start, end))
                    except ValueError:
                        continue
    
    return folder_ranges

def create_temp_folder_with_valid_frames(source_folder, frame_ranges, img_format="{:05d}.jpg"):
    """
    Create a temporary folder containing only the valid frame images.
    
    Args:
        source_folder: Path to the source folder with all images
        frame_ranges: List of (start, end) tuples for valid frames
        img_format: Format string for image filenames
    
    Returns:
        Path to temporary folder with valid frames
    """
    temp_dir = tempfile.mkdtemp()
    
    frame_count = 0
    for start, end in frame_ranges:
        for frame_num in range(start, end + 1):
            source_img = os.path.join(source_folder, img_format.format(frame_num))
            if os.path.exists(source_img):
                # Copy to temp folder with sequential naming
                dest_img = os.path.join(temp_dir, f"{frame_count:05d}.jpg")
                shutil.copy2(source_img, dest_img)
                frame_count += 1
    
    return temp_dir if frame_count > 0 else None

def convert_folders_to_videos():
    """
    Convert each folder in data/source_images3 to video files using only valid frame ranges.
    """
    # Source and output directories
    source_dir = os.path.join(ROOT, "data", "source_images3")
    output_dir = os.path.join(ROOT, "output", "videos")
    valid_images_path = os.path.join(source_dir, "valid_images.txt")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if source directory exists
    if not os.path.exists(source_dir):
        print(f"Error: Source directory does not exist: {source_dir}")
        return
    
    # Check if valid_images.txt exists
    if not os.path.exists(valid_images_path):
        print(f"Error: valid_images.txt not found: {valid_images_path}")
        return
    
    # Parse valid_images.txt
    print("Parsing valid_images.txt...")
    folder_ranges = parse_valid_images_txt(valid_images_path)
    
    if not folder_ranges:
        print("No valid folders found in valid_images.txt")
        return
    
    print(f"Found {len(folder_ranges)} folders with frame ranges:")
    for folder, ranges in folder_ranges.items():
        print(f"  - {folder}: {len(ranges)} range(s)")
        for start, end in ranges:
            print(f"    {start}-{end} ({end-start+1} frames)")
    
    successful_conversions = 0
    failed_conversions = 0
    
    # Convert each folder to video
    for folder_name, frame_ranges in folder_ranges.items():
        input_folder = os.path.join(source_dir, folder_name)
        output_video = os.path.join(output_dir, f"{folder_name}.avi")
        
        if not os.path.exists(input_folder):
            print(f"⚠ Folder not found: {input_folder}")
            failed_conversions += 1
            continue
        
        print(f"\nConverting {folder_name}...")
        print(f"  Using {len(frame_ranges)} frame range(s)")
        
        # Create temporary folder with only valid frames
        temp_folder = create_temp_folder_with_valid_frames(input_folder, frame_ranges)
        
        if temp_folder is None:
            print(f"✗ No valid frames found for {folder_name}")
            failed_conversions += 1
            continue
        
        try:
            # Build the command
            images2video_script = os.path.join(TOOLS_PATH, "images2video.py")
            cmd = [
                sys.executable,  # Use the same Python interpreter
                images2video_script,
                "-i", temp_folder,
                "-o", output_video,
                "--framerate", "15",  # Reasonable framerate for action videos
                "--sample_interval", "1"  # Use every frame
            ]
            
            # Run the conversion
            result = subprocess.run(cmd, 
                                  capture_output=True, 
                                  text=True, 
                                  cwd=TOOLS_PATH)
            
            if result.returncode == 0:
                print(f"✓ Successfully converted {folder_name} to {output_video}")
                successful_conversions += 1
            else:
                print(f"✗ Failed to convert {folder_name}")
                print(f"Error: {result.stderr}")
                failed_conversions += 1
                
        except Exception as e:
            print(f"✗ Exception while converting {folder_name}: {str(e)}")
            failed_conversions += 1
        finally:
            # Clean up temporary folder
            if temp_folder and os.path.exists(temp_folder):
                shutil.rmtree(temp_folder)
    
    # Summary
    print(f"\n{'='*50}")
    print(f"Conversion Summary:")
    print(f"  Successful: {successful_conversions}")
    print(f"  Failed: {failed_conversions}")
    print(f"  Total: {len(folder_ranges)}")
    print(f"Videos saved to: {output_dir}")

if __name__ == "__main__":
    print("Starting batch video conversion with valid frame ranges...")
    convert_folders_to_videos()
    print("Batch conversion completed!")
