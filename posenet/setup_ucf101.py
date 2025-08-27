#!/usr/bin/env python3
"""
Script to help setup UCF-101 dataset for training.
This script provides instructions and checks for the required UCF-101 dataset structure.
"""

import os
import sys
import urllib.request
import zipfile
from pathlib import Path

def check_ucf101_structure():
    """Check if UCF-101 dataset is properly structured."""
    print("Checking UCF-101 dataset structure...")
    
    # Expected structure
    required_paths = [
        "./UCF-101",  # Main dataset directory
        "./ucfTrainTestlist",  # Train/test split files
        "./ucfTrainTestlist/classInd.txt",
        "./ucfTrainTestlist/trainlist01.txt",
        "./ucfTrainTestlist/trainlist02.txt", 
        "./ucfTrainTestlist/trainlist03.txt",
        "./ucfTrainTestlist/testlist01.txt",
        "./ucfTrainTestlist/testlist02.txt",
        "./ucfTrainTestlist/testlist03.txt"
    ]
    
    missing_paths = []
    for path in required_paths:
        if not os.path.exists(path):
            missing_paths.append(path)
        else:
            print(f"✓ Found: {path}")
    
    if missing_paths:
        print("\n❌ Missing required files/directories:")
        for path in missing_paths:
            print(f"  - {path}")
        return False
    else:
        print("\n✅ UCF-101 dataset structure is complete!")
        return True

def download_train_test_splits():
    """Download UCF-101 train/test split files."""
    print("\nDownloading UCF-101 train/test split files...")
    
    base_url = "https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip"
    zip_filename = "UCF101TrainTestSplits-RecognitionTask.zip"
    
    try:
        print(f"Downloading {base_url}...")
        urllib.request.urlretrieve(base_url, zip_filename)
        
        print("Extracting split files...")
        with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
            zip_ref.extractall(".")
        
        # Clean up
        os.remove(zip_filename)
        print("✅ Train/test split files downloaded successfully!")
        
    except Exception as e:
        print(f"❌ Error downloading split files: {e}")
        print("Please manually download from:")
        print("https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip")

def print_setup_instructions():
    """Print detailed setup instructions for UCF-101."""
    print("\n" + "="*60)
    print("UCF-101 Dataset Setup Instructions")
    print("="*60)
    
    print("\n1. Download UCF-101 Videos:")
    print("   URL: https://www.crcv.ucf.edu/data/UCF101/UCF101.rar")
    print("   - Extract to create ./UCF-101/ directory")
    print("   - Should contain 101 class directories (e.g., ApplyEyeMakeup, ApplyLipstick, etc.)")
    
    print("\n2. Download Train/Test Splits:")
    print("   URL: https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip")
    print("   - Extract to create ./ucfTrainTestlist/ directory")
    print("   - Should contain classInd.txt, trainlist01-03.txt, testlist01-03.txt")
    
    print("\n3. Expected Directory Structure:")
    print("   posenet/")
    print("   ├── UCF-101/")
    print("   │   ├── ApplyEyeMakeup/")
    print("   │   │   ├── v_ApplyEyeMakeup_g01_c01.avi")
    print("   │   │   └── ...")
    print("   │   ├── ApplyLipstick/")
    print("   │   └── ...")
    print("   ├── ucfTrainTestlist/")
    print("   │   ├── classInd.txt")
    print("   │   ├── trainlist01.txt")
    print("   │   ├── trainlist02.txt")
    print("   │   ├── trainlist03.txt")
    print("   │   ├── testlist01.txt")
    print("   │   ├── testlist02.txt")
    print("   │   └── testlist03.txt")
    print("   └── src/")
    print("       └── train_videomae_on_ucf.py")
    
    print("\n4. Dataset Statistics:")
    print("   - 101 action classes")
    print("   - ~13,320 videos total")
    print("   - ~9,537 training videos")
    print("   - ~3,783 test videos")
    print("   - Each split contains roughly the same distribution")

def count_videos():
    """Count videos in UCF-101 dataset if available."""
    ucf_path = "./UCF-101"
    if not os.path.exists(ucf_path):
        print(f"UCF-101 directory not found at {ucf_path}")
        return
    
    print(f"\nAnalyzing UCF-101 dataset at {ucf_path}...")
    
    total_videos = 0
    class_counts = {}
    
    for class_dir in os.listdir(ucf_path):
        class_path = os.path.join(ucf_path, class_dir)
        if os.path.isdir(class_path):
            video_files = [f for f in os.listdir(class_path) if f.endswith(('.avi', '.mp4', '.mov'))]
            video_count = len(video_files)
            class_counts[class_dir] = video_count
            total_videos += video_count
    
    print(f"Total classes: {len(class_counts)}")
    print(f"Total videos: {total_videos}")
    
    if class_counts:
        print(f"\nTop 10 classes by video count:")
        sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
        for i, (class_name, count) in enumerate(sorted_classes[:10]):
            print(f"  {i+1:2d}. {class_name}: {count} videos")

def main():
    print("UCF-101 Dataset Setup Helper")
    print("="*30)
    
    # Check current structure
    is_complete = check_ucf101_structure()
    
    if not is_complete:
        print_setup_instructions()
        
        # Offer to download split files
        response = input("\nDo you want to download the train/test split files automatically? (y/n): ")
        if response.lower() in ['y', 'yes']:
            download_train_test_splits()
            # Re-check after download
            check_ucf101_structure()
    
    # Count videos if dataset is available
    count_videos()
    
    print("\n" + "="*60)
    print("Next Steps:")
    if is_complete:
        print("✅ Your UCF-101 dataset is ready!")
        print("Run: python src/train_videomae_on_ucf.py")
        print("Optional arguments:")
        print("  --subset_classes 10    # Use only first 10 classes for testing")
        print("  --epochs 50           # Number of training epochs")
        print("  --batch_size 2        # Reduce if GPU memory is limited")
    else:
        print("❌ Please complete the UCF-101 dataset setup first")
        print("1. Download and extract UCF-101 videos")
        print("2. Download and extract train/test splits")
        print("3. Run this script again to verify")

if __name__ == "__main__":
    main()
