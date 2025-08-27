#!/usr/bin/env python3
"""
Test script to verify PoseNet video data loading works correctly.
"""

import os
import sys

# Add parent directory to path to import train_vivit_on_feiyu
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_data_loading():
    """Test the data loading function"""
    print("Testing PoseNet data loading...")
    
    # Import the load_data function
    from train_vivit_on_feiyu import load_data
    
    # Test data loading
    data_root = "../output/videos"
    train_videos, train_labels, test_videos, test_labels, action_classes = load_data(data_root)
    
    print(f"\nData loading results:")
    print(f"Action classes: {action_classes}")
    print(f"Total training videos: {len(train_videos)}")
    print(f"Total test videos: {len(test_videos)}")
    
    # Print some example paths
    print(f"\nExample training videos:")
    for i, (video, label) in enumerate(zip(train_videos[:5], train_labels[:5])):
        action = action_classes[label]
        print(f"  {i+1}. {os.path.basename(video)} -> {action} (label: {label})")
    
    print(f"\nExample test videos:")
    for i, (video, label) in enumerate(zip(test_videos[:5], test_labels[:5])):
        action = action_classes[label]
        print(f"  {i+1}. {os.path.basename(video)} -> {action} (label: {label})")
    
    # Check if video files actually exist
    missing_train = [v for v in train_videos if not os.path.exists(v)]
    missing_test = [v for v in test_videos if not os.path.exists(v)]
    
    if missing_train:
        print(f"\nWarning: {len(missing_train)} training videos are missing:")
        for v in missing_train[:5]:  # Show first 5
            print(f"  - {v}")
    
    if missing_test:
        print(f"\nWarning: {len(missing_test)} test videos are missing:")
        for v in missing_test[:5]:  # Show first 5
            print(f"  - {v}")
    
    if not missing_train and not missing_test:
        print("\n✓ All video files exist!")
    
    print(f"\nData loading test completed!")

if __name__ == "__main__":
    test_data_loading()
