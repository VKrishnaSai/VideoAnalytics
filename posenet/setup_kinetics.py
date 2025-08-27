#!/usr/bin/env python3
"""
Script to help setup Kinetics4005per dataset for training.
This script provides instructions and checks for the required Kinetics4005per dataset structure.
"""

import os
import sys
import yaml
from pathlib import Path

def check_kinetics4005per_structure():
    """Check if Kinetics4005per dataset is properly structured."""
    print("Checking Kinetics4005per dataset structure...")
    
    # Expected structure
    required_paths = [
        "./kinetics4005per",  # Main dataset directory
        "./kinetics4005per/train",  # Train directory
        "./kinetics4005per/train/train",  # Actual train data
        "./kinetics4005per/val",  # Validation directory
        "./kinetics4005per/val/val",  # Actual validation data
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
        print("\n✅ Kinetics4005per dataset structure is complete!")
        return True

def count_videos_and_classes():
    """Count videos and classes in Kinetics4005per dataset if available."""
    kinetics_path = "./kinetics4005per"
    if not os.path.exists(kinetics_path):
        print(f"Kinetics4005per directory not found at {kinetics_path}")
        return None, None
    
    print(f"\nAnalyzing Kinetics4005per dataset at {kinetics_path}...")
    
    train_path = os.path.join(kinetics_path, "train", "train")
    val_path = os.path.join(kinetics_path, "val", "val")
    
    train_stats = {"total_videos": 0, "classes": {}}
    val_stats = {"total_videos": 0, "classes": {}}
    
    # Analyze training data
    if os.path.exists(train_path):
        print(f"\nAnalyzing training data in {train_path}...")
        for class_dir in os.listdir(train_path):
            class_path = os.path.join(train_path, class_dir)
            if os.path.isdir(class_path):
                video_files = [f for f in os.listdir(class_path) 
                              if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
                video_count = len(video_files)
                train_stats["classes"][class_dir] = video_count
                train_stats["total_videos"] += video_count
    
    # Analyze validation data
    if os.path.exists(val_path):
        print(f"Analyzing validation data in {val_path}...")
        for class_dir in os.listdir(val_path):
            class_path = os.path.join(val_path, class_dir)
            if os.path.isdir(class_path):
                video_files = [f for f in os.listdir(class_path) 
                              if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
                video_count = len(video_files)
                val_stats["classes"][class_dir] = video_count
                val_stats["total_videos"] += video_count
    
    # Get all unique classes
    all_classes = sorted(list(set(train_stats["classes"].keys()) | set(val_stats["classes"].keys())))
    
    print(f"\nDataset Statistics:")
    print(f"Total classes: {len(all_classes)}")
    print(f"Training videos: {train_stats['total_videos']}")
    print(f"Validation videos: {val_stats['total_videos']}")
    print(f"Total videos: {train_stats['total_videos'] + val_stats['total_videos']}")
    
    if all_classes:
        print(f"\nFirst 10 classes (alphabetically):")
        for i, class_name in enumerate(all_classes[:10]):
            train_count = train_stats["classes"].get(class_name, 0)
            val_count = val_stats["classes"].get(class_name, 0)
            print(f"  {i+1:2d}. {class_name}: {train_count} train, {val_count} val")
        
        if len(all_classes) > 10:
            print(f"  ... and {len(all_classes) - 10} more classes")
    
    return all_classes, {"train": train_stats, "val": val_stats}

def update_config_yaml(classes):
    """Update config.yaml with Kinetics4005per classes for PoseNet training."""
    config_path = "./config/config.yaml"
    
    if not os.path.exists("./config"):
        os.makedirs("./config")
        print("Created config directory")
    
    # Create backup if config exists
    if os.path.exists(config_path):
        backup_path = config_path + ".backup"
        import shutil
        shutil.copy2(config_path, backup_path)
        print(f"Backed up existing config to {backup_path}")
    
    # Create or update config for PoseNet training
    config = {
        "dataset": "kinetics4005per",
        "classes": classes,
        
        # PoseNet pipeline configuration
        "image_filename_format": "{:05d}.jpg",
        "skeleton_filename_format": "{:05d}.txt",
        
        "features": {
            "window_size": 5  # Number of frames for time-series features
        },
        
        # Stage 1: Skeleton detection from images
        "s1_get_skeletons_from_training_imgs.py": {
            "input": {
                "images_description_txt": "kinetics_posenet_data/kinetics_images_info.txt",
                "images_folder": "kinetics_posenet_data/source_images_kinetics/"
            },
            "output": {
                "images_info_txt": "kinetics_posenet_data/kinetics_images_info_processed.txt",
                "detected_skeletons_folder": "kinetics_posenet_data/detected_skeletons_kinetics/",
                "viz_imgs_folders": "kinetics_posenet_data/viz_imgs_kinetics/"
            },
            "openpose": {
                "model": "BODY_25",
                "img_size": "368x368"
            }
        },
        
        # Stage 2: Combine skeleton txt files
        "s2_put_skeleton_txts_to_a_single_txt.py": {
            "input": {
                "detected_skeletons_folder": "kinetics_posenet_data/detected_skeletons_kinetics/"
            },
            "output": {
                "all_skeletons_txt": "kinetics_posenet_data/kinetics_skeletons_info.txt"
            }
        },
        
        # Stage 3: Preprocess features
        "s3_preprocess_features.py": {
            "input": {
                "all_skeletons_txt": "kinetics_posenet_data/kinetics_skeletons_info.txt"
            },
            "output": {
                "processed_features": "data_proc/kinetics_features_X.csv",
                "processed_features_labels": "data_proc/kinetics_features_Y.csv"
            }
        },
        
        # Stage 4: Training
        "s4_train.py": {
            "input": {
                "processed_features": "data_proc/kinetics_features_X.csv",
                "processed_features_labels": "data_proc/kinetics_features_Y.csv"
            },
            "output": {
                "model_path": "model/kinetics_trained_classifier.pickle"
            }
        },
        
        # Kinetics adapter settings
        "kinetics_adapter": {
            "data_root": "./kinetics4005per",
            "train_path": "./kinetics4005per/train/train",
            "val_path": "./kinetics4005per/val/val",
            "frames_per_video": 10,
            "target_fps": 5,
            "output_dir": "./kinetics_posenet_data"
        }
    }
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"✅ Updated config.yaml with {len(classes)} Kinetics4005per classes for PoseNet training")
    print(f"Classes: {classes[:5]}..." if len(classes) > 5 else f"Classes: {classes}")
    return config_path

def print_setup_instructions():
    """Print detailed setup instructions for Kinetics4005per."""
    print("\n" + "="*60)
    print("Kinetics4005per Dataset Setup Instructions")
    print("="*60)
    
    print("\n1. Dataset Structure:")
    print("   Your Kinetics4005per dataset should be organized as:")
    print("   kinetics4005per/")
    print("   ├── train/")
    print("   │   └── train/")
    print("   │       ├── class1/")
    print("   │       │   ├── video1.mp4")
    print("   │       │   └── video2.mp4")
    print("   │       ├── class2/")
    print("   │       └── ...")
    print("   └── val/")
    print("       └── val/")
    print("           ├── class1/")
    print("           │   ├── video3.mp4")
    print("           │   └── video4.mp4")
    print("           ├── class2/")
    print("           └── ...")
    
    print("\n2. Data Split Strategy:")
    print("   - Training data: kinetics4005per/train/train/")
    print("   - This will be split 80/20 into train/validation")
    print("   - Test data: kinetics4005per/val/val/")
    print("   - This matches the structure from kinetics_faster.py")
    
    print("\n3. Supported Video Formats:")
    print("   - .mp4, .avi, .mov, .mkv")
    print("   - Videos should be in reasonable quality for action recognition")
    
    print("\n4. Expected Dataset Size:")
    print("   - Kinetics4005per typically contains a subset of Kinetics-400")
    print("   - Should have multiple action classes")
    print("   - Each class should have multiple video samples")

def create_requirements_file():
    """Create requirements file for Kinetics4005per training."""
    requirements_content = """# Requirements for Kinetics4005per VideoMAE Training
# Install with: pip install -r requirements_kinetics4005per.txt

# Core ML frameworks
torch>=1.9.0
torchvision>=0.10.0

# Hugging Face transformers for VideoMAE
transformers>=4.20.0
huggingface_hub>=0.10.0

# Computer vision
opencv-python>=4.5.0

# Machine learning utilities
scikit-learn>=1.0.0
numpy>=1.21.0

# Visualization
matplotlib>=3.5.0
seaborn>=0.11.0

# Progress bars
tqdm>=4.60.0

# YAML for config files
PyYAML>=6.0

# Optional: For better performance
# accelerate>=0.12.0

# Optional: For Weights & Biases logging
# wandb>=0.12.0

# Optional: For TensorBoard logging
# tensorboard>=2.8.0"""
    
    requirements_path = "./requirements_kinetics4005per.txt"
    with open(requirements_path, 'w') as f:
        f.write(requirements_content)
    
    print(f"✅ Created {requirements_path}")
    return requirements_path

def main():
    print("Kinetics4005per Dataset Setup Helper")
    print("="*40)
    
    # Check current structure
    is_complete = check_kinetics4005per_structure()
    
    if not is_complete:
        print_setup_instructions()
        print("\n" + "="*60)
        print("Please ensure your Kinetics4005per dataset follows the expected structure.")
        print("Then run this script again to proceed with setup.")
        return
    
    # Count videos and get classes
    classes, stats = count_videos_and_classes()
    
    if classes and len(classes) > 0:
        print(f"\n📊 Dataset Analysis Complete!")
        print(f"Found {len(classes)} classes in the dataset")
        
        # Update config.yaml
        response = input(f"\nDo you want to update config.yaml with these {len(classes)} classes? (y/n): ")
        if response.lower() in ['y', 'yes']:
            config_path = update_config_yaml(classes)
            print(f"Config updated: {config_path}")
        
        # Create requirements file
        response = input("\nDo you want to create a requirements file for this dataset? (y/n): ")
        if response.lower() in ['y', 'yes']:
            create_requirements_file()
    
    print("\n" + "="*60)
    print("Next Steps:")
    if is_complete and classes:
        print("✅ Your Kinetics4005per dataset is ready!")
        print("Run the PoseNet training pipeline:")
        print("")
        print("1. Convert videos to skeleton data:")
        print("   python src/kinetics_to_posenet_adapter.py")
        print("")
        print("2. Extract features from skeletons:")
        print("   python src/s3_preprocess_features.py")
        print("")
        print("3. Train PoseNet classifier:")
        print("   python src/custom_s4_train.py")
        print("")
        print("Optional arguments for step 1:")
        print("  --frames_per_video 10     # Frames to extract per video")
        print("  --target_fps 5            # Target FPS for frame extraction")
        print("  --output_dir kinetics_posenet_data  # Output directory")
        
        if len(classes) > 50:
            print(f"\n💡 Large dataset detected ({len(classes)} classes)")
            print("Consider:")
            print("  - Processing a subset first for testing")
            print("  - Using sufficient disk space for frame extraction")
            print("  - Ensuring OpenPose is properly installed")
    else:
        print("❌ Please complete the Kinetics4005per dataset setup first")
        print("1. Ensure proper directory structure")
        print("2. Place video files in appropriate class directories")
        print("3. Run this script again to verify")
    
    print("\n🔧 Additional Tools:")
    print("- Check dataset structure: python setup_kinetics.py")
    print("- PoseNet pipeline will create timestamped directories")
    print("- Training logs and models will be saved automatically")

if __name__ == "__main__":
    main()
