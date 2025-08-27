#!/usr/bin/env python3
"""
Test script for Kinetics PoseNet Adapter
Quick verification that the adapter can process the dataset structure.
"""

import os
import sys

# Include project path
ROOT = os.path.dirname(os.path.abspath(__file__)) + "/../"
sys.path.append(ROOT)

from src.kinetics_to_posenet_adapter import load_kinetics4005per_structure

def test_dataset_loading():
    """Test if the dataset can be loaded correctly."""
    print("Testing Kinetics4005per dataset loading...")
    
    # Test with default path
    data_root = "./kinetics4005per"
    
    if not os.path.exists(data_root):
        print(f"❌ Dataset not found at {data_root}")
        print("Please ensure your Kinetics4005per dataset is available.")
        print("Run: python setup_kinetics.py for setup instructions")
        return False
    
    try:
        video_data = load_kinetics4005per_structure(data_root)
        
        print(f"✅ Dataset loaded successfully!")
        print(f"   - {len(video_data['classes'])} classes found")
        print(f"   - {len(video_data['train_videos'])} training videos")
        print(f"   - {len(video_data['val_videos'])} validation videos")
        
        if len(video_data['classes']) > 0:
            print(f"   - Sample classes: {video_data['classes'][:5]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return False

def test_config_exists():
    """Test if config.yaml exists and has the right structure."""
    config_path = "./config/config.yaml"
    
    if not os.path.exists(config_path):
        print(f"⚠️  Config file not found: {config_path}")
        print("Run: python setup_kinetics.py to create it")
        return False
    
    try:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        if 'classes' in config:
            print(f"✅ Config file valid with {len(config['classes'])} classes")
            return True
        else:
            print(f"⚠️  Config file missing 'classes' field")
            return False
            
    except Exception as e:
        print(f"❌ Error reading config: {e}")
        return False

def main():
    print("Kinetics PoseNet Adapter Test")
    print("=" * 40)
    
    # Test dataset loading
    dataset_ok = test_dataset_loading()
    print()
    
    # Test config
    config_ok = test_config_exists()
    print()
    
    # Summary
    if dataset_ok and config_ok:
        print("✅ All tests passed! Ready to run the adapter.")
        print("Next step: python src/kinetics_to_posenet_adapter.py")
    else:
        print("❌ Some tests failed. Please check the setup.")
        if not dataset_ok:
            print("- Fix dataset structure and location")
        if not config_ok:
            print("- Run: python setup_kinetics.py")

if __name__ == "__main__":
    main()
