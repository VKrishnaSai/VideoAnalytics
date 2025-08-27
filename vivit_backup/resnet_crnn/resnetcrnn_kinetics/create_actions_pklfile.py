import os
import pickle
from pathlib import Path

def create_class_mapping_pkl(dataset_dir, output_pkl_path='class_mapping.pkl'):
    """
    Create a pickle file with class mappings from directory structure.
    
    Args:
        dataset_dir (str): Path to the dataset directory containing class folders
        output_pkl_path (str): Output path for the pickle file
    """
    
    # Get all subdirectories (class folders)
    dataset_path = Path(dataset_dir)
    
    if not dataset_path.exists():
        print(f"Error: Directory {dataset_dir} does not exist!")
        return
    
    # Get all directories (excluding files)
    class_folders = [d.name for d in dataset_path.iterdir() if d.is_dir()]
    class_folders.sort()  # Sort for consistent ordering
    
    # Create class to index mapping
    class_to_idx = {class_name: idx for idx, class_name in enumerate(class_folders)}
    idx_to_class = {idx: class_name for class_name, idx in class_to_idx.items()}
    
    # Create the mapping dictionary
    class_mapping = class_folders
    
    # Save to pickle file
    with open(output_pkl_path, 'wb') as f:
        pickle.dump(class_mapping, f)
    
    print(f"Class mapping saved to {output_pkl_path}")
    print(f"Found {len(class_folders)} classes:")
    for class_name, idx in class_to_idx.items():
        print(f"  {idx}: {class_name}")
    
    return class_mapping

if __name__ == "__main__":
    dataset_directory = "../kinetics4005per/train/train"
    
    output_file = "Kinetics400Actions.pkl" 
    # Create the class mapping
    mapping = create_class_mapping_pkl(dataset_directory, output_file)
