import os

def list_folders_to_classes_file(dataset_directory, output_file="truncated_classes.txt"):
    """
    List all folder names in a given directory and write them to truncated_classes.txt
    
    Args:
        dataset_directory (str): Path to the directory containing class folders
        output_file (str): Output file name (default: truncated_classes.txt)
    """
    
    # Check if the dataset directory exists
    if not os.path.exists(dataset_directory):
        print(f"Error: Directory '{dataset_directory}' does not exist!")
        return
    
    # Get all folders in the directory
    try:
        all_items = os.listdir(dataset_directory)
        folders = [item for item in all_items if os.path.isdir(os.path.join(dataset_directory, item))]
        
        # Sort folders alphabetically
        folders.sort()
        
        if not folders:
            print(f"No folders found in '{dataset_directory}'")
            return
        
        # Write to truncated_classes.txt
        output_path = os.path.join(os.path.dirname(__file__), output_file)
        
        with open(output_path, 'w') as f:
            for folder in folders:
                f.write(f"{folder}\n")
        
        print(f"Successfully wrote {len(folders)} class names to '{output_file}'")
        print(f"Output file location: {output_path}")
        print("\nClass names found:")
        for i, folder in enumerate(folders, 1):
            print(f"{i:2d}. {folder}")
            
    except Exception as e:
        print(f"Error reading directory: {e}")

def main():
    """
    Main function - modify the dataset_directory path below
    """
    
    # MODIFY THIS PATH TO YOUR DATASET DIRECTORY
    dataset_directory = input("Enter the path to your dataset directory: ").strip()
    
    # Remove quotes if user copied path with quotes
    dataset_directory = dataset_directory.strip('"').strip("'")
    
    # Call the function
    list_folders_to_classes_file(dataset_directory)

if __name__ == "__main__":
    main()