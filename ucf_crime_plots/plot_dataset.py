import os
import matplotlib.pyplot as plt
import numpy as np

def count_files_in_folders(directory_path):
    """
    Count the number of files in each subfolder of the given directory.
    
    Args:
        directory_path (str): Path to the directory containing class folders
        
    Returns:
        dict: Dictionary with folder names as keys and file counts as values
    """
    folder_file_counts = {}
    
    # Get all items in the directory
    items = os.listdir(directory_path)
    
    # Filter only directories (folders) and exclude hidden/system folders
    folders = [item for item in items if os.path.isdir(os.path.join(directory_path, item)) 
               and not item.startswith('.') and item not in ['.venv', '.qodo']]
    
    # Count files in each folder
    for folder in folders:
        folder_path = os.path.join(directory_path, folder)
        try:
            # Count only files (not subdirectories)
            file_count = len([f for f in os.listdir(folder_path) 
                            if os.path.isfile(os.path.join(folder_path, f))])
            folder_file_counts[folder] = file_count
        except PermissionError:
            print(f"Permission denied for folder: {folder}")
            folder_file_counts[folder] = 0
    
    return folder_file_counts

def plot_dataset_distribution(folder_counts, title="Dataset Distribution by Class"):
    """
    Create a bar plot showing the number of files in each class folder.
    
    Args:
        folder_counts (dict): Dictionary with folder names and file counts
        title (str): Title for the plot
    """
    # Sort folders by file count for better visualization
    sorted_folders = sorted(folder_counts.items(), key=lambda x: x[1], reverse=True)
    
    class_names = [item[0] for item in sorted_folders]
    file_counts = [item[1] for item in sorted_folders]
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    bars = plt.bar(class_names, file_counts, color='skyblue', edgecolor='navy', alpha=0.7)
    
    # Customize the plot
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Class Names', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Files', fontsize=12, fontweight='bold')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on top of each bar
    for bar, count in zip(bars, file_counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                str(count), ha='center', va='bottom', fontweight='bold')
    
    # Add grid for better readability
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Display statistics
    total_files = sum(file_counts)
    avg_files = total_files / len(file_counts) if file_counts else 0
    
    print(f"\nDataset Statistics:")
    print(f"Total number of classes: {len(class_names)}")
    print(f"Total number of files: {total_files}")
    print(f"Average files per class: {avg_files:.2f}")
    print(f"Class with most files: {class_names[0]} ({file_counts[0]} files)")
    print(f"Class with least files: {class_names[-1]} ({file_counts[-1]} files)")
    
    return plt

def main():
    """
    Main function to execute the dataset plotting.
    """
    # Get current directory
    current_directory = os.getcwd()
    print(f"Analyzing dataset in: {current_directory}")
    
    # Count files in each folder
    folder_counts = count_files_in_folders(current_directory)
    
    if not folder_counts:
        print("No folders found in the current directory.")
        return
    
    # Display folder counts
    print("\nFile counts by class:")
    for folder, count in sorted(folder_counts.items()):
        print(f"{folder}: {count} files")
    
    # Create and display the plot
    plt = plot_dataset_distribution(folder_counts, "UCF Crime Dataset - File Distribution by Class")
    
    # Show the plot
    plt.show()
    
    # Optionally save the plot
    save_plot = input("\nDo you want to save the plot? (y/n): ").lower().strip()
    if save_plot == 'y':
        filename = "dataset_distribution.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved as {filename}")

if __name__ == "__main__":
    main()