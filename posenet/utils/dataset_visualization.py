import sys
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
import random
# Set paths
ROOT = os.path.dirname(os.path.abspath(__file__)) + "/../"
CURR_PATH = os.path.dirname(os.path.abspath(__file__)) + "/"
sys.path.append(ROOT)
from utils import lib_commons
from utils.lib_skeletons_io import load_skeleton_data
# Load configuration and class names
cfg_all = lib_commons.read_yaml(ROOT + "config/config.yaml")
classes = cfg_all["classes"]
print(classes)

# Initialize class count dictionary
class_count = {cls: 0 for cls in classes}

# Load skeleton data
x_data, y_data, video_indice = load_skeleton_data(ROOT + 'data_proc/raw_skeletons/skeletons_info.txt', classes)
idx_to_label = {i: c for i, c in enumerate(classes)}

# Count occurrences of each class
for label in y_data:
    class_count[idx_to_label[label]] += 1

print(class_count)

# Compute total count
total_count = len(y_data)

def plot_data_distribution(class_count, total_count):
    # Create bar chart
    fig, ax = plt.subplots(figsize=(12, 6))  # Wider figure for better readability
    bars = ax.bar(class_count.keys(), class_count.values(), color='skyblue')

    # Set labels and title
    ax.set_xlabel('Classes')
    ax.set_ylabel('Count of Images')
    ax.set_title('Class vs Count of Images for Each Class')

    # Annotate bars with percentage
    for bar, count in zip(bars, class_count.values()):
        percentage = (count / total_count) * 100  # Compute percentage
        ax.text(
            bar.get_x() + bar.get_width() / 2,  # Center the text
            bar.get_height(),  # Position at top of bar
            f'{percentage:.1f}%',  # Format percentage
            ha='center', va='bottom', fontsize=10, fontweight='bold', color='black'
        )

    # Improve readability
    ax.set_xticklabels(class_count.keys(), rotation=45, ha="right")  # Rotate x-axis labels
    ax.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

def create_simple_action_grid(classes, save_path=None):
    """
    Create a simple 3x3 grid of action images for research paper.
    
    Args:
        classes: List of action class names
        save_path: Path to save the PNG file (optional)
    """
    # Ensure we have exactly 9 classes for 3x3 grid
    if len(classes) != 9:
        print(f"Warning: Expected 9 classes for 3x3 grid, got {len(classes)}")
        classes = (classes + [''] * 9)[:9]
    
    # Source images folder
    source_images_folder = ROOT + "data/source_images3/"
    
    # Collect (frame, class label) pairs
    frame_class_list = []
    
    for cls in classes:
        if not cls:
            continue
            
        # Find a folder that starts with this action class
        sample_image = None
        try:
            if os.path.exists(source_images_folder):
                folders = [f for f in os.listdir(source_images_folder) 
                          if os.path.isdir(os.path.join(source_images_folder, f)) 
                          and f.startswith(cls + '_')]
                
                if folders:
                    # Pick the first folder for this action
                    action_folder = folders[0]
                    folder_path = os.path.join(source_images_folder, action_folder)
                    
                    # Get all image files in the folder
                    image_files = [f for f in os.listdir(folder_path) 
                                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                    
                    if image_files:
                        # Sort files and pick middle image
                        image_files.sort()
                        mid_idx = len(image_files) // 2
                        sample_image_path = os.path.join(folder_path, image_files[mid_idx])
                        
                        # Load image
                        img = cv2.imread(sample_image_path)
                        if img is not None:
                            # Convert BGR to RGB for matplotlib
                            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            label = cls.capitalize()
                            frame_class_list.append((img_rgb, label))
                            continue
        except Exception as e:
            print(f"Error loading image for {cls}: {e}")
        
        # If no image found, create a simple placeholder
        placeholder = np.ones((200, 200, 3), dtype=np.uint8) * 128  # Gray placeholder
        label = cls.capitalize() if cls else "N/A"
        frame_class_list.append((placeholder, label))
    
    # Plot in 3×3 grid with borders and tight layout
    fig, axs = plt.subplots(3, 3, figsize=(10, 10))
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    
    for i, (frame, cls_label) in enumerate(frame_class_list):
        row, col = divmod(i, 3)
        axs[row, col].imshow(frame)
        axs[row, col].set_title("", fontsize=4)
        axs[row, col].set_xlabel(cls_label, fontsize=25)
        axs[row, col].xaxis.label.set_position((0.5, -0.05))
        axs[row, col].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        axs[row, col].spines['top'].set_visible(True)
        axs[row, col].spines['bottom'].set_visible(True)
        axs[row, col].spines['left'].set_visible(True)
        axs[row, col].spines['right'].set_visible(True)
        for edge in axs[row, col].spines.values():
            edge.set_linewidth(1.0)
    
    # Save as high-quality PNG
    if save_path is None:
        save_path = ROOT + 'results/action_grid_simple.png'
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Simple action grid saved to: {save_path}")
    plt.show()
    
    return save_path

# Call the existing data distribution plot
plot_data_distribution(class_count, total_count)

# Generate the 3x3 action labels grid with actual images
create_simple_action_grid(classes)

# Also create a version with custom save path
custom_save_path = ROOT + 'results/posenet_action_grid.png'
create_simple_action_grid(classes, custom_save_path)