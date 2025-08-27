#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to combine training plots into a single 2x2 grid:
- Top left: Training/Validation Accuracy
- Top right: Validation Precision & Recall  
- Bottom left: Validation F1 Score
- Bottom right: Training/Validation Loss
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import numpy as np

def combine_training_plots(results_dir="results", output_name="combined_training_plots.png"):
    """
    Combine existing training plots into a 2x2 grid.
    
    Args:
        results_dir: Directory containing the individual plot files
        output_name: Name of the output combined plot
    """
    
    # Define the plot files and their positions
    plot_files = {
        'accuracy_curve.png': (0, 0),      # Top left
        'precision_recall_curve.png': (0, 1),  # Top right  
        'f1_curve.png': (1, 0),           # Bottom left
        'loss_curve.png': (1, 1)          # Bottom right
    }
    
    # Check if results directory exists
    if not os.path.exists(results_dir):
        print(f"Error: Results directory '{results_dir}' not found!")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    # fig.suptitle('Training Progress Overview', fontsize=20, fontweight='bold', y=0.95)
    
    # Load and display each plot
    for filename, (row, col) in plot_files.items():
        filepath = os.path.join(results_dir, filename)
        
        if os.path.exists(filepath):
            try:
                # Load image
                img = mpimg.imread(filepath)
                
                # Display image in subplot
                axes[row, col].imshow(img)
                axes[row, col].axis('off')  # Remove axes
                
                # Add title based on filename
                if 'accuracy' in filename:
                    title = 'Training & Validation Accuracy'
                elif 'precision_recall' in filename:
                    title = 'Validation Precision & Recall'
                elif 'f1' in filename:
                    title = 'Validation F1 Score'
                elif 'loss' in filename:
                    title = 'Training & Validation Loss'
                else:
                    title = filename.replace('.png', '').replace('_', ' ').title()
                
                # axes[row, col].set_title(title, fontsize=14, fontweight='bold', pad=10)
                
                print(f"✓ Loaded: {filename}")
                
            except Exception as e:
                print(f"✗ Error loading {filename}: {e}")
                axes[row, col].text(0.5, 0.5, f'Error loading\n{filename}', 
                                  ha='center', va='center', transform=axes[row, col].transAxes,
                                  fontsize=12, color='red')
                axes[row, col].set_xlim(0, 1)
                axes[row, col].set_ylim(0, 1)
        else:
            print(f"✗ File not found: {filepath}")
            axes[row, col].text(0.5, 0.5, f'File not found:\n{filename}', 
                              ha='center', va='center', transform=axes[row, col].transAxes,
                              fontsize=12, color='red')
            axes[row, col].set_xlim(0, 1) 
            axes[row, col].set_ylim(0, 1)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, hspace=0.1, wspace=0.05)
    
    # Save combined plot
    output_path = os.path.join(results_dir, output_name)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"\n🎉 Combined plot saved as: {output_path}")
    print(f"📊 Resolution: High quality (300 DPI)")

def main():
    """Main function to run the plot combination."""
    
    # You can modify these paths as needed
    RESULTS_DIR = "./results"  # Relative to src folder
    OUTPUT_NAME = "combined_training_plots.png"
    
    print("🔄 Combining training plots...")
    print(f"📁 Looking in directory: {os.path.abspath(RESULTS_DIR)}")
    
    combine_training_plots(RESULTS_DIR, OUTPUT_NAME)

if __name__ == "__main__":
    main()
