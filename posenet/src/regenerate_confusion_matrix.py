#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to regenerate confusion matrix with smaller size and no colorbar
without retraining the model.
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from sklearn.metrics import confusion_matrix

# Include project path
ROOT = os.path.dirname(os.path.abspath(__file__)) + "/../"
sys.path.append(ROOT)

import utils.lib_commons as lib_commons

def load_model_and_data():
    """Load the trained model and test data."""
    
    # Load config
    cfg_all = lib_commons.read_yaml(ROOT + "config/config.yaml")
    cfg = cfg_all["s4_train.py"]
    CLASSES = np.array(cfg_all["classes"])
    
    # Paths
    SRC_PROCESSED_FEATURES = ROOT + cfg["input"]["processed_features"]
    SRC_PROCESSED_FEATURES_LABELS = ROOT + cfg["input"]["processed_features_labels"]
    DST_MODEL_PATH = ROOT + cfg["output"]["model_path"]
    
    print("Loading data and model...")
    
    # Load data
    X = np.loadtxt(SRC_PROCESSED_FEATURES, dtype=float)
    Y = np.loadtxt(SRC_PROCESSED_FEATURES_LABELS, dtype=int)
    
    # Load trained model
    try:
        with open(DST_MODEL_PATH, 'rb') as f:
            model, pca = pickle.load(f)
        print("✓ Model loaded successfully")
    except FileNotFoundError:
        print(f"✗ Model file not found: {DST_MODEL_PATH}")
        return None, None, None, None, None
    
    # Split data the same way as training (reproduce the same test set)
    from sklearn.model_selection import train_test_split
    
    # First split: train/test
    tr_X, te_X, tr_Y, te_Y = train_test_split(
        X, Y, test_size=0.3, random_state=1, stratify=Y)
    
    # Transform test data with PCA
    te_X_new = pca.transform(te_X)
    
    # Get predictions
    te_Y_pred = model.predict(te_X_new)
    
    print(f"✓ Test set size: {len(te_Y)} samples")
    print(f"✓ Predictions generated")
    
    return te_Y, te_Y_pred, CLASSES, model, pca

def create_compact_confusion_matrix(te_Y, te_Y_pred, CLASSES, save_path, 
                                  figsize=(8, 6), dpi=150):
    """
    Create a compact confusion matrix without colorbar.
    
    Args:
        te_Y: True labels
        te_Y_pred: Predicted labels  
        CLASSES: Class names array
        save_path: Path to save the plot
        figsize: Figure size tuple
        dpi: Resolution (lower = smaller file)
    """
    
    # Get unique classes present in data
    unique_classes_in_data = np.unique(np.concatenate([te_Y, te_Y_pred]))
    class_names_present = [CLASSES[i] for i in unique_classes_in_data]
    
    # Create confusion matrix
    cm = confusion_matrix(te_Y, te_Y_pred, labels=unique_classes_in_data)
    
    # Calculate accuracy for title
    accuracy = np.sum(te_Y == te_Y_pred) / len(te_Y)
    
    print(f"Creating compact confusion matrix...")
    print(f"Classes: {len(class_names_present)}")
    print(f"Accuracy: {accuracy:.3f}")
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Create heatmap without colorbar and annotations for readability
    sns.heatmap(cm, 
                annot=False,  # No numbers to keep it clean
                cmap='Blues', 
                xticklabels=class_names_present, 
                yticklabels=class_names_present,
                cbar=False,   # No colorbar
                square=True,  # Square cells
                linewidths=0.1,  # Thin grid lines
                linecolor='white')
    
    # Styling
    plt.xlabel('Predicted Class', fontsize=11, fontweight='bold')
    plt.ylabel('True Class', fontsize=11, fontweight='bold')
    
    # Rotate labels for better readability
    plt.xticks(rotation=90, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    
    # Tight layout
    plt.tight_layout()
    
    # Save with compression
    plt.savefig(save_path, 
                dpi=dpi, 
                bbox_inches='tight', 
                facecolor='white',
                edgecolor='none',
                format='png')
    
    plt.close()
    
    # Get file size
    file_size = os.path.getsize(save_path) / 1024  # KB
    print(f"✓ Saved: {save_path}")
    print(f"✓ File size: {file_size:.1f} KB")
    print(f"✓ Resolution: {dpi} DPI")

def create_mini_confusion_matrix(te_Y, te_Y_pred, CLASSES, save_path):
    """Create an extra small confusion matrix for thumbnails."""
    
    unique_classes_in_data = np.unique(np.concatenate([te_Y, te_Y_pred]))
    class_names_present = [CLASSES[i] for i in unique_classes_in_data]
    cm = confusion_matrix(te_Y, te_Y_pred, labels=unique_classes_in_data)
    accuracy = np.sum(te_Y == te_Y_pred) / len(te_Y)
    
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, 
                annot=False,
                cmap='Blues', 
                xticklabels=False,  # No labels for mini version
                yticklabels=False,
                cbar=False,
                square=True)
    
    plt.xlabel('Predicted', fontsize=9)
    plt.ylabel('True', fontsize=9)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=100, bbox_inches='tight', 
                facecolor='white')
    plt.close()
    
    file_size = os.path.getsize(save_path) / 1024
    print(f"✓ Mini version saved: {save_path} ({file_size:.1f} KB)")

def main():
    """Main function."""
    
    print("🔄 Regenerating confusion matrix...")
    
    # Load model and data
    te_Y, te_Y_pred, CLASSES, model, pca = load_model_and_data()
    
    if te_Y is None:
        print("❌ Failed to load model or data")
        return
    
    # Create results directory if it doesn't exist
    results_dir = ROOT + "results/"
    os.makedirs(results_dir, exist_ok=True)
    
    # Create different versions
    print("\n📊 Creating confusion matrices...")
    
    # 1. Compact version (medium size, no colorbar)
    create_compact_confusion_matrix(
        te_Y, te_Y_pred, CLASSES,
        os.path.join(results_dir, "confusion_matrix_compact.png"),
        figsize=(8, 6), dpi=150
    )
    
    # 2. Small version (smaller size, lower DPI)
    create_compact_confusion_matrix(
        te_Y, te_Y_pred, CLASSES,
        os.path.join(results_dir, "confusion_matrix_small.png"),
        figsize=(6, 4.5), dpi=100
    )
    
    # 3. Mini version (thumbnail)
    create_mini_confusion_matrix(
        te_Y, te_Y_pred, CLASSES,
        os.path.join(results_dir, "confusion_matrix_mini.png")
    )
    
    print("\n🎉 All confusion matrices generated successfully!")
    print("📁 Check the results/ folder for:")
    print("   - confusion_matrix_compact.png (medium, no colorbar)")
    print("   - confusion_matrix_small.png (small, compressed)")
    print("   - confusion_matrix_mini.png (thumbnail)")

if __name__ == "__main__":
    main()
