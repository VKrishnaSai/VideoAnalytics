#!/usr/bin/env python
# coding: utf-8

'''
This script does:
1. Load features and labels from CSV files.
2. Split the data using stratification so that each class is equally represented.
3. Perform a custom training loop using early stopping.
   It logs training and validation metrics (loss, accuracy, precision, recall, F1) per epoch.
4. Saves checkpoints (when validation loss improves) and the final model in .pickle format.
5. Evaluates the final model on the test set, printing a classification report.
6. Computes per-class accuracy, precision, recall, and F1 (using binary evaluation per class) and
   saves these metrics as bar charts.
7. Computes overall accuracy, precision, recall, and F1-score (micro, macro, weighted averages) on the test set.
8. Plots and saves detailed training curves into the /results folder, including:
   - Training and validation loss curves.
   - Accuracy curves.
   - Precision, Recall, and F1 curves.
   - A confusion matrix.
   - A bar chart of overall metrics (micro, macro, weighted).
   
Note:
The per-class metrics are computed by converting each class into a binary problem (i.e. treating the given class as "positive" and all others as "negative"). This may yield different numbers than those reported by sklearn’s classification_report, which uses a multi-class averaging scheme.
'''

import numpy as np
import time
import pickle
import matplotlib.pyplot as plt
import os
import sys
import sklearn.model_selection
from sklearn.metrics import classification_report, log_loss, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier

# Include project path
ROOT = os.path.dirname(os.path.abspath(__file__)) + "/../"
CURR_PATH = os.path.dirname(os.path.abspath(__file__)) + "/"
sys.path.append(ROOT)

import utils.lib_plot as lib_plot
import utils.lib_commons as lib_commons

def par(path):
    """Prepend ROOT to path if not absolute."""
    return ROOT + path if (path and path[0] != "/") else path

# -- Settings from config
cfg_all = lib_commons.read_yaml(ROOT + "config/config.yaml")
cfg = cfg_all["s4_train.py"]

CLASSES = np.array(cfg_all["classes"])

# Debug: Print class information from config
print(f"\n=== CONFIG DEBUG ===")
print(f"Number of classes in config: {len(CLASSES)}")
print(f"First 10 classes: {CLASSES[:10]}")
print(f"Last 10 classes: {CLASSES[-10:]}")
print(f"All classes: {list(CLASSES)}")
print("==================\n")

SRC_PROCESSED_FEATURES = par(cfg["input"]["processed_features"])
SRC_PROCESSED_FEATURES_LABELS = par(cfg["input"]["processed_features_labels"])
DST_MODEL_PATH = par(cfg["output"]["model_path"])

# Directories for saving results and checkpoints
RESULTS_DIR = par("results/")
CHECKPOINT_DIR = os.path.join(RESULTS_DIR, "checkpoints/")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def train_test_split_equal(X, Y, test_size):
    """Split data so that classes are equally represented using stratification."""
    RAND_SEED = 1
    tr_X, te_X, tr_Y, te_Y = sklearn.model_selection.train_test_split(
        X, Y, test_size=test_size, random_state=RAND_SEED, stratify=Y)
    return tr_X, te_X, tr_Y, te_Y

def plot_bar_metrics(metric_dict, title, ylabel, save_path):
    """Plot a bar chart for per-class metrics."""
    classes = list(metric_dict.keys())
    values = [metric_dict[c] for c in classes]
    
    # Adjust figure size based on number of classes
    fig_width = max(15, len(classes) * 0.3)
    fig_height = 8
    
    plt.figure(figsize=(fig_width, fig_height))
    bars = plt.bar(classes, values, color='skyblue', alpha=0.7)
    plt.xlabel('Classes', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.ylim(0, 1.05)  # Metrics in range [0,1]
    
    # Rotate labels and adjust positioning
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    
    # Add value labels on top of bars if not too many classes
    if len(classes) <= 50:
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.2f}', ha='center', va='bottom', fontsize=6)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

def plot_overall_metrics(metrics_dict, title, save_path):
    """Plot a bar chart for overall metrics (micro, macro, weighted)."""
    labels = ['Micro', 'Macro', 'Weighted']
    precision = [metrics_dict['precision_micro'], metrics_dict['precision_macro'], metrics_dict['precision_weighted']]
    recall = [metrics_dict['recall_micro'], metrics_dict['recall_macro'], metrics_dict['recall_weighted']]
    f1 = [metrics_dict['f1_micro'], metrics_dict['f1_macro'], metrics_dict['f1_weighted']]

    x = np.arange(len(labels))
    width = 0.25

    plt.figure(figsize=(10, 6))
    plt.bar(x - width, precision, width, label='Precision', color='skyblue')
    plt.bar(x, recall, width, label='Recall', color='lightgreen')
    plt.bar(x + width, f1, width, label='F1 Score', color='salmon')
    plt.xlabel('Averaging Method', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xticks(x, labels)
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

def main():
    # -- Load preprocessed data
    print("\nReading CSV files of features and labels ...")
    X = np.loadtxt(SRC_PROCESSED_FEATURES, dtype=float)
    Y = np.loadtxt(SRC_PROCESSED_FEATURES_LABELS, dtype=int)
    
    # Debug: Print data information
    print(f"\n=== DATA DEBUG ===")
    print(f"X shape: {X.shape}")
    print(f"Y shape: {Y.shape}")
    print(f"Unique labels in Y: {sorted(np.unique(Y))}")
    print(f"Number of unique labels: {len(np.unique(Y))}")
    print(f"Min label: {np.min(Y)}, Max label: {np.max(Y)}")
    
    # Count samples per class
    unique_labels, counts = np.unique(Y, return_counts=True)
    print(f"Label distribution:")
    for label, count in zip(unique_labels, counts):
        class_name = CLASSES[label] if label < len(CLASSES) else f"UNKNOWN_{label}"
        print(f"  Class {label} ({class_name}): {count} samples")
    print("================\n")
    
    # -- Stratified train-test split (equal class distribution)
    tr_X, te_X, tr_Y, te_Y = train_test_split_equal(X, Y, test_size=0.3)
    print("\nAfter train-test split:")
    print("Training data shape:    ", tr_X.shape)
    print("Training samples:       ", len(tr_Y))
    print("Testing samples:        ", len(te_Y))
    
    # -- Further split training set into training and validation sets (80/20 split)
    tr_X, val_X, tr_Y, val_Y = sklearn.model_selection.train_test_split(
        tr_X, tr_Y, test_size=0.2, random_state=1, stratify=tr_Y)
    print("Training set shape: ", tr_X.shape, " Validation set shape: ", val_X.shape)
    
    # -- Dimensionality Reduction using PCA
    NUM_FEATURES_FROM_PCA = min(50, tr_X.shape[1])
    pca = PCA(n_components=NUM_FEATURES_FROM_PCA, whiten=True)
    pca.fit(tr_X)
    tr_X_new = pca.transform(tr_X)
    val_X_new = pca.transform(val_X)
    te_X_new = pca.transform(te_X)
    print("After PCA, training data shape: ", tr_X_new.shape)
    
    # -- Initialize MLPClassifier with warm_start for incremental training.
    # We'll train one epoch at a time.
    model = MLPClassifier(hidden_layer_sizes=(20, 30, 40),
                          warm_start=True,    # So that calling fit() does not reinitialize weights.
                          max_iter=1,         # We will loop over epochs manually.
                          random_state=1)
    
    max_epochs = 500
    patience = 10
    best_val_loss = np.inf
    epochs_without_improvement = 0
    
    # Lists for logging metrics per epoch
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    val_precisions = []
    val_recalls = []
    val_f1s = []
    
    # Ensure the first call to partial_fit is made with all classes.
    classes_for_partial = np.unique(tr_Y)
    model.partial_fit(tr_X_new, tr_Y, classes=classes_for_partial)
    
    # -- Custom Training Loop with Early Stopping and Logging
    for epoch in range(max_epochs):
        # Shuffle training data at each epoch
        indices = np.arange(tr_X_new.shape[0])
        np.random.shuffle(indices)
        tr_X_new = tr_X_new[indices]
        tr_Y = tr_Y[indices]
        
        # Train for one epoch (using warm_start, fit() continues training)
        model.fit(tr_X_new, tr_Y)
        
        # Get training loss from model attribute (after the last iteration)
        train_loss = model.loss_
        # Evaluate on training data
        tr_Y_pred = model.predict(tr_X_new)
        train_accuracy = accuracy_score(tr_Y, tr_Y_pred)
        
        # Evaluate on validation data
        val_proba = model.predict_proba(val_X_new)
        val_loss = log_loss(val_Y, val_proba)
        val_Y_pred = model.predict(val_X_new)
        val_accuracy = accuracy_score(val_Y, val_Y_pred)
        # For multi-class, classification_report uses a one-vs-rest approach.
        # Here, we compute macro averages over all classes.
        val_precision = precision_score(val_Y, val_Y_pred, average='macro', zero_division=0)
        val_recall = recall_score(val_Y, val_Y_pred, average='macro', zero_division=0)
        val_f1 = f1_score(val_Y, val_Y_pred, average='macro', zero_division=0)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        val_precisions.append(val_precision)
        val_recalls.append(val_recall)
        val_f1s.append(val_f1)
        
        print(f"Epoch {epoch+1:03d}: Train loss {train_loss:.4f}, Train acc {train_accuracy:.4f}, " +
              f"Val loss {val_loss:.4f}, Val acc {val_accuracy:.4f}, " +
              f"Val prec {val_precision:.4f}, Val rec {val_recall:.4f}, Val f1 {val_f1:.4f}")
        
        # Checkpoint model if validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"model_epoch_{epoch+1:03d}.pickle")
            with open(checkpoint_path, 'wb') as f:
                pickle.dump((model, pca), f)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print("Early stopping triggered.")
                break

    # -- Evaluate final model on test set
    print("\nEvaluating on test set ...")
    te_Y_pred = model.predict(te_X_new)
    
    # Handle missing classes - some UCF-101 classes may not have detected skeletons
    unique_classes_in_data = np.unique(np.concatenate([te_Y, te_Y_pred]))
    
    # Validate that all class indices are within bounds
    max_class_idx = np.max(unique_classes_in_data)
    if max_class_idx >= len(CLASSES):
        print(f"ERROR: Class index {max_class_idx} exceeds available classes ({len(CLASSES)})")
        print("This indicates a mismatch between your data and config file.")
        return
    
    # Create class names for present classes only
    class_names_present = [CLASSES[i] for i in unique_classes_in_data]
    
    # Identify missing classes
    missing_class_indices = [i for i in range(len(CLASSES)) if i not in unique_classes_in_data]
    missing_class_names = [CLASSES[i] for i in missing_class_indices]
    
    print(f"Total UCF-101 classes in config: {len(CLASSES)}")
    print(f"Classes with detected skeletons: {len(unique_classes_in_data)}")
    print(f"Missing classes: {len(missing_class_indices)}")
    if missing_class_indices:
        print(f"Missing class indices: {missing_class_indices}")
        print(f"Missing class names: {missing_class_names[:5]}{'...' if len(missing_class_names) > 5 else ''}")
    
    # Use labels parameter to specify which classes are present
    try:
        print(classification_report(te_Y, te_Y_pred, labels=unique_classes_in_data, target_names=class_names_present))
    except Exception as e:
        print(f"Error in classification report: {e}")
        print("Generating basic classification report...")
        print(classification_report(te_Y, te_Y_pred))
    
    # -- Debug: Print label information
    print("\n=== DEBUG INFORMATION ===")
    print(f"Total CLASSES array length: {len(CLASSES)}")
    print(f"First 10 classes in CLASSES: {CLASSES[:10]}")
    print(f"Last 10 classes in CLASSES: {CLASSES[-10:]}")
    print(f"Unique labels in Y (original): {sorted(np.unique(Y))}")
    print(f"Unique labels in te_Y: {sorted(te_Y)}")
    print(f"Unique labels in te_Y_pred: {sorted(te_Y_pred)}")
    print(f"unique_classes_in_data: {sorted(unique_classes_in_data)}")
    print(f"Length of class_names_present: {len(class_names_present)}")
    print(f"class_names_present[:10]: {class_names_present[:10]}")
    print(f"Min label in te_Y: {np.min(te_Y)}, Max label in te_Y: {np.max(te_Y)}")
    print(f"Min label in te_Y_pred: {np.min(te_Y_pred)}, Max label in te_Y_pred: {np.max(te_Y_pred)}")
    
    # Check for any issues with class indexing
    if np.max(te_Y) >= len(CLASSES):
        print(f"ERROR: Maximum label {np.max(te_Y)} exceeds CLASSES array length {len(CLASSES)}")
    if np.max(te_Y_pred) >= len(CLASSES):
        print(f"ERROR: Maximum predicted label {np.max(te_Y_pred)} exceeds CLASSES array length {len(CLASSES)}")
    
    print("========================\n")

    # -- Plot confusion matrix for test set predictions (for PoseNet)
    print("Plotting confusion matrix for PoseNet...")
    
    # Create a simple confusion matrix using sklearn directly (more reliable)
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    try:
        # Use only the classes that are present in the data
        cm = confusion_matrix(te_Y, te_Y_pred, labels=unique_classes_in_data)
        
        # Create figure with appropriate size based on number of classes
        fig_size = max(12, len(class_names_present) * 0.5)
        plt.figure(figsize=(fig_size, fig_size))
        
        # Create heatmap with proper annotations
        if len(class_names_present) <= 50:  # Only annotate if not too many classes
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=class_names_present, yticklabels=class_names_present,
                       cbar_kws={'label': 'Number of Samples'})
        else:
            sns.heatmap(cm, annot=False, cmap='Blues', 
                       xticklabels=class_names_present, yticklabels=class_names_present,
                       cbar_kws={'label': 'Number of Samples'})
        
        plt.xlabel('Predicted Class', fontsize=12)
        plt.ylabel('True Class', fontsize=12)
        plt.title(f'Confusion Matrix ({len(class_names_present)} classes)', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        confusion_plot_path = os.path.join(RESULTS_DIR, 'confusion_matrix.png')
        plt.savefig(confusion_plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print("Confusion matrix saved successfully!")
        
    except Exception as e:
        print(f"Error plotting confusion matrix: {e}")
        print("Creating a basic confusion matrix plot...")
        
        # Fallback: create a very simple confusion matrix
        cm = confusion_matrix(te_Y, te_Y_pred)
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation='nearest', cmap='Blues')
        plt.title('Confusion Matrix (Basic)')
        plt.colorbar()
        plt.xlabel('Predicted')
        plt.ylabel('True')
        
        confusion_plot_path = os.path.join(RESULTS_DIR, 'confusion_matrix_basic.png')
        plt.savefig(confusion_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print("Basic confusion matrix saved successfully!")
    
    # -- Compute Per-Class Metrics on Test Set using a one-vs-rest (binary) approach:
    per_class_acc = {}
    per_class_prec = {}
    per_class_rec = {}
    per_class_f1 = {}
    
    # Only compute metrics for classes that actually appear in the data
    present_classes = [CLASSES[i] for i in unique_classes_in_data]
    
    print(f"\nComputing per-class metrics for {len(present_classes)} classes...")
    
    for i in unique_classes_in_data:
        try:
            class_name = CLASSES[i]
            # Create binary labels: current class vs. rest.
            y_true_bin = (te_Y == i).astype(int)
            y_pred_bin = (te_Y_pred == i).astype(int)
            
            # Only compute metrics if the class has samples
            if np.sum(y_true_bin) > 0:
                acc = accuracy_score(y_true_bin, y_pred_bin)
                prec = precision_score(y_true_bin, y_pred_bin, average='binary', zero_division=0)
                rec = recall_score(y_true_bin, y_pred_bin, average='binary', zero_division=0)
                f1 = f1_score(y_true_bin, y_pred_bin, average='binary', zero_division=0)
            else:
                acc, prec, rec, f1 = 0, 0, 0, 0
                
            per_class_acc[class_name] = acc
            per_class_prec[class_name] = prec
            per_class_rec[class_name] = rec
            per_class_f1[class_name] = f1
            
        except Exception as e:
            print(f"Error computing metrics for class {i} ({CLASSES[i] if i < len(CLASSES) else 'UNKNOWN'}): {e}")
            # Set default values for problematic classes
            class_name = CLASSES[i] if i < len(CLASSES) else f"UNKNOWN_{i}"
            per_class_acc[class_name] = 0
            per_class_prec[class_name] = 0
            per_class_rec[class_name] = 0
            per_class_f1[class_name] = 0

    # Print per-class metrics
    print("\nPer-Class Metrics on Test Set (binary one-vs-rest):")
    print(f"Note: Showing {len(present_classes)} classes with detected skeletons out of {len(CLASSES)} total UCF-101 classes")
    print(f"Missing classes ({len(missing_class_names)}): {', '.join(missing_class_names)}")
    print("-" * 80)
    
    for class_name in present_classes:
        print(f"{class_name:20s}: Acc={per_class_acc[class_name]:.3f}, Prec={per_class_prec[class_name]:.3f}, " +
              f"Rec={per_class_rec[class_name]:.3f}, F1={per_class_f1[class_name]:.3f}")
    
    # Calculate and display summary statistics
    acc_values = [per_class_acc[c] for c in present_classes]
    prec_values = [per_class_prec[c] for c in present_classes]
    rec_values = [per_class_rec[c] for c in present_classes]
    f1_values = [per_class_f1[c] for c in present_classes]
    
    print("-" * 80)
    print(f"Per-class averages: Acc={np.mean(acc_values):.3f}, Prec={np.mean(prec_values):.3f}, " +
          f"Rec={np.mean(rec_values):.3f}, F1={np.mean(f1_values):.3f}")
    print(f"Per-class std dev:  Acc={np.std(acc_values):.3f}, Prec={np.std(prec_values):.3f}, " +
          f"Rec={np.std(rec_values):.3f}, F1={np.std(f1_values):.3f}")

    # -- Compute Overall Metrics on Test Set
    overall_accuracy = accuracy_score(te_Y, te_Y_pred)
    precision_micro = precision_score(te_Y, te_Y_pred, average='micro', zero_division=0)
    recall_micro = recall_score(te_Y, te_Y_pred, average='micro', zero_division=0)
    f1_micro = f1_score(te_Y, te_Y_pred, average='micro', zero_division=0)
    precision_macro = precision_score(te_Y, te_Y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(te_Y, te_Y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(te_Y, te_Y_pred, average='macro', zero_division=0)
    precision_weighted = precision_score(te_Y, te_Y_pred, average='weighted', zero_division=0)
    recall_weighted = recall_score(te_Y, te_Y_pred, average='weighted', zero_division=0)
    f1_weighted = f1_score(te_Y, te_Y_pred, average='weighted', zero_division=0)

    # Store overall metrics in a dictionary
    overall_metrics = {
        'accuracy': overall_accuracy,
        'precision_micro': precision_micro,
        'recall_micro': recall_micro,
        'f1_micro': f1_micro,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted
    }

    # Print overall metrics
    print("\nOverall Metrics on Test Set:")
    print(f"Accuracy: {overall_accuracy:.4f}")
    print("Micro-Averaged:")
    print(f"  Precision: {precision_micro:.4f}")
    print(f"  Recall: {recall_micro:.4f}")
    print(f"  F1 Score: {f1_micro:.4f}")
    print("Macro-Averaged:")
    print(f"  Precision: {precision_macro:.4f}")
    print(f"  Recall: {recall_macro:.4f}")
    print(f"  F1 Score: {f1_macro:.4f}")
    print("Weighted-Averaged:")
    print(f"  Precision: {precision_weighted:.4f}")
    print(f"  Recall: {recall_weighted:.4f}")
    print(f"  F1 Score: {f1_weighted:.4f}")

    # Save per-class and overall metrics to a text file
    per_class_metrics_path = os.path.join(RESULTS_DIR, "per_class_metrics.txt")
    with open(per_class_metrics_path, "w") as f:
        f.write(f"Per-Class Metrics (binary one-vs-rest) - {len(present_classes)}/{len(CLASSES)} classes with detected skeletons:\n")
        for class_name in present_classes:
            f.write(f"{class_name}: Accuracy: {per_class_acc[class_name]:.4f}, Precision: {per_class_prec[class_name]:.4f}, " +
                    f"Recall: {per_class_rec[class_name]:.4f}, F1: {per_class_f1[class_name]:.4f}\n")
        
        # List missing classes
        missing_classes = [CLASSES[i] for i in range(len(CLASSES)) if i not in unique_classes_in_data]
        f.write(f"\nMissing classes (no detected skeletons): {len(missing_classes)}\n")
        for class_name in missing_classes:
            f.write(f"{class_name}\n")
            
        f.write("\nOverall Metrics:\n")
        f.write(f"Accuracy: {overall_accuracy:.4f}\n")
        f.write("Micro-Averaged:\n")
        f.write(f"  Precision: {precision_micro:.4f}\n")
        f.write(f"  Recall: {recall_micro:.4f}\n")
        f.write(f"  F1 Score: {f1_micro:.4f}\n")
        f.write("Macro-Averaged:\n")
        f.write(f"  Precision: {precision_macro:.4f}\n")
        f.write(f"  Recall: {recall_macro:.4f}\n")
        f.write(f"  F1 Score: {f1_macro:.4f}\n")
        f.write("Weighted-Averaged:\n")
        f.write(f"  Precision: {precision_weighted:.4f}\n")
        f.write(f"  Recall: {recall_weighted:.4f}\n")
        f.write(f"  F1 Score: {f1_weighted:.4f}\n")
        
    print(f"Training complete. Detailed results and plots saved in", RESULTS_DIR)

    # Plot per-class metrics as bar charts
    plot_bar_metrics(per_class_acc, "Per-Class Accuracy", "Accuracy", os.path.join(RESULTS_DIR, "per_class_accuracy.png"))
    plot_bar_metrics(per_class_prec, "Per-Class Precision", "Precision", os.path.join(RESULTS_DIR, "per_class_precision.png"))
    plot_bar_metrics(per_class_rec, "Per-Class Recall", "Recall", os.path.join(RESULTS_DIR, "per_class_recall.png"))
    plot_bar_metrics(per_class_f1, "Per-Class F1 Score", "F1 Score", os.path.join(RESULTS_DIR, "per_class_f1.png"))
    
    # Plot overall metrics
    plot_overall_metrics(overall_metrics, "Overall Metrics (Micro, Macro, Weighted)", os.path.join(RESULTS_DIR, "overall_metrics.png"))
    
    # -- Save the final model in .pickle format
    final_model_path = DST_MODEL_PATH
    print("\nSaving final model to " + final_model_path)
    with open(final_model_path, 'wb') as f:
        pickle.dump((model, pca), f)
    
    # -- Plot and save training curves
    epochs_range = range(1, len(train_losses) + 1)
    
    # Loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, train_losses, 'bo-', label='Training Loss')
    plt.plot(epochs_range, val_losses, 'ro-', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    loss_plot_path = os.path.join(RESULTS_DIR, 'loss_curve.png')
    plt.savefig(loss_plot_path)
    plt.close()
    
    # Accuracy curve
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, train_accuracies, 'bo-', label='Training Accuracy')
    plt.plot(epochs_range, val_accuracies, 'ro-', label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    acc_plot_path = os.path.join(RESULTS_DIR, 'accuracy_curve.png')
    plt.savefig(acc_plot_path)
    plt.close()
    
    # Precision and Recall curve (Validation only)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, val_precisions, 'go-', label='Validation Precision')
    plt.plot(epochs_range, val_recalls, 'mo-', label='Validation Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Validation Precision and Recall')
    plt.legend()
    pr_plot_path = os.path.join(RESULTS_DIR, 'precision_recall_curve.png')
    plt.savefig(pr_plot_path)
    plt.close()
    
    # F1 Curve for Validation
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, val_f1s, 'co-', label='Validation F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('Validation F1 Score')
    plt.legend()
    f1_plot_path = os.path.join(RESULTS_DIR, 'f1_curve.png')
    plt.savefig(f1_plot_path)
    plt.close()
    
    print("Training complete. Detailed results and plots saved in", RESULTS_DIR)

if __name__ == "__main__":
    main()