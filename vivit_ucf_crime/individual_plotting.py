import matplotlib.pyplot as plt
import numpy as np

def plot_individual_metrics(metrics, class_names):
    """
    Plot individual metric files for detailed analysis (Option C)
    """
    epochs = range(1, len(metrics["train_loss"]) + 1)
    
    # Set style for better looking plots
    plt.style.use('default')
    colors = {
        'train': '#2E86AB',  # Blue
        'val': '#A23B72',    # Purple-red
        'grid': '#E5E5E5'    # Light gray
    }
    
    # 1. Loss Plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, metrics["train_loss"], label='Training Loss', color=colors['train'], linewidth=2.5)
    plt.plot(epochs, metrics["val_loss"], label='Validation Loss', color=colors['val'], linewidth=2.5)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3, color=colors['grid'])
    plt.tight_layout()
    plt.savefig("loss_plot.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Accuracy Plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, [acc*100 for acc in metrics["train_accuracy"]], label='Training Accuracy', color=colors['train'], linewidth=2.5)
    plt.plot(epochs, [acc*100 for acc in metrics["val_accuracy"]], label='Validation Accuracy', color=colors['val'], linewidth=2.5)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3, color=colors['grid'])
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig("accuracy_plot.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. F1 Score Plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, [f1*100 for f1 in metrics["train_f1"]], label='Training F1 Score', color=colors['train'], linewidth=2.5)
    plt.plot(epochs, [f1*100 for f1 in metrics["val_f1"]], label='Validation F1 Score', color=colors['val'], linewidth=2.5)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('F1 Score (%)', fontsize=12)
    plt.title('Training and Validation F1 Score', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3, color=colors['grid'])
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig("f1_plot.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Precision Plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, [p*100 for p in metrics["train_precision"]], label='Training Precision', color=colors['train'], linewidth=2.5)
    plt.plot(epochs, [p*100 for p in metrics["val_precision"]], label='Validation Precision', color=colors['val'], linewidth=2.5)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Precision (%)', fontsize=12)
    plt.title('Training and Validation Precision', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3, color=colors['grid'])
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig("precision_plot.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Recall Plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, [r*100 for r in metrics["train_recall"]], label='Training Recall', color=colors['train'], linewidth=2.5)
    plt.plot(epochs, [r*100 for r in metrics["val_recall"]], label='Validation Recall', color=colors['val'], linewidth=2.5)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Recall (%)', fontsize=12)
    plt.title('Training and Validation Recall', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3, color=colors['grid'])
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig("recall_plot.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. Class-wise Evolution Plot
    if metrics["classwise_f1"] and len(metrics["classwise_f1"]) > 0:
        plt.figure(figsize=(14, 8))
        
        # Convert class-wise metrics to arrays for easier plotting
        classwise_f1_array = np.array(metrics["classwise_f1"])  # Shape: (epochs, num_classes)
        
        # Plot evolution of each class's F1 score
        for class_idx, class_name in enumerate(class_names):
            if class_idx < classwise_f1_array.shape[1]:  # Check if class exists in metrics
                plt.plot(epochs, classwise_f1_array[:, class_idx] * 100, 
                        label=class_name, linewidth=2, alpha=0.8)
        
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('F1 Score (%)', fontsize=12)
        plt.title('Class-wise F1 Score Evolution', fontsize=14, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        plt.grid(True, alpha=0.3, color=colors['grid'])
        plt.ylim(0, 100)
        plt.tight_layout()
        plt.savefig("classwise_evolution.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    print("📊 Individual metric plots saved:")
    print("  📈 loss_plot.png - Training and validation loss")
    print("  📈 accuracy_plot.png - Training and validation accuracy")
    print("  📈 f1_plot.png - Training and validation F1 score")
    print("  📈 precision_plot.png - Training and validation precision")
    print("  📈 recall_plot.png - Training and validation recall")
    print("  📈 classwise_evolution.png - Class-wise F1 score evolution")