import os
import time
import threading
import asyncio
import logging
import copy
from collections import defaultdict
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset

from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report)
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

# Import your ViViT model and load_video function from vivit.py
from vivit import ViViT, load_video

# Try to import Hugging Face transformers for pre-trained weights
try:
    from transformers import VivitModel, VivitConfig
    HUGGINGFACE_AVAILABLE = True
    print("✅ Hugging Face transformers available")
except ImportError:
    HUGGINGFACE_AVAILABLE = False
    print("❌ Hugging Face transformers not available. Install with: pip install transformers")

# Try to import timm as fallback
try:
    import timm
    TIMM_AVAILABLE = True
    print("✅ timm available as fallback")
except ImportError:
    TIMM_AVAILABLE = False
    print("❌ timm not available")

# ----------------------- Global Training Status -----------------------
training_status = {
    "epoch": 0,
    "loss": 0.0,
    "accuracy": 0.0,
    "progress": 0.0  # percentage (0-100)
}

# ----------------------- Logging Setup -----------------------
logging.basicConfig(
    filename='vivit_training_enhanced.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s'
)
logging.info("Enhanced ViViT training started.")

# ----------------------- Data Preparation -----------------------
# Read the file and organize categories
class_file = "truncated_classes.txt"
data_subset = []

# Simple reading - each line is a class name
with open(class_file, "r") as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("//"):
            data_subset.append(line)

print(f"📋 Found {len(data_subset)} classes:")
for i, cls in enumerate(data_subset):
    print(f"  {i+1:2d}. {cls}")

# Data root for videos - MODIFY THIS PATH TO YOUR DATASET
data_root = input("Enter the path to your dataset directory: ").strip().strip('"').strip("'")

# ----------------------- Custom Dataset -----------------------
class VideoDataset(Dataset):
    def __init__(self, video_paths, labels, num_frames=16, image_size=224):
        self.video_paths = video_paths
        self.labels = labels
        self.num_frames = num_frames
        self.image_size = image_size

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        # Use your load_video function
        video = load_video(self.video_paths[idx], num_frames=self.num_frames, image_size=self.image_size)
        # Remove extra batch dimension if present (expected shape: [1, T, C, H, W])
        if video.ndim == 5 and video.shape[0] == 1:
            video = video.squeeze(0)  # now [T, C, H, W]
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return video, label

# Build video paths and labels lists
videos = []
labels = []
label_map = {cls: idx for idx, cls in enumerate(data_subset)}  # Map class names to indices

print(f"🔍 Scanning dataset directory: {data_root}")
for cls in data_subset:
    cls_path = os.path.join(data_root, cls)
    if not os.path.isdir(cls_path):
        print(f"⚠️  Warning: Class directory {cls_path} does not exist!")
        continue
    
    video_files = [f for f in os.listdir(cls_path) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    print(f"  📁 {cls}: {len(video_files)} videos")
    
    for video in video_files:
        videos.append(os.path.join(cls_path, video))
        labels.append(label_map[cls])

print(f"📊 Total loaded: {len(videos)} videos across {len(data_subset)} classes")

# Create the full dataset
full_dataset = VideoDataset(videos, labels)

# ----------------------- Split Data (70/20/10) -----------------------
# First split: 70% train, 30% temp (which will be split into 20% val, 10% test)
all_indices = list(range(len(full_dataset)))
train_indices, temp_indices = train_test_split(
    all_indices, test_size=0.30, random_state=42, stratify=labels
)

# Second split: Split the 30% temp into 20% val and 10% test (2/3 and 1/3 of temp)
temp_labels = [labels[i] for i in temp_indices]
val_indices, test_indices = train_test_split(
    temp_indices, test_size=0.33333, random_state=42, stratify=temp_labels  # 1/3 of 30% = 10% of total
)

print(f"\n📊 Dataset Split:")
print(f"  🚂 Train samples: {len(train_indices)} ({len(train_indices)/len(all_indices)*100:.1f}%)")
print(f"  🔍 Validation samples: {len(val_indices)} ({len(val_indices)/len(all_indices)*100:.1f}%)")
print(f"  🧪 Test samples: {len(test_indices)} ({len(test_indices)/len(all_indices)*100:.1f}%)")
print(f"  📈 Total samples: {len(all_indices)}")

train_dataset = Subset(full_dataset, train_indices)
val_dataset = Subset(full_dataset, val_indices)
test_dataset = Subset(full_dataset, test_indices)

# Create DataLoaders
trainloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
valloader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)
testloader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2)

# ----------------------- Checkpoints Directory -----------------------
checkpoints_dir = "checkpoints_vivit_enhanced"
if not os.path.exists(checkpoints_dir):
    os.makedirs(checkpoints_dir)

# ----------------------- Pre-trained Weight Loading -----------------------
def load_pretrained_vivit_weights(model, device='cuda'):
    """
    Load pre-trained ViViT-B weights from Hugging Face or other sources
    """
    print("🔍 Attempting to load pre-trained ViViT weights...")
    
    # Try Hugging Face first
    if HUGGINGFACE_AVAILABLE:
        try:
            print("📥 Trying to load from Hugging Face...")
            # Try different model names
            model_names = [
                "google/vivit-b-16x2-kinetics400",
                "MCG-NJU/videomae-base-finetuned-kinetics",
                "facebook/videomae-base-finetuned-kinetics"
            ]
            
            for model_name in model_names:
                try:
                    print(f"  Trying: {model_name}")
                    pretrained_model = VivitModel.from_pretrained(model_name)
                    print(f"✅ Successfully loaded {model_name}")
                    
                    # Transfer compatible weights
                    model_dict = model.state_dict()
                    pretrained_dict = pretrained_model.state_dict()
                    
                    # Filter out incompatible keys (mainly classification head)
                    compatible_dict = {}
                    for k, v in pretrained_dict.items():
                        if k in model_dict and v.shape == model_dict[k].shape:
                            compatible_dict[k] = v
                    
                    model_dict.update(compatible_dict)
                    model.load_state_dict(model_dict, strict=False)
                    
                    print(f"✅ Transferred {len(compatible_dict)} weight tensors")
                    return True, f"Hugging Face: {model_name}"
                    
                except Exception as e:
                    print(f"  ❌ Failed: {e}")
                    continue
                    
        except Exception as e:
            print(f"❌ Hugging Face loading failed: {e}")
    
    # Try timm as fallback
    if TIMM_AVAILABLE:
        try:
            print("📥 Trying timm models...")
            vivit_models = timm.list_models('*vivit*')
            video_models = timm.list_models('*video*')
            
            all_models = vivit_models + video_models
            
            for model_name in all_models:
                try:
                    print(f"  Trying: {model_name}")
                    pretrained_model = timm.create_model(model_name, pretrained=True)
                    print(f"✅ Successfully loaded {model_name}")
                    
                    # Transfer compatible weights
                    model_dict = model.state_dict()
                    pretrained_dict = pretrained_model.state_dict()
                    
                    compatible_dict = {}
                    for k, v in pretrained_dict.items():
                        if k in model_dict and v.shape == model_dict[k].shape:
                            compatible_dict[k] = v
                    
                    model_dict.update(compatible_dict)
                    model.load_state_dict(model_dict, strict=False)
                    
                    print(f"✅ Transferred {len(compatible_dict)} weight tensors")
                    return True, f"timm: {model_name}"
                    
                except Exception as e:
                    print(f"  ❌ Failed: {e}")
                    continue
                    
        except Exception as e:
            print(f"❌ timm loading failed: {e}")
    
    print("❌ No pre-trained weights found. Training from scratch.")
    return False, "Training from scratch"

# ----------------------- Enhanced Training Function with Early Stopping -----------------------
def train_model_with_early_stopping(model, train_dataloader, val_dataloader, criterion, optimizer, 
                                   num_epochs=300, patience=20, device='cuda'):
    """
    Enhanced training function with early stopping and detailed metrics tracking
    """
    global training_status
    
    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience_counter = 0
    total_train_samples = len(train_dataloader.dataset)
    
    # Comprehensive metrics tracking
    metrics = {
        "train_loss": [], "train_accuracy": [], "train_precision": [], "train_recall": [], "train_f1": [],
        "val_loss": [], "val_accuracy": [], "val_precision": [], "val_recall": [], "val_f1": [],
        "classwise_precision": [], "classwise_recall": [], "classwise_f1": []
    }
    
    print(f"🚀 Starting training for up to {num_epochs} epochs with early stopping patience of {patience}")
    
    for epoch in range(num_epochs):
        # =================== TRAINING PHASE ===================
        model.train()
        epoch_loss = 0.0
        total_count = 0
        all_train_preds = []
        all_train_labels = []
        
        train_pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for videos_batch, labels_batch in train_pbar:
            videos_batch, labels_batch = videos_batch.to(device), labels_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(videos_batch)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()
            
            # Track metrics
            batch_size = labels_batch.size(0)
            epoch_loss += loss.item() * batch_size
            total_count += batch_size
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_train_preds.extend(preds)
            all_train_labels.extend(labels_batch.cpu().numpy())
            
            # Update progress
            current_progress = (((epoch * total_train_samples) + total_count) / (num_epochs * total_train_samples)) * 100
            training_status["progress"] = current_progress
            
            # Update progress bar
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Progress': f'{current_progress:.1f}%'
            })
        
        # Calculate training metrics
        train_loss_avg = epoch_loss / total_count
        train_accuracy = accuracy_score(all_train_labels, all_train_preds)
        train_precision = precision_score(all_train_labels, all_train_preds, average='macro', zero_division=0)
        train_recall = recall_score(all_train_labels, all_train_preds, average='macro', zero_division=0)
        train_f1 = f1_score(all_train_labels, all_train_preds, average='macro', zero_division=0)
        
        # Store training metrics
        metrics["train_loss"].append(train_loss_avg)
        metrics["train_accuracy"].append(train_accuracy)
        metrics["train_precision"].append(train_precision)
        metrics["train_recall"].append(train_recall)
        metrics["train_f1"].append(train_f1)
        
        # =================== VALIDATION PHASE ===================
        model.eval()
        val_loss = 0.0
        val_count = 0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            val_pbar = tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
            for videos_batch, labels_batch in val_pbar:
                videos_batch, labels_batch = videos_batch.to(device), labels_batch.to(device)
                outputs = model(videos_batch)
                loss = criterion(outputs, labels_batch)
                
                batch_size = labels_batch.size(0)
                val_loss += loss.item() * batch_size
                val_count += batch_size
                preds = outputs.argmax(dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_labels.extend(labels_batch.cpu().numpy())
                
                val_pbar.set_postfix({'Val Loss': f'{loss.item():.4f}'})
        
        # Calculate validation metrics
        val_loss_avg = val_loss / val_count
        val_accuracy = accuracy_score(val_labels, val_preds)
        val_precision = precision_score(val_labels, val_preds, average='macro', zero_division=0)
        val_recall = recall_score(val_labels, val_preds, average='macro', zero_division=0)
        val_f1 = f1_score(val_labels, val_preds, average='macro', zero_division=0)
        
        # Class-wise metrics
        classwise_precision = precision_score(val_labels, val_preds, average=None, zero_division=0)
        classwise_recall = recall_score(val_labels, val_preds, average=None, zero_division=0)
        classwise_f1 = f1_score(val_labels, val_preds, average=None, zero_division=0)
        
        # Store validation metrics
        metrics["val_loss"].append(val_loss_avg)
        metrics["val_accuracy"].append(val_accuracy)
        metrics["val_precision"].append(val_precision)
        metrics["val_recall"].append(val_recall)
        metrics["val_f1"].append(val_f1)
        metrics["classwise_precision"].append(classwise_precision)
        metrics["classwise_recall"].append(classwise_recall)
        metrics["classwise_f1"].append(classwise_f1)
        
        # Update global training status
        training_status["epoch"] = epoch + 1
        training_status["loss"] = val_loss_avg
        training_status["accuracy"] = val_accuracy
        
        # =================== LOGGING ===================
        log_msg = (f"Epoch {epoch+1}/{num_epochs} | "
                  f"Train Loss: {train_loss_avg:.4f}, Train Acc: {train_accuracy:.4f} | "
                  f"Val Loss: {val_loss_avg:.4f}, Val Acc: {val_accuracy:.4f}, "
                  f"Val F1: {val_f1:.4f}")
        print(log_msg)
        logging.info(log_msg)
        
        # =================== CHECKPOINTING ===================
        # Save regular checkpoint
        checkpoint_path = os.path.join(checkpoints_dir, f"checkpoint_epoch_{epoch+1}.pth")
        torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss_avg,
            'val_loss': val_loss_avg,
            'val_accuracy': val_accuracy,
            'metrics': metrics
        }, checkpoint_path)
        
        # =================== EARLY STOPPING LOGIC ===================
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            best_val_acc = val_accuracy
            patience_counter = 0
            
            # Save best model
            best_checkpoint_path = os.path.join(checkpoints_dir, "best_checkpoint.pth")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss_avg,
                'val_loss': val_loss_avg,
                'val_accuracy': val_accuracy,
                'metrics': metrics
            }, best_checkpoint_path)
            
            print(f"🎉 New best model! Val Loss: {val_loss_avg:.4f}, Val Acc: {val_accuracy:.4f}")
            logging.info(f"New best model saved at epoch {epoch+1}")
            
        else:
            patience_counter += 1
            print(f"⏳ No improvement for {patience_counter}/{patience} epochs")
            
            if patience_counter >= patience:
                print(f"🛑 Early stopping triggered after {epoch+1} epochs")
                print(f"🏆 Best validation loss: {best_val_loss:.4f}, Best validation accuracy: {best_val_acc:.4f}")
                logging.info(f"Early stopping triggered at epoch {epoch+1}")
                break
    
    training_status["progress"] = 100
    return best_val_loss, metrics

# ----------------------- Enhanced Plotting Functions -----------------------
def plot_comprehensive_metrics(metrics, class_names):
    """
    Plot comprehensive training metrics including class-wise performance
    """
    epochs = range(1, len(metrics["train_loss"]) + 1)
    
    # Create a large figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('ViViT Training Metrics', fontsize=16, fontweight='bold')
    
    # Plot 1: Loss
    axes[0, 0].plot(epochs, metrics["train_loss"], label='Train Loss', color='blue')
    axes[0, 0].plot(epochs, metrics["val_loss"], label='Val Loss', color='red')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss vs Epoch')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Accuracy
    axes[0, 1].plot(epochs, metrics["train_accuracy"], label='Train Accuracy', color='blue')
    axes[0, 1].plot(epochs, metrics["val_accuracy"], label='Val Accuracy', color='red')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Accuracy vs Epoch')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: F1 Score
    axes[0, 2].plot(epochs, metrics["train_f1"], label='Train F1', color='blue')
    axes[0, 2].plot(epochs, metrics["val_f1"], label='Val F1', color='red')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('F1 Score')
    axes[0, 2].set_title('F1 Score vs Epoch')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Precision
    axes[1, 0].plot(epochs, metrics["train_precision"], label='Train Precision', color='blue')
    axes[1, 0].plot(epochs, metrics["val_precision"], label='Val Precision', color='red')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].set_title('Precision vs Epoch')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Recall
    axes[1, 1].plot(epochs, metrics["train_recall"], label='Train Recall', color='blue')
    axes[1, 1].plot(epochs, metrics["val_recall"], label='Val Recall', color='red')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].set_title('Recall vs Epoch')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Class-wise F1 (final epoch)
    if metrics["classwise_f1"]:
        final_classwise_f1 = metrics["classwise_f1"][-1]
        axes[1, 2].bar(range(len(final_classwise_f1)), final_classwise_f1, color='skyblue')
        axes[1, 2].set_xlabel('Class')
        axes[1, 2].set_ylabel('F1 Score')
        axes[1, 2].set_title('Final Class-wise F1 Score')
        axes[1, 2].set_xticks(range(len(class_names)))
        axes[1, 2].set_xticklabels(class_names, rotation=45, ha='right')
        axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("comprehensive_training_metrics.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("📊 Comprehensive metrics plot saved as 'comprehensive_training_metrics.png'")

def evaluate_test_comprehensive(model, test_dataloader, class_names, device='cuda'):
    """
    Comprehensive test evaluation with detailed metrics and visualizations
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    print("🧪 Evaluating on test set...")
    with torch.no_grad():
        for videos_batch, labels_batch in tqdm(test_dataloader, desc="Testing"):
            videos_batch, labels_batch = videos_batch.to(device), labels_batch.to(device)
            outputs = model(videos_batch)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels_batch.cpu().numpy())
    
    # Overall metrics
    test_accuracy = accuracy_score(all_labels, all_preds)
    test_precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    test_recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    test_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    # Class-wise metrics
    classwise_precision = precision_score(all_labels, all_preds, average=None, zero_division=0)
    classwise_recall = recall_score(all_labels, all_preds, average=None, zero_division=0)
    classwise_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)
    
    # Print results
    print(f"\n🎯 Test Results:")
    print(f"  Overall Accuracy: {test_accuracy:.4f}")
    print(f"  Overall Precision: {test_precision:.4f}")
    print(f"  Overall Recall: {test_recall:.4f}")
    print(f"  Overall F1 Score: {test_f1:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("True", fontsize=12)
    plt.title("Confusion Matrix on Test Set", fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig("confusion_matrix_detailed.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Class-wise performance bar chart
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    x_pos = np.arange(len(class_names))
    
    ax1.bar(x_pos, classwise_precision, color='lightcoral', alpha=0.7)
    ax1.set_xlabel('Class')
    ax1.set_ylabel('Precision')
    ax1.set_title('Class-wise Precision')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(class_names, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    ax2.bar(x_pos, classwise_recall, color='lightgreen', alpha=0.7)
    ax2.set_xlabel('Class')
    ax2.set_ylabel('Recall')
    ax2.set_title('Class-wise Recall')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(class_names, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    ax3.bar(x_pos, classwise_f1, color='lightblue', alpha=0.7)
    ax3.set_xlabel('Class')
    ax3.set_ylabel('F1 Score')
    ax3.set_title('Class-wise F1 Score')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(class_names, rotation=45, ha='right')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("classwise_performance.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save detailed classification report
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv("classification_report_detailed.csv")
    
    print("📊 Detailed evaluation plots and reports saved:")
    print("  - confusion_matrix_detailed.png")
    print("  - classwise_performance.png") 
    print("  - classification_report_detailed.csv")
    
    return {
        'accuracy': test_accuracy,
        'precision': test_precision,
        'recall': test_recall,
        'f1': test_f1,
        'classwise_precision': classwise_precision,
        'classwise_recall': classwise_recall,
        'classwise_f1': classwise_f1
    }

# ----------------------- FastAPI Setup -----------------------
app = FastAPI()

@app.get("/", response_class=HTMLResponse)
async def index():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>ViViT Training Dashboard</title>
        <style>
            body { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; display: flex; justify-content: center; align-items: center; height: 100vh; }
            .container { background: rgba(255,255,255,0.95); border-radius: 20px; padding: 30px 50px; text-align: center; box-shadow: 0 20px 40px rgba(0,0,0,0.3); backdrop-filter: blur(10px); }
            h1 { margin-bottom: 30px; color: #333; font-size: 2.5em; text-shadow: 2px 2px 4px rgba(0,0,0,0.1); }
            .progress-container { width: 100%; background: #e0e0e0; border-radius: 25px; margin: 30px 0; overflow: hidden; height: 40px; box-shadow: inset 0 2px 4px rgba(0,0,0,0.1); }
            .progress-bar { width: 0%; height: 100%; background: linear-gradient(90deg, #4CAF50, #45a049); transition: width 0.3s ease; position: relative; }
            .progress-bar::after { content: ''; position: absolute; top: 0; left: 0; right: 0; bottom: 0; background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent); animation: shimmer 2s infinite; }
            @keyframes shimmer { 0% { transform: translateX(-100%); } 100% { transform: translateX(100%); } }
            .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-top: 30px; }
            .stat-card { background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); padding: 20px; border-radius: 15px; box-shadow: 0 5px 15px rgba(0,0,0,0.1); }
            .stat-label { font-size: 0.9em; color: #666; margin-bottom: 5px; }
            .stat-value { font-size: 1.8em; font-weight: bold; color: #333; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎬 ViViT Training Dashboard</h1>
            <div class="progress-container">
                <div class="progress-bar" id="progress-bar"></div>
            </div>
            <div class="stats">
                <div class="stat-card">
                    <div class="stat-label">Epoch</div>
                    <div class="stat-value" id="epoch">0</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Loss</div>
                    <div class="stat-value" id="loss">0.0000</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Accuracy</div>
                    <div class="stat-value" id="accuracy">0.00%</div>
                </div>
            </div>
        </div>
        <script>
            var ws = new WebSocket("ws://" + location.host + "/ws");
            ws.onmessage = function(event) {
                var data = JSON.parse(event.data);
                document.getElementById("epoch").innerText = data.epoch;
                document.getElementById("loss").innerText = data.loss.toFixed(4);
                document.getElementById("accuracy").innerText = (data.accuracy * 100).toFixed(2) + "%";
                document.getElementById("progress-bar").style.width = data.progress + "%";
            };
            ws.onclose = function() { console.log("WebSocket connection closed."); };
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            await websocket.send_json(training_status)
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        print("WebSocket disconnected")

# ----------------------- Main Training Function -----------------------
def start_enhanced_training():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️  Using device: {device}")
    logging.info(f"Using device: {device}")
    
    # Initialize ViViT-B model for your classes
    print(f"🏗️  Initializing ViViT-B model for {len(data_subset)} classes...")
    model = ViViT(
        image_size=224, 
        patch_size=16, 
        num_classes=len(data_subset), 
        num_frames=16,
        dim=768,  # ViViT-B embedding dimension
        depth=12,  # ViViT-B layers
        heads=12   # ViViT-B attention heads
    ).to(device)
    
    # Try to load pre-trained weights
    pretrained_loaded, pretrained_source = load_pretrained_vivit_weights(model, device)
    
    # Set learning rate based on whether we loaded pre-trained weights
    if pretrained_loaded:
        learning_rate = 1e-5  # Smaller LR for fine-tuning
        print(f"🎯 Fine-tuning with learning rate: {learning_rate}")
    else:
        learning_rate = 1e-4  # Normal LR for training from scratch
        print(f"🎯 Training from scratch with learning rate: {learning_rate}")
    
    # Setup training components
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    print(f"📋 Training Configuration:")
    print(f"  Model: ViViT-B")
    print(f"  Classes: {len(data_subset)}")
    print(f"  Pre-trained: {pretrained_source}")
    print(f"  Learning Rate: {learning_rate}")
    print(f"  Max Epochs: 300")
    print(f"  Early Stopping Patience: 20")
    print(f"  Batch Size: 4")
    
    # Start training with early stopping
    best_val_loss, metrics = train_model_with_early_stopping(
        model=model,
        train_dataloader=trainloader,
        val_dataloader=valloader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=300,
        patience=20,
        device=device
    )
    
    # Plot comprehensive training metrics
    plot_comprehensive_metrics(metrics, data_subset)
    logging.info("Comprehensive training metrics plots saved.")
    
    # Evaluate on test set with detailed analysis
    test_results = evaluate_test_comprehensive(model, testloader, data_subset, device=device)
    logging.info("Comprehensive test evaluation complete.")
    
    # Save final model
    final_model_path = "vivit_model_final.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_names': data_subset,
        'test_results': test_results,
        'training_metrics': metrics,
        'pretrained_source': pretrained_source
    }, final_model_path)
    
    print(f"🎉 Training completed!")
    print(f"📁 Final model saved as: {final_model_path}")
    print(f"🏆 Best validation loss: {best_val_loss:.4f}")
    print(f"🎯 Final test accuracy: {test_results['accuracy']:.4f}")
    
    logging.info(f"Training completed. Final model saved as {final_model_path}")

@app.on_event("startup")
async def on_startup():
    threading.Thread(target=start_enhanced_training, daemon=True).start()

# ----------------------- Run the App -----------------------
if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting ViViT training server...")
    print("📊 Open http://localhost:8000 to monitor training progress")
    uvicorn.run("fun_training_enhanced:app", host="0.0.0.0", port=8000, reload=False)