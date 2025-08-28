#!/usr/bin/env python3
"""
ViViT Training Script - No Web Interface Version
For background training without port conflicts
"""

import os
import time
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

# ----------------------- Logging Setup -----------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('vivit_training_only.log'),
        logging.StreamHandler()
    ]
)
logging.info("ViViT training started (no web interface).")

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

# Data root for videos
data_root = "../../ucf_crime/"  # Use the path from your log

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
checkpoints_dir = "checkpoints_vivit_final"
if not os.path.exists(checkpoints_dir):
    os.makedirs(checkpoints_dir)

# ----------------------- Pre-trained Weight Loading -----------------------
def load_pretrained_vivit_weights(model, device='cuda'):
    """
    Load pre-trained ViViT-B weights from Hugging Face or other sources
    """
    print("🔍 Attempting to load pre-trained ViViT weights...")
    
    # Skip pre-trained loading for faster startup - train from scratch
    print("⚡ Skipping pre-trained weights for faster startup - training from scratch")
    return False, "Training from scratch (fast startup)"

# ----------------------- Enhanced Training Function with Early Stopping -----------------------
def train_model_with_early_stopping(model, train_dataloader, val_dataloader, criterion, optimizer, 
                                   num_epochs=300, patience=20, device='cuda'):
    """
    Enhanced training function with early stopping and detailed metrics tracking
    """
    
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
            
            # Update progress bar
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Epoch': f'{epoch+1}/{num_epochs}'
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
    
    return best_val_loss, metrics

# ----------------------- Individual Plotting Functions -----------------------
def plot_individual_metrics(metrics, class_names):
    """Plot individual metric files"""
    epochs = range(1, len(metrics["train_loss"]) + 1)
    
    colors = {'train': '#2E86AB', 'val': '#A23B72', 'grid': '#E5E5E5'}
    
    # Loss Plot
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
    
    # Accuracy Plot
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
    
    print("📊 Individual metric plots saved:")
    print("  📈 loss_plot.png - Training and validation loss")
    print("  📈 accuracy_plot.png - Training and validation accuracy")

# ----------------------- Main Training Function -----------------------
def main():
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
    
    # Skip pre-trained weights for faster startup
    pretrained_loaded, pretrained_source = load_pretrained_vivit_weights(model, device)
    
    # Set learning rate
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
    
    # Plot individual training metrics
    plot_individual_metrics(metrics, data_subset)
    logging.info("Individual training metrics plots saved.")
    
    # Save final model
    final_model_path = "vivit_model_final.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_names': data_subset,
        'training_metrics': metrics,
        'pretrained_source': pretrained_source
    }, final_model_path)
    
    print(f"🎉 Training completed!")
    print(f"📁 Final model saved as: {final_model_path}")
    print(f"🏆 Best validation loss: {best_val_loss:.4f}")
    
    logging.info(f"Training completed. Final model saved as {final_model_path}")

if __name__ == "__main__":
    main()