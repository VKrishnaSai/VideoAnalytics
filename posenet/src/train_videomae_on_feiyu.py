import os
import time
import logging
import argparse
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from collections import Counter
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Import VideoMAE components
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
from huggingface_hub import login
import cv2
import pathlib

# ----------------------- Logging Setup -----------------------
log_filename = f'posenet_videomae_training_{datetime.now().strftime("%Y%m%d_%H%M")}.log'
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s'
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)
logger = logging.getLogger(__name__)
logger.info("PoseNet VideoMAE Training started")

# ----------------------- Performance Settings -----------------------
NUM_GPUS = torch.cuda.device_count()
NUM_WORKERS = 8
PIN_MEMORY = True
BENCHMARK = True
PREFETCH_FACTOR = 2
torch.backends.cudnn.benchmark = BENCHMARK
BATCH_SIZE = 4  # Reduced batch size for VideoMAE to avoid memory issues
MIXED_PRECISION = True

# VideoMAE specific settings
MODEL_CKPT = "MCG-NJU/videomae-base"

# ----------------------- Video Loading Function for VideoMAE -----------------------
def load_video_for_videomae(video_path, num_frames=16, image_size=224):
    """
    Load and preprocess video for VideoMAE model.
    Returns video tensor with shape (num_frames, channels, height, width)
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        logger.warning(f"No frames found in video: {video_path}")
        # Return dummy tensor
        return torch.zeros((num_frames, 3, image_size, image_size))
    
    # Sample frames uniformly
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    for i, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if ret:
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Resize frame
            frame = cv2.resize(frame, (image_size, image_size))
            # Convert to tensor and normalize to [0, 1]
            frame = torch.from_numpy(frame).float() / 255.0
            # Change from HWC to CHW
            frame = frame.permute(2, 0, 1)
            frames.append(frame)
        else:
            # If we can't read the frame, duplicate the last valid frame
            if frames:
                frames.append(frames[-1].clone())
            else:
                # If no valid frames yet, create a zero frame
                frames.append(torch.zeros((3, image_size, image_size)))
    
    cap.release()
    
    # Stack frames: (num_frames, channels, height, width)
    video_tensor = torch.stack(frames)
    
    return video_tensor

# ----------------------- Custom Dataset for VideoMAE -----------------------
class VideoMAEDataset(Dataset):
    def __init__(self, video_paths, labels, image_processor, num_frames=16, image_size=224, cache_size=100):
        self.video_paths = video_paths
        self.labels = labels
        self.image_processor = image_processor
        self.num_frames = num_frames
        self.image_size = image_size
        self.cache = {}
        self.cache_size = cache_size

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        if idx in self.cache:
            video = self.cache[idx]
        else:
            video = load_video_for_videomae(self.video_paths[idx], num_frames=self.num_frames, image_size=self.image_size)
            
            if len(self.cache) >= self.cache_size:
                remove_key = list(self.cache.keys())[0]
                del self.cache[remove_key]
            self.cache[idx] = video

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return video, label

# ----------------------- Data Preparation -----------------------
def load_data(data_root="./video_clips"):
    """
    Load PoseNet video data from the output/video_clips directory (10-second clips).
    """
    # Define action classes based on your config
    action_classes = ['stand', 'walk', 'run', 'jump', 'sit', 'squat', 'kick', 'punch', 'wave']
    
    # Create class to index mapping
    class_to_idx = {class_name: idx for idx, class_name in enumerate(action_classes)}
    
    logger.info(f"Action classes: {action_classes}")
    
    # Get all video files from the output/videos directory
    video_files = []
    video_labels = []
    
    if not os.path.exists(data_root):
        logger.error(f"Data directory does not exist: {data_root}")
        return [], [], [], [], action_classes
    
    # Collect all .avi files from both flat structure and subdirectories
    # First check for flat structure (original videos)
    for filename in os.listdir(data_root):
        if filename.endswith('.avi'):
            # Extract action label from filename (e.g., "jump_03-02-12-34-01-795.avi" -> "jump")
            action_name = filename.split('_')[0]
            
            if action_name in class_to_idx:
                video_path = os.path.join(data_root, filename)
                video_files.append(video_path)
                video_labels.append(class_to_idx[action_name])
                logger.debug(f"Added video: {filename} -> {action_name} (label: {class_to_idx[action_name]})")
            else:
                logger.warning(f"Unknown action class '{action_name}' in filename: {filename}")
    
    # Then check for subdirectory structure (video clips)
    for item in os.listdir(data_root):
        item_path = os.path.join(data_root, item)
        if os.path.isdir(item_path) and item in class_to_idx:
            # This is an action subdirectory
            action_name = item
            for clip_filename in os.listdir(item_path):
                if clip_filename.endswith('.avi'):
                    clip_path = os.path.join(item_path, clip_filename)
                    video_files.append(clip_path)
                    video_labels.append(class_to_idx[action_name])
                    logger.debug(f"Added clip: {clip_filename} -> {action_name} (label: {class_to_idx[action_name]})")
    
    if not video_files:
        logger.error(f"No video files found in {data_root}")
        return [], [], [], [], action_classes
    
    logger.info(f"Found {len(video_files)} video files")
    
    # Count videos per class
    from collections import Counter
    label_counts = Counter(video_labels)
    for class_name, class_idx in class_to_idx.items():
        count = label_counts.get(class_idx, 0)
        logger.info(f"  {class_name}: {count} videos")
    
    # Smart train/val/test split strategy for small datasets
    min_samples_per_class = min(label_counts.values()) if label_counts else 0
    total_samples = len(video_files)
    
    logger.info(f"Total samples: {total_samples}, Min samples per class: {min_samples_per_class}")
    
    if total_samples < 30:
        # Very small dataset: use 70/15/15 split or ensure at least 1 sample per split
        logger.warning("Very small dataset detected. Using flexible split strategy.")
        train_ratio, val_ratio, test_ratio = 0.7, 0.15, 0.15
    elif min_samples_per_class == 1:
        # Classes with only 1 sample: put all single-sample classes in training
        logger.warning("Classes with only 1 sample detected. Allocating single samples to training set.")
        
        # Separate single-sample classes from multi-sample classes
        single_sample_videos = []
        single_sample_labels = []
        multi_sample_videos = []
        multi_sample_labels = []
        
        for video, label in zip(video_files, video_labels):
            if label_counts[label] == 1:
                single_sample_videos.append(video)
                single_sample_labels.append(label)
            else:
                multi_sample_videos.append(video)
                multi_sample_labels.append(label)
        
        logger.info(f"Single-sample classes: {len(single_sample_videos)} videos")
        logger.info(f"Multi-sample classes: {len(multi_sample_videos)} videos")
        
        # Split multi-sample classes normally
        if len(multi_sample_videos) >= 6:  # Need at least 6 for 3-way split
            train_multi, temp_multi, train_labels_multi, temp_labels_multi = train_test_split(
                multi_sample_videos, multi_sample_labels, 
                test_size=0.4, random_state=42, stratify=multi_sample_labels
            )
            val_multi, test_multi, val_labels_multi, test_labels_multi = train_test_split(
                temp_multi, temp_labels_multi,
                test_size=0.5, random_state=42, stratify=temp_labels_multi
            )
        else:
            # Too few multi-sample videos for 3-way split
            logger.warning("Too few multi-sample videos for proper 3-way split. Using simple allocation.")
            train_multi = multi_sample_videos[:-2] if len(multi_sample_videos) > 2 else multi_sample_videos
            val_multi = [multi_sample_videos[-2]] if len(multi_sample_videos) > 2 else []
            test_multi = [multi_sample_videos[-1]] if len(multi_sample_videos) > 1 else []
            
            train_labels_multi = multi_sample_labels[:-2] if len(multi_sample_labels) > 2 else multi_sample_labels
            val_labels_multi = [multi_sample_labels[-2]] if len(multi_sample_labels) > 2 else []
            test_labels_multi = [multi_sample_labels[-1]] if len(multi_sample_labels) > 1 else []
        
        # Combine single-sample videos with training set
        train_videos = single_sample_videos + train_multi
        train_labels = single_sample_labels + train_labels_multi
        val_videos = val_multi
        val_labels = val_labels_multi
        test_videos = test_multi
        test_labels = test_labels_multi
        
    else:
        # Standard case: all classes have >= 2 samples
        logger.info("Using standard train/val/test split with stratification.")
        
        # First split: train vs (val + test)
        train_videos, temp_videos, train_labels, temp_labels = train_test_split(
            video_files, video_labels, 
            test_size=0.3, random_state=42, stratify=video_labels
        )
        
        # Second split: val vs test from the temp set
        val_videos, test_videos, val_labels, test_labels = train_test_split(
            temp_videos, temp_labels,
            test_size=0.5, random_state=42, stratify=temp_labels
        )
    
    logger.info(f"Final split - Training: {len(train_videos)}, Validation: {len(val_videos)}, Test: {len(test_videos)}")
    
    # Log distribution for all splits
    train_counts = Counter(train_labels)
    val_counts = Counter(val_labels)
    test_counts = Counter(test_labels)
    
    logger.info("Training distribution:")
    for class_name, class_idx in class_to_idx.items():
        count = train_counts.get(class_idx, 0)
        logger.info(f"  {class_name}: {count} videos")
    
    logger.info("Validation distribution:")
    for class_name, class_idx in class_to_idx.items():
        count = val_counts.get(class_idx, 0)
        logger.info(f"  {class_name}: {count} videos")
    
    logger.info("Test distribution:")
    for class_name, class_idx in class_to_idx.items():
        count = test_counts.get(class_idx, 0)
        logger.info(f"  {class_name}: {count} videos")
    
    return train_videos, train_labels, val_videos, val_labels, test_videos, test_labels, action_classes

# ----------------------- Multi-GPU Support -----------------------
class ModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.module = model

    def forward(self, x):
        return self.module(x)

def prepare_model(model, num_gpus=1):
    if num_gpus > 1:
        logger.info(f"Using {num_gpus} GPUs for training")
        model = nn.DataParallel(model)
    return model

# ----------------------- Utility Function for Saving Metrics -----------------------
def save_metrics(all_labels, all_preds, class_names, save_dir, mode='test', epoch=None):
    prefix = f"epoch_{epoch+1}_" if mode == 'val' and epoch is not None else ""
    metrics_dir = os.path.join(save_dir, f"{mode}_metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    # Compute confusion matrix for per-class accuracy
    cm = confusion_matrix(all_labels, all_preds)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    per_class_acc = np.nan_to_num(per_class_acc, nan=0.0)  # Handle classes with zero instances

    overall_metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision_macro': precision_score(all_labels, all_preds, average='macro', zero_division=0),
        'recall_macro': recall_score(all_labels, all_preds, average='macro', zero_division=0),
        'f1_macro': f1_score(all_labels, all_preds, average='macro', zero_division=0),
        'precision_micro': precision_score(all_labels, all_preds, average='micro', zero_division=0),
        'recall_micro': recall_score(all_labels, all_preds, average='micro', zero_division=0),
        'f1_micro': f1_score(all_labels, all_preds, average='micro', zero_division=0),
        'precision_weighted': precision_score(all_labels, all_preds, average='weighted', zero_division=0),
        'recall_weighted': recall_score(all_labels, all_preds, average='weighted', zero_division=0),
        'f1_weighted': f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    }

    per_class_metrics = {
        'accuracy': per_class_acc,
        'precision': precision_score(all_labels, all_preds, average=None, zero_division=0),
        'recall': recall_score(all_labels, all_preds, average=None, zero_division=0),
        'f1': f1_score(all_labels, all_preds, average=None, zero_division=0)
    }

    # Save to text file
    metrics_file = os.path.join(metrics_dir, f"{prefix}detailed_metrics.txt")
    with open(metrics_file, "w") as f:
        f.write("===== Overall Metrics =====\n")
        for metric, value in overall_metrics.items():
            f.write(f"{metric.replace('_', ' ').title()}: {value:.4f}\n")
        f.write("\n===== Per-Class Metrics =====\n")
        f.write(f"{'Class':<30} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}\n")
        f.write("-" * 74 + "\n")
        for i, class_name in enumerate(class_names):
            f.write(f"{class_name:<30} {per_class_metrics['accuracy'][i]:>10.4f} "
                   f"{per_class_metrics['precision'][i]:>10.4f} {per_class_metrics['recall'][i]:>10.4f} "
                   f"{per_class_metrics['f1'][i]:>10.4f}\n")

    # Save to CSV file
    metrics_csv = os.path.join(metrics_dir, f"{prefix}metrics.csv")
    with open(metrics_csv, "w") as f:
        f.write("Metric,Value\n")
        for metric, value in overall_metrics.items():
            f.write(f"{metric},{value:.4f}\n")
        f.write("\nClass,Accuracy,Precision,Recall,F1-Score\n")
        for i, class_name in enumerate(class_names):
            f.write(f"{class_name},{per_class_metrics['accuracy'][i]:.4f},"
                   f"{per_class_metrics['precision'][i]:.4f},{per_class_metrics['recall'][i]:.4f},"
                   f"{per_class_metrics['f1'][i]:.4f}\n")

    # Generate per-class F1 score bar plot
    plt.figure(figsize=(15, 8))
    plt.bar(range(len(class_names)), per_class_metrics['f1'])
    plt.xticks(range(len(class_names)), class_names, rotation=90)
    plt.xlabel('Classes')
    plt.ylabel('F1 Score')
    plt.title(f'{mode.capitalize()} Per-Class F1 Scores')
    plt.tight_layout()
    plt.savefig(os.path.join(metrics_dir, f"{prefix}per_class_f1.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # Generate confusion matrix plot for validation and test
    if mode != 'train':
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'{mode.capitalize()} Confusion Matrix')
        plt.savefig(os.path.join(metrics_dir, f"{prefix}confusion_matrix.png"), dpi=300, bbox_inches='tight')
        plt.close()

    return overall_metrics, per_class_metrics

# ----------------------- VideoMAE Collate Function -----------------------
def collate_fn(examples):
    """The collation function to be used by DataLoader to prepare data batches for VideoMAE."""
    # VideoMAE expects pixel_values with shape (batch_size, num_frames, num_channels, height, width)
    pixel_values = torch.stack([example[0] for example in examples])  # Stack video tensors
    labels = torch.tensor([example[1] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

# ----------------------- Optimized Training Function -----------------------
def train_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch, total_epochs, class_names, save_dir):
    model.train()
    epoch_loss = 0.0
    total_count = 0
    all_train_preds = []
    all_train_labels = []
    batch_times = []

    start_time = time.time()
    for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs}")):
        batch_start = time.time()
        pixel_values = batch["pixel_values"].to(device, non_blocking=PIN_MEMORY)
        labels = batch["labels"].to(device, non_blocking=PIN_MEMORY)

        if MIXED_PRECISION:
            with autocast('cuda'):
                outputs = model(pixel_values=pixel_values)
                loss = criterion(outputs.logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(pixel_values=pixel_values)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            optimizer.step()

        optimizer.zero_grad(set_to_none=True)

        batch_size = labels.size(0)
        epoch_loss += loss.item() * batch_size
        total_count += batch_size

        preds = outputs.logits.argmax(dim=1).detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        all_train_preds.extend(preds)
        all_train_labels.extend(labels_np)

        batch_times.append(time.time() - batch_start)

        if (batch_idx + 1) % 10 == 0:
            logger.info(f"Epoch {epoch+1} Batch {batch_idx+1}/{len(train_loader)} - "
                        f"Loss: {loss.item():.4f}, Batch time: {batch_times[-1]:.4f}s")

    epoch_time = time.time() - start_time
    epoch_loss_avg = epoch_loss / total_count

    overall_metrics, per_class_metrics = save_metrics(all_train_labels, all_train_preds, class_names, save_dir, mode='train', epoch=epoch)

    avg_batch_time = sum(batch_times) / len(batch_times)
    logger.info(f"Epoch {epoch+1} completed in {epoch_time:.2f}s (avg batch: {avg_batch_time:.4f}s) - "
                f"Train Loss: {epoch_loss_avg:.4f}, "
                f"Train Acc: {overall_metrics['accuracy']:.4f}, "
                f"Train Prec (macro): {overall_metrics['precision_macro']:.4f}, "
                f"Train Rec (macro): {overall_metrics['recall_macro']:.4f}, "
                f"Train F1 (macro): {overall_metrics['f1_macro']:.4f}")

    return epoch_loss_avg, overall_metrics, per_class_metrics

# ----------------------- Validation Function -----------------------
def validate(model, val_loader, criterion, device, class_names, save_dir, epoch):
    model.eval()
    val_loss = 0.0
    val_count = 0
    val_preds = []
    val_labels = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            pixel_values = batch["pixel_values"].to(device, non_blocking=PIN_MEMORY)
            labels = batch["labels"].to(device, non_blocking=PIN_MEMORY)

            if MIXED_PRECISION:
                with autocast('cuda'):
                    outputs = model(pixel_values=pixel_values)
                    loss = criterion(outputs.logits, labels)
            else:
                outputs = model(pixel_values=pixel_values)
                loss = criterion(outputs.logits, labels)

            batch_size = labels.size(0)
            val_loss += loss.item() * batch_size
            val_count += batch_size

            preds = outputs.logits.argmax(dim=1).cpu().numpy()
            val_labels.extend(labels.cpu().numpy())
            val_preds.extend(preds)

    val_loss_avg = val_loss / val_count
    overall_metrics, per_class_metrics = save_metrics(val_labels, val_preds, class_names, save_dir, mode='val', epoch=epoch)

    logger.info(f"Validation - Loss: {val_loss_avg:.4f}, "
                f"Acc: {overall_metrics['accuracy']:.4f}, "
                f"Prec (macro): {overall_metrics['precision_macro']:.4f}, "
                f"Rec (macro): {overall_metrics['recall_macro']:.4f}, "
                f"F1 (macro): {overall_metrics['f1_macro']:.4f}")

    return val_loss_avg, overall_metrics, per_class_metrics

# ----------------------- Test Evaluation Function -----------------------
def evaluate_test(model, test_loader, device, class_names, save_dir):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            pixel_values = batch["pixel_values"].to(device, non_blocking=PIN_MEMORY)
            labels = batch["labels"].to(device, non_blocking=PIN_MEMORY)

            with autocast('cuda', enabled=MIXED_PRECISION):
                outputs = model(pixel_values=pixel_values)

            preds = outputs.logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    overall_metrics, per_class_metrics = save_metrics(all_labels, all_preds, class_names, save_dir, mode='test')

    logger.info(f"Test Accuracy: {overall_metrics['accuracy']:.4f}")
    logger.info(f"Test Precision (macro): {overall_metrics['precision_macro']:.4f}")
    logger.info(f"Test Recall (macro): {overall_metrics['recall_macro']:.4f}")
    logger.info(f"Test F1 Score (macro): {overall_metrics['f1_macro']:.4f}")
    logger.info(f"Test Precision (micro): {overall_metrics['precision_micro']:.4f}")
    logger.info(f"Test Recall (micro): {overall_metrics['recall_micro']:.4f}")
    logger.info(f"Test F1 Score (micro): {overall_metrics['f1_micro']:.4f}")
    logger.info(f"Test Precision (weighted): {overall_metrics['precision_weighted']:.4f}")
    logger.info(f"Test Recall (weighted): {overall_metrics['recall_weighted']:.4f}")
    logger.info(f"Test F1 Score (weighted): {overall_metrics['f1_weighted']:.4f}")

    return overall_metrics, per_class_metrics

# ----------------------- Main Training Loop -----------------------
def train_model(model, train_loader, val_loader, test_loader, criterion, optimizer,
                scheduler, num_epochs, patience, device, checkpoints_dir, class_names, start_epoch=0):

    scaler = GradScaler('cuda') if MIXED_PRECISION else None
    best_val_loss = float('inf')
    epochs_no_improve = 0

    metrics_history = {
        "train_loss": [], "train_accuracy": [], "train_precision": [], "train_recall": [], "train_f1": [],
        "val_loss": [], "val_accuracy": [], "val_precision": [], "val_recall": [], "val_f1": []
    }

    total_start_time = time.time()

    for epoch in range(start_epoch, start_epoch + num_epochs):
        epoch_start_time = time.time()

        train_loss, train_metrics, _ = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch,
            start_epoch + num_epochs, class_names, checkpoints_dir
        )

        if scheduler:
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(f"Current learning rate: {current_lr:.7f}")

        val_loss, val_metrics, _ = validate(
            model, val_loader, criterion, device, class_names, checkpoints_dir, epoch
        )

        metrics_history["train_loss"].append(train_loss)
        metrics_history["train_accuracy"].append(train_metrics['accuracy'])
        metrics_history["train_precision"].append(train_metrics['precision_macro'])
        metrics_history["train_recall"].append(train_metrics['recall_macro'])
        metrics_history["train_f1"].append(train_metrics['f1_macro'])
        metrics_history["val_loss"].append(val_loss)
        metrics_history["val_accuracy"].append(val_metrics['accuracy'])
        metrics_history["val_precision"].append(val_metrics['precision_macro'])
        metrics_history["val_recall"].append(val_metrics['recall_macro'])
        metrics_history["val_f1"].append(val_metrics['f1_macro'])

        checkpoint_path = os.path.join(checkpoints_dir, f"checkpoint_epoch_{epoch+1}.pth")
        if isinstance(model, nn.DataParallel):
            model_state_dict = model.module.state_dict()
        else:
            model_state_dict = model.state_dict()

        torch.save({
            'epoch': epoch+1,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'train_loss': train_loss,
            'val_loss': val_loss,
        }, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_checkpoint_path = os.path.join(checkpoints_dir, "best_checkpoint.pth")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model_state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, best_checkpoint_path)
            logger.info(f"New best model saved at: {best_checkpoint_path}")
        else:
            epochs_no_improve += 1
            logger.info(f"No improvement for {epochs_no_improve} epoch(s).")

        if epochs_no_improve >= patience:
            logger.info(f"Early stopping triggered after {epoch+1} epochs.")
            break

        epoch_time = time.time() - epoch_start_time
        logger.info(f"Epoch {epoch+1} completed in {epoch_time:.2f} seconds")

    total_training_time = time.time() - total_start_time
    logger.info(f"Total training time: {total_training_time/60:.2f} minutes")

    model = load_best_model(model, os.path.join(checkpoints_dir, "best_checkpoint.pth"), device)
    test_metrics, _ = evaluate_test(model, test_loader, device, class_names, checkpoints_dir)

    return best_val_loss, metrics_history

# ----------------------- Utility Functions -----------------------
def load_best_model(model, checkpoint_path, device):
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded best model from {checkpoint_path}")
    return model

def plot_metrics(metrics, save_dir='.'):
    epochs = range(1, len(metrics["train_loss"]) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, metrics["train_loss"], 'b-', label='Train Loss')
    plt.plot(epochs, metrics["val_loss"], 'r-', label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "loss_plot.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, metrics["train_accuracy"], 'b-', label='Train Accuracy')
    plt.plot(epochs, metrics["val_accuracy"], 'r-', label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "accuracy_plot.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, metrics["train_precision"], 'b-', label='Train Precision')
    plt.plot(epochs, metrics["val_precision"], 'r-', label='Val Precision')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.title('Training and Validation Precision')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "precision_plot.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, metrics["train_recall"], 'b-', label='Train Recall')
    plt.plot(epochs, metrics["val_recall"], 'r-', label='Val Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.title('Training and Validation Recall')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "recall_plot.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, metrics["train_f1"], 'b-', label='Train F1 Score')
    plt.plot(epochs, metrics["val_f1"], 'r-', label='Val F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('Training and Validation F1 Score')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "f1_plot.png"), dpi=300)
    plt.close()

# ----------------------- Main Execution -----------------------
def main():
    parser = argparse.ArgumentParser(description="Train VideoMAE on PoseNet video clips")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train (default: 50)")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience (default: 12)")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size (default: 4)")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Checkpoint directory to resume from")
    args = parser.parse_args()
    
    # Update global batch size if specified
    global BATCH_SIZE
    BATCH_SIZE = args.batch_size
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        logger.info(f"CUDA version: {torch.version.cuda}")
    else:
        device = torch.device("cpu")
        logger.info("No GPU available, using CPU")

    checkpoints_dir = f"posenet_videomae_checkpoints_{datetime.now().strftime('%Y%m%d_%H%M')}"
    logger.info(f"Creating checkpoints directory: {checkpoints_dir}")
    os.makedirs(checkpoints_dir, exist_ok=True)

    # Load data from video clips directory (10-second clips created by split_videos_to_clips.py)
    logger.info("Loading video clips data (10-second clips)...")
    
    # Try multiple possible data paths
    possible_paths = [
        "./video_clips",
        "./output/video_clips", 
        "output/video_clips",
        "../output/videos",
        "./videos"
    ]
    
    data_path = None
    for path in possible_paths:
        if os.path.exists(path):
            data_path = path
            logger.info(f"Found data directory: {path}")
            break
    
    if data_path is None:
        logger.error("No data directory found! Tried paths:")
        for path in possible_paths:
            logger.error(f"  - {path}")
        raise FileNotFoundError("No video data directory found")
    
    train_videos, train_labels, val_videos, val_labels, test_videos, test_labels, action_classes = load_data(data_path)

    # Initialize VideoMAE components
    logger.info("Initializing VideoMAE model and processor...")
    
    # Try to authenticate with Hugging Face (optional - you can skip this if you have a token)
    try:
        # Uncomment the next line and add your HF token if needed
        # login(token="your_huggingface_token_here")
        logger.info("Attempting to load VideoMAE model...")
        
        image_processor = VideoMAEImageProcessor.from_pretrained(MODEL_CKPT)
        
        # Create class mappings for VideoMAE
        label2id = {label: i for i, label in enumerate(action_classes)}
        id2label = {i: label for label, i in label2id.items()}
        
        model = VideoMAEForVideoClassification.from_pretrained(
            MODEL_CKPT,
            label2id=label2id,
            id2label=id2label,
            ignore_mismatched_sizes=True,
        )
        logger.info("VideoMAE model loaded successfully!")
        
    except Exception as e:
        logger.error(f"Failed to load VideoMAE model: {e}")
        logger.info("Trying alternative VideoMAE model...")
        
        # Try alternative model that might not require authentication
        try:
            alt_model = "facebook/videomae-base"
            logger.info(f"Trying alternative model: {alt_model}")
            
            image_processor = VideoMAEImageProcessor.from_pretrained(alt_model)
            
            # Create class mappings for VideoMAE
            label2id = {label: i for i, label in enumerate(action_classes)}
            id2label = {i: label for label, i in label2id.items()}
            
            model = VideoMAEForVideoClassification.from_pretrained(
                alt_model,
                label2id=label2id,
                id2label=id2label,
                ignore_mismatched_sizes=True,
            )
            logger.info("Alternative VideoMAE model loaded successfully!")
            
        except Exception as e2:
            logger.error(f"Failed to load alternative model: {e2}")
            logger.info("Please authenticate with Hugging Face or check your internet connection.")
            logger.info("You can authenticate by:")
            logger.info("1. Running: huggingface-cli login")
            logger.info("2. Or get a token from https://huggingface.co/settings/tokens")
            logger.info("3. Or uncomment and set the login token in the code")
            raise e2
    
    logger.info("Creating datasets...")
    train_dataset = VideoMAEDataset(train_videos, train_labels, image_processor)
    val_dataset = VideoMAEDataset(val_videos, val_labels, image_processor)
    test_dataset = VideoMAEDataset(test_videos, test_labels, image_processor)

    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    logger.info(f"Test samples: {len(test_dataset)}")

    trainloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        prefetch_factor=PREFETCH_FACTOR,
        persistent_workers=True if NUM_WORKERS > 0 else False,
        collate_fn=collate_fn
    )

    valloader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        prefetch_factor=PREFETCH_FACTOR,
        persistent_workers=True if NUM_WORKERS > 0 else False,
        collate_fn=collate_fn
    )

    testloader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        prefetch_factor=PREFETCH_FACTOR,
        persistent_workers=True if NUM_WORKERS > 0 else False,
        collate_fn=collate_fn
    )

    logger.info("Preparing model for training...")
    model = prepare_model(model, num_gpus=NUM_GPUS)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)

    # Updated checkpoint path for PoseNet VideoMAE training
    if args.checkpoint_dir:
        checkpoint_path = os.path.join(args.checkpoint_dir, "best_checkpoint.pth")
        logger.info(f"Using specified checkpoint directory: {args.checkpoint_dir}")
    else:
        checkpoint_path = os.path.join("posenet_videomae_checkpoints_previous", "best_checkpoint.pth")
    
    start_epoch = 0

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        logger.info(f"Resuming training from epoch {start_epoch}")
    else:
        logger.info("No checkpoint found, starting training from scratch.")

    logger.info("Starting VideoMAE training on video clips...")
    logger.info(f"Training for {args.epochs} epochs with patience {args.patience}")
    best_val_loss, metrics = train_model(
        model, trainloader, valloader, testloader,
        criterion, optimizer, scheduler,
        args.epochs, args.patience, device,
        checkpoints_dir, action_classes, start_epoch
    )

    plot_metrics(metrics, save_dir=checkpoints_dir)
    logger.info(f"Training metrics plots saved to {checkpoints_dir}")

    final_model_path = os.path.join(checkpoints_dir, "posenet_videomae_model_final.pth")
    if isinstance(model, nn.DataParallel):
        torch.save(model.module.state_dict(), final_model_path)
    else:
        torch.save(model.state_dict(), final_model_path)
    logger.info(f"Final model saved as {final_model_path}")

    logger.info("PoseNet VideoMAE training complete!")

if __name__ == "__main__":
    main()
