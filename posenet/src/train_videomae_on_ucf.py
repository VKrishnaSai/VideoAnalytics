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
from torch.cuda.amp import autocast, GradScaler
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
log_filename = f'ucf101_videomae_training_{datetime.now().strftime("%Y%m%d_%H%M")}.log'
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s'
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)
logger = logging.getLogger(__name__)
logger.info("UCF-101 VideoMAE Training started")

# ----------------------- Performance Settings -----------------------
NUM_GPUS = torch.cuda.device_count()
NUM_WORKERS = 16  # Set to 0 for cluster stability - multiprocessing can cause hangs
PIN_MEMORY = True  # Disable pin_memory for cluster compatibility
BENCHMARK = True
PREFETCH_FACTOR = 2
torch.backends.cudnn.benchmark = BENCHMARK
BATCH_SIZE = 32 # Reduced batch size for VideoMAE to avoid memory issues
MIXED_PRECISION = True

# VideoMAE specific settings
MODEL_CKPT = "MCG-NJU/videomae-base"

# ----------------------- Video Loading Function for VideoMAE -----------------------
import decord

def load_video_for_videomae(video_path, num_frames=16, image_size=224):
    """
    Load and preprocess video using the much faster Decord library.
    """
    try:
        # Use decord.VideoReader for fast, random frame access
        vr = decord.VideoReader(video_path, width=image_size, height=image_size)
        total_frames = len(vr)

        if total_frames == 0:
            logger.warning(f"No frames found in video: {video_path}")
            return torch.zeros((num_frames, 3, image_size, image_size))

        # Sample frame indices uniformly
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

        # Get all frames in one go, which is highly optimized
        frames = vr.get_batch(frame_indices).asnumpy()

        # Convert to tensor, normalize, and permute
        # (N, H, W, C) -> (N, C, H, W)
        video_tensor = torch.from_numpy(frames).float() / 255.0
        video_tensor = video_tensor.permute(0, 3, 1, 2)

        return video_tensor

    except decord.DECORDError as e:
        logger.error(f"Decord error loading video {video_path}: {e}")
        # Return a dummy tensor if video is corrupt or unreadable
        return torch.zeros((num_frames, 3, image_size, image_size))

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
def load_ucf101_data(data_root="./UCF-101", subset_classes=None):
    """
    Load UCF-101 video data using all three official train/test splits (trainlist01-03, testlist01-03).
    """
    # UCF-101 class indices file
    class_index_file = "./ucfTrainTestlist/classInd.txt"
    
    # Load class mappings
    class_indices = {}
    if os.path.exists(class_index_file):
        with open(class_index_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    idx, class_name = parts
                    class_indices[int(idx)] = class_name
    else:
        logger.warning(f"Class index file not found: {class_index_file}")
        # Fallback: scan directories
        if os.path.exists(data_root):
            dirs = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]
            class_indices = {i+1: class_name for i, class_name in enumerate(sorted(dirs))}
        else:
            logger.error(f"Data directory does not exist: {data_root}")
            return [], [], [], [], [], [], []
    
    # Get all available classes
    all_classes = [class_indices[i] for i in sorted(class_indices.keys())]
    
    # Use subset if specified
    if subset_classes and subset_classes < len(all_classes):
        action_classes = all_classes[:subset_classes]
        logger.info(f"Using subset of {subset_classes} classes: {action_classes}")
    else:
        action_classes = all_classes
        logger.info(f"Using all {len(action_classes)} UCF-101 classes")
    
    # Create class to index mapping for our subset
    class_to_idx = {class_name: idx for idx, class_name in enumerate(action_classes)}
    
    # Load all train/test splits (1, 2, 3)
    trainlist_files = [
        "./ucfTrainTestlist/trainlist01.txt",
        "./ucfTrainTestlist/trainlist02.txt", 
        "./ucfTrainTestlist/trainlist03.txt"
    ]
    testlist_files = [
        "./ucfTrainTestlist/testlist01.txt",
        "./ucfTrainTestlist/testlist02.txt",
        "./ucfTrainTestlist/testlist03.txt"
    ]
    
    # Check if split files exist
    missing_files = []
    for file_list in [trainlist_files, testlist_files]:
        for file_path in file_list:
            if not os.path.exists(file_path):
                missing_files.append(file_path)
    
    if missing_files:
        logger.error("Missing UCF-101 train/test split files:")
        for missing in missing_files:
            logger.error(f"  - {missing}")
        logger.error("")
        logger.error("Please download train/test splits from:")
        logger.error("https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip")
        logger.error("Or run: python setup_ucf101.py for automated setup")
        logger.error("")
        raise FileNotFoundError("Missing UCF-101 train/test split files")
    
    train_videos = []
    train_labels = []
    
    # Load training videos from all three splits
    for trainlist in trainlist_files:
        if os.path.exists(trainlist):
            logger.info(f"Loading training data from {trainlist}")
            with open(trainlist, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) >= 2:
                        # Format: video_path label
                        video_rel_path = parts[0]
                        label = int(parts[1]) - 1  # Convert to 0-based indexing
                    else:
                        # Format: video_path only
                        video_rel_path = parts[0]
                        class_name = video_rel_path.split('/')[0]
                        if class_name not in class_to_idx:
                            continue
                        label = class_to_idx[class_name]
                    
                    # Check if label is within our subset range
                    if label < len(action_classes):
                        video_path = os.path.join(data_root, video_rel_path)
                        if os.path.isfile(video_path):
                            train_videos.append(video_path)
                            train_labels.append(label)
                        else:
                            logger.warning(f"Training video file not found: {video_path}")
        else:
            logger.warning(f"Training list file not found: {trainlist}")
    
    # Load test videos from all three splits
    test_videos = []
    test_labels = []
    
    for testlist in testlist_files:
        if os.path.exists(testlist):
            logger.info(f"Loading test data from {testlist}")
            with open(testlist, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) >= 2:
                        # Format: video_path label
                        video_rel_path = parts[0]
                        label = int(parts[1]) - 1  # Convert to 0-based indexing
                    else:
                        # Format: video_path only - extract class from path
                        video_rel_path = parts[0]
                        class_name = video_rel_path.split('/')[0]
                        if class_name not in class_to_idx:
                            continue
                        label = class_to_idx[class_name]
                    
                    # Check if label is within our subset range
                    if label < len(action_classes):
                        video_path = os.path.join(data_root, video_rel_path)
                        if os.path.isfile(video_path):
                            test_videos.append(video_path)
                            test_labels.append(label)
                        else:
                            logger.warning(f"Test video file not found: {video_path}")
        else:
            logger.warning(f"Test list file not found: {testlist}")
    
    if not train_videos and not test_videos:
        logger.error(f"No video files found in {data_root}")
        return [], [], [], [], [], [], action_classes
    
    logger.info(f"Found {len(train_videos)} training videos and {len(test_videos)} test videos")
    
    # Count videos per class
    train_label_counts = Counter(train_labels)
    test_label_counts = Counter(test_labels)
    
    logger.info("Training distribution:")
    for i, class_name in enumerate(action_classes):
        count = train_label_counts.get(i, 0)
        logger.info(f"  {class_name}: {count} videos")
    
    logger.info("Test distribution:")
    for i, class_name in enumerate(action_classes):
        count = test_label_counts.get(i, 0)
        logger.info(f"  {class_name}: {count} videos")
    
    # Create validation split from training data (20% of training)
    if len(train_videos) > 0:
        # Stratified split to maintain class balance
        train_videos_split, val_videos, train_labels_split, val_labels = train_test_split(
            train_videos, train_labels, 
            test_size=0.2, random_state=42, 
            stratify=train_labels
        )
        train_videos = train_videos_split
        train_labels = train_labels_split
    else:
        val_videos = []
        val_labels = []
    
    logger.info(f"Final split - Training: {len(train_videos)}, Validation: {len(val_videos)}, Test: {len(test_videos)}")
    
    # Log distribution for all splits
    train_counts = Counter(train_labels)
    val_counts = Counter(val_labels)
    test_counts = Counter(test_labels)
    
    logger.info("Final training distribution:")
    for i, class_name in enumerate(action_classes):
        count = train_counts.get(i, 0)
        logger.info(f"  {class_name}: {count} videos")
    
    logger.info("Validation distribution:")
    for i, class_name in enumerate(action_classes):
        count = val_counts.get(i, 0)
        logger.info(f"  {class_name}: {count} videos")
    
    logger.info("Test distribution:")
    for i, class_name in enumerate(action_classes):
        count = test_counts.get(i, 0)
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
        
        # Debug: Log first few batches to identify hanging point
        if batch_idx < 5:
            logger.info(f"Processing batch {batch_idx+1}")
        
        try:
            pixel_values = batch["pixel_values"].to(device, non_blocking=PIN_MEMORY)
            labels = batch["labels"].to(device, non_blocking=PIN_MEMORY)
            
            if batch_idx < 5:
                logger.info(f"Batch {batch_idx+1}: Data moved to device, shapes: {pixel_values.shape}, {labels.shape}")
        except Exception as e:
            logger.error(f"Error loading batch {batch_idx}: {e}")
            continue

        try:
            if batch_idx < 5:
                # Log memory usage for first few batches
                if torch.cuda.is_available():
                    memory_allocated = torch.cuda.memory_allocated() / 1024**3
                    memory_reserved = torch.cuda.memory_reserved() / 1024**3
                    logger.info(f"Batch {batch_idx+1}: GPU memory - Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB")
            
            if MIXED_PRECISION:
                with torch.cuda.amp.autocast( enabled=MIXED_PRECISION):
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
            
            if batch_idx < 5:
                logger.info(f"Batch {batch_idx+1}: Forward pass completed, loss: {loss.item():.4f}")
                
        except RuntimeError as e:
            if "out of memory" in str(e):
                logger.error(f"GPU out of memory error in batch {batch_idx}: {e}")
                logger.error("Try reducing batch size or using gradient accumulation")
                # Clear cache and continue
                torch.cuda.empty_cache()
                optimizer.zero_grad(set_to_none=True)
                continue
            else:
                logger.error(f"Runtime error in forward pass for batch {batch_idx}: {e}")
                raise e
        except Exception as e:
            logger.error(f"Error in forward pass for batch {batch_idx}: {e}")
            # Clear gradients and continue
            optimizer.zero_grad(set_to_none=True)
            continue

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
                with torch.cuda.amp.autocast( enabled=MIXED_PRECISION):
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

            with torch.cuda.amp.autocast( enabled=MIXED_PRECISION):
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

    scaler = torch.cuda.amp.GradScaler(enabled=MIXED_PRECISION)
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
    parser = argparse.ArgumentParser(description="Train VideoMAE on UCF-101 dataset")
    parser.add_argument("--epochs", type=int, default=300, help="Number of epochs to train (default: 300)")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience (default: 15)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Checkpoint directory to resume from")
    parser.add_argument("--subset_classes", type=int, default=None, help="Number of classes to use (default: all 101 classes)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with minimal data")
    args = parser.parse_args()
    
    # Update global batch size if specified
    global BATCH_SIZE
    BATCH_SIZE = args.batch_size
    
    # Debug mode: Use minimal settings for testing
    if args.debug:
        logger.info("DEBUG MODE: Using minimal settings for testing")
        args.subset_classes = 2  # Only 2 classes
        args.epochs = 2  # Only 2 epochs
        BATCH_SIZE = 1  # Single batch for debugging
        logger.info(f"Debug settings: {args.subset_classes} classes, {args.epochs} epochs, batch_size={BATCH_SIZE}")
        
        # Additional debug settings for memory optimization
        torch.cuda.empty_cache()  # Clear any existing cache
        logger.info("GPU cache cleared for debug mode")
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        logger.info(f"CUDA version: {torch.version.cuda}")
    else:
        device = torch.device("cpu")
        logger.info("No GPU available, using CPU")

    checkpoints_dir = f"ucf101_videomae_checkpoints_{datetime.now().strftime('%Y%m%d_%H%M')}"
    logger.info(f"Creating checkpoints directory: {checkpoints_dir}")
    os.makedirs(checkpoints_dir, exist_ok=True)

    # Load UCF-101 dataset using all three train/test splits
    logger.info("Loading UCF-101 dataset...")
    
    # Try multiple possible UCF-101 data paths
    possible_paths = [
        "./UCF-101/",
        "../UCF-101", 
        "./data/UCF-101",
        "./UCF101/UCF-101",
        "C:/datasets/UCF-101",
        "/datasets/UCF-101"
    ]
    
    data_path = None
    for path in possible_paths:
        if os.path.exists(path):
            data_path = path
            logger.info(f"Found UCF-101 data directory: {path}")
            break
    
    if data_path is None:
        logger.error("No UCF-101 data directory found! Tried paths:")
        for path in possible_paths:
            logger.error(f"  - {path}")
        logger.error("")
        logger.error("Please ensure UCF-101 dataset is available. You can:")
        logger.error("1. Download UCF-101 from: https://www.crcv.ucf.edu/data/UCF101/UCF101.rar")
        logger.error("2. Extract to create UCF-101/ directory with 101 class folders")
        logger.error("3. Run: python setup_ucf101.py for automated setup help")
        logger.error("")
        raise FileNotFoundError("No UCF-101 data directory found")
    
    # Use subset of classes for testing (set to None for all 101 classes)
    subset_classes = args.subset_classes  # Use all 101 classes by default, or specify subset for testing
    
    train_videos, train_labels, val_videos, val_labels, test_videos, test_labels, action_classes = load_ucf101_data(
        data_root=data_path, subset_classes=subset_classes
    )

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

    # Test data loading before creating full dataloaders
    logger.info("Testing data loading...")
    try:
        test_sample = train_dataset[0]
        logger.info(f"Sample loaded successfully: video shape {test_sample[0].shape}, label {test_sample[1]}")
    except Exception as e:
        logger.error(f"Error loading test sample: {e}")
        raise e

    trainloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        prefetch_factor=PREFETCH_FACTOR if NUM_WORKERS > 0 else None,
        persistent_workers=False,  # Disable for cluster stability
        collate_fn=collate_fn,
        drop_last=True  # Ensure consistent batch sizes
    )

    valloader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        prefetch_factor=PREFETCH_FACTOR if NUM_WORKERS > 0 else None,
        persistent_workers=False,  # Disable for cluster stability
        collate_fn=collate_fn,
        drop_last=False
    )

    testloader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        prefetch_factor=PREFETCH_FACTOR if NUM_WORKERS > 0 else None,
        persistent_workers=False,  # Disable for cluster stability
        collate_fn=collate_fn,
        drop_last=False
    )

    logger.info("Preparing model for training...")
    model = prepare_model(model, num_gpus=NUM_GPUS)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)

    # Updated checkpoint path for UCF-101 VideoMAE training
    if args.checkpoint_dir:
        checkpoint_path = os.path.join(args.checkpoint_dir, "best_checkpoint.pth")
        logger.info(f"Using specified checkpoint directory: {args.checkpoint_dir}")
    else:
        checkpoint_path = os.path.join("ucf101_videomae_checkpoints_previous", "best_checkpoint.pth")
    
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

    logger.info("Starting UCF-101 VideoMAE training...")
    logger.info(f"Training for {args.epochs} epochs with patience {args.patience}")
    best_val_loss, metrics = train_model(
        model, trainloader, valloader, testloader,
        criterion, optimizer, scheduler,
        args.epochs, args.patience, device,
        checkpoints_dir, action_classes, start_epoch
    )

    plot_metrics(metrics, save_dir=checkpoints_dir)
    logger.info(f"Training metrics plots saved to {checkpoints_dir}")

    final_model_path = os.path.join(checkpoints_dir, "ucf101_videomae_model_final.pth")
    if isinstance(model, nn.DataParallel):
        torch.save(model.module.state_dict(), final_model_path)
    else:
        torch.save(model.state_dict(), final_model_path)
    logger.info(f"Final model saved as {final_model_path}")

    logger.info("UCF-101 VideoMAE training complete!")

if __name__ == "__main__":
    main()
