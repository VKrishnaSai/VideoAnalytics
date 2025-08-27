import os
import time
import logging
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
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Import your ViViT model and load_video function from vivit.py
from vivit import ViViT, load_video, load_video_new

# ----------------------- Logging Setup -----------------------
log_filename = f'vivit_training_16thApril.log'
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s'
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)
logger = logging.getLogger(__name__)
logger.info("Training started")

# ----------------------- Performance Settings -----------------------
NUM_GPUS = torch.cuda.device_count()
NUM_WORKERS = 8
PIN_MEMORY = True
BENCHMARK = True
PREFETCH_FACTOR = 2
torch.backends.cudnn.benchmark = BENCHMARK
BATCH_SIZE = 16
MIXED_PRECISION = True

# ----------------------- Custom Dataset with Caching -----------------------
class VideoDataset(Dataset):
    def __init__(self, video_paths, labels, num_frames=16, image_size=224, cache_size=100):
        self.video_paths = video_paths
        self.labels = labels
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
            video = load_video_new(self.video_paths[idx], num_frames=self.num_frames, image_size=self.image_size)
            if video.ndim == 5 and video.shape[0] == 1:
                video = video.squeeze(0)
            if len(self.cache) >= self.cache_size:
                remove_key = list(self.cache.keys())[0]
                del self.cache[remove_key]
            self.cache[idx] = video
            
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return video, label

# ----------------------- Data Preparation -----------------------
def load_data(data_root="./UCF101/UCF-101"):
    class_index_file = "./ucfTrainTestlist/classInd.txt"
    class_indices = {}
    with open(class_index_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                idx, class_name = parts
                class_indices[int(idx)] = class_name

    data_subset = [class_indices[i] for i in sorted(class_indices.keys())]
    logger.info(f"Classes: {data_subset}")

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

    train_videos = []
    train_labels = []

    for trainlist in trainlist_files:
        with open(trainlist, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                video_rel_path, label_str = line.split()
                label = int(label_str) - 1
                video_path = os.path.join(data_root, video_rel_path)
                if not os.path.isfile(video_path):
                    logger.warning(f"Warning: Video file {video_path} does not exist!")
                    continue
                train_videos.append(video_path)
                train_labels.append(label)

    logger.info(f"Loaded {len(train_videos)} training videos.")

    testlist_videos = []
    testlist_labels = []

    for testlist in testlist_files:
        with open(testlist, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) == 2:
                    video_rel_path, label_str = parts
                    label = int(label_str) - 1
                else:
                    video_rel_path = parts[0]
                    class_name = video_rel_path.split('/')[0]
                    try:
                        label = data_subset.index(class_name)
                    except ValueError:
                        logger.warning(f"Warning: Class {class_name} not found in data_subset!")
                        continue

                video_path = os.path.join(data_root, video_rel_path)
                if not os.path.isfile(video_path):
                    logger.warning(f"Warning: Video file {video_path} does not exist!")
                    continue
                testlist_videos.append(video_path)
                testlist_labels.append(label)

    logger.info(f"Loaded {len(testlist_videos)} videos from testlist files.")
    
    return train_videos, train_labels, testlist_videos, testlist_labels, data_subset

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

# ----------------------- Optimized Training Function -----------------------
def train_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch, total_epochs, class_names, save_dir):
    model.train()
    epoch_loss = 0.0
    total_count = 0
    all_train_preds = []
    all_train_labels = []
    batch_times = []
    
    start_time = time.time()
    for batch_idx, (videos_batch, labels_batch) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs}")):
        batch_start = time.time()
        videos_batch, labels_batch = videos_batch.to(device, non_blocking=PIN_MEMORY), labels_batch.to(device, non_blocking=PIN_MEMORY)
        
        if MIXED_PRECISION:
            with autocast('cuda'):
                outputs = model(videos_batch)
                loss = criterion(outputs, labels_batch)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(videos_batch)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()
            
        optimizer.zero_grad(set_to_none=True)
        
        batch_size = labels_batch.size(0)
        epoch_loss += loss.item() * batch_size
        total_count += batch_size
        
        preds = outputs.argmax(dim=1).detach().cpu().numpy()
        labels_np = labels_batch.detach().cpu().numpy()
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
        for videos_batch, labels_batch in tqdm(val_loader, desc="Validation"):
            videos_batch = videos_batch.to(device, non_blocking=PIN_MEMORY)
            labels_batch = labels_batch.to(device, non_blocking=PIN_MEMORY)
            
            if MIXED_PRECISION:
                with autocast('cuda'): # added cuda 10th april
                    outputs = model(videos_batch)
                    loss = criterion(outputs, labels_batch)
            else:
                outputs = model(videos_batch)
                loss = criterion(outputs, labels_batch)
                
            batch_size = labels_batch.size(0)
            val_loss += loss.item() * batch_size
            val_count += batch_size
            
            preds = outputs.argmax(dim=1).cpu().numpy()
            val_labels.extend(labels_batch.cpu().numpy())
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
        for videos_batch, labels_batch in tqdm(test_loader, desc="Testing"):
            videos_batch = videos_batch.to(device, non_blocking=PIN_MEMORY)
            labels_batch = labels_batch.to(device, non_blocking=PIN_MEMORY)
            
            with autocast('cuda', enabled=MIXED_PRECISION):
                outputs = model(videos_batch)
            
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels_batch.cpu().numpy())
    
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
    if torch.cuda.is_available():
        device = torch.device("cuda")
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        logger.info(f"CUDA version: {torch.version.cuda}")
    else:
        device = torch.device("cpu")
        logger.info("No GPU available, using CPU")
    
    checkpoints_dir = f"checkpoints_{datetime.now().strftime('%Y%m%d_%H%M')}"
    logger.info(f"Creating checkpoints directory: {checkpoints_dir}")
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    train_videos, train_labels, testlist_videos, testlist_labels, data_subset = load_data()
    
    logger.info("Creating datasets...")
    train_dataset = VideoDataset(train_videos, train_labels)
    testlist_dataset = VideoDataset(testlist_videos, testlist_labels)
    
    all_indices = list(range(len(testlist_dataset)))
    val_indices, test_indices = train_test_split(
        all_indices, test_size=0.5, random_state=42, stratify=testlist_labels
    )
    
    logger.info(f"Validation samples: {len(val_indices)}, Test samples: {len(test_indices)}")
    
    val_dataset = Subset(testlist_dataset, val_indices)
    test_dataset = Subset(testlist_dataset, test_indices)
    
    trainloader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        prefetch_factor=PREFETCH_FACTOR,
        persistent_workers=True if NUM_WORKERS > 0 else False
    )
    
    valloader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        prefetch_factor=PREFETCH_FACTOR,
        persistent_workers=True if NUM_WORKERS > 0 else False
    )
    
    testloader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        prefetch_factor=PREFETCH_FACTOR,
        persistent_workers=True if NUM_WORKERS > 0 else False
    )
    
    logger.info("Initializing model...")
    model = ViViT(image_size=224, patch_size=16, num_classes=len(data_subset), num_frames=16)
    model = prepare_model(model, num_gpus=NUM_GPUS)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)
    
    checkpoint_path = os.path.join("checkpoints_28feb", "none.pth")
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
    
    logger.info("Starting training...")
    additional_epochs = 40
    patience = 5
    best_val_loss, metrics = train_model(
        model, trainloader, valloader, testloader, 
        criterion, optimizer, scheduler, 
        additional_epochs, patience, device, 
        checkpoints_dir, data_subset, start_epoch
    )
    
    plot_metrics(metrics, save_dir=checkpoints_dir)
    logger.info(f"Training metrics plots saved to {checkpoints_dir}")
    
    final_model_path = os.path.join(checkpoints_dir, "vivit_model_final.pth")
    if isinstance(model, nn.DataParallel):
        torch.save(model.module.state_dict(), final_model_path)
    else:
        torch.save(model.state_dict(), final_model_path)
    logger.info(f"Final model saved as {final_model_path}")
    
    logger.info("Training complete!")

if __name__ == "__main__":
    main()