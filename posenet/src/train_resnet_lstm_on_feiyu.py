"""
ResNet + LSTM (CRNN) Training Script for PoseNet Video Clips
Adapted from hhtseng github to work with video clips created by split_videos_to_clips.py
"""

import os
import time
import logging
import argparse
import numpy as np
from datetime import datetime
from collections import Counter
import signal
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import torch.utils.data as data
from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Import CRNN models and functions
from resnetrcnn_functions import ResCNNEncoder, DecoderRNN, Dataset_CRNN
from vivit_old import load_video  # For video loading

# ----------------------- Early Stopping Class -----------------------
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            restore_best_weights (bool): Whether to restore model weights from the best epoch.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, val_loss, model, save_dir, epoch):
        """
        Call method to check if training should stop.
        
        Args:
            val_loss (float): Current validation loss
            model (tuple): (cnn_encoder, rnn_decoder) model tuple
            save_dir (str): Directory to save best model
            epoch (int): Current epoch number
        """
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.save_checkpoint(model, save_dir)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                logger.info(f'Early stopping triggered! Best validation loss: {self.best_loss:.6f} at epoch {self.best_epoch + 1}')
        else:
            self.best_score = score
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.save_checkpoint(model, save_dir)
            self.counter = 0

    def save_checkpoint(self, model, save_dir):
        """Saves model when validation loss decreases."""
        cnn_encoder, rnn_decoder = model
        try:
            # Handle DataParallel models
            if hasattr(cnn_encoder, 'module'):
                cnn_state_dict = cnn_encoder.module.state_dict()
            else:
                cnn_state_dict = cnn_encoder.state_dict()
                
            if hasattr(rnn_decoder, 'module'):
                rnn_state_dict = rnn_decoder.module.state_dict()
            else:
                rnn_state_dict = rnn_decoder.state_dict()
            
            # Save with CPU mapping to avoid GPU memory issues
            torch.save(cnn_state_dict, os.path.join(save_dir, 'early_stop_best_cnn_encoder.pth'))
            torch.save(rnn_state_dict, os.path.join(save_dir, 'early_stop_best_rnn_decoder.pth'))
            
            # Clear cache after saving
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            logger.info(f'Early stopping checkpoint saved (val_loss: {self.best_loss:.6f})')
        except Exception as e:
            logger.error(f'Error saving early stopping checkpoint: {e}')

    def load_best_weights(self, model, save_dir, device):
        """Load the best model weights."""
        if self.restore_best_weights:
            cnn_encoder, rnn_decoder = model
            try:
                # Load state dicts
                cnn_state_dict = torch.load(os.path.join(save_dir, 'early_stop_best_cnn_encoder.pth'), map_location=device)
                rnn_state_dict = torch.load(os.path.join(save_dir, 'early_stop_best_rnn_decoder.pth'), map_location=device)
                
                # Handle DataParallel models
                if hasattr(cnn_encoder, 'module'):
                    cnn_encoder.module.load_state_dict(cnn_state_dict)
                else:
                    cnn_encoder.load_state_dict(cnn_state_dict)
                    
                if hasattr(rnn_decoder, 'module'):
                    rnn_decoder.module.load_state_dict(rnn_state_dict)
                else:
                    rnn_decoder.load_state_dict(rnn_state_dict)
                
                # Clear cache after loading
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
                logger.info(f'Loaded best model weights from epoch {self.best_epoch + 1}')
            except Exception as e:
                logger.error(f'Error loading best weights: {e}')
                return False
            return True
        return False

# ----------------------- Logging Setup -----------------------
log_filename = f'posenet_resnet_lstm_training_{datetime.now().strftime("%Y%m%d_%H%M")}.log'
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s'
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)
logger = logging.getLogger(__name__)
logger.info("PoseNet ResNet+LSTM Training started")

# ----------------------- Performance Settings -----------------------
NUM_GPUS = torch.cuda.device_count()
MIXED_PRECISION = True

# ----------------------- Model Parameters -----------------------
# ResNet CNN Encoder parameters
CNN_fc_hidden1, CNN_fc_hidden2 = 1024, 768
CNN_embed_dim = 512   # latent dim extracted by 2D CNN
res_size = 224        # ResNet image size
dropout_p = 0.3       # dropout probability

# LSTM Decoder parameters
RNN_hidden_layers = 3
RNN_hidden_nodes = 512
RNN_FC_dim = 256

# Training parameters
epochs = 50          # training epochs
batch_size = 8       # smaller batch size for video data
learning_rate = 1e-4
weight_decay = 1e-5
log_interval = 10    # interval for displaying training info

# Early stopping parameters
early_stop_patience = 10  # Number of epochs to wait for improvement
early_stop_min_delta = 0.001  # Minimum change to qualify as improvement

# Video parameters - use fewer frames for CRNN vs ViViT
num_frames = 8       # Number of frames to sample from each video
begin_frame, end_frame, skip_frame = 0, num_frames, 1

# ----------------------- Video Dataset for CRNN -----------------------
class VideoDataset_CRNN(Dataset):
    """Dataset class that loads video clips and converts them to frame sequences for CRNN"""
    def __init__(self, video_paths, labels, num_frames=8, image_size=224, transform=None):
        self.video_paths = video_paths
        self.labels = labels
        self.num_frames = num_frames
        self.image_size = image_size
        self.transform = transform

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]

        try:
            # Load video and convert to frame sequence
            video_tensor = load_video(video_path, num_frames=self.num_frames, image_size=self.image_size)
            
            # video_tensor shape: (1, num_frames, 3, H, W) or (num_frames, 3, H, W)
            if video_tensor.ndim == 5 and video_tensor.shape[0] == 1:
                video_tensor = video_tensor.squeeze(0)  # Remove batch dimension: (num_frames, 3, H, W)
            
            # Ensure we have the correct shape
            if video_tensor.ndim != 4:
                raise ValueError(f"Expected 4D tensor (num_frames, 3, H, W), got shape {video_tensor.shape}")
            
            # Apply additional transforms if specified
            if self.transform:
                frames = []
                for i in range(video_tensor.shape[0]):
                    frame = video_tensor[i]  # (3, H, W)
                    # Convert to PIL Image for transform
                    frame_pil = transforms.ToPILImage()(frame)
                    frame_transformed = self.transform(frame_pil)
                    frames.append(frame_transformed)
                video_tensor = torch.stack(frames, dim=0)

            return video_tensor, torch.tensor(label, dtype=torch.long)
            
        except Exception as e:
            logger.error(f"Error loading video {video_path}: {str(e)}")
            # Return a dummy tensor with correct shape
            dummy_tensor = torch.zeros(self.num_frames, 3, self.image_size, self.image_size)
            if self.transform:
                frames = []
                for i in range(self.num_frames):
                    frame_pil = transforms.ToPILImage()(dummy_tensor[i])
                    frame_transformed = self.transform(frame_pil)
                    frames.append(frame_transformed)
                dummy_tensor = torch.stack(frames, dim=0)
            return dummy_tensor, torch.tensor(label, dtype=torch.long)

# ----------------------- Data Loading Function -----------------------
def load_data(data_root="./video_clips"):
    """
    Load PoseNet video data from the output/video_clips directory.
    """
    # Define action classes based on your config
    action_classes = ['stand', 'walk', 'run', 'jump', 'sit', 'squat', 'kick', 'punch', 'wave']
    
    # Create class to index mapping
    class_to_idx = {class_name: idx for idx, class_name in enumerate(action_classes)}
    
    logger.info(f"Action classes: {action_classes}")
    
    # Get all video files from the output/video_clips directory
    video_files = []
    video_labels = []
    
    if not os.path.exists(data_root):
        logger.error(f"Data directory does not exist: {data_root}")
        return [], [], [], [], [], [], action_classes
    
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
        return [], [], [], [], [], [], action_classes
    
    logger.info(f"Found {len(video_files)} video files")
    
    # Count videos per class
    label_counts = Counter(video_labels)
    for class_name, class_idx in class_to_idx.items():
        count = label_counts.get(class_idx, 0)
        logger.info(f"  {class_name}: {count} videos")
    
    # Smart train/val/test split strategy for small datasets
    min_samples_per_class = min(label_counts.values()) if label_counts else 0
    total_samples = len(video_files)
    
    logger.info(f"Total samples: {total_samples}, Min samples per class: {min_samples_per_class}")
    
    if min_samples_per_class == 1:
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
    
    return train_videos, train_labels, val_videos, val_labels, test_videos, test_labels, action_classes

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

# ----------------------- Training Function -----------------------
def train_epoch(model, device, train_loader, optimizer, scaler, epoch, log_interval, class_names, save_dir):
    # set model as training mode
    cnn_encoder, rnn_decoder = model
    cnn_encoder.train()
    rnn_decoder.train()

    losses = []
    all_train_preds = []
    all_train_labels = []
    N_count = 0   # counting total trained sample in one epoch
    batch_times = []
    
    start_time = time.time()
    for batch_idx, (X, y) in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch+1}")):
        batch_start = time.time()
        
        # Check tensor shapes and move to device safely
        try:
            # Print debug info for first batch
            if batch_idx == 0:
                logger.info(f"Input batch shape: {X.shape}, Labels shape: {y.shape}")
                logger.info(f"Input dtype: {X.dtype}, Labels dtype: {y.dtype}")
            
            # distribute data to device
            X, y = X.to(device, non_blocking=False), y.to(device, non_blocking=False).view(-1, )
            N_count += X.size(0)

            optimizer.zero_grad()
        except Exception as e:
            logger.error(f"Error moving batch {batch_idx} to device: {str(e)}")
            continue
        
        if MIXED_PRECISION:
            with autocast('cuda'):
                output = rnn_decoder(cnn_encoder(X))  # output has dim = (batch, number of classes)
                loss = F.cross_entropy(output, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            output = rnn_decoder(cnn_encoder(X))  # output has dim = (batch, number of classes)
            loss = F.cross_entropy(output, y)
            loss.backward()
            optimizer.step()

        # record loss and predictions
        losses.append(loss.item())
        y_pred = torch.max(output, 1)[1]  # y_pred != output
        
        # Collect all predictions and labels for detailed metrics
        all_train_preds.extend(y_pred.cpu().numpy())
        all_train_labels.extend(y.cpu().numpy())
        
        batch_times.append(time.time() - batch_start)

        # show information
        if (batch_idx + 1) % log_interval == 0:
            step_score = accuracy_score(y.cpu().data.squeeze().numpy(), y_pred.cpu().data.squeeze().numpy())
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\\tLoss: {:.6f}, Accu: {:.2f}%'.format(
                epoch + 1, N_count, len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader), loss.item(), 100 * step_score))

    epoch_time = time.time() - start_time
    epoch_loss_avg = sum(losses) / len(losses)
    
    # Compute comprehensive metrics for training epoch
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
def validate(model, device, test_loader, class_names, save_dir, epoch=None):
    # set model as testing mode
    cnn_encoder, rnn_decoder = model
    cnn_encoder.eval()
    rnn_decoder.eval()

    test_loss = 0
    all_y = []
    all_y_pred = []
    with torch.no_grad():
        for X, y in tqdm(test_loader, desc="Validation"):
            try:
                # distribute data to device
                X, y = X.to(device, non_blocking=False), y.to(device, non_blocking=False).view(-1, )

                if MIXED_PRECISION:
                    with autocast('cuda'):
                        output = rnn_decoder(cnn_encoder(X))
                        loss = F.cross_entropy(output, y, reduction='sum')
                else:
                    output = rnn_decoder(cnn_encoder(X))
                    loss = F.cross_entropy(output, y, reduction='sum')

                test_loss += loss.item()                 # sum up batch loss
                y_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability

                # collect all y and y_pred in all batches
                all_y.extend(y.cpu().numpy())
                all_y_pred.extend(y_pred.cpu().numpy().squeeze())
                
            except Exception as e:
                logger.error(f"Error in validation batch: {str(e)}")
                continue

    test_loss /= len(test_loader.dataset)

    # Compute comprehensive metrics
    overall_metrics, per_class_metrics = save_metrics(all_y, all_y_pred, class_names, save_dir, 
                                                     mode='val' if epoch is not None else 'test', epoch=epoch)

    # show information
    logger.info('\\nValidation set ({:d} samples): Average loss: {:.4f}, '
                'Accuracy: {:.2f}%, Precision: {:.2f}%, Recall: {:.2f}%, F1: {:.2f}%\\n'.format(
        len(all_y), test_loss, 
        100 * overall_metrics['accuracy'], 
        100 * overall_metrics['precision_macro'], 
        100 * overall_metrics['recall_macro'], 
        100 * overall_metrics['f1_macro']))

    return test_loss, overall_metrics, per_class_metrics

# ----------------------- Main Function -----------------------
def main():
    parser = argparse.ArgumentParser(description="Train ResNet+LSTM on PoseNet video clips")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train (default: 50)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    parser.add_argument("--num_frames", type=int, default=8, help="Number of frames per video (default: 8)")
    parser.add_argument("--data_root", type=str, default="./video_clips", help="Data root directory")
    parser.add_argument("--save_dir", type=str, default="./ResNetLSTM_PoseNet_ckpt/", help="Directory to save checkpoints")
    parser.add_argument("--early_stop_patience", type=int, default=10, help="Early stopping patience (default: 10)")
    parser.add_argument("--early_stop_min_delta", type=float, default=0.001, help="Early stopping minimum delta (default: 0.001)")
    parser.add_argument("--disable_early_stop", action="store_true", help="Disable early stopping")
    args = parser.parse_args()

    # Update global parameters
    global epochs, batch_size, learning_rate, num_frames, early_stop_patience, early_stop_min_delta
    epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    num_frames = args.num_frames
    early_stop_patience = args.early_stop_patience
    early_stop_min_delta = args.early_stop_min_delta

    # Device configuration with better error handling
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    if torch.cuda.is_available():
        # Clear CUDA cache and set memory fraction
        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(0.7)  # Use 70% of GPU memory instead of 80%
        
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            logger.info(f"GPU {i} Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
        
        # Simple CUDA functionality test with minimal memory usage
        try:
            test_tensor = torch.tensor([1.0], device=device)  # Minimal tensor
            test_result = test_tensor * 2
            logger.info("CUDA functionality test passed")
            del test_tensor, test_result
            torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f"CUDA functionality test failed: {e}")
            device = torch.device("cpu")
            logger.info("Falling back to CPU")
    else:
        logger.info("CUDA not available, using CPU")

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Load data
    logger.info("Loading video data...")
    train_videos, train_labels, val_videos, val_labels, test_videos, test_labels, action_classes = load_data(args.data_root)
    
    num_classes = len(action_classes)
    logger.info(f"Number of classes: {num_classes}")

    # Data transforms
    transform = transforms.Compose([
        transforms.Resize([res_size, res_size]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create datasets
    train_dataset = VideoDataset_CRNN(train_videos, train_labels, num_frames=num_frames, image_size=res_size, transform=transform)
    val_dataset = VideoDataset_CRNN(val_videos, val_labels, num_frames=num_frames, image_size=res_size, transform=transform)
    test_dataset = VideoDataset_CRNN(test_videos, test_labels, num_frames=num_frames, image_size=res_size, transform=transform)

    # Create data loaders - disable pin_memory to avoid CUDA pinning errors
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=False)

    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    logger.info(f"Test samples: {len(test_dataset)}")

    # Create model
    cnn_encoder = ResCNNEncoder(fc_hidden1=CNN_fc_hidden1, fc_hidden2=CNN_fc_hidden2, 
                               drop_p=dropout_p, CNN_embed_dim=CNN_embed_dim).to(device)
    rnn_decoder = DecoderRNN(CNN_embed_dim=CNN_embed_dim, h_RNN_layers=RNN_hidden_layers, 
                            h_RNN=RNN_hidden_nodes, h_FC_dim=RNN_FC_dim, 
                            drop_p=dropout_p, num_classes=num_classes).to(device)

    # Parallelize model to multiple GPUs
    if torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs!")
        cnn_encoder = nn.DataParallel(cnn_encoder)
        rnn_decoder = nn.DataParallel(rnn_decoder)
        
        # Combine all parameters
        crnn_params = (
            list(cnn_encoder.module.fc1.parameters()) +
            list(cnn_encoder.module.bn1.parameters()) +
            list(cnn_encoder.module.fc2.parameters()) +
            list(cnn_encoder.module.bn2.parameters()) +
            list(cnn_encoder.module.fc3.parameters()) +
            list(rnn_decoder.parameters())
        )
    else:
        crnn_params = (
            list(cnn_encoder.fc1.parameters()) +
            list(cnn_encoder.bn1.parameters()) +
            list(cnn_encoder.fc2.parameters()) +
            list(cnn_encoder.bn2.parameters()) +
            list(cnn_encoder.fc3.parameters()) +
            list(rnn_decoder.parameters())
        )

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(crnn_params, lr=learning_rate, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = GradScaler('cuda') if MIXED_PRECISION else None

    # Training loop
    logger.info("Starting training...")
    
    # Initialize early stopping
    early_stopping = None
    if not args.disable_early_stop:
        early_stopping = EarlyStopping(
            patience=early_stop_patience, 
            min_delta=early_stop_min_delta, 
            restore_best_weights=True
        )
        logger.info(f"Early stopping enabled - patience: {early_stop_patience}, min_delta: {early_stop_min_delta}")
    else:
        logger.info("Early stopping disabled")
    
    # Metrics tracking
    metrics_history = {
        "train_loss": [], "train_accuracy": [], "train_precision": [], "train_recall": [], "train_f1": [],
        "val_loss": [], "val_accuracy": [], "val_precision": [], "val_recall": [], "val_f1": []
    }
    
    best_val_score = 0.0
    total_start_time = time.time()
    
    # Graceful shutdown handler
    def signal_handler(signum, frame):
        logger.info("\\n" + "="*50)
        logger.info("Training interrupted! Saving current progress...")
        
        # Save current training history
        np.save(os.path.join(args.save_dir, 'train_losses_interrupted.npy'), metrics_history["train_loss"])
        np.save(os.path.join(args.save_dir, 'val_losses_interrupted.npy'), metrics_history["val_loss"])
        np.save(os.path.join(args.save_dir, 'val_scores_interrupted.npy'), metrics_history["val_accuracy"])
        
        # Plot current progress
        if len(metrics_history["train_loss"]) > 0:
            plot_comprehensive_metrics(metrics_history, save_dir=args.save_dir)
            logger.info("Training curves saved!")
        
        # Run final test evaluation if we have a best model
        best_encoder_path = os.path.join(args.save_dir, 'best_cnn_encoder.pth')
        best_decoder_path = os.path.join(args.save_dir, 'best_rnn_decoder.pth')
        early_stop_encoder_path = os.path.join(args.save_dir, 'early_stop_best_cnn_encoder.pth')
        early_stop_decoder_path = os.path.join(args.save_dir, 'early_stop_best_rnn_decoder.pth')
        
        # Try early stopping model first, then fall back to best model
        model_loaded = False
        if os.path.exists(early_stop_encoder_path) and os.path.exists(early_stop_decoder_path):
            logger.info("Running final test evaluation with early stopping best model...")
            try:
                # Load early stopping best model
                if isinstance(cnn_encoder, nn.DataParallel):
                    cnn_encoder.module.load_state_dict(torch.load(early_stop_encoder_path, map_location=device))
                    rnn_decoder.module.load_state_dict(torch.load(early_stop_decoder_path, map_location=device))
                else:
                    cnn_encoder.load_state_dict(torch.load(early_stop_encoder_path, map_location=device))
                    rnn_decoder.load_state_dict(torch.load(early_stop_decoder_path, map_location=device))
                model_loaded = True
                logger.info("Loaded early stopping best model")
            except Exception as e:
                logger.error(f"Error loading early stopping model: {e}")
        
        if not model_loaded and os.path.exists(best_encoder_path) and os.path.exists(best_decoder_path):
            logger.info("Running final test evaluation with best validation model...")
            try:
                # Load best model
                if isinstance(cnn_encoder, nn.DataParallel):
                    cnn_encoder.module.load_state_dict(torch.load(best_encoder_path, map_location=device))
                    rnn_decoder.module.load_state_dict(torch.load(best_decoder_path, map_location=device))
                else:
                    cnn_encoder.load_state_dict(torch.load(best_encoder_path, map_location=device))
                    rnn_decoder.load_state_dict(torch.load(best_decoder_path, map_location=device))
                model_loaded = True
                logger.info("Loaded best validation model")
            except Exception as e:
                logger.error(f"Error loading best model: {e}")
        
        if model_loaded:
            try:
                # Final test evaluation
                test_loss, test_metrics, _ = validate((cnn_encoder, rnn_decoder), device, test_loader, 
                                                    action_classes, args.save_dir, epoch=None)
                
                logger.info(f"Final Test Results (Best Model):")
                logger.info(f"  Accuracy: {test_metrics['accuracy']:.4f}")
                logger.info(f"  F1-Score (macro): {test_metrics['f1_macro']:.4f}")
                logger.info(f"  Best validation accuracy: {best_val_score:.4f}")
                
            except Exception as e:
                logger.error(f"Error in final test evaluation: {e}")
        
        logger.info(f"All results saved to: {args.save_dir}")
        logger.info("Training gracefully terminated!")
        sys.exit(0)
    
    # Register signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    for epoch in range(epochs):
        epoch_start_time = time.time()
        
        # Training
        train_loss, train_metrics, _ = train_epoch((cnn_encoder, rnn_decoder), device, train_loader, 
                                                 optimizer, scaler, epoch, log_interval, action_classes, args.save_dir)
        
        # Validation
        val_loss, val_metrics, _ = validate((cnn_encoder, rnn_decoder), device, val_loader, 
                                          action_classes, args.save_dir, epoch)
        
        # Update scheduler
        if scheduler:
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(f"Current learning rate: {current_lr:.7f}")

        # Record metrics history
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

        # Save best model
        if val_metrics['accuracy'] > best_val_score:
            best_val_score = val_metrics['accuracy']
            logger.info(f"New best validation accuracy: {best_val_score:.4f}")
            
            # Save best model with proper DataParallel handling
            try:
                if hasattr(cnn_encoder, 'module'):
                    torch.save(cnn_encoder.module.state_dict(), os.path.join(args.save_dir, 'best_cnn_encoder.pth'))
                    torch.save(rnn_decoder.module.state_dict(), os.path.join(args.save_dir, 'best_rnn_decoder.pth'))
                else:
                    torch.save(cnn_encoder.state_dict(), os.path.join(args.save_dir, 'best_cnn_encoder.pth'))
                    torch.save(rnn_decoder.state_dict(), os.path.join(args.save_dir, 'best_rnn_decoder.pth'))
                torch.save(optimizer.state_dict(), os.path.join(args.save_dir, 'best_optimizer.pth'))
                
                # Clear cache after saving
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                logger.error(f"Error saving best model: {e}")

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            try:
                if hasattr(cnn_encoder, 'module'):
                    torch.save(cnn_encoder.module.state_dict(), os.path.join(args.save_dir, f'cnn_encoder_epoch_{epoch+1}.pth'))
                    torch.save(rnn_decoder.module.state_dict(), os.path.join(args.save_dir, f'rnn_decoder_epoch_{epoch+1}.pth'))
                else:
                    torch.save(cnn_encoder.state_dict(), os.path.join(args.save_dir, f'cnn_encoder_epoch_{epoch+1}.pth'))
                    torch.save(rnn_decoder.state_dict(), os.path.join(args.save_dir, f'rnn_decoder_epoch_{epoch+1}.pth'))
                torch.save(optimizer.state_dict(), os.path.join(args.save_dir, f'optimizer_epoch_{epoch+1}.pth'))
                
                # Clear cache after saving
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
                logger.info(f"Checkpoint saved at epoch {epoch+1}")
            except Exception as e:
                logger.error(f"Error saving checkpoint at epoch {epoch+1}: {e}")
        
        # Early stopping check
        if early_stopping is not None:
            early_stopping(val_loss, (cnn_encoder, rnn_decoder), args.save_dir, epoch)
            if early_stopping.early_stop:
                logger.info("="*50)
                logger.info("Early stopping triggered!")
                
                # Load best weights
                if early_stopping.load_best_weights((cnn_encoder, rnn_decoder), args.save_dir, device):
                    logger.info("Restored best model weights")
                
                logger.info(f"Training stopped at epoch {epoch+1}")
                logger.info(f"Best validation loss: {early_stopping.best_loss:.6f} at epoch {early_stopping.best_epoch+1}")
                break
        
        epoch_time = time.time() - epoch_start_time
        logger.info(f"Epoch {epoch+1} completed in {epoch_time:.2f} seconds")
        
        # Clear CUDA cache after each epoch to prevent memory accumulation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    total_training_time = time.time() - total_start_time
    logger.info(f"Total training time: {total_training_time/60:.2f} minutes")

    # Final evaluation on test set
    logger.info("\\n" + "="*50)
    logger.info("Final evaluation on test set...")
    
    # Use early stopping best model if available, otherwise use latest
    early_stop_encoder_path = os.path.join(args.save_dir, 'early_stop_best_cnn_encoder.pth')
    early_stop_decoder_path = os.path.join(args.save_dir, 'early_stop_best_rnn_decoder.pth')
    
    model_loaded_for_test = False
    if os.path.exists(early_stop_encoder_path) and os.path.exists(early_stop_decoder_path):
        try:
            logger.info("Loading early stopping best model for final evaluation...")
            if isinstance(cnn_encoder, nn.DataParallel):
                cnn_encoder.module.load_state_dict(torch.load(early_stop_encoder_path, map_location=device))
                rnn_decoder.module.load_state_dict(torch.load(early_stop_decoder_path, map_location=device))
            else:
                cnn_encoder.load_state_dict(torch.load(early_stop_encoder_path, map_location=device))
                rnn_decoder.load_state_dict(torch.load(early_stop_decoder_path, map_location=device))
            model_loaded_for_test = True
            logger.info("Successfully loaded early stopping best model")
        except Exception as e:
            logger.error(f"Error loading early stopping model for test: {e}")
            logger.info("Using current model state for final evaluation")
    
    test_loss, test_metrics, _ = validate((cnn_encoder, rnn_decoder), device, test_loader, 
                                        action_classes, args.save_dir, epoch=None)
    
    logger.info(f"Final Test Results{'(Early Stopping Best Model)' if model_loaded_for_test else ''}:")
    logger.info(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"  Precision (macro): {test_metrics['precision_macro']:.4f}")
    logger.info(f"  Recall (macro): {test_metrics['recall_macro']:.4f}")
    logger.info(f"  F1-Score (macro): {test_metrics['f1_macro']:.4f}")
    logger.info(f"  Precision (micro): {test_metrics['precision_micro']:.4f}")
    logger.info(f"  Recall (micro): {test_metrics['recall_micro']:.4f}")
    logger.info(f"  F1-Score (micro): {test_metrics['f1_micro']:.4f}")
    logger.info(f"  Precision (weighted): {test_metrics['precision_weighted']:.4f}")
    logger.info(f"  Recall (weighted): {test_metrics['recall_weighted']:.4f}")
    logger.info(f"  F1-Score (weighted): {test_metrics['f1_weighted']:.4f}")

    # Early stopping summary
    if early_stopping is not None and early_stopping.early_stop:
        logger.info(f"\\nEarly Stopping Summary:")
        logger.info(f"  Training stopped at epoch: {epoch+1}")
        logger.info(f"  Best epoch: {early_stopping.best_epoch+1}")
        logger.info(f"  Best validation loss: {early_stopping.best_loss:.6f}")
        logger.info(f"  Final epochs trained: {epoch+1}/{epochs}")

    # Save training history
    np.save(os.path.join(args.save_dir, 'train_losses.npy'), metrics_history["train_loss"])
    np.save(os.path.join(args.save_dir, 'train_scores.npy'), metrics_history["train_accuracy"])
    np.save(os.path.join(args.save_dir, 'val_losses.npy'), metrics_history["val_loss"])
    np.save(os.path.join(args.save_dir, 'val_scores.npy'), metrics_history["val_accuracy"])

    # Plot comprehensive training curves
    plot_comprehensive_metrics(metrics_history, save_dir=args.save_dir)

    final_message = f"Training completed! Best validation accuracy: {best_val_score:.4f}"
    if early_stopping is not None and early_stopping.early_stop:
        final_message += f" (Early stopped at epoch {early_stopping.best_epoch+1})"
    logger.info(final_message)
    logger.info(f"Models and results saved to: {args.save_dir}")

def plot_comprehensive_metrics(metrics, save_dir='.'):
    """Plot comprehensive training metrics similar to ViViT training"""
    epochs = range(1, len(metrics["train_loss"]) + 1)

    # Loss plot
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

    # Accuracy plot
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

    # Precision plot
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

    # Recall plot
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

    # F1 Score plot
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

    # Combined metrics plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Loss
    axes[0, 0].plot(epochs, metrics["train_loss"], 'b-', label='Train')
    axes[0, 0].plot(epochs, metrics["val_loss"], 'r-', label='Val')
    axes[0, 0].set_title('Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Accuracy
    axes[0, 1].plot(epochs, metrics["train_accuracy"], 'b-', label='Train')
    axes[0, 1].plot(epochs, metrics["val_accuracy"], 'r-', label='Val')
    axes[0, 1].set_title('Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Precision
    axes[0, 2].plot(epochs, metrics["train_precision"], 'b-', label='Train')
    axes[0, 2].plot(epochs, metrics["val_precision"], 'r-', label='Val')
    axes[0, 2].set_title('Precision')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Precision')
    axes[0, 2].legend()
    axes[0, 2].grid(True)
    
    # Recall
    axes[1, 0].plot(epochs, metrics["train_recall"], 'b-', label='Train')
    axes[1, 0].plot(epochs, metrics["val_recall"], 'r-', label='Val')
    axes[1, 0].set_title('Recall')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Recall')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # F1 Score
    axes[1, 1].plot(epochs, metrics["train_f1"], 'b-', label='Train')
    axes[1, 1].plot(epochs, metrics["val_f1"], 'r-', label='Val')
    axes[1, 1].set_title('F1 Score')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('F1 Score')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    # Summary metrics
    axes[1, 2].plot(epochs, metrics["val_accuracy"], 'g-', label='Val Accuracy', linewidth=2)
    axes[1, 2].plot(epochs, metrics["val_f1"], 'orange', label='Val F1', linewidth=2)
    axes[1, 2].plot(epochs, metrics["val_precision"], 'purple', label='Val Precision', linewidth=1)
    axes[1, 2].plot(epochs, metrics["val_recall"], 'brown', label='Val Recall', linewidth=1)
    axes[1, 2].set_title('Validation Metrics Summary')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Score')
    axes[1, 2].legend()
    axes[1, 2].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "all_metrics_combined.png"), dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main()
