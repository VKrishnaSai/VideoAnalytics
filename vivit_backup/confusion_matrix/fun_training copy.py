import os
import time
import threading
import asyncio
import logging
import copy
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset

from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix)
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns



# Import your ViViT model and load_video function from vivit.py
from vivit import ViViT, load_video

# ----------------------- Global Training Status -----------------------
training_status = {
    "epoch": 0,
    "loss": 0.0,
    "accuracy": 0.0,
    "progress": 0.0  # percentage (0-100)
}

# ----------------------- Logging Setup -----------------------
logging.basicConfig(
    filename='training.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s'
)
logging.info("Training started.")

# ----------------------- Data Preparation -----------------------
# Read the file and organize categories
class_file = "truncated_classes.txt"
data_subset = []
categories = defaultdict(list)
current_category = None
category_order = []  # Maintain order of categories

with open(class_file, "r") as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("//"):
            if line.startswith("//"):
                current_category = line.strip("//").strip()
                if current_category not in category_order:
                    category_order.append(current_category)
        else:
            categories[current_category].append(line)
            data_subset.append(line)

print("Classes:", data_subset)
print("Categories:", dict(categories))

# Data root for videos
data_root = './kinetics400_5per/train/'

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
for cls in data_subset:
    cls_path = os.path.join(data_root, cls)
    if not os.path.isdir(cls_path):
        print(f"Warning: Class directory {cls_path} does not exist!")
        continue
    for video in os.listdir(cls_path):
        if video.endswith('.mp4'):
            videos.append(os.path.join(cls_path, video))
            labels.append(label_map[cls])

print(f"Loaded {len(videos)} videos.")

# Create the full dataset
full_dataset = VideoDataset(videos, labels)

# ----------------------- Split Data -----------------------
# Split full dataset into train (70%) and test (30%)
all_indices = list(range(len(full_dataset)))
train_indices, test_indices = train_test_split(
    all_indices, test_size=0.30, random_state=42, stratify=labels
)
# Further split train_indices into train (80%) and val (20%)
train_labels_subset = [labels[i] for i in train_indices]
train_indices, val_indices = train_test_split(
    train_indices, test_size=0.20, random_state=42, stratify=train_labels_subset
)

print(f"Train samples: {len(train_indices)}, Validation samples: {len(val_indices)}, Test samples: {len(test_indices)}")

train_dataset = Subset(full_dataset, train_indices)
val_dataset = Subset(full_dataset, val_indices)
test_dataset = Subset(full_dataset, test_indices)

# Create DataLoaders
trainloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
valloader = DataLoader(val_dataset, batch_size=4, shuffle=False)
testloader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# ----------------------- Checkpoints Directory -----------------------
checkpoints_dir = "checkpoints_28feb"
if not os.path.exists(checkpoints_dir):
    os.makedirs(checkpoints_dir)

# ----------------------- Model Components -----------------------
# Tubelet Embedding: using 3D convolution
class TubeletEmbedding(nn.Module):
    def __init__(self, embed_dim, patch_size):
        super().__init__()
        # Input videos are expected as [B, T, C, H, W]. Permute to [B, C, T, H, W] for Conv3d.
        self.projection = nn.Conv3d(in_channels=3, out_channels=embed_dim,
                                    kernel_size=patch_size, stride=patch_size, padding=0)
    def forward(self, videos):
        # videos: [B, T, C, H, W] -> permute to [B, C, T, H, W]
        videos = videos.permute(0, 2, 1, 3, 4)
        x = self.projection(videos)
        B, C, T, H, W = x.shape
        x = x.view(B, C, T * H * W).transpose(1, 2)  # [B, N, embed_dim]
        return x

# Positional Encoder: add learnable positional embedding
class PositionalEncoder(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, encoded_tokens):
        # encoded_tokens: [B, N, embed_dim]
        B, N, _ = encoded_tokens.shape
        positions = torch.arange(N, device=encoded_tokens.device).unsqueeze(0).expand(B, N)
        # Learnable positional embeddings
        pos_embedding = nn.Embedding(N, self.embed_dim).to(encoded_tokens.device)
        encoded_positions = pos_embedding(positions)
        return encoded_tokens + encoded_positions

# ----------------------- Full ViViT Model Wrapper -----------------------
def create_vivit_classifier(tubelet_embedder, positional_encoder,
                            input_shape=(16, 3, 224, 224),  # [T, C, H, W]
                            transformer_layers=8, num_heads=8,
                            embed_dim=128, layer_norm_eps=1e-6,
                            num_classes=len(data_subset)):
    # For simplicity, we assume ViViT is implemented to accept these parameters.
    model = ViViT(image_size=224, patch_size=16, num_classes=num_classes, num_frames=input_shape[0],
                  hidden_size=embed_dim, num_hidden_layers=transformer_layers, num_attention_heads=num_heads)
    return model

# ----------------------- Training Function (with Validation and Metric Recording) -----------------------
def train_model(model, train_dataloader, val_dataloader, criterion, optimizer, num_epochs=10, device='cuda'):
    global training_status
    best_val_loss = float('inf')
    total_train_samples = len(train_dataloader.dataset)

    # Lists to record metrics
    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []
    val_precision_list = []
    val_recall_list = []
    val_f1_list = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        total_count = 0
        all_train_preds = []
        all_train_labels = []
        for videos_batch, labels_batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            videos_batch, labels_batch = videos_batch.to(device), labels_batch.to(device)
            optimizer.zero_grad()
            outputs = model(videos_batch)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()
            
            batch_size = labels_batch.size(0)
            epoch_loss += loss.item() * batch_size
            total_count += batch_size
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_train_preds.extend(preds)
            all_train_labels.extend(labels_batch.cpu().numpy())
            
            training_status["progress"] = (((epoch * total_train_samples) + total_count) / (num_epochs * total_train_samples)) * 100
        
        epoch_loss_avg = epoch_loss / total_count
        train_accuracy = accuracy_score(all_train_labels, all_train_preds)
        train_loss_list.append(epoch_loss_avg)
        train_acc_list.append(train_accuracy)
        training_status["epoch"] = epoch + 1
        training_status["loss"] = epoch_loss_avg
        training_status["accuracy"] = train_accuracy
        
        log_msg = (f"Epoch {epoch+1}/{num_epochs} -- Train Loss: {epoch_loss_avg:.4f}, Train Acc: {train_accuracy:.4f}")
        print(log_msg)
        logging.info(log_msg)
        
        # Evaluate on validation set
        model.eval()
        val_loss = 0.0
        val_count = 0
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for videos_batch, labels_batch in val_dataloader:
                videos_batch, labels_batch = videos_batch.to(device), labels_batch.to(device)
                outputs = model(videos_batch)
                loss = criterion(outputs, labels_batch)
                batch_size = labels_batch.size(0)
                val_loss += loss.item() * batch_size
                val_count += batch_size
                preds = outputs.argmax(dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_labels.extend(labels_batch.cpu().numpy())
        val_loss_avg = val_loss / val_count
        val_accuracy = accuracy_score(val_labels, val_preds)
        val_precision = precision_score(val_labels, val_preds, average='macro', zero_division=0)
        val_recall = recall_score(val_labels, val_preds, average='macro', zero_division=0)
        val_f1 = f1_score(val_labels, val_preds, average='macro', zero_division=0)
        val_loss_list.append(val_loss_avg)
        val_acc_list.append(val_accuracy)
        val_precision_list.append(val_precision)
        val_recall_list.append(val_recall)
        val_f1_list.append(val_f1)

        val_msg = (f"Validation Loss: {val_loss_avg:.4f}, Val Acc: {val_accuracy:.4f}, "
                   f"Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}")
        print(val_msg)
        logging.info(val_msg)
        
        # Save checkpoint for this epoch
        checkpoint_path = os.path.join(checkpoints_dir, f"checkpoint_epoch_{epoch+1}.pth")
        torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': epoch_loss_avg,
            'val_loss': val_loss_avg,
        }, checkpoint_path)
        logging.info(f"Saved checkpoint: {checkpoint_path}")
        
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            best_checkpoint_path = os.path.join(checkpoints_dir, "best_checkpoint.pth")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': epoch_loss_avg,
                'val_loss': val_loss_avg,
            }, best_checkpoint_path)
            logging.info(f"New best model saved at: {best_checkpoint_path}")
    training_status["progress"] = 100

    # Return recorded metrics for plotting
    metrics = {
        "train_loss": train_loss_list,
        "train_accuracy": train_acc_list,
        "val_loss": val_loss_list,
        "val_accuracy": val_acc_list,
        "val_precision": val_precision_list,
        "val_recall": val_recall_list,
        "val_f1": val_f1_list,
    }
    return best_val_loss, metrics

# ----------------------- Plotting Functions -----------------------
def plot_metrics(metrics):
    epochs = range(1, len(metrics["train_loss"]) + 1)
    
    # Plot Loss
    plt.figure()
    plt.plot(epochs, metrics["train_loss"], label='Train Loss')
    plt.plot(epochs, metrics["val_loss"], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs Epoch')
    plt.legend()
    plt.savefig("loss_plot.png")
    plt.close()

    # Plot Accuracy
    plt.figure()
    plt.plot(epochs, metrics["train_accuracy"], label='Train Accuracy')
    plt.plot(epochs, metrics["val_accuracy"], label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Epoch')
    plt.legend()
    plt.savefig("accuracy_plot.png")
    plt.close()

    # Plot Precision, Recall, F1 Score for validation
    plt.figure()
    plt.plot(epochs, metrics["val_precision"], label='Precision')
    plt.plot(epochs, metrics["val_recall"], label='Recall')
    plt.plot(epochs, metrics["val_f1"], label='F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Validation Metrics vs Epoch')
    plt.legend()
    plt.savefig("val_metrics_plot.png")
    plt.close()

def evaluate_test(model, test_dataloader, device='cuda'):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for videos_batch, labels_batch in test_dataloader:
            videos_batch, labels_batch = videos_batch.to(device), labels_batch.to(device)
            outputs = model(videos_batch)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels_batch.cpu().numpy())
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix on Test Set")
    plt.savefig("confusion_matrix.png")
    plt.close()
    # Also print out the test metrics
    test_accuracy = accuracy_score(all_labels, all_preds)
    test_precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    test_recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    test_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    print(f"Test F1 Score: {test_f1:.4f}")

# ----------------------- Grid Search for Hyperparameters -----------------------
def grid_search(model_class, train_dataloader, param_grid, num_epochs=3, device='cuda'):
    best_lr = None
    best_loss = float('inf')
    best_model_state = None

    for lr in param_grid.get("learning_rate", [1e-4]):
        print(f"Testing learning rate: {lr}")
        # Initialize a fresh model for each candidate
        model = model_class(image_size=224, patch_size=16, num_classes=len(data_subset), num_frames=16).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        loss, _ = train_model(model, train_dataloader, valloader, criterion, optimizer, num_epochs=num_epochs, device=device)
        print(f"Learning rate {lr} yielded val loss: {loss:.4f}")
        if loss < best_loss:
            best_loss = loss
            best_lr = lr
            best_model_state = copy.deepcopy(model.state_dict())
    print(f"Best learning rate: {best_lr} with val loss: {best_loss:.4f}")
    return best_lr, best_model_state

# Define the hyperparameter grid (expandable)
param_grid = {
    "learning_rate": [1e-3, 1e-4, 1e-5]
}



# ----------------------- Background Training -----------------------
def start_training():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device:", device)
    logging.info(f"Using device: {device}")
    
    # First, perform grid search on the training set (using valloader for validation)
    # best_lr, best_model_state = grid_search(ViViT, trainloader, param_grid, num_epochs=5, device=device)
    
    # Initialize final model with best found learning rate
    model = ViViT(image_size=224, patch_size=16, num_classes=len(data_subset), num_frames=16).to(device)
    # model.load_state_dict(best_model_state)
    best_lr = 1e-4
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=best_lr)
    print(f"Starting full training with best learning rate: {best_lr}")
    logging.info(f"Starting full training with best learning rate: {best_lr}")
    
    # Train for full epochs using trainloader and evaluate on valloader each epoch
    best_val_loss, metrics = train_model(model, trainloader, valloader, criterion, optimizer, num_epochs=20, device=device)
    
    # Plot training metrics
    plot_metrics(metrics)
    logging.info("Training metrics plots saved.")
    
    # Evaluate on test set and plot confusion matrix
    evaluate_test(model, testloader, device=device)
    logging.info("Test evaluation complete; confusion matrix saved.")
    
    # Save final model
    torch.save(model.state_dict(), "vivit_model.pth")
    logging.info("Final model saved as vivit_model.pth")
    print("Model training complete and saved.")
