import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from collections import defaultdict
from vivit import ViViT, load_video  # Ensure these are imported correctly

# For progress monitoring and metrics calculations
from tqdm import tqdm
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Set up logging
logging.basicConfig(
    filename='training.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s'
)
logging.info("Training started.")

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

print(data_subset)
print()
print(categories)

# Data root for videos
data_root = './kinetics400_5per/train/'

# Custom VideoDataset (data-related parts unchanged)
class VideoDataset(Dataset):
    def __init__(self, video_paths, labels, num_frames=16, image_size=224):
        self.video_paths = video_paths
        self.labels = labels
        self.num_frames = num_frames
        self.image_size = image_size

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video = load_video(self.video_paths[idx], num_frames=self.num_frames, image_size=self.image_size)
        # FIX: Remove extra batch dimension if present (should be [T, C, H, W])
        if video.ndim == 5 and video.shape[0] == 1:
            video = video.squeeze(0)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return video, label

# Load video paths and labels
videos = []
labels = []
label_map = {cls: idx for idx, cls in enumerate(data_subset)}  # Map class names to indices
for cls in data_subset:
    cls_path = os.path.join(data_root, cls)
    for video in os.listdir(cls_path):
        if video.endswith('.mp4'):
            videos.append(os.path.join(cls_path, video))
            labels.append(label_map[cls])

print("Loaded", len(videos), "videos.")
print("Sample video paths:", videos[:5])
print("Sample labels:", labels[:5])

# Create checkpoints directory if not exists
checkpoints_dir = "checkpoints"
if not os.path.exists(checkpoints_dir):
    os.makedirs(checkpoints_dir)

def train_model(model, dataloader, criterion, optimizer, num_epochs=10, device='cuda'):
    best_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        total_samples = 0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for videos, labels in pbar:
            videos, labels = videos.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(videos)  # Expected shape: [B, T, C, H, W] -> ViViT processes accordingly
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix(loss=loss.item())
        
        # Calculate epoch metrics
        epoch_loss_avg = epoch_loss / total_samples
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        
        log_msg = (f"Epoch {epoch+1}/{num_epochs} -- Loss: {epoch_loss_avg:.4f}, "
                   f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, "
                   f"Recall: {recall:.4f}, F1: {f1:.4f}")
        print(log_msg)
        logging.info(log_msg)
        
        # Save checkpoint each epoch
        checkpoint_path = os.path.join(checkpoints_dir, f"checkpoint_epoch_{epoch+1}.pth")
        torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss_avg
        }, checkpoint_path)
        logging.info(f"Saved checkpoint: {checkpoint_path}")
        
        # Save best model
        if epoch_loss_avg < best_loss:
            best_loss = epoch_loss_avg
            best_checkpoint_path = os.path.join(checkpoints_dir, "best_checkpoint.pth")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss_avg
            }, best_checkpoint_path)
            logging.info(f"New best model saved at: {best_checkpoint_path}")

# Main script
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device:", device)
    logging.info(f"Using device: {device}")
    
    # Create dataset & dataloader
    dataset = VideoDataset(videos, labels)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Model initialization: update num_classes to match data_subset length
    model = ViViT(image_size=224, patch_size=16, num_classes=len(data_subset), num_frames=16).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # Train model
    train_model(model, dataloader, criterion, optimizer, num_epochs=10, device=device)
    
    # Save final trained model
    torch.save(model.state_dict(), "vivit_model.pth")
    logging.info("Final model saved as vivit_model.pth")
    print("Model training complete and saved.")
