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

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

# Import your ViViT model and load_video function from vivit.py
from vivit import ViViT, load_video, load_video_new

# ----------------------- Global Training Status -----------------------
training_status = {
    "epoch": 0,
    "loss": 0.0,
    "accuracy": 0.0,
    "progress": 0.0  # percentage (0-100)
}

# ----------------------- Logging Setup -----------------------
logging.basicConfig(
    filename='resume_training_second_time.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s'
)
logging.info("Resumed training started.")

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
        # Use your load_video function to load the video frames
        video = load_video_new(self.video_paths[idx], num_frames=self.num_frames, image_size=self.image_size)
        # Remove extra batch dimension if present (expected shape: [1, T, C, H, W])
        if video.ndim == 5 and video.shape[0] == 1:
            video = video.squeeze(0)  # now shape: [T, C, H, W]
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return video, label

# ----------------------- Data Preparation -----------------------
# Read classInd.txt to get mapping from class id to class name
class_index_file = "./ucfTrainTestlist/classInd.txt"
class_indices = {}
with open(class_index_file, "r") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) == 2:
            idx, class_name = parts
            class_indices[int(idx)] = class_name

# Build list of class names in order (0-indexed)
data_subset = [class_indices[i] for i in sorted(class_indices.keys())]
print("Classes:", data_subset)

# Define trainlist and testlist files
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

# Data root for UCF101 videos – update this path as needed.
data_root = "./UCF101/UCF-101"

# ----------------------- Build Training Dataset -----------------------
train_videos = []
train_labels = []

# Process each trainlist file
for trainlist in trainlist_files:
    with open(trainlist, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Each line format: "relative_video_path label"
            video_rel_path, label_str = line.split()
            # Convert label to zero-indexed integer (UCF101 labels are 1-indexed)
            label = int(label_str) - 1
            video_path = os.path.join(data_root, video_rel_path)
            if not os.path.isfile(video_path):
                print(f"Warning: Video file {video_path} does not exist!")
                continue
            train_videos.append(video_path)
            train_labels.append(label)

print(f"Loaded {len(train_videos)} training videos.")

# Create training dataset
train_dataset = VideoDataset(train_videos, train_labels)

# ----------------------- Build Testlist Dataset -----------------------
testlist_videos = []
testlist_labels = []

# Process each testlist file
for testlist in testlist_files:
    with open(testlist, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # For testlist files, lines can be in two formats:
            # Format 1: "relative_video_path label"
            # Format 2: "relative_video_path" (in which case, infer label from folder)
            parts = line.split()
            if len(parts) == 2:
                video_rel_path, label_str = parts
                label = int(label_str) - 1  # Convert label to zero-indexed
            else:
                video_rel_path = parts[0]
                # Infer label from directory name (assumes the first part of the path is the class)
                class_name = video_rel_path.split('/')[0]
                try:
                    label = data_subset.index(class_name)
                except ValueError:
                    print(f"Warning: Class {class_name} not found in data_subset!")
                    continue

            video_path = os.path.join(data_root, video_rel_path)
            if not os.path.isfile(video_path):
                print(f"Warning: Video file {video_path} does not exist!")
                continue
            testlist_videos.append(video_path)
            testlist_labels.append(label)

print(f"Loaded {len(testlist_videos)} videos from testlist files.")

# Create full dataset for testlist files
testlist_dataset = VideoDataset(testlist_videos, testlist_labels)

# ----------------------- Split Testlist Dataset into Validation and Test -----------------------
# Here, we split the testlist dataset into validation and test sets.
# Adjust test_size (e.g., 0.5 for a 50/50 split) as needed.
all_indices = list(range(len(testlist_dataset)))
val_indices, test_indices = train_test_split(
    all_indices, test_size=0.5, random_state=42, stratify=testlist_labels
)

print(f"Validation samples: {len(val_indices)}, Test samples: {len(test_indices)}")

val_dataset = Subset(testlist_dataset, val_indices)
test_dataset = Subset(testlist_dataset, test_indices)

# ----------------------- Create DataLoaders -----------------------
trainloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
valloader = DataLoader(val_dataset, batch_size=4, shuffle=False)
testloader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# ----------------------- Checkpoints Directory -----------------------
checkpoints_dir = "checkpoints_28feb"
if not os.path.exists(checkpoints_dir):
    os.makedirs(checkpoints_dir)

# ----------------------- Resume Training Function with Early Stopping -----------------------
def resume_training(model, train_dataloader, val_dataloader, criterion, optimizer, start_epoch, additional_epochs, patience=3, device='cuda'):
    global training_status
    best_loss = float('inf')
    total_samples = len(train_dataloader.dataset)
    
    # Lists to record metrics for plotting
    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []
    val_precision_list = []
    val_recall_list = []
    val_f1_list = []
    
    epochs_no_improve = 0  # Early stopping counter

    for epoch in range(start_epoch, start_epoch + additional_epochs):
        model.train()
        epoch_loss = 0.0
        total_count = 0
        all_train_preds = []
        all_train_labels = []
        for videos_batch, labels_batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{start_epoch+additional_epochs}"):
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
            
            training_status["progress"] = (((epoch - start_epoch) * total_samples + total_count) / (additional_epochs * total_samples)) * 100
        
        epoch_loss_avg = epoch_loss / total_count
        train_accuracy = accuracy_score(all_train_labels, all_train_preds)
        train_loss_list.append(epoch_loss_avg)
        train_acc_list.append(train_accuracy)
        training_status["epoch"] = epoch + 1
        training_status["loss"] = epoch_loss_avg
        training_status["accuracy"] = train_accuracy
        
        log_msg = (f"Epoch {epoch+1} -- Train Loss: {epoch_loss_avg:.4f}, Train Acc: {train_accuracy:.4f}")
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
        checkpoint_path = os.path.join(checkpoints_dir, f"resume_checkpoint_epoch_{epoch+1}.pth")
        torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': epoch_loss_avg,
            'val_loss': val_loss_avg,
        }, checkpoint_path)
        logging.info(f"Saved checkpoint: {checkpoint_path}")
        
        # Check if current validation loss is the best so far
        if val_loss_avg < best_loss:
            best_loss = val_loss_avg
            epochs_no_improve = 0  # Reset early stopping counter
            best_checkpoint_path = os.path.join(checkpoints_dir, "resume_best_checkpoint.pth")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': epoch_loss_avg,
                'val_loss': val_loss_avg,
            }, best_checkpoint_path)
            logging.info(f"New best model saved at: {best_checkpoint_path}")
        else:
            epochs_no_improve += 1
            logging.info(f"No improvement for {epochs_no_improve} epoch(s).")
        
        # Check for early stopping
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            logging.info(f"Early stopping triggered after {epoch+1} epochs.")
            break
    
    training_status["progress"] = 100
    # Package the recorded metrics for plotting
    metrics = {
        "train_loss": train_loss_list,
        "train_accuracy": train_acc_list,
        "val_loss": val_loss_list,
        "val_accuracy": val_acc_list,
        "val_precision": val_precision_list,
        "val_recall": val_recall_list,
        "val_f1": val_f1_list,
    }
    return best_loss, metrics

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
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix on Test Set')
    plt.savefig("confusion_matrix.png")
    plt.close()
    test_accuracy = accuracy_score(all_labels, all_preds)
    test_precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    test_recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    test_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    print(f"Test F1 Score: {test_f1:.4f}")

# ----------------------- FastAPI Setup -----------------------
app = FastAPI()

@app.get("/", response_class=HTMLResponse)
async def index():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Resumed Training Progress Dashboard</title>
        <style>
            body { background: #f0f8ff; font-family: 'Comic Sans MS', cursive, sans-serif; margin: 0; display: flex; justify-content: center; align-items: center; height: 100vh; }
            .container { background: #fff; border-radius: 10px; padding: 20px 40px; text-align: center; box-shadow: 0 0 20px rgba(0,0,0,0.2); }
            h1 { margin-bottom: 20px; animation: hue 10s infinite linear; }
            @keyframes hue { from { filter: hue-rotate(0deg); } to { filter: hue-rotate(360deg); } }
            .progress-container { width: 80%; background: #ddd; border-radius: 5px; margin: 20px auto; overflow: hidden; height: 30px; }
            .progress-bar { width: 0%; height: 100%; background: linear-gradient(90deg, #ff8a00, #e52e71); animation: pulse 2s infinite; }
            @keyframes pulse { 0% { opacity: 0.8; } 50% { opacity: 1; } 100% { opacity: 0.8; } }
            .stats p { font-size: 1.2em; margin: 10px 0; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Resumed Training Progress Dashboard</h1>
            <div class="progress-container">
                <div class="progress-bar" id="progress-bar"></div>
            </div>
            <div class="stats">
                <p>Epoch: <span id="epoch">0</span></p>
                <p>Loss: <span id="loss">0.0</span></p>
                <p>Accuracy: <span id="accuracy">0.0%</span></p>
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

# ----------------------- Background Training -----------------------
def start_resumed_training():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device:", device)
    logging.info(f"Using device: {device}")
    
    # Initialize model, criterion, and optimizer using our split datasets
    model = ViViT(image_size=224, patch_size=16, num_classes=len(data_subset), num_frames=16).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # Try to load a previously saved checkpoint for resumed training
    checkpoint_path = os.path.join(checkpoints_dir, "none.pth")
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Resuming training from epoch {start_epoch}")
        logging.info(f"Resuming training from epoch {start_epoch}")
    else:
        print("No checkpoint found, starting training from scratch.")
        logging.info("No checkpoint found, starting training from scratch.")
    
    additional_epochs = 10  # Specify additional epochs to train
    patience = 3  # Early stopping patience
    best_val_loss, metrics = resume_training(model, trainloader, valloader, criterion, optimizer, start_epoch, additional_epochs, patience=patience, device=device)
    
    # Plot training metrics and confusion matrix on the test set
    plot_metrics(metrics)
    logging.info("Training metrics plots saved.")
    evaluate_test(model, testloader, device=device)
    logging.info("Test evaluation complete; confusion matrix saved.")
    
    # Save final resumed model
    torch.save(model.state_dict(), "vivit_model_resume.pth")
    logging.info("Final resumed model saved as vivit_model_resume.pth")
    print("Resumed model training complete and saved.")

@app.on_event("startup")
async def on_startup():
    # Launch resumed training in a background thread so it doesn't block FastAPI.
    threading.Thread(target=start_resumed_training, daemon=True).start()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("ucf_training:app", host="0.0.0.0", port=8000, reload=True)
