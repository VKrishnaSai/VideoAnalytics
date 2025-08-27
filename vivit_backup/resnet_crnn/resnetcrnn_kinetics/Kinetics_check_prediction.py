import os
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import matplotlib.pyplot as plt
from functions import *
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import pandas as pd
import pickle
import multiprocessing
import argparse
import cv2
import shutil
from tqdm import tqdm
import tempfile
from PIL import Image
import seaborn as sns

# Function to extract frames from a video
def extract_frames(video_path, output_dir, num_frames=16):
    """Extract frames from a video file and save to output directory"""
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file {video_path}")
        return False

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Select frames to extract
    if frame_count <= num_frames:
        frame_indices = list(range(frame_count))
    else:
        frame_indices = np.linspace(0, frame_count-1, num_frames, dtype=int)

    frame_idx = 0
    saved_count = 0

    while saved_count < num_frames and frame_idx < frame_count:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx in frame_indices:
            # Resize to 256x256
            frame = cv2.resize(frame, (256, 256))

            # Save frame
            frame_path = os.path.join(output_dir, f"frame{saved_count+1:06d}.jpg")
            cv2.imwrite(frame_path, frame)

            saved_count += 1

        frame_idx += 1

    cap.release()
    return saved_count == num_frames

# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, classes, output_file):
    """
    Generate and save a confusion matrix visualization

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        classes: List of class names
        output_file: Path to save the confusion matrix image
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=classes)

    # Normalize the confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Create figure (set size based on number of classes)
    plt.figure(figsize=(12, 10))

    # Use seaborn for better visualization
    sns.heatmap(cm_normalized, annot=False, fmt='.2f', cmap='Blues',
                xticklabels=classes, yticklabels=classes)

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Normalized Confusion Matrix')

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Confusion matrix saved to {output_file}")

    # Close the figure to free memory
    plt.close()

# Custom dataset class for on-the-fly frame extraction
class VideoDataset(data.Dataset):
    def __init__(self, video_dir, video_paths, labels, selected_frames, transform=None, temp_dir=None):
        self.video_dir = video_dir
        self.video_paths = video_paths  # Full relative paths like "ApplyEyeMakeup/v_ApplyEyeMakeup_g03_c01.avi"
        self.labels = labels
        self.selected_frames = selected_frames
        self.transform = transform
        self.temp_dir = temp_dir or tempfile.mkdtemp()

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, index):
        # Get video path and label
        video_rel_path = self.video_paths[index]  # e.g., "ApplyEyeMakeup/v_ApplyEyeMakeup_g03_c01.avi"
        label = self.labels[index]

        # Create temporary directory for frames
        video_name = os.path.splitext(os.path.basename(video_rel_path))[0]  # e.g., "v_ApplyEyeMakeup_g03_c01"
        frames_dir = os.path.join(self.temp_dir, video_name)

        # Extract frames
        video_path = os.path.join(self.video_dir, video_rel_path)  # Full path to video file
        success = extract_frames(video_path, frames_dir, len(self.selected_frames))

        if not success:
            print(f"Failed to extract frames from {video_path}")
            # Return empty tensor with correct shape and label
            return torch.zeros(len(self.selected_frames), 3, 224, 224), label

        # Load frames
        X = []
        for i in range(1, len(self.selected_frames) + 1):  # Frame numbers start from 1
            frame_path = os.path.join(frames_dir, f"frame{i:06d}.jpg")
            if os.path.exists(frame_path):
                frame = Image.open(frame_path)

                if self.transform:
                    frame = self.transform(frame)

                X.append(frame)
            else:
                print(f"Warning: Frame {frame_path} not found")
                # Create a blank frame if missing
                X.append(torch.zeros(3, 224, 224))

        # Clean up temporary frames
        shutil.rmtree(frames_dir, ignore_errors=True)

        # Stack frames
        X = torch.stack(X)
        return X, label

def main():
    parser = argparse.ArgumentParser(description='Run inference on test videos')
    parser.add_argument('--dataset_path', default='../kinetics4005per/val/val', type=str, help='Path to the validation video dataset')
    parser.add_argument('--model_path', type=str, default='./ResNetCRNN_Kinetics_ckpt/', help='Path to saved models')
    parser.add_argument('--action_names', type=str, default='./Kinetics400Actions.pkl', help='Path to action names pickle file')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for inference')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Directory to save results')
    parser.add_argument('--model_epoch', type=int, default=120, help='Epoch number of the model to load')
    args = parser.parse_args()

    # set paths
    data_path = args.dataset_path  # Your validation video directory
    action_name_path = args.action_names
    save_model_path = args.model_path
    output_dir = args.output_dir
    model_epoch = args.model_epoch

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # CNN and RNN parameters
    CNN_fc_hidden1, CNN_fc_hidden2 = 1024, 768
    CNN_embed_dim = 512   # latent dim extracted by 2D CNN
    res_size = 224        # ResNet image size
    dropout_p = 0.0       # dropout probability

    RNN_hidden_layers = 3
    RNN_hidden_nodes = 512
    RNN_FC_dim = 256


     ############ CHANGED FOR KINETICS ##################
    # training parameters - this will be auto-detected from pickle file
    begin_frame, end_frame, skip_frame = 1, 29, 1  # Match training script

    # Load action names (auto-generated from training)
    with open(action_name_path, 'rb') as f:
        action_names = pickle.load(f)
    
    k = len(action_names)  # Auto-detect number of classes
    print(f"Number of classes: {k}")

    # convert labels -> category
    le = LabelEncoder()
    le.fit(action_names)

    # show how many classes there are
    print(f"Number of classes: {len(list(le.classes_))}")

    # convert category -> 1-hot
    action_category = le.transform(action_names).reshape(-1, 1)
    enc = OneHotEncoder()
    enc.fit(action_category)

    # MODIFIED: Create test dataset from validation folder structure
    test_video_paths = []
    test_labels = []

    # Scan validation directory (similar to training structure)
    for class_name in action_names:
        class_path = os.path.join(data_path, class_name)
        if os.path.isdir(class_path):
            for video_file in os.listdir(class_path):
                if video_file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
                    # Store relative path like "class_name/video_file.ext"
                    relative_path = os.path.join(class_name, video_file)
                    test_video_paths.append(relative_path)
                    test_labels.append(class_name)

    print(f"Loaded {len(test_video_paths)} validation videos")

    # Convert labels to categorical
    test_y_list = labels2cat(le, test_labels)

    # data loading parameters
    use_cuda = torch.cuda.is_available()                   # check if GPU exists
    device = torch.device("cuda" if use_cuda else "cpu")   # use CPU or GPU
    print(f"Using device: {device}")

    transform = transforms.Compose([transforms.Resize([res_size, res_size]),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    selected_frames = np.arange(begin_frame, end_frame, skip_frame).tolist()

    # Create temporary directory for frame extraction
    temp_dir = tempfile.mkdtemp()
    print(f"Created temporary directory for frames: {temp_dir}")

    # Create dataset and dataloader
    test_params = {'batch_size': batch_size, 'shuffle': False, 'num_workers': 0, 'pin_memory': True} if use_cuda else {}
    test_dataset = VideoDataset(data_path, test_video_paths, test_y_list, selected_frames, transform=transform, temp_dir=temp_dir)
    test_loader = data.DataLoader(test_dataset, **test_params)

    # Load CRNN model
    cnn_encoder = ResCNNEncoder(fc_hidden1=CNN_fc_hidden1, fc_hidden2=CNN_fc_hidden2, drop_p=dropout_p, CNN_embed_dim=CNN_embed_dim).to(device)
    rnn_decoder = DecoderRNN(CNN_embed_dim=CNN_embed_dim, h_RNN_layers=RNN_hidden_layers, h_RNN=RNN_hidden_nodes,
                             h_FC_dim=RNN_FC_dim, drop_p=dropout_p, num_classes=k).to(device)

    # MODIFIED: Load model weights with correct epoch number
    try:
        cnn_encoder.load_state_dict(torch.load(os.path.join(save_model_path, f'cnn_encoder_epoch{model_epoch}.pth'), weights_only=True))
        rnn_decoder.load_state_dict(torch.load(os.path.join(save_model_path, f'rnn_decoder_epoch{model_epoch}.pth'), weights_only=True))
    except TypeError:
        cnn_encoder.load_state_dict(torch.load(os.path.join(save_model_path, f'cnn_encoder_epoch{model_epoch}.pth')))
        rnn_decoder.load_state_dict(torch.load(os.path.join(save_model_path, f'rnn_decoder_epoch{model_epoch}.pth')))

    print('CRNN model loaded successfully!')

    # Make predictions
    print(f'Predicting {len(test_loader.dataset)} test videos:')
    test_y_pred = CRNN_final_prediction([cnn_encoder, rnn_decoder], device, test_loader)

    # Convert predictions and ground truth to labels
    y_true = cat2labels(le, test_y_list)
    y_pred = cat2labels(le, test_y_pred)

    # Calculate overall metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average='weighted')

    print("\n===== Overall Metrics =====")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Calculate per-class metrics
    per_class_precision, per_class_recall, per_class_f1, per_class_support = precision_recall_fscore_support(y_true, y_pred, labels=list(le.classes_), zero_division=0)

    # Calculate per-class accuracy
    per_class_accuracy = []
    y_true_array = np.array(y_true)
    y_pred_array = np.array(y_pred)

    for class_name in le.classes_:
        mask = (y_true_array == class_name)
        if np.any(mask):  # Check if there are any samples for this class
            class_accuracy = accuracy_score(y_true_array[mask], y_pred_array[mask])
        else:
            class_accuracy = 0.0  # Handle case where no samples exist for this class
        per_class_accuracy.append(class_accuracy)


    # Create DataFrame for per-class metrics
    per_class_metrics_df = pd.DataFrame({
        'class': le.classes_,
        'accuracy': per_class_accuracy,
        'precision': per_class_precision,
        'recall': per_class_recall,
        'f1_score': per_class_f1,
        'support': per_class_support
    })

    # Create DataFrame for overall metrics
    overall_metrics_df = pd.DataFrame({
        'metric': ['accuracy', 'precision', 'recall', 'f1_score'],
        'value': [accuracy, precision, recall, f1]
    })

    # Save metrics to CSV files
    metrics_dir = os.path.join(output_dir, f"metrics_split{test_split}")
    os.makedirs(metrics_dir, exist_ok=True)

    per_class_metrics_df.to_csv(os.path.join(metrics_dir, 'per_class_metrics.csv'), index=False)
    overall_metrics_df.to_csv(os.path.join(metrics_dir, 'overall_metrics.csv'), index=False)

    print(f"\nMetrics saved to {metrics_dir}")

    # Generate confusion matrix
    confusion_matrix_file = os.path.join(metrics_dir, 'confusion_matrix.png')
    plot_confusion_matrix(y_true, y_pred, list(le.classes_), confusion_matrix_file)

    # Clean up temporary directory
    shutil.rmtree(temp_dir, ignore_errors=True)
    print(f"Cleaned up temporary directory: {temp_dir}")

# Add this block to properly handle multiprocessing
if __name__ == "__main__":
    # Add freeze_support for Windows
    multiprocessing.freeze_support()
    main()
