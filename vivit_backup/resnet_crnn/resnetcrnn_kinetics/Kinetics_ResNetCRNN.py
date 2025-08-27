import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import torch.utils.data as data
import torchvision
from torch.autograd import Variable
import matplotlib.pyplot as plt
from functions import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from tqdm import tqdm
import pickle
import warnings
import pandas as pd
import seaborn as sns

# Ignore the specific UserWarning from sklearn
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


def print_gpu_memory():
    if torch.cuda.is_available():
        print(f"GPU Memory: {torch.cuda.memory_allocated()/1024**2:.2f} MB allocated, "
              f"{torch.cuda.memory_reserved()/1024**2:.2f} MB reserved")

def save_metrics_and_plots(epoch_train_metrics, epoch_val_metrics, action_names, output_dir="./training_results"):
    """Save comprehensive metrics and generate plots"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract metrics for plotting
    train_losses = [metrics['loss'] for metrics in epoch_train_metrics]
    train_accuracies = [metrics['accuracy'] for metrics in epoch_train_metrics]
    train_precisions = [metrics['precision'] for metrics in epoch_train_metrics]
    train_recalls = [metrics['recall'] for metrics in epoch_train_metrics]
    train_f1s = [metrics['f1'] for metrics in epoch_train_metrics]
    
    val_losses = [metrics['loss'] for metrics in epoch_val_metrics]
    val_accuracies = [metrics['accuracy'] for metrics in epoch_val_metrics]
    val_precisions = [metrics['precision'] for metrics in epoch_val_metrics]
    val_recalls = [metrics['recall'] for metrics in epoch_val_metrics]
    val_f1s = [metrics['f1'] for metrics in epoch_val_metrics]
    
    epochs_range = range(1, len(train_losses) + 1)
    
    # Create comprehensive plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Loss plot
    axes[0, 0].plot(epochs_range, train_losses, 'b-', label='Training Loss')
    axes[0, 0].plot(epochs_range, val_losses, 'r-', label='Validation Loss')
    axes[0, 0].set_title('Model Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Accuracy plot
    axes[0, 1].plot(epochs_range, train_accuracies, 'b-', label='Training Accuracy')
    axes[0, 1].plot(epochs_range, val_accuracies, 'r-', label='Validation Accuracy')
    axes[0, 1].set_title('Model Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Precision plot
    axes[0, 2].plot(epochs_range, train_precisions, 'b-', label='Training Precision')
    axes[0, 2].plot(epochs_range, val_precisions, 'r-', label='Validation Precision')
    axes[0, 2].set_title('Model Precision')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Precision')
    axes[0, 2].legend()
    axes[0, 2].grid(True)
    
    # Recall plot
    axes[1, 0].plot(epochs_range, train_recalls, 'b-', label='Training Recall')
    axes[1, 0].plot(epochs_range, val_recalls, 'r-', label='Validation Recall')
    axes[1, 0].set_title('Model Recall')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Recall')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # F1 Score plot
    axes[1, 1].plot(epochs_range, train_f1s, 'b-', label='Training F1')
    axes[1, 1].plot(epochs_range, val_f1s, 'r-', label='Validation F1')
    axes[1, 1].set_title('Model F1 Score')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('F1 Score')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    # Remove the empty subplot
    axes[1, 2].remove()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save metrics to CSV
    overall_metrics_df = pd.DataFrame({
        'epoch': epochs_range,
        'train_loss': train_losses,
        'train_accuracy': train_accuracies,
        'train_precision': train_precisions,
        'train_recall': train_recalls,
        'train_f1': train_f1s,
        'val_loss': val_losses,
        'val_accuracy': val_accuracies,
        'val_precision': val_precisions,
        'val_recall': val_recalls,
        'val_f1': val_f1s
    })
    overall_metrics_df.to_csv(os.path.join(output_dir, 'overall_metrics.csv'), index=False)
    
    # Save final epoch confusion matrix
    if epoch_val_metrics:
        final_val_metrics = epoch_val_metrics[-1]
        cm = confusion_matrix(final_val_metrics['y_true'], final_val_metrics['y_pred'])
        
        # Plot confusion matrix
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=action_names, yticklabels=action_names)
        plt.title('Confusion Matrix - Final Epoch')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confusion_matrix_final.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save per-class metrics for final epoch
        per_class_df = pd.DataFrame({
            'class_name': action_names,
            'accuracy': final_val_metrics['per_class_accuracy'],
            'precision': final_val_metrics['per_class_precision'],
            'recall': final_val_metrics['per_class_recall'],
            'f1': final_val_metrics['per_class_f1'],
            'support': final_val_metrics['per_class_support']
        })
        per_class_df.to_csv(os.path.join(output_dir, 'per_class_metrics_final.csv'), index=False)
    
    print(f"Metrics and plots saved to {output_dir}")

def calculate_train_metrics(losses, scores, y_true, y_pred, action_names):
    """Calculate comprehensive training metrics"""
    train_loss = np.mean(losses)
    train_accuracy = np.mean(scores)
    
    # Calculate precision, recall, f1
    train_precision, train_recall, train_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    
    # Per-class metrics
    per_class_precision, per_class_recall, per_class_f1, per_class_support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(action_names)), zero_division=0
    )
    
    # Calculate per-class accuracy
    per_class_accuracy = []
    for class_idx in range(len(action_names)):
        mask = (y_true == class_idx)
        if np.any(mask):
            class_acc = accuracy_score(y_true[mask], y_pred[mask])
        else:
            class_acc = 0.0
        per_class_accuracy.append(class_acc)
    
    return {
        'loss': train_loss,
        'accuracy': train_accuracy,
        'precision': train_precision,
        'recall': train_recall,
        'f1': train_f1,
        'per_class_accuracy': per_class_accuracy,
        'per_class_precision': per_class_precision,
        'per_class_recall': per_class_recall,
        'per_class_f1': per_class_f1,
        'per_class_support': per_class_support,
        'y_true': y_true,
        'y_pred': y_pred
    }

############## to change ##################
# set path
data_path = "./jpegs_256/"    # define Kinetics RGB data path
action_name_path = './Kinetics400Actions.pkl'  # this will be auto-generated now
save_model_path = "./ResNetCRNN_Kinetics_ckpt/"

# EncoderCNN architecture
CNN_fc_hidden1, CNN_fc_hidden2 = 1024, 768
CNN_embed_dim = 512   # latent dim extracted by 2D CNN
res_size = 224        # ResNet image size
dropout_p = 0.0       # dropout probability

# DecoderRNN architecture
RNN_hidden_layers = 3
RNN_hidden_nodes = 512
RNN_FC_dim = 256

#################### to change ##################
# training parameters
k = 400             # number of target category
epochs = 320        # training epochs
batch_size = 40
learning_rate = 1e-3
log_interval = 10   # interval for displaying training info

# Early stopping parameters
patience = 15       # early stopping patience
best_val_loss = float('inf')
patience_counter = 0
early_stop = False

# Select which frame to begin & end in videos
begin_frame, end_frame, skip_frame = 1, 14, 1


def train(log_interval, model, device, train_loader, optimizer, epoch):
    # set model as training mode
    cnn_encoder, rnn_decoder = model
    cnn_encoder.train()
    rnn_decoder.train()

    losses = []
    scores = []
    all_y_true = []
    all_y_pred = []
    N_count = 0   # counting total trained sample in one epoch
    
    # Progress bar with proper positioning
    pbar = tqdm(train_loader, desc=f'Training Epoch {epoch+1:3d}/{epochs}', 
                position=1, leave=True, ncols=150)
    
    for batch_idx, (X, y) in enumerate(pbar):
        # distribute data to device
        X, y = X.to(device), y.to(device).view(-1, )

        N_count += X.size(0)
        
        optimizer.zero_grad()
        output = rnn_decoder(cnn_encoder(X))   # output has dim = (batch, number of classes)

        loss = F.cross_entropy(output, y)
        losses.append(loss.item())

        # to compute accuracy
        y_pred = torch.max(output, 1)[1]  # y_pred != output
        step_score = accuracy_score(y.cpu().data.squeeze().numpy(), y_pred.cpu().data.squeeze().numpy())
        scores.append(step_score)         # computed on CPU
        
        # Collect all predictions for comprehensive metrics
        all_y_true.extend(y.cpu().data.squeeze().numpy())
        all_y_pred.extend(y_pred.cpu().data.squeeze().numpy())
        
        loss.backward()
        optimizer.step()

        # Update progress bar with current metrics
        avg_loss = np.mean(losses[-10:]) if len(losses) >= 10 else np.mean(losses)
        avg_acc = np.mean(scores[-10:]) if len(scores) >= 10 else np.mean(scores)
        
        pbar.set_postfix({
            'Loss': f'{loss.item():.3f}',
            'Acc': f'{step_score*100:.1f}%',
            'AvgL': f'{avg_loss:.3f}',
            'AvgA': f'{avg_acc*100:.1f}%'
        })

        # Reduced frequency of detailed logging to avoid interference
        if (batch_idx + 1) % (log_interval * 5) == 0:
            tqdm.write(f'  Detailed: Epoch {epoch + 1} [{N_count}/{len(train_loader.dataset)} ({100. * (batch_idx + 1) / len(train_loader):.0f}%)]\tLoss: {loss.item():.4f}, Acc: {step_score*100:.2f}%')

    # Convert to numpy arrays for metrics calculation
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    
    # Close the progress bar properly
    pbar.close()
    
    return losses, scores, all_y_true, all_y_pred


def validation(model, device, optimizer, test_loader, epoch, action_names, le, save_best_only=False):
    # set model as testing mode
    cnn_encoder, rnn_decoder = model
    cnn_encoder.eval()
    rnn_decoder.eval()

    test_loss = 0
    all_y = []
    all_y_pred = []

    # Add progress bar for validation
    with torch.no_grad():
        val_pbar = tqdm(test_loader, desc=f'Validation Epoch {epoch+1}', 
                       position=2, leave=True, ncols=80)
        for X, y in val_pbar:
            # distribute data to device
            X, y = X.to(device), y.to(device).view(-1, )

            output = rnn_decoder(cnn_encoder(X))

            loss = F.cross_entropy(output, y, reduction='sum')
            test_loss += loss.item()                 # sum up batch loss
            y_pred = output.max(1, keepdim=True)[1]  # (y_pred != output) get the index of the max log-probability

            # collect all y and y_pred in all batches
            all_y.extend(y)
            all_y_pred.extend(y_pred)

            # Update validation progress
            val_pbar.set_postfix({'Loss': f'{loss.item():.3f}'})
        
        val_pbar.close()

    test_loss /= len(test_loader.dataset)

    # Convert to numpy for metrics calculation
    y_true_np = torch.stack(all_y, dim=0).cpu().data.squeeze().numpy()
    y_pred_np = torch.stack(all_y_pred, dim=0).cpu().data.squeeze().numpy()
    
    # Calculate comprehensive metrics
    test_accuracy = accuracy_score(y_true_np, y_pred_np)
    test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(y_true_np, y_pred_np, average='weighted', zero_division=0)
    
    # Per-class metrics
    per_class_precision, per_class_recall, per_class_f1, per_class_support = precision_recall_fscore_support(
        y_true_np, y_pred_np, labels=range(len(action_names)), zero_division=0
    )
    
    # Calculate per-class accuracy
    per_class_accuracy = []
    for class_idx in range(len(action_names)):
        mask = (y_true_np == class_idx)
        if np.any(mask):
            class_acc = accuracy_score(y_true_np[mask], y_pred_np[mask])
        else:
            class_acc = 0.0
        per_class_accuracy.append(class_acc)

    # show information
    print(f'\nValidation Results - Epoch {epoch + 1}:')
    print(f'Loss: {test_loss:.4f} | Accuracy: {test_accuracy*100:.2f}% | Precision: {test_precision:.4f} | Recall: {test_recall:.4f} | F1: {test_f1:.4f}\n')

    # save Pytorch models (save every epoch or only best)
    if not save_best_only:
        torch.save(cnn_encoder.state_dict(), os.path.join(save_model_path, f'cnn_encoder_epoch{epoch + 1}.pth'))
        torch.save(rnn_decoder.state_dict(), os.path.join(save_model_path, f'rnn_decoder_epoch{epoch + 1}.pth'))
        torch.save(optimizer.state_dict(), os.path.join(save_model_path, f'optimizer_epoch{epoch + 1}.pth'))
        print(f"Epoch {epoch + 1} model saved!")

    # Return comprehensive metrics
    metrics = {
        'loss': test_loss,
        'accuracy': test_accuracy,
        'precision': test_precision,
        'recall': test_recall,
        'f1': test_f1,
        'per_class_accuracy': per_class_accuracy,
        'per_class_precision': per_class_precision,
        'per_class_recall': per_class_recall,
        'per_class_f1': per_class_f1,
        'per_class_support': per_class_support,
        'y_true': y_true_np,
        'y_pred': y_pred_np
    }

    return metrics

# Detect devices
use_cuda = torch.cuda.is_available()                   # check if GPU exists
device = torch.device("cuda" if use_cuda else "cpu")   # use CPU or GPU

# Data loading parameters
# params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 4, 'pin_memory': True} if use_cuda else {}

# Temporarily change this for debugging:
params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 4, 'pin_memory': True} if use_cuda else {}

# Modified section: Create action_names from folder structure
def create_action_names_from_folders(data_path, action_name_path):
    """Generate action_names list from folder structure and save as pickle"""
    if not os.path.exists(data_path):
        raise ValueError(f"Data path {data_path} does not exist!")
    
    # Get all subdirectories (class folders)
    class_folders = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    class_folders.sort()  # Sort for consistent ordering
   
    print(f"Data path: {data_path}")
    print(f"Found directories: {class_folders}")

    if len(class_folders) == 0:
        raise ValueError(f"No class directories found in {data_path}!")


    # Save as pickle file (matching your existing variable name)
    with open(action_name_path, 'wb') as f:
        pickle.dump(class_folders, f)
    
    print(f"Created {action_name_path} with {len(class_folders)} classes")
    return class_folders

# Check if action_name_path exists, if not create it from folder structure
if not os.path.exists(action_name_path):
    print(f"Creating {action_name_path} from folder structure...")
    action_names = create_action_names_from_folders(data_path, action_name_path)
else:
    # load existing action names
    with open(action_name_path, 'rb') as f:
        action_names = pickle.load(f)

'''
print(f"Action names type: {type(action_names)}")
print(f"Action names content: {action_names}")
print(f"Number of classes: {len(action_names) if action_names else 0}")
'''

# Ensure action_names is a proper list
if not isinstance(action_names, list) or len(action_names) == 0:
    raise ValueError(f"action_names must be a non-empty list, got: {action_names}")

# Update k to match your dataset
k = len(action_names)  # number of target category

# convert labels -> category
le = LabelEncoder()
le.fit(action_names)

# show how many classes there are
list(le.classes_)

# convert category -> 1-hot
action_category = le.transform(action_names).reshape(-1, 1)
enc = OneHotEncoder()
enc.fit(action_category)

#Scan files from folder structure instead of parsing filenames

# Modified section: Scan files from folder structure instead of parsing filenames
actions = []
all_names = []

'''
print(f"Scanning data_path: {data_path}")
print(f"Action names: {action_names}")
'''

# Scan all files in class directories
for class_name in action_names:
    class_path = os.path.join(data_path, class_name)
    
    
   #  print(f"Checking class path: {class_path}")

    if os.path.isdir(class_path):
        
        # print(f"  Directory exists, listing contents...")
        

        filenames = os.listdir(class_path)
        
        # print(f"  Found {len(filenames)} items: {filenames[:5]}...")  # Show first 5 items
        
        video_count = 0
        for filename in filenames:
            file_path = os.path.join(class_path, filename)
            
            # Check if it's a directory (video folder) - THIS IS THE KEY ISSUE
            if os.path.isdir(file_path):
                # Store relative path from data_path
                relative_path = os.path.join(class_name, filename)
                all_names.append(relative_path)
                actions.append(class_name)
                video_count += 1
        
        # print(f"  Added {video_count} video folders from {class_name}")
    else:
        print(f"  Directory {class_path} does not exist!")

# print(f"Total video folders found: {len(all_names)}")
# print(f"Sample video folders: {all_names[:5]}")

if len(all_names) == 0:
    print("ERROR: No video folders found!")
    print("Expected structure:")
    print("jpegs_256/")
    print("├── class1/")
    print("│   ├── video1/")
    print("│   │   ├── frame_001.jpg")
    print("│   │   └── ...")
    print("│   └── video2/")
    print("└── class2/")
    exit(1)

# print(f"Found {len(all_names)} files across {len(action_names)} classes")

# list all data files (keeping your variable names)
all_X_list = all_names                  # all video file names
all_y_list = labels2cat(le, actions)    # all video labels

# train, test split
train_list, test_list, train_label, test_label = train_test_split(all_X_list, all_y_list, test_size=0.2, random_state=42)

transform = transforms.Compose([transforms.Resize([res_size, res_size]),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

selected_frames = np.arange(begin_frame, end_frame, skip_frame).tolist()

train_set, valid_set = Dataset_CRNN(data_path, train_list, train_label, selected_frames, transform=transform), \
                       Dataset_CRNN(data_path, test_list, test_label, selected_frames, transform=transform)

train_loader = data.DataLoader(train_set, **params)
valid_loader = data.DataLoader(valid_set, **params)

'''
# Add this after creating train_loader and valid_loader, before the model creation
print("="*50)
print("DEBUGGING: Testing data loading...")

try:
    # Test loading one batch
    print("Attempting to load first batch...")
    first_batch = next(iter(train_loader))
    X_batch, y_batch = first_batch
    
    print(f"✓ First batch loaded successfully!")
    print(f"  Batch shape: {X_batch.shape}")
    print(f"  Batch dtype: {X_batch.dtype}")
    print(f"  Labels shape: {y_batch.shape}")
    print(f"  Labels dtype: {y_batch.dtype}")
    print(f"  Memory usage: {X_batch.element_size() * X_batch.nelement() / 1024**2:.2f} MB")
    
    # Test moving to device
    print("Testing device transfer...")
    X_batch = X_batch.to(device)
    y_batch = y_batch.to(device)
    print(f"✓ Successfully moved to device: {device}")
    
except Exception as e:
    print(f"✗ ERROR loading first batch: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("="*50)
print_gpu_memory()
'''

# Create model
cnn_encoder = ResCNNEncoder(fc_hidden1=CNN_fc_hidden1, fc_hidden2=CNN_fc_hidden2, drop_p=dropout_p, CNN_embed_dim=CNN_embed_dim).to(device)
rnn_decoder = DecoderRNN(CNN_embed_dim=CNN_embed_dim, h_RNN_layers=RNN_hidden_layers, h_RNN=RNN_hidden_nodes,
                         h_FC_dim=RNN_FC_dim, drop_p=dropout_p, num_classes=k).to(device)

'''
# Add this after creating cnn_encoder and rnn_decoder, before the training loop
print("DEBUGGING: Testing model forward pass...")

try:
    # Test with a small batch first
    print("Testing CNN encoder...")
    cnn_encoder.eval()  # Set to eval mode for testing
    
    with torch.no_grad():
        # Use the first batch we already loaded
        print(f"Input shape to CNN: {X_batch.shape}")
        cnn_features = cnn_encoder(X_batch)
        print(f"✓ CNN encoder output shape: {cnn_features.shape}")
        print(f"CNN output dtype: {cnn_features.dtype}")
        
        print("Testing RNN decoder...")
        rnn_decoder.eval()  # Set to eval mode for testing
        rnn_output = rnn_decoder(cnn_features)
        print(f"✓ RNN decoder output shape: {rnn_output.shape}")
        print(f"Expected output shape: ({X_batch.shape[0]}, {k})")
        
        # Test loss computation
        print("Testing loss computation...")
        y_batch_flat = y_batch.view(-1)
        loss = F.cross_entropy(rnn_output, y_batch_flat)
        print(f"✓ Loss computed successfully: {loss.item()}")
        
except Exception as e:
    print(f"✗ ERROR in model forward pass: {e}")
    import traceback
    traceback.print_exc()
    exit(1)


# Set models back to train mode
cnn_encoder.train()
rnn_decoder.train()
print("✓ Model testing completed successfully!")
print("="*50)

print_gpu_memory()
'''

# Parallelize model to multiple GPUs
if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs!")
    # cnn_encoder = nn.DataParallel(cnn_encoder)
    # rnn_decoder = nn.DataParallel(rnn_decoder)

    # Combine all EncoderCNN + DecoderRNN parameters
    # crnn_params = list(cnn_encoder.module.fc1.parameters()) + list(cnn_encoder.module.bn1.parameters()) + \
    #              list(cnn_encoder.module.fc2.parameters()) + list(cnn_encoder.module.bn2.parameters()) + \
    #              list(cnn_encoder.module.fc3.parameters()) + list(rnn_decoder.parameters())

if torch.cuda.device_count() >= 1:
    print("Using", torch.cuda.device_count(), "GPU!")
    # Combine all EncoderCNN + DecoderRNN parameters
    crnn_params = list(cnn_encoder.fc1.parameters()) + list(cnn_encoder.bn1.parameters()) + \
                  list(cnn_encoder.fc2.parameters()) + list(cnn_encoder.bn2.parameters()) + \
                  list(cnn_encoder.fc3.parameters()) + list(rnn_decoder.parameters())

optimizer = torch.optim.Adam(crnn_params, lr=learning_rate)


# record training process
epoch_train_metrics = []
epoch_val_metrics = []

# Enhanced training loop with early stopping
print(f"Starting training for {epochs} epochs with early stopping (patience: {patience})...")
print(f"Dataset: {len(train_list)} train samples, {len(test_list)} validation samples")
print(f"Batch size: {batch_size}, Learning rate: {learning_rate}")
print("="*80)

# start training
for epoch in range(epochs):
    print(f"\n{'='*20} EPOCH {epoch+1}/{epochs} {'='*20}")

    # train model
    train_losses, train_scores, train_y_true, train_y_pred = train(log_interval, [cnn_encoder, rnn_decoder], device, train_loader, optimizer, epoch)
    
    # Calculate comprehensive training metrics
    train_metrics = calculate_train_metrics(train_losses, train_scores, train_y_true, train_y_pred, action_names)
    epoch_train_metrics.append(train_metrics)
    
    # validation
    val_metrics = validation([cnn_encoder, rnn_decoder], device, optimizer, valid_loader, epoch, action_names, le, save_best_only=True)
    epoch_val_metrics.append(val_metrics)
    
    # Print epoch summary
    print(f"\nEPOCH {epoch+1} SUMMARY:")
    print(f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']*100:.2f}%, F1: {train_metrics['f1']:.4f}")
    print(f"Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']*100:.2f}%, F1: {val_metrics['f1']:.4f}")

    # Early stopping logic
    current_val_loss = val_metrics['loss']
    if current_val_loss < best_val_loss:
        best_val_loss = current_val_loss
        patience_counter = 0
        
        # Save best model
        print(f"🎯 NEW BEST! Validation loss: {best_val_loss:.4f} - Saving best model...")
        torch.save(cnn_encoder.state_dict(), os.path.join(save_model_path, 'best_cnn_encoder1.pth'))
        torch.save(rnn_decoder.state_dict(), os.path.join(save_model_path, 'best_rnn_decoder1.pth'))
        torch.save(optimizer.state_dict(), os.path.join(save_model_path, 'best_optimizer1.pth'))
    else:
        patience_counter += 1
        print(f"⏳ No improvement. Patience: {patience_counter}/{patience}")
        
        if patience_counter >= patience:
            print(f"🛑 Early stopping triggered after {epoch + 1} epochs!")
            early_stop = True
            break
    
    # Save metrics every 10 epochs and at the end
    if (epoch + 1) % 10 == 0 or epoch == epochs - 1 or early_stop:
        save_metrics_and_plots(epoch_train_metrics, epoch_val_metrics, action_names)
    
    print(f"{'='*60}")

if early_stop:
    print(f"\nTraining stopped early at epoch {epoch + 1} due to no improvement in validation loss.")
else:
    print(f"\nTraining completed after {epochs} epochs!")

# Save final comprehensive metrics and plots
save_metrics_and_plots(epoch_train_metrics, epoch_val_metrics, action_names)

print(f"Best validation loss: {best_val_loss:.4f}")
print("Best model saved as best_cnn_encoder.pth and best_rnn_decoder.pth")

print("Training process completed successfully!")
print("All metrics, plots, and model checkpoints have been saved.")


