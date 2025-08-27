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
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import pickle
import warnings

# Ignore the specific UserWarning from sklearn
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


def print_gpu_memory():
    if torch.cuda.is_available():
        print(f"GPU Memory: {torch.cuda.memory_allocated()/1024**2:.2f} MB allocated, "
              f"{torch.cuda.memory_reserved()/1024**2:.2f} MB reserved")

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
epochs = 120        # training epochs
batch_size = 40
learning_rate = 1e-3
log_interval = 10   # interval for displaying training info

# Select which frame to begin & end in videos
begin_frame, end_frame, skip_frame = 1, 14, 1


def train(log_interval, model, device, train_loader, optimizer, epoch):
    # set model as training mode

    cnn_encoder, rnn_decoder = model
    # print("Training cnn encoder")    
    cnn_encoder.train()
    # print("Training rnn decoder")
    rnn_decoder.train()

    losses = []
    scores = []
    N_count = 0   # counting total trained sample in one epoch
    
    # Progress bar
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1:3d}/{epochs}')
    
    for batch_idx, (X, y) in enumerate(train_loader):
        # distribute data to device
        X, y = X.to(device), y.to(device).view(-1, )

        N_count += X.size(0)
        
        # print("optimizer step")
        optimizer.zero_grad()
        output = rnn_decoder(cnn_encoder(X))   # output has dim = (batch, number of classes)

        loss = F.cross_entropy(output, y)
        losses.append(loss.item())

        # to compute accuracy
        y_pred = torch.max(output, 1)[1]  # y_pred != output
        step_score = accuracy_score(y.cpu().data.squeeze().numpy(), y_pred.cpu().data.squeeze().numpy())
        scores.append(step_score)         # computed on CPU
        
        
        loss.backward()
        optimizer.step()
        
        '''
        # show information
        if (batch_idx + 1) % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accu: {:.2f}%'.format(
                epoch + 1, N_count, len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader), loss.item(), 100 * step_score))
        
        '''

         # Update progress bar with current metrics
        avg_loss = np.mean(losses[-10:]) if len(losses) >= 10 else np.mean(losses)
        avg_acc = np.mean(scores[-10:]) if len(scores) >= 10 else np.mean(scores)
        
        # print("updated loss and acc")
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{step_score*100:.2f}%',
            'AvgLoss': f'{avg_loss:.4f}',
            'AvgAcc': f'{avg_acc*100:.2f}%'
        })
        '''
        if batch_idx == 0:
            print("✓ First training batch completed successfully!")
            print(f"Loss: {loss.item()}, Accuracy: {step_score}")
            print_gpu_memory()
            # Remove this return statement after confirming it works:
            return losses, scores  # Early exit for testing
        '''

        # Still show detailed information at intervals
        if (batch_idx + 1) % log_interval == 0:
            tqdm.write(f'Train Epoch: {epoch + 3} [{N_count}/{len(train_loader.dataset)} ({100. * (batch_idx + 1) / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}, Accu: {step_score*100:.2f}%')

    # print("Returning losses and scores")
    return losses, scores


def validation(model, device, optimizer, test_loader):
    # set model as testing mode
    cnn_encoder, rnn_decoder = model
    cnn_encoder.eval()
    rnn_decoder.eval()

    test_loss = 0
    all_y = []
    all_y_pred = []

    ####################### commented for progress bar #############################
    '''
    with torch.no_grad():
        for X, y in test_loader:
            # distribute data to device
            X, y = X.to(device), y.to(device).view(-1, )

            output = rnn_decoder(cnn_encoder(X))

            loss = F.cross_entropy(output, y, reduction='sum')
            test_loss += loss.item()                 # sum up batch loss
            y_pred = output.max(1, keepdim=True)[1]  # (y_pred != output) get the index of the max log-probability

            # collect all y and y_pred in all batches
            all_y.extend(y)
            all_y_pred.extend(y_pred)

    test_loss /= len(test_loader.dataset)

    # compute accuracy
    all_y = torch.stack(all_y, dim=0)
    all_y_pred = torch.stack(all_y_pred, dim=0)
    test_score = accuracy_score(all_y.cpu().data.squeeze().numpy(), all_y_pred.cpu().data.squeeze().numpy())

    # show information
    print('\nTest set ({:d} samples): Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(len(all_y), test_loss, 100* test_score))

    # save Pytorch models of best record
    torch.save(cnn_encoder.state_dict(), os.path.join(save_model_path, 'cnn_encoder_epoch{}.pth'.format(epoch + 1)))  # save spatial_encoder
    torch.save(rnn_decoder.state_dict(), os.path.join(save_model_path, 'rnn_decoder_epoch{}.pth'.format(epoch + 1)))  # save motion_encoder
    torch.save(optimizer.state_dict(), os.path.join(save_model_path, 'optimizer_epoch{}.pth'.format(epoch + 1)))      # save optimizer
    print("Epoch {} model saved!".format(epoch + 1))

    return test_loss, test_score
    '''
    ############################## commented for progress bar ###########################
    # Add progress bar for validation
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Validation', leave=False)
        for X, y in pbar:
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
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})

    test_loss /= len(test_loader.dataset)

    # compute accuracy
    all_y = torch.stack(all_y, dim=0)
    all_y_pred = torch.stack(all_y_pred, dim=0)
    test_score = accuracy_score(all_y.cpu().data.squeeze().numpy(), all_y_pred.cpu().data.squeeze().numpy())

    # show information
    print(f'\nTest set ({len(all_y):d} samples): Average loss: {test_loss:.4f}, Accuracy: {test_score*100:.2f}%\n')

    # save Pytorch models of best record
    torch.save(cnn_encoder.state_dict(), os.path.join(save_model_path, 'cnn_encoder_epoch{}.pth'.format(epoch + 1)))  # save spatial_encoder
    torch.save(rnn_decoder.state_dict(), os.path.join(save_model_path, 'rnn_decoder_epoch{}.pth'.format(epoch + 1)))  # save motion_encoder
    torch.save(optimizer.state_dict(), os.path.join(save_model_path, 'optimizer_epoch{}.pth'.format(epoch + 1)))      # save optimizer
    print("Epoch {} model saved!".format(epoch + 1))

    return test_loss, test_score

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
epoch_train_losses = []
epoch_train_scores = []
epoch_test_losses = []
epoch_test_scores = []


###################################### commented for progress bar ###################################
'''
# start training
for epoch in range(epochs):
    # train, test model
    train_losses, train_scores = train(log_interval, [cnn_encoder, rnn_decoder], device, train_loader, optimizer, epoch)
    epoch_test_loss, epoch_test_score = validation([cnn_encoder, rnn_decoder], device, optimizer, valid_loader)

    # save results
    epoch_train_losses.append(train_losses)
    epoch_train_scores.append(train_scores)
    epoch_test_losses.append(epoch_test_loss)
    epoch_test_scores.append(epoch_test_score)

    # save all train test results
    A = np.array(epoch_train_losses)
    B = np.array(epoch_train_scores)
    C = np.array(epoch_test_losses)
    D = np.array(epoch_test_scores)
    np.save('./CRNN_epoch_training_losses.npy', A)
    np.save('./CRNN_epoch_training_scores.npy', B)
    np.save('./CRNN_epoch_test_loss.npy', C)
    np.save('./CRNN_epoch_test_score.npy', D)
'''
######################## commented for enhancing with progress bar ################################

# Enhanced training loop with overall progress
print(f"Starting training for {epochs} epochs...")
print(f"Dataset: {len(train_list)} train samples, {len(test_list)} validation samples")
print(f"Batch size: {batch_size}, Learning rate: {learning_rate}")
print("="*80)

# Create overall epoch progress bar
epoch_pbar = tqdm(range(epochs), desc='Overall Progress', position=0)

# start training
print("About to start training loop...")

for epoch in epoch_pbar:
    print(f"\n=== Starting Epoch {epoch+1}/{epochs} ===")

    # Update epoch progress bar description
    epoch_pbar.set_description(f'Epoch {epoch+1}/{epochs}')

    # train, test model
    print("Calling train function...")
    train_losses, train_scores = train(log_interval, [cnn_encoder, rnn_decoder], device, train_loader, optimizer, epoch)
    print("Calling validation function...")
    epoch_test_loss, epoch_test_score = validation([cnn_encoder, rnn_decoder], device, optimizer, valid_loader)

    # save results
    epoch_train_losses.append(train_losses)
    epoch_train_scores.append(train_scores)
    epoch_test_losses.append(epoch_test_loss)
    epoch_test_scores.append(epoch_test_score)
    
    print(f"Completed Epoch {epoch+1}")

    # Calculate epoch averages
    avg_train_loss = np.mean(train_losses)
    avg_train_score = np.mean(train_scores)

    # Update overall progress bar with epoch summary
    epoch_pbar.set_postfix({
        'T_Loss': f'{avg_train_loss:.4f}',
        'T_Acc': f'{avg_train_score*100:.2f}%',
        'V_Loss': f'{epoch_test_loss:.4f}',
        'V_Acc': f'{epoch_test_score*100:.2f}%'
    })

    # save all train test results
    A = np.array(epoch_train_losses)
    B = np.array(epoch_train_scores)
    C = np.array(epoch_test_losses)
    D = np.array(epoch_test_scores)
    np.save('./CRNN_epoch_training_losses.npy', A)
    np.save('./CRNN_epoch_training_scores.npy', B)
    np.save('./CRNN_epoch_test_loss.npy', C)
    np.save('./CRNN_epoch_test_score.npy', D)

print("\nTraining completed!")

# plot
fig = plt.figure(figsize=(10, 4))
plt.subplot(121)
plt.plot(np.arange(1, epochs + 1), A[:, -1])  # train loss (on epoch end)
plt.plot(np.arange(1, epochs + 1), C)         #  test loss (on epoch end)
plt.title("model loss")
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(['train', 'test'], loc="upper left")
# 2nd figure
plt.subplot(122)
plt.plot(np.arange(1, epochs + 1), B[:, -1])  # train accuracy (on epoch end)
plt.plot(np.arange(1, epochs + 1), D)         #  test accuracy (on epoch end)
plt.title("training scores")
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend(['train', 'test'], loc="upper left")
title = "./fig_Kinetics_ResNetCRNN.png"
plt.savefig(title, dpi=600)
# plt.close(fig)
plt.show()


