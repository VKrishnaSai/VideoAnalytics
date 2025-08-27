# UCF-101 VideoMAE Training

This script trains a VideoMAE model on the UCF-101 action recognition dataset using all three official train/test splits.

## Prerequisites

### 1. Install Dependencies

Make sure you have the required packages installed:

```bash
pip install torch torchvision transformers huggingface_hub opencv-python scikit-learn matplotlib seaborn numpy tqdm
```

### 2. Setup UCF-101 Dataset

You need to download and setup the UCF-101 dataset:

#### Option A: Automated Setup (Recommended)
```bash
python setup_ucf101.py
```

#### Option B: Manual Setup

1. **Download UCF-101 Videos:**
   - URL: https://www.crcv.ucf.edu/data/UCF101/UCF101.rar
   - Extract to create `UCF-101/` directory
   - Should contain 101 class directories (e.g., ApplyEyeMakeup, ApplyLipstick, etc.)

2. **Download Train/Test Splits:**
   - URL: https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip
   - Extract to create `ucfTrainTestlist/` directory
   - Should contain `classInd.txt`, `trainlist01-03.txt`, `testlist01-03.txt`

### Expected Directory Structure

```
posenet/
├── UCF-101/
│   ├── ApplyEyeMakeup/
│   │   ├── v_ApplyEyeMakeup_g01_c01.avi
│   │   └── ...
│   ├── ApplyLipstick/
│   └── ...
├── ucfTrainTestlist/
│   ├── classInd.txt
│   ├── trainlist01.txt
│   ├── trainlist02.txt
│   ├── trainlist03.txt
│   ├── testlist01.txt
│   ├── testlist02.txt
│   └── testlist03.txt
└── src/
    └── train_videomae_on_ucf.py
```

## Running the Training

### Basic Usage

```bash
python src/train_videomae_on_ucf.py
```

### Advanced Usage

```bash
# Use subset of classes for testing (e.g., first 10 classes)
python src/train_videomae_on_ucf.py --subset_classes 10

# Reduce batch size for limited GPU memory
python src/train_videomae_on_ucf.py --batch_size 2

# Train for fewer epochs
python src/train_videomae_on_ucf.py --epochs 20

# Resume from checkpoint
python src/train_videomae_on_ucf.py --checkpoint_dir path/to/checkpoint/directory
```

### Command Line Arguments

- `--epochs`: Number of training epochs (default: 100)
- `--patience`: Early stopping patience (default: 10)
- `--batch_size`: Batch size (default: 4)
- `--subset_classes`: Number of classes to use (default: all 101 classes)
- `--checkpoint_dir`: Checkpoint directory to resume from

## Features

### Dataset Handling
- Uses all three UCF-101 train/test splits (trainlist01-03, testlist01-03)
- Automatic train/validation/test split creation
- Supports subset training for quick testing
- Robust error handling for missing files

### Model Training
- VideoMAE model from Hugging Face Transformers
- Mixed precision training for memory efficiency
- Multi-GPU support with DataParallel
- Early stopping with validation monitoring
- Cosine annealing learning rate scheduler

### Monitoring & Evaluation
- Comprehensive metrics (accuracy, precision, recall, F1)
- Per-class and overall performance evaluation
- Confusion matrix visualization
- Training curves (loss, accuracy, precision, recall, F1)
- Detailed logging and checkpoints

### Output Files

The script creates a timestamped directory with:
- `ucf101_videomae_checkpoints_YYYYMMDD_HHMM/`
  - `best_checkpoint.pth`: Best model weights
  - `checkpoint_epoch_N.pth`: Per-epoch checkpoints
  - `ucf101_videomae_model_final.pth`: Final model
  - `train_metrics/`, `val_metrics/`, `test_metrics/`: Performance reports
  - `*.png`: Training curves and confusion matrices

### Log Files
- `ucf101_videomae_training_YYYYMMDD_HHMM.log`: Detailed training log

## Dataset Statistics

- **Classes**: 101 action categories
- **Total Videos**: ~13,320
- **Training Videos**: ~9,537 (from 3 splits combined)
- **Test Videos**: ~3,783 (from 3 splits combined)
- **Validation**: 20% of training data

## GPU Requirements

- **Minimum**: 6GB GPU memory with batch_size=2
- **Recommended**: 12GB+ GPU memory with batch_size=4+
- **Multi-GPU**: Automatically detected and used

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   python src/train_videomae_on_ucf.py --batch_size 2
   ```

2. **Missing Dataset Files**
   ```bash
   python setup_ucf101.py
   ```

3. **Hugging Face Authentication Error**
   ```bash
   huggingface-cli login
   ```

### Performance Tips

- Use `--subset_classes 10` for quick testing
- Reduce `--batch_size` if GPU memory is limited
- Use `--epochs 20` for faster experimentation
- Monitor GPU utilization with `nvidia-smi`

## References

- UCF-101 Dataset: [https://www.crcv.ucf.edu/data/UCF101.php](https://www.crcv.ucf.edu/data/UCF101.php)
- VideoMAE Paper: [https://arxiv.org/abs/2203.12602](https://arxiv.org/abs/2203.12602)
- Hugging Face VideoMAE: [https://huggingface.co/MCG-NJU/videomae-base](https://huggingface.co/MCG-NJU/videomae-base)
