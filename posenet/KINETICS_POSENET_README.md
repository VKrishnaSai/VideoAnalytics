# Kinetics4005per PoseNet Training

This guide shows how to train PoseNet (skeleton-based action recognition) on the Kinetics4005per dataset following the original PoseNet pipeline.

## Overview

The PoseNet training pipeline consists of several stages:
1. **Video to Frames**: Extract frames from Kinetics videos
2. **Skeleton Detection**: Use OpenPose to detect human skeletons in frames
3. **Feature Processing**: Convert skeleton data to time-series features
4. **Training**: Train a classifier on the processed features

## Prerequisites

### 1. Install Dependencies

```bash
pip install -r requirements_kinetics_posenet.txt
```

### 2. Install OpenPose

This is the most critical dependency. Follow the official OpenPose installation guide:
- [OpenPose Installation](https://github.com/CMU-Perceptual-Computing-Lab/openpose)
- Ensure Python bindings are available
- Test with: `python -c "import sys; sys.path.append('path/to/openpose/build/python'); import openpose"`

### 3. Setup Kinetics4005per Dataset

Your dataset should be structured as:
```
kinetics4005per/
├── train/
│   └── train/
│       ├── class1/
│       │   ├── video1.mp4
│       │   └── video2.mp4
│       ├── class2/
│       └── ...
└── val/
    └── val/
        ├── class1/
        │   ├── video3.mp4
        │   └── video4.mp4
        ├── class2/
        └── ...
```

## Setup and Configuration

### 1. Check Dataset Structure

```bash
python setup_kinetics.py
```

This will:
- Verify the dataset structure
- Count videos and classes
- Update `config.yaml` with Kinetics classes
- Provide next steps

### 2. Verify Configuration

The setup script creates a `config.yaml` with all necessary paths and settings for the PoseNet pipeline.

## Training Pipeline

### Step 1: Convert Videos to Skeleton Data

```bash
python src/kinetics_to_posenet_adapter.py
```

**Options:**
- `--data_root ./kinetics4005per`: Path to dataset
- `--output_dir ./kinetics_posenet_data`: Output directory
- `--frames_per_video 10`: Frames to extract per video
- `--target_fps 5`: Target FPS for frame extraction

**What this does:**
- Extracts frames from all videos
- Runs OpenPose skeleton detection on frames
- Creates skeleton data files
- Saves visualizations for verification

**Output:**
- `kinetics_posenet_data/source_images_kinetics/`: Extracted frames
- `kinetics_posenet_data/detected_skeletons_kinetics/`: Skeleton data
- `kinetics_posenet_data/viz_imgs_kinetics/`: Visualization images
- `kinetics_posenet_data/kinetics_skeletons_info.txt`: Combined skeleton data

### Step 2: Extract Features

```bash
python src/s3_preprocess_features.py
```

**What this does:**
- Loads skeleton data
- Processes into time-series features (velocities, normalized positions, etc.)
- Applies sliding window approach
- Saves features and labels to CSV

**Output:**
- `data_proc/kinetics_features_X.csv`: Feature vectors
- `data_proc/kinetics_features_Y.csv`: Class labels

### Step 3: Train Classifier

```bash
python src/custom_s4_train.py
```

**What this does:**
- Loads processed features
- Splits data (train/validation/test)
- Trains MLP classifier with early stopping
- Evaluates performance with detailed metrics
- Saves trained model

**Output:**
- `model/kinetics_trained_classifier.pickle`: Trained model
- `results/`: Training curves and performance metrics
- Detailed classification report

## Configuration Details

The `config.yaml` file contains all pipeline settings:

### Key Settings
- `classes`: List of action classes from your dataset
- `features.window_size`: Number of frames for time-series features (default: 5)
- `openpose`: OpenPose model and image size settings
- File paths for each pipeline stage

### Customization
- Modify `frames_per_video` to extract more/fewer frames
- Adjust `window_size` for different temporal context
- Change OpenPose model for different accuracy/speed tradeoffs

## Expected Processing Times

For a typical Kinetics4005per dataset:
- **Frame Extraction**: 10-30 minutes (depends on video count/length)
- **Skeleton Detection**: 1-4 hours (depends on GPU and frame count)
- **Feature Processing**: 5-15 minutes
- **Training**: 10-60 minutes (depends on data size)

## Output and Results

### Training Metrics
- Per-class accuracy, precision, recall, F1-score
- Overall performance metrics (micro, macro, weighted averages)
- Confusion matrix
- Training curves (loss, accuracy, etc.)

### Model Files
- `kinetics_trained_classifier.pickle`: Main trained model
- Checkpoint files during training
- Performance plots and reports

## Troubleshooting

### Common Issues

1. **OpenPose Not Found**
   ```bash
   # Ensure OpenPose Python bindings are in your path
   export PYTHONPATH=$PYTHONPATH:/path/to/openpose/build/python
   ```

2. **Out of Memory**
   - Reduce `frames_per_video`
   - Process dataset in smaller batches
   - Use CPU-only mode if GPU memory is limited

3. **No Skeletons Detected**
   - Check video quality
   - Adjust OpenPose confidence thresholds
   - Verify people are visible in frames

4. **Config Errors**
   ```bash
   # Reset configuration
   python setup_kinetics.py
   ```

### Performance Tips

- Use GPU for OpenPose detection (much faster)
- Process videos in parallel if you have multiple GPUs
- Monitor disk space during frame extraction
- Use SSD storage for better I/O performance

## Advanced Usage

### Custom Class Subset
```bash
# Modify config.yaml to include only specific classes
# Then run the pipeline normally
```

### Different Frame Sampling
```bash
# Extract more frames per video for better temporal coverage
python src/kinetics_to_posenet_adapter.py --frames_per_video 20 --target_fps 10
```

### Feature Engineering
- Modify `s3_preprocess_features.py` for custom features
- Adjust window size in `config.yaml`
- Add data augmentation techniques

## Integration with Existing PoseNet

This pipeline is designed to work with the existing PoseNet codebase:
- Uses the same skeleton format and feature processing
- Compatible with existing visualization tools
- Can be used with the real-time demo after training

## References

- Original PoseNet: [felixchenfy/Realtime-Action-Recognition](https://github.com/felixchenfy/Realtime-Action-Recognition)
- OpenPose: [CMU-Perceptual-Computing-Lab/openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)
- Kinetics Dataset: [DeepMind Kinetics](https://deepmind.com/research/open-source/kinetics)
