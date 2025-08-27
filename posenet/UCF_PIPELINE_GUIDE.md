# UCF-101 to PoseNet Pipeline Integration

This guide shows how to train PoseNet on UCF-101 dataset using your existing pipeline (s1-s5).

## Quick Start

### Option 1: Using Batch Script (Windows)
```bash
# Full dataset (all 101 classes)
run_ucf_pipeline.bat "C:\path\to\UCF-101"

# Subset for testing (first 10 classes only)
run_ucf_pipeline.bat "C:\path\to\UCF-101" 10
```

### Option 2: Using PowerShell
```powershell
# Full dataset
.\run_ucf_pipeline.ps1 -UCFRoot "C:\path\to\UCF-101"

# Subset with custom settings
.\run_ucf_pipeline.ps1 -UCFRoot "C:\path\to\UCF-101" -SubsetClasses 10 -FramesPerVideo 30 -SampleInterval 3
```

### Option 3: Manual Step-by-Step

```bash
# Step 0: Convert UCF videos to frames
python src/s0_ucf_to_posenet_adapter.py --ucf_root "C:\path\to\UCF-101" --subset_classes 10

# Step 1: Extract skeletons from frames
python src/s1_get_skeletons_from_training_imgs.py

# Step 2: Combine skeleton data
python src/s2_put_skeleton_txts_to_a_single_txt.py

# Step 3: Process features
python src/s3_preprocess_features.py

# Step 4: Train model
python src/s4_train.py

# Step 5: Test model (optional)
python src/s5_test.py --model_path model/trained_classifier.pickle --data_type video --data_path "test_video.avi" --output_folder output
```

## Pipeline Overview

```
UCF-101 Videos → [s0] → Frames → [s1] → Skeletons → [s2] → Combined → [s3] → Features → [s4] → Model → [s5] → Test
```

1. **s0**: `s0_ucf_to_posenet_adapter.py` - Converts UCF videos to frames
2. **s1**: `s1_get_skeletons_from_training_imgs.py` - Extracts skeletons using OpenPose
3. **s2**: `s2_put_skeleton_txts_to_a_single_txt.py` - Combines skeleton data
4. **s3**: `s3_preprocess_features.py` - Processes features from skeletons
5. **s4**: `s4_train.py` - Trains classifier on skeleton features
6. **s5**: `s5_test.py` - Tests trained model

## Performance Optimizations for Your Hardware

Given your constraints (4GB GPU, 16GB RAM, 10GB free space):

### Recommended Settings
```bash
# Use subset for initial testing
--subset_classes 10           # Use only 10 classes instead of 101
--frames_per_video 30         # Extract 30 frames per video (vs 50+)
--sample_interval 3           # Sample every 3rd frame (faster processing)
```

### Storage Usage
- **10 classes**: ~2-3GB frames + ~500MB skeletons = **~3.5GB total**
- **50 classes**: ~8-10GB frames + ~2GB skeletons = **~12GB total** (exceeds your limit)
- **101 classes**: ~20GB+ (not feasible with 10GB free space)

## Time Estimates

### With Subset (10 classes)
- **s0** (Frame extraction): 20-30 minutes
- **s1** (Skeleton detection): 1-2 hours  
- **s2-s3** (Data processing): 5-10 minutes
- **s4** (Training): 10-30 minutes
- **Total**: ~2-3 hours

### With Full Dataset (101 classes)
- Would require ~20GB+ storage (not feasible)
- Would take 8-12 hours total

## Configuration

The adapter automatically updates `config/config.yaml`:

```yaml
# Before (original PoseNet classes)
classes: ['stand', 'walk', 'run', 'jump', 'sit', 'squat', 'kick', 'punch', 'wave']

# After (UCF-101 classes)
classes: ['ApplyEyeMakeup', 'ApplyLipstick', 'Archery', 'BabyCrawling', ...]
```

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce `--subset_classes` or `--frames_per_video`
2. **Disk Space**: Use `--subset_classes 5` for minimal testing
3. **Slow Processing**: Increase `--sample_interval` to 4-5

### Error Recovery
If pipeline fails at any step, you can resume:
```bash
# If s1 fails, just re-run from s1
python src/s1_get_skeletons_from_training_imgs.py

# If s4 fails, re-run from s4  
python src/s4_train.py
```

### Restore Original Config
```bash
# Restore original config.yaml
cp config/config.yaml.backup config/config.yaml
```

## Expected Results

With 10 UCF classes, you should expect:
- **Training Accuracy**: 70-85%
- **Test Accuracy**: 60-75%
- **Model Size**: ~500KB-2MB
- **Inference Speed**: <100ms per frame

## Next Steps

1. **Start with subset**: Use 5-10 classes first
2. **Validate pipeline**: Ensure all steps work
3. **Scale up**: Gradually increase classes if storage allows
4. **Optimize**: Tune hyperparameters in `s4_train.py`

## Files Created

- `src/s0_ucf_to_posenet_adapter.py` - Main adapter script
- `run_ucf_pipeline.bat` - Windows batch script
- `run_ucf_pipeline.ps1` - PowerShell script  
- `data/source_images_ucf/` - Extracted frames
- `config/config.yaml.backup` - Original config backup
