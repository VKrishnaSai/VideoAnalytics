@echo off
REM UCF-101 PoseNet Training Pipeline
REM ================================
REM This script runs the complete pipeline from UCF-101 videos to trained PoseNet model

echo UCF-101 PoseNet Training Pipeline
echo ================================

REM Check if UCF-101 path is provided
if "%1"=="" (
    echo Usage: run_ucf_pipeline.bat "C:\path\to\UCF-101" [subset_classes]
    echo Example: run_ucf_pipeline.bat "D:\datasets\UCF-101" 10
    echo.
    echo Arguments:
    echo   UCF-101 path: Path to UCF-101 dataset folder
    echo   subset_classes: Optional - Use only first N classes ^(default: all 101^)
    exit /b 1
)

set UCF_ROOT=%1
set SUBSET_CLASSES=%2

echo UCF-101 Dataset: %UCF_ROOT%
if "%SUBSET_CLASSES%"=="" (
    echo Using all 101 classes
    set SUBSET_ARG=
) else (
    echo Using subset: %SUBSET_CLASSES% classes
    set SUBSET_ARG=--subset_classes %SUBSET_CLASSES%
)

echo.
echo Starting pipeline...

REM Step 0: Convert UCF-101 videos to frames
echo ========================================
echo Step 0: Converting UCF-101 videos to frames...
echo ========================================
python src/s0_ucf_to_posenet_adapter.py --ucf_root "%UCF_ROOT%" %SUBSET_ARG% --frames_per_video 30 --sample_interval 3

if %ERRORLEVEL% neq 0 (
    echo ERROR: Step 0 failed
    exit /b 1
)

REM Step 1: Extract skeletons from images
echo ========================================
echo Step 1: Extracting skeletons from images...
echo ========================================
python src/s1_get_skeletons_from_training_imgs.py

if %ERRORLEVEL% neq 0 (
    echo ERROR: Step 1 failed
    exit /b 1
)

REM Step 2: Combine skeleton data
echo ========================================
echo Step 2: Combining skeleton data...
echo ========================================
python src/s2_put_skeleton_txts_to_a_single_txt.py

if %ERRORLEVEL% neq 0 (
    echo ERROR: Step 2 failed
    exit /b 1
)

REM Step 3: Process features
echo ========================================
echo Step 3: Processing features...
echo ========================================
python src/s3_preprocess_features.py

if %ERRORLEVEL% neq 0 (
    echo ERROR: Step 3 failed
    exit /b 1
)

REM Step 4: Train model
echo ========================================
echo Step 4: Training model...
echo ========================================
python src/s4_train.py

if %ERRORLEVEL% neq 0 (
    echo ERROR: Step 4 failed
    exit /b 1
)

echo.
echo ========================================
echo ✅ Pipeline completed successfully!
echo ========================================
echo.
echo Trained model saved to: model/trained_classifier.pickle
echo.
echo To test the model, run:
echo python src/s5_test.py --model_path model/trained_classifier.pickle --data_type video --data_path "path/to/test/video.avi" --output_folder output
echo.
pause
