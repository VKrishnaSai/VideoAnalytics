#!/usr/bin/env pwsh
# UCF-101 PoseNet Training Pipeline
# ================================
# This script runs the complete pipeline from UCF-101 videos to trained PoseNet model

param(
    [Parameter(Mandatory=$true)]
    [string]$UCFRoot,
    
    [Parameter(Mandatory=$false)]
    [int]$SubsetClasses = $null,
    
    [Parameter(Mandatory=$false)]
    [int]$FramesPerVideo = 30,
    
    [Parameter(Mandatory=$false)]
    [int]$SampleInterval = 3,
    
    [Parameter(Mandatory=$false)]
    [string]$Split = "01"
)

Write-Host "UCF-101 PoseNet Training Pipeline" -ForegroundColor Green
Write-Host "================================" -ForegroundColor Green

Write-Host "UCF-101 Dataset: $UCFRoot"
if ($SubsetClasses) {
    Write-Host "Using subset: $SubsetClasses classes"
    $subsetArg = "--subset_classes $SubsetClasses"
} else {
    Write-Host "Using all 101 classes"
    $subsetArg = ""
}

Write-Host "Frames per video: $FramesPerVideo"
Write-Host "Sample interval: $SampleInterval"
Write-Host "Split: $Split"
Write-Host ""

function Invoke-PipelineStep {
    param(
        [string]$StepName,
        [string]$Command,
        [string]$Arguments
    )
    
    Write-Host "========================================" -ForegroundColor Yellow
    Write-Host "Step $StepName" -ForegroundColor Yellow
    Write-Host "========================================" -ForegroundColor Yellow
    
    $fullCommand = "python $Command $Arguments"
    Write-Host "Running: $fullCommand"
    
    $process = Start-Process -FilePath "python" -ArgumentList "$Command $Arguments" -Wait -PassThru -NoNewWindow
    
    if ($process.ExitCode -ne 0) {
        Write-Host "ERROR: Step $StepName failed with exit code $($process.ExitCode)" -ForegroundColor Red
        exit 1
    }
    
    Write-Host "✅ Step $StepName completed successfully" -ForegroundColor Green
    Write-Host ""
}

try {
    # Step 0: Convert UCF-101 videos to frames
    $step0Args = "src/s0_ucf_to_posenet_adapter.py --ucf_root `"$UCFRoot`" $subsetArg --frames_per_video $FramesPerVideo --sample_interval $SampleInterval --split $Split"
    Invoke-PipelineStep "0: Converting UCF-101 videos to frames" "" $step0Args

    # Step 1: Extract skeletons from images
    Invoke-PipelineStep "1: Extracting skeletons from images" "src/s1_get_skeletons_from_training_imgs.py" ""

    # Step 2: Combine skeleton data
    Invoke-PipelineStep "2: Combining skeleton data" "src/s2_put_skeleton_txts_to_a_single_txt.py" ""

    # Step 3: Process features
    Invoke-PipelineStep "3: Processing features" "src/s3_preprocess_features.py" ""

    # Step 4: Train model
    Invoke-PipelineStep "4: Training model" "src/s4_train.py" ""

    Write-Host "========================================" -ForegroundColor Green
    Write-Host "✅ Pipeline completed successfully!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Trained model saved to: model/trained_classifier.pickle"
    Write-Host ""
    Write-Host "To test the model, run:"
    Write-Host "python src/s5_test.py --model_path model/trained_classifier.pickle --data_type video --data_path `"path/to/test/video.avi`" --output_folder output"
    Write-Host ""

} catch {
    Write-Host "ERROR: Pipeline failed - $_" -ForegroundColor Red
    exit 1
}
