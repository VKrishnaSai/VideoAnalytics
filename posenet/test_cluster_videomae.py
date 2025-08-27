#!/usr/bin/env python3
"""
Cluster-friendly test script for VideoMAE training
Tests basic functionality before running full training
"""
import os
import sys
import torch
import logging
from datetime import datetime

# Setup logging
log_filename = f'cluster_test_{datetime.now().strftime("%Y%m%d_%H%M")}.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def test_cuda_setup():
    """Test CUDA setup and memory"""
    logger.info("Testing CUDA setup...")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA available: True")
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"Number of GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            
        # Test basic GPU operation
        try:
            device = torch.device("cuda:0")
            test_tensor = torch.randn(100, 100).to(device)
            result = torch.mm(test_tensor, test_tensor.T)
            logger.info("GPU computation test: PASSED")
            
            # Test memory
            total_memory = torch.cuda.get_device_properties(0).total_memory
            logger.info(f"GPU 0 total memory: {total_memory / 1024**3:.1f} GB")
            
            del test_tensor, result
            torch.cuda.empty_cache()
            
        except Exception as e:
            logger.error(f"GPU computation test FAILED: {e}")
            return False
    else:
        logger.warning("CUDA not available - will use CPU")
    
    return True

def test_data_loading():
    """Test basic data loading"""
    logger.info("Testing data loading...")
    
    # Try to import required libraries
    try:
        from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
        import cv2
        import numpy as np
        logger.info("Required libraries imported successfully")
    except ImportError as e:
        logger.error(f"Import error: {e}")
        return False
    
    # Test VideoMAE model loading
    try:
        logger.info("Testing VideoMAE model loading...")
        model_name = "MCG-NJU/videomae-base"
        
        # Try to load processor (this tests internet connection)
        processor = VideoMAEImageProcessor.from_pretrained(model_name)
        logger.info("VideoMAE processor loaded successfully")
        
        # Try to load model
        model = VideoMAEForVideoClassification.from_pretrained(
            model_name,
            num_labels=10,  # Test with 10 classes
            ignore_mismatched_sizes=True
        )
        logger.info("VideoMAE model loaded successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"VideoMAE loading failed: {e}")
        logger.info("Trying alternative model...")
        
        try:
            alt_model = "facebook/videomae-base"
            processor = VideoMAEImageProcessor.from_pretrained(alt_model)
            model = VideoMAEForVideoClassification.from_pretrained(
                alt_model,
                num_labels=10,
                ignore_mismatched_sizes=True
            )
            logger.info("Alternative VideoMAE model loaded successfully")
            return True
        except Exception as e2:
            logger.error(f"Alternative model also failed: {e2}")
            return False

def test_dataloader():
    """Test DataLoader with minimal settings"""
    logger.info("Testing DataLoader...")
    
    try:
        from torch.utils.data import DataLoader, Dataset
        import torch
        
        # Create dummy dataset
        class DummyDataset(Dataset):
            def __init__(self, size=10):
                self.size = size
                
            def __len__(self):
                return self.size
                
            def __getitem__(self, idx):
                # Simulate video data: (frames, channels, height, width)
                video = torch.randn(16, 3, 224, 224)
                label = torch.randint(0, 10, (1,)).item()
                return {"pixel_values": video, "labels": label}
        
        dataset = DummyDataset(20)
        
        # Test with cluster-friendly settings
        dataloader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=True,
            num_workers=0,  # No multiprocessing
            pin_memory=False,
            timeout=10
        )
        
        logger.info("DataLoader created successfully")
        
        # Test iteration
        for i, batch in enumerate(dataloader):
            if i >= 2:  # Only test first 2 batches
                break
            logger.info(f"Batch {i+1}: pixel_values shape {batch['pixel_values'].shape}, labels shape {len(batch['labels'])}")
        
        logger.info("DataLoader iteration test: PASSED")
        return True
        
    except Exception as e:
        logger.error(f"DataLoader test FAILED: {e}")
        return False

def test_training_step():
    """Test a single training step"""
    logger.info("Testing training step...")
    
    try:
        import torch.nn as nn
        from torch.cuda.amp import autocast, GradScaler
        
        # Create dummy model and data
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Simple model for testing
        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 3 * 224 * 224, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        ).to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scaler = GradScaler() if torch.cuda.is_available() else None
        
        # Dummy batch
        pixel_values = torch.randn(2, 16, 3, 224, 224).to(device)
        labels = torch.randint(0, 10, (2,)).to(device)
        
        logger.info(f"Test data created on device: {device}")
        
        # Test forward pass
        if torch.cuda.is_available() and scaler:
            with autocast():
                outputs = model(pixel_values)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(pixel_values)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        optimizer.zero_grad()
        
        logger.info(f"Training step completed - Loss: {loss.item():.4f}")
        logger.info("Training step test: PASSED")
        return True
        
    except Exception as e:
        logger.error(f"Training step test FAILED: {e}")
        return False

def main():
    logger.info("Starting cluster compatibility tests...")
    logger.info("="*50)
    
    tests = [
        ("CUDA Setup", test_cuda_setup),
        ("Data Loading", test_data_loading),
        ("DataLoader", test_dataloader),
        ("Training Step", test_training_step)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n--- Running {test_name} Test ---")
        try:
            result = test_func()
            results[test_name] = result
            status = "PASSED" if result else "FAILED"
            logger.info(f"{test_name} Test: {status}")
        except Exception as e:
            logger.error(f"{test_name} Test FAILED with exception: {e}")
            results[test_name] = False
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info("TEST SUMMARY")
    logger.info("="*50)
    
    all_passed = True
    for test_name, result in results.items():
        status = "PASSED" if result else "FAILED"
        logger.info(f"{test_name}: {status}")
        if not result:
            all_passed = False
    
    if all_passed:
        logger.info("\nAll tests PASSED! Your cluster environment should work.")
        logger.info("You can now run the full training with:")
        logger.info("python src/train_videomae_on_ucf.py --debug --subset_classes 2 --epochs 2")
    else:
        logger.info("\nSome tests FAILED. Please fix the issues before running full training.")
    
    logger.info(f"Test log saved to: {log_filename}")

if __name__ == "__main__":
    main()
