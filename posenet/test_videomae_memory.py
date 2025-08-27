#!/usr/bin/env python3
"""
Quick memory test for VideoMAE model to identify hanging issues
"""
import torch
import torch.nn as nn
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
import logging
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_videomae_memory():
    """Test VideoMAE model with different batch sizes"""
    
    if not torch.cuda.is_available():
        logger.error("CUDA not available - this test requires GPU")
        return
    
    device = torch.device("cuda")
    logger.info(f"Using device: {device}")
    
    # Test different batch sizes
    batch_sizes = [1, 2, 4]
    num_frames = 16
    image_size = 224
    num_classes = 10
    
    for batch_size in batch_sizes:
        logger.info(f"\n=== Testing batch size {batch_size} ===")
        
        try:
            # Clear cache
            torch.cuda.empty_cache()
            
            # Create dummy data
            pixel_values = torch.randn(batch_size, num_frames, 3, image_size, image_size).to(device)
            labels = torch.randint(0, num_classes, (batch_size,)).to(device)
            
            logger.info(f"Created dummy data: {pixel_values.shape}")
            
            # Log initial memory
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            logger.info(f"Initial GPU memory - Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB")
            
            # Try to load model
            logger.info("Loading VideoMAE model...")
            
            try:
                model = VideoMAEForVideoClassification.from_pretrained(
                    "MCG-NJU/videomae-base",
                    num_labels=num_classes,
                    ignore_mismatched_sizes=True,
                ).to(device)
                logger.info("Model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load MCG-NJU model: {e}")
                logger.info("Trying facebook model...")
                model = VideoMAEForVideoClassification.from_pretrained(
                    "facebook/videomae-base",
                    num_labels=num_classes,
                    ignore_mismatched_sizes=True,
                ).to(device)
                logger.info("Facebook model loaded successfully")
            
            # Log memory after model loading
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            logger.info(f"After model load - Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB")
            
            # Test forward pass
            logger.info("Testing forward pass...")
            start_time = time.time()
            
            model.eval()
            with torch.no_grad():
                outputs = model(pixel_values=pixel_values)
                
            forward_time = time.time() - start_time
            logger.info(f"Forward pass completed in {forward_time:.2f}s")
            logger.info(f"Output shape: {outputs.logits.shape}")
            
            # Log memory after forward pass
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            logger.info(f"After forward pass - Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB")
            
            # Test training step
            logger.info("Testing training step...")
            model.train()
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
            
            start_time = time.time()
            outputs = model(pixel_values=pixel_values)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            training_time = time.time() - start_time
            logger.info(f"Training step completed in {training_time:.2f}s, loss: {loss.item():.4f}")
            
            # Final memory check
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            logger.info(f"After training step - Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB")
            
            logger.info(f"✅ Batch size {batch_size} PASSED")
            
            # Clean up
            del model, outputs, loss
            torch.cuda.empty_cache()
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                logger.error(f"❌ Batch size {batch_size} FAILED - Out of memory: {e}")
                torch.cuda.empty_cache()
            else:
                logger.error(f"❌ Batch size {batch_size} FAILED - Runtime error: {e}")
                break
        except Exception as e:
            logger.error(f"❌ Batch size {batch_size} FAILED - Unexpected error: {e}")
            break

def test_gpu_info():
    """Test basic GPU information"""
    logger.info("=== GPU Information ===")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA available: True")
        logger.info(f"Number of GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            logger.info(f"GPU {i}: {props.name}")
            logger.info(f"  Total memory: {props.total_memory / 1024**3:.1f} GB")
            logger.info(f"  Multiprocessors: {props.multi_processor_count}")
        
        # Test basic operations
        try:
            device = torch.device("cuda:0")
            test_tensor = torch.randn(1000, 1000).to(device)
            result = torch.mm(test_tensor, test_tensor.T)
            logger.info("✅ Basic GPU operations working")
            
            # Memory info
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            logger.info(f"Current GPU memory - Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB")
            
        except Exception as e:
            logger.error(f"❌ Basic GPU operations failed: {e}")
    else:
        logger.error("❌ CUDA not available")

def main():
    logger.info("Starting VideoMAE memory test...")
    
    # Test GPU basics first
    test_gpu_info()
    
    # Test VideoMAE model with different batch sizes
    test_videomae_memory()
    
    logger.info("\nTest completed! Check the results above.")
    logger.info("If batch size 1 fails, the issue is likely model size vs GPU memory.")
    logger.info("If batch size 1 works but training hangs, the issue might be elsewhere.")

if __name__ == "__main__":
    main()
