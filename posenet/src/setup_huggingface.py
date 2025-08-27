#!/usr/bin/env python3
"""
Setup script for Hugging Face authentication
Run this before training VideoMAE model
"""

import os
from huggingface_hub import login

def setup_huggingface():
    print("Setting up Hugging Face authentication...")
    print("You have several options:")
    print("1. Use the command line: huggingface-cli login")
    print("2. Enter your token here")
    print("3. Set HF_TOKEN environment variable")
    
    # Check if already authenticated
    try:
        from huggingface_hub import whoami
        user_info = whoami()
        print(f"Already authenticated as: {user_info['name']}")
        return True
    except:
        pass
    
    # Option 1: Check environment variable
    token = os.getenv('HF_TOKEN')
    if token:
        print("Using token from HF_TOKEN environment variable...")
        try:
            login(token=token)
            print("Successfully authenticated with environment token!")
            return True
        except Exception as e:
            print(f"Failed to authenticate with environment token: {e}")
    
    # Option 2: Manual token input
    print("\nTo get a token:")
    print("1. Go to https://huggingface.co/settings/tokens")
    print("2. Create a new token (read access is sufficient)")
    print("3. Copy and paste it below")
    
    token = input("\nEnter your Hugging Face token (or press Enter to skip): ").strip()
    
    if token:
        try:
            login(token=token)
            print("Successfully authenticated!")
            return True
        except Exception as e:
            print(f"Failed to authenticate: {e}")
            return False
    else:
        print("Skipping authentication. You may need to authenticate later.")
        return False

def test_videomae_access():
    """Test if we can access VideoMAE models"""
    print("\nTesting VideoMAE model access...")
    
    models_to_test = [
        "MCG-NJU/videomae-base",
        "facebook/videomae-base"
    ]
    
    for model_name in models_to_test:
        try:
            from transformers import VideoMAEImageProcessor
            processor = VideoMAEImageProcessor.from_pretrained(model_name)
            print(f"✓ Successfully accessed {model_name}")
            return model_name
        except Exception as e:
            print(f"✗ Failed to access {model_name}: {e}")
    
    print("Could not access any VideoMAE models. Please check authentication.")
    return None

if __name__ == "__main__":
    print("Hugging Face Setup for VideoMAE Training")
    print("=" * 50)
    
    # Setup authentication
    auth_success = setup_huggingface()
    
    # Test model access
    accessible_model = test_videomae_access()
    
    if accessible_model:
        print(f"\n✓ Ready to train! Use model: {accessible_model}")
    else:
        print(f"\n✗ Setup incomplete. Please resolve authentication issues.")
        print("\nAlternative solutions:")
        print("1. Run: pip install --upgrade transformers huggingface_hub")
        print("2. Try running: huggingface-cli login")
        print("3. Check your internet connection")
