@echo off
echo Setting up VideoMAE Training Environment
echo ========================================

echo.
echo Step 1: Installing/Upgrading required packages...
pip install --upgrade -r requirements_videomae.txt

echo.
echo Step 2: Setting up Hugging Face authentication...
python src/setup_huggingface.py

echo.
echo Step 3: Ready to start training!
echo To start training, run:
echo   python src/train_videomae_on_feiyu.py
echo.
echo Or with custom parameters:
echo   python src/train_videomae_on_feiyu.py --epochs 30 --batch_size 2 --patience 8
echo.

pause
