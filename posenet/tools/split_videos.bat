@echo off
echo Splitting videos into 10-second clips...
cd /d "%~dp0"
python split_videos_to_clips.py --input_dir ../output/videos --output_dir ../output/video_clips --clip_duration 10 --flat_structure
echo.
echo Done! Check the ../output/video_clips folder for the results.
pause
