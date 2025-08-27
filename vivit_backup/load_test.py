import cv2

video_path = "./UCF101/UCF-101/Biking/v_Biking_g02_c06.avi"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Unable to open video.")

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Total Frames: {frame_count}")

for i in range(frame_count):  # Read first 10 frames
    ret, frame = cap.read()
    if not ret:
        print(f"Failed to read frame {i}")
    else:
        print(f"Frame {i} read successfully.")

cap.release()
