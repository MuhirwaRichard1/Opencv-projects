
import cv2
import sys
import os

video_path = r"C:\Users\Robotic Muhirwa\Downloads\WhatsApp Video 2025-01-14 at 5.57.13 AM.mp4"
print(f"Path exists: {os.path.exists(video_path)}")

cap = cv2.VideoCapture(video_path)
print(f"Opened: {cap.isOpened()}")
print(f"Frame count: {cap.get(cv2.CAP_PROP_FRAME_COUNT)}")

ret, frame = cap.read()
print(f"Read success: {ret}")
if ret:
    print(f"Frame shape: {frame.shape}")

cap.release()
