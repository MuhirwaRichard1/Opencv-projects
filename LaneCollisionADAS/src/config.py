
import cv2
import numpy as np

# Video Config
# Update this path to your actual video
VIDEO_SOURCE = r"C:\Users\Robotic Muhirwa\Downloads\WhatsApp Video 2025-01-14 at 5.57.13 AM.mp4"

# Display settings
TARGET_WIDTH = 1280
TARGET_HEIGHT = 720

# Camera Calibration (Estimated for generic dashcam)
# F_pixel = (ImageWidth / 2) / tan(FOV / 2)
# Assuming ~90 deg FOV for dashcam, tan(45) = 1 => F_pixel = 640
FOCAL_LENGTH_X = 1100.0
FOCAL_LENGTH_Y = 1100.0
OPTICAL_CENTER_X = TARGET_WIDTH / 2.0
OPTICAL_CENTER_Y = TARGET_HEIGHT / 2.0 + 50.0  # Slightly below center for horizon
CAMERA_HEIGHT_METERS = 1.6  # Height of camera from road

# Lane Segmentation Settings
SEG_MODEL_ID = "nvidia/segformer-b0-finetuned-cityscapes-512-1024"
# Cityscapes classes: 0=road, 1=sidewalk, ... (road is usually index 0 or 1 depending on mapping)
# For this model: 0: road, 1: sidewalk, ...

# Object Detection
DET_MODEL_PATH = "yolov8m.pt"  # Use the one in your root folder
CONF_THRESHOLD = 0.5

# Tracking & TTC
FPS = 30.0
LANE_WIDTH_METERS = 3.7
RISK_DISTANCE_THRESH_METERS = 15.0
RISK_TTC_THRESH_SECONDS = 2.5
