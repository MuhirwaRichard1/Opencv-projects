
import os
import sys
import cv2
import numpy as np
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import config
from detection import ObjectDetector
from segmentation import LaneSegmenter
from geometry import GeometryEngine
from tracking import Tracker

def main():
    # 1. Initialize Modules
    print("Initializing ADAS System...")
    
    # Check if video exists
    if not os.path.exists(config.VIDEO_SOURCE):
        print(f"Error: Video not found at {config.VIDEO_SOURCE}")
        return

    cap = cv2.VideoCapture(config.VIDEO_SOURCE)
    
    if not cap.isOpened():
        print(f"FAILED TO OPEN VIDEO CAPTURE: {config.VIDEO_SOURCE}")
        return

    print(f"Video opened successfully. Source: {config.VIDEO_SOURCE}")
    print(f"FPS: {cap.get(cv2.CAP_PROP_FPS)}, Frames: {cap.get(cv2.CAP_PROP_FRAME_COUNT)}")
    
    # Test read immediately
    ret_test, frame_test = cap.read()
    print(f"Immediate read test: {ret_test}")
    # Reset
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Models
    # Finding model relative to this script or configured path
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", config.DET_MODEL_PATH))
    if not os.path.exists(model_path):
        # Fallback to just filename if in current dir or download needed
        model_path = config.DET_MODEL_PATH
        
    detector = ObjectDetector(model_path)
    segmenter = LaneSegmenter(config.SEG_MODEL_ID)
    
    # Geometry & Tracker
    geom_engine = GeometryEngine(
        focal_len=config.FOCAL_LENGTH_X, 
        cam_height=config.CAMERA_HEIGHT_METERS,
        optical_center=(config.OPTICAL_CENTER_X, config.OPTICAL_CENTER_Y)
    )
    tracker = Tracker()
    
    # Loop
    while cap.isOpened():
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            print("Video ended.")
            break
            
        # Resize for consistent processing
        frame = cv2.resize(frame, (config.TARGET_WIDTH, config.TARGET_HEIGHT))
        display_frame = frame.copy()
        
        # 1. Lane Segmentation
        # We run this every few frames or on downscaled image for speed if needed
        # For demo, run every frame
        road_mask = segmenter.segment_road(frame)
        
        # Overlay Lane (Green)
        # Create a boolean mask from road_mask
        mask_bool = road_mask > 0
        
        # Create a green overlay
        green_overlay = np.zeros_like(display_frame)
        green_overlay[mask_bool] = [0, 255, 0]
        
        # Blend
        cv2.addWeighted(green_overlay, 0.3, display_frame, 1.0, 0, display_frame)
        
        # 2. Object Detection
        detections = detector.detect(frame)
        
        # 3. Tracking & Estimation
        tracks = tracker.update(detections, geom_engine)
        
        # 4. Visualization & Risk
        for tr_id, track in tracks.items():
            x1, y1, x2, y2 = map(int, track['bbox'])
            dist = track['dist']
            ttc = track['ttc']
            
            # Risk Color
            color = (0, 255, 255) # Yellow default
            warning = ""
            
            if dist < config.RISK_DISTANCE_THRESH_METERS:
                color = (0, 165, 255) # Orange
            
            if ttc < config.RISK_TTC_THRESH_SECONDS:
                color = (0, 0, 255) # Red
                warning = "COLLISION WARNING!"
                
            # Draw Box
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw Info
            label = f"ID: {tr_id}"
            info = f"Dist: {dist:.1f}m TTC: {ttc:.1f}s"
            
            cv2.putText(display_frame, label, (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.putText(display_frame, info, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            if warning:
                cv2.putText(display_frame, warning, (x1, y1 - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
        # 5. BEV Visualization (Picture-in-Picture)
        bev_img, _ = geom_engine.get_bev_projection(frame)
        # Resize BEV to small box
        bev_small = cv2.resize(bev_img, (320, 240))
        # Overlay on Top Right
        h_bev, w_bev = bev_small.shape[:2]
        display_frame[20:20+h_bev, config.TARGET_WIDTH-20-w_bev:config.TARGET_WIDTH-20] = bev_small
        cv2.rectangle(display_frame, (config.TARGET_WIDTH-20-w_bev, 20), (config.TARGET_WIDTH-20, 20+h_bev), (255, 255, 255), 2)
        cv2.putText(display_frame, "BEV", (config.TARGET_WIDTH-w_bev-15, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # FPS
        fps = 1.0 / (time.time() - start_time)
        cv2.putText(display_frame, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("LaneCollisionADAS", display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
