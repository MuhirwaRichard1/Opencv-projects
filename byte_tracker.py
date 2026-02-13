"""
Drone Object Tracking with ByteTrack + YOLOv8
===============================================
Click-to-Track: Video plays immediately with all detections visible.
Click on any detected object to lock onto it and start tracking.

Install dependencies:
    pip install ultralytics opencv-python numpy

Controls:
    - LEFT CLICK on a detected box  → Lock onto that target
    - Press 'c'                     → Clear / unlock current target
    - Press 'q'                     → Quit
"""

import cv2
import numpy as np
from ultralytics import YOLO


# ─── Configuration ────────────────────────────────────────────────────────────
VIDEO_PATH = r"C:\Users\Robotic Muhirwa\Downloads\WhatsApp Video 2025-01-14 at 5.57.13 AM.mp4"
MODEL_PATH = "yolo26m.pt"          # YOLOv8 nano (fast); use yolov8s.pt for better accuracy
CONFIDENCE_THRESHOLD = 0.3
IOU_THRESHOLD = 0.5
TRACKER_CONFIG = "bytetrack.yaml"

# Classes to detect (COCO). Set to None for all classes.
# 0=person, 1=bicycle, 2=car, 3=motorcycle, 5=bus, 7=truck
TARGET_CLASSES = None

# ─── Globals for mouse callback ───────────────────────────────────────────────
click_point = None  # Stores the latest mouse click coordinates


def mouse_callback(event, x, y, flags, param):
    """Capture left mouse clicks."""
    global click_point
    if event == cv2.EVENT_LBUTTONDOWN:
        click_point = (x, y)


def point_in_box(point, box):
    """Check if a point (x, y) is inside a box [x1, y1, x2, y2]."""
    return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]


def get_box_center(box):
    """Get center point of a bounding box [x1, y1, x2, y2]."""
    return (int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2))


def parse_tracks(results):
    """Parse tracking results into a list of dicts."""
    detections = []
    if results[0].boxes is None or results[0].boxes.id is None:
        return detections
    # Each detection dict: {"id": track_id, "box": [x1, y1, x2, y2], "conf": confidence, "cls": class_id}
    boxes = results[0].boxes
    for i in range(len(boxes)):
        detections.append({
            "id": int(boxes.id[i].item()),
            "box": boxes.xyxy[i].cpu().numpy().tolist(),
            "conf": float(boxes.conf[i].item()),
            "cls": int(boxes.cls[i].item()),
        })
    return detections


def find_clicked_detection(point, detections):
    """Find which detection the user clicked on. Returns track ID or None."""
    for det in detections:
        if point_in_box(point, det["box"]):
            return det["id"]
    return None


def draw_detection(frame, det, is_locked=False):
    """Draw a single detection box."""
    x1, y1, x2, y2 = [int(v) for v in det["box"]]

    if is_locked:
        # Locked target: bold blue box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        label = f"TARGET ID:{det['id']} ({det['conf']:.2f})"
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    else:
        # Unlocked: green box, clickable
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
        label = f"ID:{det['id']} ({det['conf']:.2f})"
        cv2.putText(frame, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)


def draw_tracking_overlay(frame, box, frame_center):
    """Draw crosshair, center line, and offset for the locked target."""
    center_x, center_y = get_box_center(box)
    cx_frame, cy_frame = frame_center

    # Crosshair on target
    cv2.line(frame, (center_x - 100, center_y), (center_x + 100, center_y), (0, 255, 0), 2)
    cv2.line(frame, (center_x, center_y - 100), (center_x, center_y + 100), (0, 255, 0), 2)
    cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

    # Frame center
    cv2.circle(frame, (cx_frame, cy_frame), 5, (0, 0, 255), -1)

    # Line from frame center to target
    cv2.line(frame, (cx_frame, cy_frame), (center_x, center_y), (0, 0, 255), 2)

    # Offset text
    offset_x = center_x - cx_frame
    offset_y = cy_frame - center_y
    cv2.putText(frame, f"Offset: ({offset_x}, {offset_y})",
                (center_x + 15, center_y + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 255), 1)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    global click_point

    # Load model
    print(f"Loading YOLOv8 model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    # Open video
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_center = (width // 2, height // 2)

    # Setup window with mouse callback
    window_name = "ByteTrack + YOLOv8 Tracker"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    # Tracking state
    locked_track_id = None
    frames_lost = 0
    max_lost_frames = 30

    print("\nVideo playing. Click on any detected object to track it.")
    print("Controls: CLICK = select target | 'c' = clear target | 'q' = quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video.")
            break

        # Run YOLOv8 + ByteTrack
        results = model.track(
            frame,
            persist=True,
            tracker=TRACKER_CONFIG,
            conf=CONFIDENCE_THRESHOLD,
            iou=IOU_THRESHOLD,
            classes=TARGET_CLASSES,
            verbose=False,
        )

        detections = parse_tracks(results)

        # ── Handle mouse click: lock onto clicked detection ──
        if click_point is not None:
            clicked_id = find_clicked_detection(click_point, detections)
            if clicked_id is not None:
                locked_track_id = clicked_id
                frames_lost = 0
                print(f"Locked onto Track ID: {locked_track_id}")
            click_point = None  # Consume the click

        # ── Find locked target in current detections ──
        target_det = None
        if locked_track_id is not None:
            for det in detections:
                if det["id"] == locked_track_id:
                    target_det = det
                    break

        # ── Draw all detections ──
        for det in detections:
            is_target = (det["id"] == locked_track_id)
            draw_detection(frame, det, is_locked=is_target)

        # ── Draw tracking overlay if target is found ──
        if target_det is not None:
            frames_lost = 0
            draw_tracking_overlay(frame, target_det["box"], frame_center)
            cv2.putText(frame, "TRACKING", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        elif locked_track_id is not None:
            # Target was locked but not found this exact frame
            frames_lost += 1
            if frames_lost < max_lost_frames:
                cv2.putText(frame, f"Target occluded ({frames_lost}/{max_lost_frames})",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 165, 255), 2)
            else:
                cv2.putText(frame, "TARGET LOST - Click a new object",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                locked_track_id = None

        else:
            # No target locked
            cv2.putText(frame, "Click on an object to track it",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (200, 200, 200), 2)

        # Info bar
        cv2.putText(frame, f"Detections: {len(detections)}", (10, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Display
        cv2.imshow(window_name, frame)

        # Key handling
        key = cv2.waitKey(1) & 0xFF

        # Quit or clear target
        if key == ord("q"):
            break
        elif key == ord("c"):
            locked_track_id = None
            frames_lost = 0
            print("Target cleared. Click a new object to track.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()