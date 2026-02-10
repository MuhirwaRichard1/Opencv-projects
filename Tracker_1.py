import cv2
import os


def draw_corner_brackets(img, pt1, pt2, color, thickness=2, length=18):
    x1, y1 = pt1
    x2, y2 = pt2
    # Top-left
    cv2.line(img, (x1, y1), (x1 + length, y1), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x1, y1), (x1, y1 + length), color, thickness, cv2.LINE_AA)
    # Top-right
    cv2.line(img, (x2, y1), (x2 - length, y1), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x2, y1), (x2, y1 + length), color, thickness, cv2.LINE_AA)
    # Bottom-left
    cv2.line(img, (x1, y2), (x1 + length, y2), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x1, y2), (x1, y2 - length), color, thickness, cv2.LINE_AA)
    # Bottom-right
    cv2.line(img, (x2, y2), (x2 - length, y2), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x2, y2), (x2, y2 - length), color, thickness, cv2.LINE_AA)


def draw_dashed_rect(img, pt1, pt2, color, thickness=2, dash=12, gap=6):
    x1, y1 = pt1
    x2, y2 = pt2
    # Horizontal edges
    for x in range(x1, x2, dash + gap):
        cv2.line(img, (x, y1), (min(x + dash, x2), y1), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x, y2), (min(x + dash, x2), y2), color, thickness, cv2.LINE_AA)
    # Vertical edges
    for y in range(y1, y2, dash + gap):
        cv2.line(img, (x1, y), (x1, min(y + dash, y2)), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x2, y), (x2, min(y + dash, y2)), color, thickness, cv2.LINE_AA)


def draw_crosshair(img, cx, cy, color=(0, 200, 255), thickness=1):
    h, w = img.shape[:2]
    cv2.line(img, (0, cy), (w, cy), color, thickness, cv2.LINE_AA)
    cv2.line(img, (cx, 0), (cx, h), color, thickness, cv2.LINE_AA)


def draw_center_tick(img, cx, cy, color=(255, 255, 255), length=8, thickness=2):
    cv2.line(img, (cx - length // 2, cy), (cx + length // 2, cy), color, thickness, cv2.LINE_AA)


# Create tracker - choose one of these alternatives:
# tracker = cv2.TrackerCSRT_create()  # CSRT not available in this build
tracker = cv2.TrackerMIL_create()     # Fallback to MIL

# Load the video
video_path = r"C:\Users\Robotic Muhirwa\Downloads\WhatsApp Video 2025-01-14 at 5.57.13 AM.mp4"
cap = cv2.VideoCapture(video_path)

# Check if the video was opened successfully
if not cap.isOpened():
    print(f"Error: Could not open video file {video_path}")
    exit()

# Read the first frame
ret, frame = cap.read()
if not ret:
    print("Error: Could not read the first frame from the video.")
    cap.release()
    exit()

# Allow the user to select the bounding box (ROI) on the first frame
print("Select the target to track, then press ENTER or SPACE.")
bbox = cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=True)
cv2.destroyWindow("Frame")

# Check if the user canceled the ROI selection
if bbox == (0, 0, 0, 0):
    print("ROI selection was canceled. Exiting the program.")
    cap.release()
    exit()

# Initialize the tracker
tracker.init(frame, bbox)

# Main tracking loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video.")
        break

    # Update the tracker
    success, bbox = tracker.update(frame)

    # If tracking is successful, draw the overlays
    if success:
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))

        # Overlay styles: dashed perimeter + corner brackets
        draw_dashed_rect(frame, p1, p2, (0, 255, 255), thickness=2, dash=12, gap=6)
        draw_corner_brackets(frame, p1, p2, (0, 255, 0), thickness=2, length=18)

        # Crosshair from box center across the frame
        center_x = int(bbox[0] + bbox[2] / 2)
        center_y = int(bbox[1] + bbox[3] / 2)
        draw_crosshair(frame, center_x, center_y, color=(0, 200, 255), thickness=1)
        draw_center_tick(frame, center_x, center_y, color=(255, 255, 255), length=10, thickness=2)

    else:
        cv2.putText(frame, "Tracking failure detected", (100, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow("Tracker", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()