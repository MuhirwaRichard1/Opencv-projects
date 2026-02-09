import cv2
import time
import os



# Define the path to the GOTURN model files
GOTURN_MODEL_PATH = "c:/Users/Robotic Muhirwa/Downloads/Tracking_Opencv/goturn-files-master/"
PROTOTXT_PATH = os.path.join(GOTURN_MODEL_PATH, "goturn.prototxt")
CAFFEMODEL_PATH = os.path.join(GOTURN_MODEL_PATH, "goturn.caffemodel")


# Initialize the GOTURN tracker
tracker = cv2.TrackerGOTURN_create()
#tracker = cv2.TrackerGOTURN_create(PROTOTXT_PATH, CAFFEMODEL_PATH)

# Load the video file
video_path = r"C:\Users\Robotic Muhirwa\Downloads\WhatsApp Video 2025-01-14 at 5.57.13 AM.mp4"  # video file
cap = cv2.VideoCapture(video_path)

# Check if the video was opened correctly
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Read the first frame
ret, frame = cap.read()
if not ret:
    print("Error: Cannot read video file.")
    cap.release()
    exit()

# Select the Region of Interest (ROI) for tracking
bbox = cv2.selectROI("Select Object", frame, fromCenter=False, showCrosshair=True)
cv2.destroyAllWindows()

# Initialize the tracker with the selected bounding box
tracker.init(frame, bbox)

# Start tracking
while True:
    ret, frame = cap.read()
    if not ret:
        break  # End of video

    height, width = frame.shape[:2]

    print("Frame width:", width)
    print("Frame Height:", height)
    # Update the tracker
    success, bbox = tracker.update(frame)

    if success:
        # Draw the tracking box
        x, y, w, h = [int(v) for v in bbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2, 1)
    else:
        cv2.putText(frame, "Tracking Lost", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow("GOTURN Object Tracking", frame)

    # Press 'q' to exit
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
