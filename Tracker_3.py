import cv2
import os

# Print the current working directory
print(f"Current working directory: {os.getcwd()}")

# Check if GOTURN model files exist
goturn_prototxt = "goturn.prototxt"
goturn_caffemodel = "goturn.caffemodel"

if not (os.path.isfile(goturn_prototxt) and os.path.isfile(goturn_caffemodel)):
    raise FileNotFoundError(
        f"GOTURN model files not found! Please download {goturn_prototxt} and "
        f"{goturn_caffemodel} and place them in the following directory:\n{os.getcwd()}"
    )

# Initialize the GOTURN tracker
tracker = cv2.TrackerGOTURN_create()

# Load the video
video_path = r"C:\Users\Robotic Muhirwa\Downloads\WhatsApp Video 2025-01-14 at 5.57.13 AM.mp4"  # video file path
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

# bbox = (x, y, width, height)
bbox = cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=True)
cv2.destroyWindow("Frame")

# Check if the user canceled the ROI selection
if bbox == (0, 0, 0, 0):
    print("ROI selection was canceled. Exiting the program.")
    cap.release()
    exit()

# Initialize the tracker with the first frame and bounding box
tracker.init(frame, bbox)

# Main tracking loop
while True:
    # Read a new frame
    ret, frame = cap.read()
    if not ret:
        print("End of video.")
        break
    
    # Read the actual size of the frame.
    height, width = frame.shape[:2]

    centre_frame_x = int(width / 2)
    centre_frame_y = int(height / 2)
    # Update the tracker
    success, bbox = tracker.update(frame)

    # If tracking is successful, draw the bounding box
    if success:
        p1 = (int(bbox[0]), int(bbox[1]))  # Top-left corner
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))  # Bottom-right corner
        cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)  # Draw the rectangle

        # Calculate the center point of the bounding box
        center_x = int(bbox[0] + bbox[2] / 2)
        center_y = int(bbox[1] + bbox[3] / 2)

        # Draw cross lines at the center of the bounding box
        cv2.line(frame, (center_x - 100, center_y), (center_x + 100, center_y), (0, 255, 0), 2)  # Horizontal line
        cv2.line(frame, (center_x, center_y - 100), (center_x, center_y + 100), (0, 255, 0), 2)  # Vertical line

        # Add dot at the center
        cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

        # draw circle to the center of the frame
        cv2.circle(frame, (centre_frame_x, centre_frame_y), 5, (0, 0, 255), -1)

        # draw a line between center frame and center target
        cv2.line(frame, (centre_frame_x, centre_frame_y), (center_x, center_y), (0, 0, 255), 2)

        # Display the center coordinates
        cv2.putText(frame, f"Center off: ({center_x - centre_frame_x}, {centre_frame_y - center_y})", (center_x + 15, center_y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 255), 1)


    else:
        # Display failure message
        cv2.putText(frame, "Tracking failure detected", (100, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow("GOTURN Tracker", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
