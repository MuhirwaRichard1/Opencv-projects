import cv2
import numpy as np

# Initialize the video capture
cap = cv2.VideoCapture(0)

# Initialize ArUco detector
# For DICT_4X4_50: markerSize=4, maxId=50
aruco_dict = cv2.aruco.Dictionary(
    cv2.aruco.DICT_4X4_50,  # Dictionary id
    50,                     # Number of markers
    4                       # Marker size (4x4)
)
aruco_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect ArUco markers
    corners, ids, rejected = detector.detectMarkers(gray)

    # If markers are detected
    if ids is not None and len(ids) > 0:
        # Draw the markers
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        # Draw bounding boxes
        for i in range(len(ids)):
            # Get the corners of the current marker
            marker_corners = corners[i][0]
            
            # Convert to integers
            marker_corners = marker_corners.astype(int)
            
            # Get the top-left and bottom-right corners for the bounding box
            top_left = np.min(marker_corners, axis=0)
            bottom_right = np.max(marker_corners, axis=0)
            
            # Draw the bounding box
            cv2.rectangle(
                frame,
                tuple(top_left),
                tuple(bottom_right),
                (0, 255, 0),  # Green color
                2
            )
            
            # Add marker ID text
            cv2.putText(
                frame,
                f"ID: {ids[i][0]}",
                tuple(top_left + [0, -10]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

    # Display the frame
    cv2.imshow('ArUco Marker Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()