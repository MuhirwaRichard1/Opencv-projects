import cv2
import numpy as np

# Define the ArUco dictionary and parameters
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
aruco_params = cv2.aruco.DetectorParameters()

# Function to perform camera calibration
def calibrate_camera(images, chessboard_size, square_size):
    obj_points = []  # 3D points in real world space
    img_points = []  # 2D points in image plane

    # Prepare object points (0,0,0), (1,0,0), ..., (8,5,0)
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp *= square_size

    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

        if ret:
            obj_points.append(objp)
            img_points.append(corners)

    # Calibrate the camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)
    return ret, mtx, dist, rvecs, tvecs

# Function to detect ArUco markers
def detect_aruco_markers(frame, mtx, dist):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)

    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.05, mtx, dist)
        for i in range(len(ids)):
            cv2.drawFrameAxes(frame, mtx, dist, rvecs[i], tvecs[i], 0.1)
    return frame, ids, corners

# Main function
def main():
    # Load calibration images (chessboard images)
    calibration_images = [cv2.imread(f'calib_img_{i}.jpg') for i in range(1, 11)]  # Adjust the range as needed
    chessboard_size = (9, 6)  # Number of inner corners per a chessboard row and column
    square_size = 0.025  # Size of a square in meters

    # Calibrate the camera
    ret, mtx, dist, rvecs, tvecs = calibrate_camera(calibration_images, chessboard_size, square_size)
    if not ret:
        print("Camera calibration failed!")
        return

    # Start video capture
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect ArUco markers
        frame, ids, corners = detect_aruco_markers(frame, mtx, dist)

        # Display the frame
        cv2.imshow('ArUco Marker Detection', frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()