from ultralytics import YOLO
import cv2

# "yolo26" is likely a typo for YOLOv8 (current standard) or similar. 
# Using YOLOv8 Medium model (yolov8m.pt) as requested ("medium model").

def run_yolo_inference():
    # Load the YOLOv8 Medium model
    # It will download 'yolov8m.pt' automatically if not present
    model = YOLO('yolov8m.pt')

    # Path to video file (taken from Tracker_2.py context)
    video_path = r"C:\Users\Robotic Muhirwa\Downloads\WhatsApp Video 2025-01-14 at 5.57.13 AM.mp4"
    
    # Open the video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    print("Starting YOLOv8 Medium Inference... Press 'q' to exit.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Run YOLOv8 inference on the frame
        # conf=0.5 sets detection threshold
        results = model(frame, conf=0.5)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLOv8 Medium Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run_yolo_inference()
