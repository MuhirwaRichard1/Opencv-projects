from ultralytics import YOLO
import cv2


def run_video_demo():
	"""Run YOLO26 medium on a video and display frames live."""

	# Load the medium model (detection). Use the seg variant if you need segmentation masks.
	model = YOLO("yolo26s-seg.pt")

	# Path to your video file
	video_path = r"C:\Users\Robotic Muhirwa\Downloads\WhatsApp Video 2025-01-14 at 5.57.13 AM.mp4"

	# Stream inference so we can display frames as they are processed
	for result in model(source=video_path, stream=True, conf=0.4):
		frame = result.plot()  # draw boxes/masks on the frame
		cv2.imshow("YOLO26s-seg video", frame)

		# Exit on 'q'
		if cv2.waitKey(1) & 0xFF == ord("q"):
			break

	cv2.destroyAllWindows()


if __name__ == "__main__":
	run_video_demo()