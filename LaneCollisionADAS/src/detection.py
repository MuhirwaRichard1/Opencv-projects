
import cv2
import numpy as np
from ultralytics import YOLO

class ObjectDetector:
    def __init__(self, model_path, conf_thresh=0.5):
        print(f"Loading YOLOv8 model from {model_path}...")
        self.model = YOLO(model_path)
        self.conf_thresh = conf_thresh
        self.classes_of_interest = [2, 3, 5, 7] # car, motorcycle, bus, truck (COCO indices)

    def detect(self, frame):
        """
        Returns list of detections: [x1, y1, x2, y2, score, class_id]
        """
        results = self.model(frame, conf=self.conf_thresh, verbose=False)
        detections = []
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                if cls_id in self.classes_of_interest:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    score = float(box.conf[0])
                    detections.append([x1, y1, x2, y2, score, cls_id])
        
        return detections
