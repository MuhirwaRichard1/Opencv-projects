
import cv2
import numpy as np

class GeometryEngine:
    def __init__(self, focal_len, cam_height, optical_center):
        self.f = focal_len
        self.H = cam_height
        self.cx, self.cy = optical_center
        
    def estimate_distance(self, bbox_bottom_y):
        """
        Estimate distance to the object using flat ground assumption.
        Z = (f * H) / (y_img - y_horizon)
        """
        # Ensure we don't divide by zero or get negative distances (above horizon)
        if bbox_bottom_y <= self.cy:
            return 999.0 # Horizon or above
        
        pixel_offset = bbox_bottom_y - self.cy
        distance_meters = (self.f * self.H) / pixel_offset
        return distance_meters

    def get_bev_projection(self, frame):
        """
        Simple IPM transform for visualization.
        """
        h, w = frame.shape[:2]
        
        # Source points: trapezoid covering the lanes
        # Tuning these requires calibration, using heuristics here:
        # Bottom: full width, Top: near horizon, narrower
        src_points = np.float32([
            [w * 0.45, h * 0.55], # Top Left (near horizon)
            [w * 0.55, h * 0.55], # Top Right
            [w * 1.0,  h * 0.95], # Bottom Right
            [w * 0.0,  h * 0.95]  # Bottom Left
        ])
        
        # Dest points: Rectangle (BEV)
        # Map to a 400x400 top-down view
        bev_w, bev_h = 400, 400
        dst_points = np.float32([
            [0, 0],
            [bev_w, 0],
            [bev_w, bev_h],
            [0, bev_h]
        ])
        
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        bev_img = cv2.warpPerspective(frame, matrix, (bev_w, bev_h))
        
        return bev_img, matrix
