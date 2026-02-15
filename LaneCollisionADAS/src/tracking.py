
import cv2
import numpy as np

class LongitudinalKalmanFilter:
    def __init__(self, initial_dist):
        # State: [Distance, RelativeVelocity]
        # x = [z, v]
        self.kalman = cv2.KalmanFilter(2, 1, 0) # 2 dynamic params, 1 measurement, 0 control
        
        # Transition matrix (F)
        # z_k = z_{k-1} + v_{k-1}*dt
        # v_k = v_{k-1}
        dt = 1/30.0 # Approx 30 FPS
        self.kalman.transitionMatrix = np.array([[1, dt],
                                                 [0, 1]], np.float32)
        
        # Measurement matrix (H) - we measure distance (z)
        self.kalman.measurementMatrix = np.array([[1, 0]], np.float32)
        
        # Process Noise Covariance (Q)
        self.kalman.processNoiseCov = np.array([[1e-2, 0],
                                                [0, 1e-1]], np.float32)
        
        # Measurement Noise Covariance (R)
        self.kalman.measurementNoiseCov = np.array([[1.0]], np.float32) # Noise in distance measurement
        
        # Error Covariance (P)
        self.kalman.errorCovPost = np.eye(2, dtype=np.float32)
        
        # Initial State
        self.kalman.statePost = np.array([[initial_dist], [0]], np.float32)

    def update(self, measurement_dist):
        """
        Predict and Correct.
        """
        # Predict
        self.kalman.predict()
        
        # Correct
        meas = np.array([[measurement_dist]], np.float32)
        self.kalman.correct(meas)
        
        return self.kalman.statePost

class Tracker:
    def __init__(self):
        # Dict of ID -> { 'kf': KalmanFilter, 'bbox': ..., 'missed': 0 }
        self.tracks = {}
        self.next_id = 0
        self.max_missed = 5
        self.dist_thresh = 50.0 # Pixels (for association)

    def update(self, detections, geom_engine):
        """
        detections: list of [x1, y1, x2, y2, score, cls]
        geom_engine: instance of GeometryEngine
        """
        updated_tracks_map = {} # detection_idx -> track_id
        
        # Simple centroid matching
        # (A production system would use SORT/DeepSORT)
        
        active_track_ids = list(self.tracks.keys())
        used_detections = set()
        
        # Convert measurement to centroids
        det_centroids = []
        for det in detections:
            cx = (det[0] + det[2]) / 2
            cy = det[3] # Use bottom center for tracking distance
            det_centroids.append((cx, cy))
            
        # Match
        measurement_matches = {} # track_id -> det_index
        
        for tr_id in active_track_ids:
            track = self.tracks[tr_id]
            last_bbox = track['bbox']
            tcx = (last_bbox[0] + last_bbox[2]) / 2
            tcy = last_bbox[3]
            
            best_dist = self.dist_thresh
            best_idx = -1
            
            for i, (dcx, dcy) in enumerate(det_centroids):
                if i in used_detections: continue
                dist = np.hypot(tcx - dcx, tcy - dcy)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i
            
            if best_idx != -1:
                measurement_matches[tr_id] = best_idx
                used_detections.add(best_idx)
        
        # Update Tracks
        current_risks = []
        
        for tr_id in list(self.tracks.keys()):
            if tr_id in measurement_matches:
                det_idx = measurement_matches[tr_id]
                det = detections[det_idx]
                bbox_bottom = det[3]
                
                # Estimate measurement distance
                measured_z = geom_engine.estimate_distance(bbox_bottom)
                
                # Kalman Update
                state = self.tracks[tr_id]['kf'].update(measured_z)
                est_dist = state[0][0]
                est_vel = state[1][0] # m/s (negative means closing in)
                
                # TTC Calculation
                ttc = 99.9
                if est_vel < -0.1: # Moving closer
                    ttc = est_dist / abs(est_vel)
                
                self.tracks[tr_id]['bbox'] = det[:4]
                self.tracks[tr_id]['dist'] = est_dist
                self.tracks[tr_id]['ttc'] = ttc
                self.tracks[tr_id]['missed'] = 0
                
            else:
                self.tracks[tr_id]['missed'] += 1
                if self.tracks[tr_id]['missed'] > self.max_missed:
                    del self.tracks[tr_id]
                    
        # Create new tracks
        for i, det in enumerate(detections):
            if i not in used_detections:
                bbox_bottom = det[3]
                measured_z = geom_engine.estimate_distance(bbox_bottom)
                
                new_track = {
                    'kf': LongitudinalKalmanFilter(measured_z),
                    'bbox': det[:4],
                    'dist': measured_z,
                    'ttc': 99.9,
                    'missed': 0,
                    'cls': det[5]
                }
                self.tracks[self.next_id] = new_track
                self.next_id += 1
                
        return self.tracks
