import cv2
import numpy as np
import math

class BehaviorAnalyzer:
    def __init__(self, frame_shape):
        """
        Initializes the behavioral intelligence engine.
        Needs frame_shape (height, width) to approximate the focal length and camera matrix.
        """
        h, w = frame_shape[:2]
        self.focal_length = w
        self.camera_matrix = np.array([
            [self.focal_length, 0, w / 2],
            [0, self.focal_length, h / 2],
            [0, 0, 1]
        ], dtype=np.float64)
        
        self.dist_coeffs = np.zeros((4, 1))
        
        # Generic 3D facial model for solvePnP
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip 1
            (0.0, -330.0, -65.0),        # Chin 152
            (-225.0, 170.0, -135.0),     # Left eye left corner 33
            (225.0, 170.0, -135.0),      # Right eye right corner 263
            (-150.0, -150.0, -125.0),    # Left Mouth corner 61
            (150.0, -150.0, -125.0)      # Right mouth corner 291
        ])
        
        # Historical tracking for smoothing
        # Dictionary mapping tracking_id -> state
        self.tracking_history = {}

    def get_head_pose(self, landmarks_2d):
        """
        Calculates Head Pose (Yaw, Pitch, Roll) from 2D landmarks.
        """
        # Extract the specific 6 landmarks
        image_points = np.array([
            landmarks_2d[1],   # Nose tip
            landmarks_2d[152], # Chin
            landmarks_2d[33],  # Left eye left corner
            landmarks_2d[263], # Right eye right corner
            landmarks_2d[61],  # Left mouth corner
            landmarks_2d[291]  # Right mouth corner
        ], dtype=np.float64)

        success, rotation_vector, translation_vector = cv2.solvePnP(
            self.model_points, image_points, self.camera_matrix, self.dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not success:
            return 0, 0, 0

        # Convert rotation vector to rotation matrix
        rmat, _ = cv2.Rodrigues(rotation_vector)
        
        # Get Euler angles
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
        pitch = angles[0]
        yaw = angles[1]
        roll = angles[2]
        
        # Normalize/adjust based on the generic 3D model orientation
        if pitch > 90:
            pitch -= 180
        elif pitch < -90:
            pitch += 180
            
        return pitch, yaw, roll

    def get_ear(self, landmarks_2d):
        """
        Calculates Eye Aspect Ratio (EAR) to detect blinking or micro-sleep.
        """
        def ear_calc(eye):
            # Compute the euclidean distances between the two sets of vertical eye landmarks
            A = np.linalg.norm(eye[1] - eye[5])
            B = np.linalg.norm(eye[2] - eye[4])
            # Compute the euclidean distance between the horizontal eye landmark
            C = np.linalg.norm(eye[0] - eye[3])
            return (A + B) / (2.0 * C) if C != 0 else 0

        # MediaPipe Eye indices
        right_eye_indices = [33, 160, 158, 133, 153, 144]
        left_eye_indices = [362, 385, 387, 263, 373, 380]

        right_eye = landmarks_2d[right_eye_indices]
        left_eye = landmarks_2d[left_eye_indices]

        ear_right = ear_calc(right_eye)
        ear_left = ear_calc(left_eye)

        return (ear_right + ear_left) / 2.0

    def calculate_focus_score(self, pitch, yaw, ear, track_id):
        """
        Calculates a dynamic focus score (0-100) based on pose and eye state.
        Maintains temporal tracking for smoothing.
        """
        if track_id not in self.tracking_history:
            self.tracking_history[track_id] = {
                'score': 100.0,
                'frames_distracted': 0,
                'frames_closed_eyes': 0
            }
            
        history = self.tracking_history[track_id]
        
        # Define thresholds
        YAW_THRESH = 25  # degrees
        PITCH_THRESH = 20 # degrees
        EAR_THRESH = 0.21 # typical threshold for closed eyes
        
        is_looking_away = abs(yaw) > YAW_THRESH or abs(pitch) > PITCH_THRESH
        is_eyes_closed = ear < EAR_THRESH
        
        focus_penalty = 0
        
        if is_looking_away:
            history['frames_distracted'] += 1
            focus_penalty += 2.0 * history['frames_distracted'] # Exponential decay
        else:
            history['frames_distracted'] = max(0, history['frames_distracted'] - 2)
            
        if is_eyes_closed:
            history['frames_closed_eyes'] += 1
            if history['frames_closed_eyes'] > 5: # Micro-sleep detected
                focus_penalty += 5.0 * (history['frames_closed_eyes'] - 5)
        else:
            history['frames_closed_eyes'] = 0
            
        if not is_looking_away and not is_eyes_closed:
            # Regain focus slowly
            history['score'] = min(100.0, history['score'] + 1.5)
            
        # Apply penalty
        history['score'] = max(0.0, history['score'] - focus_penalty)
        
        distracted = history['score'] < 50
        
        return history['score'], distracted
