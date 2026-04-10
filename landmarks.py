import cv2
import mediapipe as mp
import numpy as np

class LandmarkExtractor:
    def __init__(self, max_faces=5):
        """
        Initializes MediaPipe FaceMesh to extract 478 precise facial landmarks.
        `refine_landmarks=True` ensures we get detailed eye center and iris tracking.
        """
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=max_faces,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def extract_landmarks(self, frame):
        """
        Extracts landmarks for all faces in the frame.
        Returns a list of dictionaries with 2D (pixels) and 3D (normalized) landmarks.
        """
        # MediaPipe expects RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Performance optimization: pass by reference
        rgb_frame.flags.writeable = False
        results = self.face_mesh.process(rgb_frame)
        rgb_frame.flags.writeable = True
        
        extracted_faces = []
        h, w = frame.shape[:2]
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks_2d = []
                landmarks_3d = []
                
                # Track bounds for easy IoU matching with YOLO bounding boxes
                x_min, y_min = w, h
                x_max, y_max = 0, 0
                
                for lm in face_landmarks.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    landmarks_2d.append((x, y))
                    landmarks_3d.append((lm.x, lm.y, lm.z))
                    
                    x_min = min(x_min, x)
                    y_min = min(y_min, y)
                    x_max = max(x_max, x)
                    y_max = max(y_max, y)
                
                extracted_faces.append({
                    'landmarks_2d': np.array(landmarks_2d),
                    'landmarks_3d': np.array(landmarks_3d),
                    'bbox': (max(0, x_min), max(0, y_min), min(w, x_max), min(h, y_max))
                })
                
        return extracted_faces
