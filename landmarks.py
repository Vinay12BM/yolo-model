import cv2
import mediapipe as mp
import numpy as np

class LandmarkExtractor:
    def __init__(self, max_faces=5):
        """
        Initializes MediaPipe FaceLandmarker via the new Tasks API.
        Extracts 478 precise facial landmarks.
        """
        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode
        
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path='models/face_landmarker.task'),
            running_mode=VisionRunningMode.IMAGE,
            num_faces=max_faces,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False
        )
        self.landmarker = FaceLandmarker.create_from_options(options)

    def extract_landmarks(self, frame):
        """
        Extracts landmarks for all faces in the frame.
        Returns a list of dictionaries with 2D (pixels) and 3D (normalized) landmarks.
        """
        # MediaPipe expects RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        results = self.landmarker.detect(mp_image)
        
        extracted_faces = []
        h, w = frame.shape[:2]
        
        if results.face_landmarks:
            for face_landmarks in results.face_landmarks:
                landmarks_2d = []
                landmarks_3d = []
                
                # Track bounds for easy IoU matching with YOLO bounding boxes
                x_min, y_min = w, h
                x_max, y_max = 0, 0
                
                for lm in face_landmarks:
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
