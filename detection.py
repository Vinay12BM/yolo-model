import cv2
from ultralytics import YOLO

class FaceDetector:
    def __init__(self, model_path="models/yolov8n-face.pt", conf_threshold=0.5):
        """
        Initializes the YOLOv8 face detector.
        """
        try:
            self.model = YOLO(model_path)
            self.conf_threshold = conf_threshold
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            print("Please run download_models.py first to fetch the model weights.")
            raise

    def detect_and_track(self, frame):
        """
        Detects and tracks faces in a frame using YOLO.
        Returns a list of dictionaries containing face info:
        [{'bbox': (x1, y1, x2, y2), 'conf': float, 'id': int}, ...]
        """
        # We use model.track to maintain consistent IDs across frames for behavioral tracking
        results = self.model.track(frame, persist=True, conf=self.conf_threshold, verbose=False)
        
        faces = []
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                
                # Extract tracking IDs if tracking is successful
                if result.boxes.id is not None:
                    track_ids = result.boxes.id.int().cpu().numpy()
                else:
                    track_ids = [-1] * len(boxes)
                
                for box, conf, track_id in zip(boxes, confs, track_ids):
                    x1, y1, x2, y2 = map(int, box)
                    
                    # Basic bound checking
                    h, w = frame.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    
                    if x2 <= x1 or y2 <= y1:
                        continue
                        
                    faces.append({
                        'bbox': (x1, y1, x2, y2),
                        'conf': float(conf),
                        'id': int(track_id)
                    })
                    
        return faces
