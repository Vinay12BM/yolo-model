import cv2
from ultralytics import YOLO

# COCO dataset class names for animals
ANIMAL_CLASSES = {
    14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep',
    19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe'
}

class AnimalDetector:
    def __init__(self, model_path="yolov8n.pt", conf_threshold=0.4):
        """
        Initializes the YOLOv8 object detector for animals.
        Note: yolov8n.pt will automatically download if not present in the current dir.
        """
        try:
            self.model = YOLO(model_path)
            self.conf_threshold = conf_threshold
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            raise

    def detect_and_track(self, frame):
        """
        Detects and tracks animals in a frame using YOLO.
        Returns a list of dictionaries containing animal info:
        [{'bbox': (x1, y1, x2, y2), 'conf': float, 'id': int, 'class_name': str, 'class_id': int}, ...]
        """
        # We use model.track to maintain consistent IDs across frames for tracking
        # We pass classes list to YOLO to only detect those classes
        target_classes = list(ANIMAL_CLASSES.keys())
        results = self.model.track(frame, persist=True, conf=self.conf_threshold, classes=target_classes, verbose=False)
        
        animals = []
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.int().cpu().numpy()
                
                # Extract tracking IDs if tracking is successful
                if result.boxes.id is not None:
                    track_ids = result.boxes.id.int().cpu().numpy()
                else:
                    track_ids = [-1] * len(boxes)
                
                for box, conf, track_id, cls_id in zip(boxes, confs, track_ids, class_ids):
                    x1, y1, x2, y2 = map(int, box)
                    
                    # Basic bound checking
                    h, w = frame.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    
                    if x2 <= x1 or y2 <= y1:
                        continue
                        
                    animals.append({
                        'bbox': (x1, y1, x2, y2),
                        'conf': float(conf),
                        'id': int(track_id),
                        'class_id': int(cls_id),
                        'class_name': ANIMAL_CLASSES.get(int(cls_id), "Unknown Animal")
                    })
                    
        return animals
