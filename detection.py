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
        Optimized for CPU-based cloud environments (no tracking).
        """
        try:
            self.model = YOLO(model_path)
            self.conf_threshold = conf_threshold
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            raise

    def detect_and_track(self, frame):
        """
        Detects animals in a frame using YOLO Predict.
        Note: We switched from .track() to .predict() to save memory/CPU on Render.
        """
        target_classes = list(ANIMAL_CLASSES.keys())
        
        # We use .predict instead of .track for lower resource consumption
        # We also use persist=False to stop tracking ID management
        results = self.model.predict(
            frame, 
            conf=self.conf_threshold, 
            classes=target_classes, 
            verbose=False,
            device='cpu' # Force CPU to avoid GPU-related RAM overhead
        )
        
        animals = []
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.int().cpu().numpy()
                
                for box, conf, cls_id in zip(boxes, confs, class_ids):
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
                        'id': -1, # Tracking disabled for memory efficiency
                        'class_id': int(cls_id),
                        'class_name': ANIMAL_CLASSES.get(int(cls_id), "Unknown Animal")
                    })
                    
        return animals
