import cv2
import time
from detection import AnimalDetector
from notification import Notifier

def main():
    print("Initializing components...")
    
    # --- CONFIGURATION ---
    # Put your Render URL here once deployed, e.g., "https://agri-guardian.onrender.com"
    RENDER_URL = None 
    
    # Initialize YOLOv8 default detector for animals
    detector = AnimalDetector(model_path="yolov8n.pt", conf_threshold=0.5)
    # Initialize notification engine with optional remote dashboard support
    notifier = Notifier(log_file="intrusion_log.txt", cooldown_seconds=10, remote_url=RENDER_URL)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
        
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab initial frame.")
        return
        
    prev_frame_time = 0

    print("System starting. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Detect animals with YOLO
        animals = detector.detect_and_track(frame)

        # 2. Draw bounds and trigger notifications
        for animal in animals:
            track_id = animal['id']
            class_name = animal['class_name']
            conf = animal['conf']
            x1, y1, x2, y2 = animal['bbox']
            
            # Draw bounding box (Red to convey alert/recording state)
            color = (0, 0, 255) 
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            
            # Info Label
            label = f"INTRUDER: {class_name.upper()} (ID: {track_id}) {conf*100:.1f}%"
            cv2.putText(frame, label, (x1, max(20, y1 - 10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # Trigger notification (handles cooldown state internally)
            notifier.trigger_alert(class_name, conf, bbox=(x1, y1, x2, y2))

        # 3. Global Performance metric
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time) if prev_frame_time > 0 else 0
        prev_frame_time = new_frame_time
        
        # Display FPS
        cv2.putText(frame, f"SYS FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Display Current System State
        cv2.putText(frame, "AGRICULTURAL SECURITY MODE", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Agriculture Security System", frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
