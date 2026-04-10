import cv2
import time
import numpy as np
from detection import FaceDetector
from landmarks import LandmarkExtractor
from behavior import BehaviorAnalyzer

def calculate_iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    Box format: (x1, y1, x2, y2)
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union = area1 + area2 - intersection
    if union == 0:
        return 0
    return intersection / union

def main():
    print("Initializing components...")
    
    # It might take a moment to load the models
    detector = FaceDetector(model_path="models/yolov8n-face.pt", conf_threshold=0.5)
    landmark_ext = LandmarkExtractor(max_faces=5)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
        
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab initial frame.")
        return
        
    # Initialize the behavioral intelligence engine with the camera's resolution
    analyzer = BehaviorAnalyzer(frame.shape)

    prev_frame_time = 0

    print("System starting. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Detect faces with YOLO (Tracking enabled)
        yolo_faces = detector.detect_and_track(frame)

        # 2. Extract Landmarks with MediaPipe
        mp_faces = landmark_ext.extract_landmarks(frame)

        # 3. Match YOLO to MediaPipe using IoU
        matched_faces = []
        
        # O(N*M) matching, fine since max faces is typically small (e.g. < 5)
        for m_face in mp_faces:
            best_iou = 0
            best_yolo = None
            for y_face in yolo_faces:
                iou = calculate_iou(m_face['bbox'], y_face['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_yolo = y_face
            
            # If a YOLO face matches with reasonable overlap
            if best_iou > 0.1 and best_yolo is not None:
                matched_faces.append((best_yolo, m_face))

        # 4. Behavioral Intelligence and Visualization
        for y_face, m_face in matched_faces:
            track_id = y_face['id']
            landmarks_2d = m_face['landmarks_2d']
            
            # Geometry Analysis
            pitch, yaw, roll = analyzer.get_head_pose(landmarks_2d)
            ear = analyzer.get_ear(landmarks_2d)
            focus_score, is_distracted = analyzer.calculate_focus_score(pitch, yaw, ear, track_id)

            # --- Drawing logic ---
            x1, y1, x2, y2 = y_face['bbox']
            
            # Dynamically color the bounding box and UI based on state
            color = (0, 0, 255) if is_distracted else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Tracking ID header
            cv2.putText(frame, f"ID {track_id}", (x1, max(20, y1 - 10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Mesh visualization (sub-sampled to maintain real-time performance)
            # Drawing all 478 points is too expensive
            for x, y in landmarks_2d[::7]:
                 cv2.circle(frame, (int(x), int(y)), 1, (200, 200, 200), -1)
            
            # Highlight key points used for behavioral metrics (Nose, Eye corners, Mouth corners)
            key_pts = [1, 152, 33, 263, 61, 291]
            for pt in key_pts:
                cv2.circle(frame, (int(landmarks_2d[pt][0]), int(landmarks_2d[pt][1])), 3, (0, 255, 255), -1)

            # Draw behavioral metrics UI near the face
            info_text = [
                f"P:{int(pitch)} Y:{int(yaw)}",
                f"EAR: {ear:.2f}",
                f"Focus: {int(focus_score)}",
                "ALERT: DISTRACTED" if is_distracted else "FOCUSED"
            ]
            
            y_offset = y2 + 20
            for text in info_text:
                if "ALERT" in text:
                    cv2.putText(frame, text, (x1, y_offset), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 255), 2)
                else:
                    cv2.putText(frame, text, (x1, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                y_offset += 20

        # Global Performance metric
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time) if prev_frame_time > 0 else 0
        prev_frame_time = new_frame_time
        
        # Display FPS in corner
        cv2.putText(frame, f"SYS FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow("Next-Gen Real-Time Facial Intelligence", frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
