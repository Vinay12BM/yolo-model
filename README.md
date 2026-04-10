# Next-Gen Real-Time Facial Intelligence System

A high-performance real-time computer vision system that interprets human attention, behavior, and facial structure using YOLOv8 and MediaPipe.

## 🌟 Core Features
- **YOLOv8 Multi-Face Tracking:** Robust, occlusion-resistant face detection and persistence tracking.
- **MediaPipe Facial Mesh:** High-precision 478-point 3D facial landmark extraction.
- **Dynamic Geometric Engine:**
  - **Head Pose Estimation:** 6D pose (Pitch, Yaw, Roll) using `cv2.solvePnP` on key canonical 3D facial features.
  - **Focus & Attention Scoring:** A temporal intelligent scoring system (0-100) that models focus based on head alignment and eye aspect ratio (EAR).
  - **Distraction & Micro-sleep Detection:** Real-time visual alerts for eyes-closed events and head aversion.

## 🛠️ Architecture

1. `detection.py`: Instantiates the YOLO face tracking object. Maintains bounding boxes and cross-frame IDs.
2. `landmarks.py`: Instantiates the MediaPipe FaceMesh engine. Generates sub-millimeter precision facial nodes.
3. `behavior.py`: The mathematics layer. Computes the geometric intersection of gaze boundaries and blink threshold ranges. Tracks individual users over time using an EMA (Exponential Moving Average) penalty decay model.
4. `main.py`: The orchestrator bounding rendering loops, webcam ingress, and YOLO/MediaPipe IoU crossover logic.

## 🚀 Setup Instructions

1. Ensure you have Python 3.9+ installed and a webcam connected.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the model downloader to fetch the open-source YOLOv8 face weights:
   ```bash
   python download_models.py
   ```
4. Start the system:
   ```bash
   python main.py
   ```

## 🧠 Behavioral Math Intuition
The Focus Score logic rests on a penalty-recovery state machine anchored per Tracking ID:
- **Spatial Bounds:** If $|\text{Yaw}| > 25^\circ$ or $|\text{Pitch}| > 20^\circ$, a rapid penalty is induced simulating distraction.
- **Eye Aspect Ratio (EAR):** Using the equation `EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)`, an EAR $< 0.21$ over prolonged frames flags a micro-sleep or heavy blinking event, exponentially decaying the focus score.
- **Symmetry & Recovery:** When bounds return to normal, the temporal decay lifts, simulating a user "refocusing" on their screen.

## 🚫 Constraints
- GPU is recommended for optimal performance but the code is heavily vectorized and uses the lightweight YOLO `nano` architecture to allow 20+ FPS on modern CPUs.
- If no webcam is found, verify your device index in `cv2.VideoCapture(0)`.
