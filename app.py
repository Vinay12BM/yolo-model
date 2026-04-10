import time
import json
import base64
import numpy as np
import cv2
from flask import Flask, Response, render_template, request, jsonify
from detection import AnimalDetector

app = Flask(__name__)

# Initialize the detector on the server
detector = AnimalDetector(model_path="yolov8n.pt", conf_threshold=0.5)

# Global storage for the latest alert
# We use a timestamp to allow the local client to detect "new" alerts
latest_alert = {
    "detected": False,
    "class_name": "",
    "timestamp": 0
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    global latest_alert
    try:
        data = request.json
        if not data or 'image' not in data:
            return jsonify({"status": "error", "message": "No image"}), 400

        # Decode base64 image
        img_data = base64.b64decode(data['image'].split(',')[1])
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"status": "error", "message": "Invalid image"}), 400

        # Run Detection
        animals = detector.detect_and_track(frame)
        
        # Explicitly clear image from memory
        del frame
        
        found = False
        class_name = ""
        
        if animals:
            # We found something! Update the global alert
            animal = animals[0]
            latest_alert = {
                "detected": True,
                "class_name": animal['class_name'],
                "timestamp": time.time()
            }
            print(f"Cloud Detected: {animal['class_name']}")
            found = True
            class_name = animal['class_name']
        
        # Immediate cleanup for Render memory limits
        import gc
        gc.collect()
        
        return jsonify({"status": "success", "found": found, "class": class_name})

    except Exception as e:
        print(f"Processing Error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/poll', methods=['GET'])
def poll():
    """Endpoint for the local main.py to check for alerts."""
    return jsonify(latest_alert)

@app.route('/clear', methods=['POST'])
def clear():
    """Allow the local client to clear the alert state after beeping."""
    global latest_alert
    latest_alert["detected"] = False
    return jsonify({"status": "cleared"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
