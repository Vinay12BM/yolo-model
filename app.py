import time
import json
from flask import Flask, Response, render_template, request, jsonify

app = Flask(__name__)

# Basic storage for the most recent intrusion
# In a production app, this would be a database or a redis queue
latest_intrusion = None

def event_stream():
    """Returns a server-sent event if a new intrusion is detected."""
    global latest_intrusion
    last_sent = None
    
    while True:
        if latest_intrusion and latest_intrusion != last_sent:
            yield f"data: {json.dumps(latest_intrusion)}\n\n"
            last_sent = latest_intrusion.copy()
        time.sleep(1)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/report', methods=['POST'])
def report():
    global latest_intrusion
    data = request.json
    if not data:
        return jsonify({"status": "error", "message": "No data"}), 400
    
    # Store the intrusion details
    latest_intrusion = {
        "class_name": data.get("class_name", "Unknown"),
        "confidence": data.get("confidence", 0),
        "timestamp": data.get("timestamp", ""),
        "id": time.time() # Unique ID to trigger the event stream
    }
    
    print(f"Intrusion Reported: {latest_intrusion['class_name']}")
    return jsonify({"status": "success"}), 200

@app.route('/stream')
def stream():
    return Response(event_stream(), mimetype="text/event-stream")

if __name__ == '__main__':
    # For local testing
    app.run(host='0.0.0.0', port=5000, debug=True)
