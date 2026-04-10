import os
import time
import datetime
import requests
import platform

# Only import winsound on Windows
if platform.system() == "Windows":
    try:
        import winsound
    except ImportError:
        winsound = None
else:
    winsound = None

class Notifier:
    def __init__(self, log_file="intrusion_log.txt", cooldown_seconds=10, remote_url=None):
        self.log_file = log_file
        self.cooldown_seconds = cooldown_seconds
        self.remote_url = remote_url
        
        # Track last notification time per class to avoid spam
        self.last_notified = {}

    def trigger_alert(self, class_name, confidence, bbox=None):
        current_time = time.time()
        
        # Check cooldown
        if class_name in self.last_notified:
            if current_time - self.last_notified[class_name] < self.cooldown_seconds:
                return False 

        self.last_notified[class_name] = current_time
        
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 1. Log locally
        log_msg = f"[{timestamp}] ALARM: {class_name.upper()} detected with {confidence*100:.1f}% confidence\n"
        with open(self.log_file, "a") as f:
            f.write(log_msg)
            
        print(f"\n!!! INTRUDER ALERT: {class_name.upper()} !!!")
        
        # 2. Local sound alert (Windows only)
        if winsound:
            try:
                for _ in range(3):
                    winsound.Beep(2500, 500)
                    time.sleep(0.1)
            except Exception as e:
                print(f"Local alert sound failed: {e}")
        else:
            print("Local buzzer unavailable (Non-Windows environment).")

        # 3. Remote notification to Render
        if self.remote_url:
            try:
                payload = {
                    "class_name": class_name,
                    "confidence": float(confidence),
                    "timestamp": timestamp,
                    "bbox": bbox
                }
                # Use a short timeout to prevent blocking the detection loop
                response = requests.post(f"{self.remote_url}/report", json=payload, timeout=2)
                if response.status_code == 200:
                    print(f"Cloud update sent to {self.remote_url}")
                else:
                    print(f"Cloud update failed (Status {response.status_code})")
            except Exception as e:
                print(f"Cloud notification error: {e}")
            
        return True
