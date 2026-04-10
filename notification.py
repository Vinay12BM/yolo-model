import os
import time
import datetime
import winsound

class Notifier:
    def __init__(self, log_file="intrusion_log.txt", cooldown_seconds=10):
        self.log_file = log_file
        self.cooldown_seconds = cooldown_seconds
        
        # Track last notification time per class to avoid spam
        self.last_notified = {}

    def trigger_alert(self, class_name, confidence, bbox=None):
        current_time = time.time()
        
        # Check cooldown
        if class_name in self.last_notified:
            if current_time - self.last_notified[class_name] < self.cooldown_seconds:
                return False # Still in cooldown

        self.last_notified[class_name] = current_time
        
        # 1. Log the event
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] ALARM: {class_name.upper()} detected with {confidence*100:.1f}% confidence"
        if bbox:
             log_msg += f" at bounding box {bbox}"
        log_msg += "\n"
        
        with open(self.log_file, "a") as f:
            f.write(log_msg)
            
        print(f"\n{'='*40}")
        print(f"!!! INTRUDER ALERT: {class_name.upper()} !!!")
        print(f"{'='*40}\n")
        
        # 2. Play sound alert
        try:
            # Play a repeating beep
            for _ in range(3):
                winsound.Beep(2500, 500)
                time.sleep(0.1)
        except Exception as e:
            print(f"Alert sound failed: {e}")
            
        return True
