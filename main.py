import time
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

def trigger_local_notification(class_name):
    print(f"\n[{time.strftime('%H:%M:%S')}] !!! CLOUD ALERT: {class_name.upper()} DETECTED !!!")
    
    if winsound:
        try:
            # High-pitched alert beeps
            for _ in range(3):
                winsound.Beep(3000, 500)
                time.sleep(0.1)
        except Exception as e:
            print(f"Local sound failed: {e}")
    else:
        print("\a") # fallback system bell

def main():
    # --- CONFIGURATION ---
    # Replace with your actual Render URL
    RENDER_URL = "https://yolo-model-qrt1.onrender.com" 
    POLL_INTERVAL = 1.0 # Check every second
    
    print("="*40)
    print("AGRI-GUARDIAN LOCAL ALARM SYSTEM")
    print(f"Listening to: {RENDER_URL}")
    print("="*40)
    print("Press Ctrl+C to stop.")

    last_alert_time = 0

    while True:
        try:
            response = requests.get(f"{RENDER_URL}/poll", timeout=5)
            if response.status_code == 200:
                data = response.json()
                
                # Check if there is a new detection
                if data.get("detected") and data.get("timestamp") > last_alert_time:
                    trigger_local_notification(data.get("class_name", "Animal"))
                    last_alert_time = data.get("timestamp")
                    
                    # Optionally clear it on the server
                    requests.post(f"{RENDER_URL}/clear")
            
        except requests.exceptions.RequestException:
            # Silent fail for network hiccups, typical on public wifi/cellular
            pass
        except Exception as e:
            print(f"Listener Error: {e}")
            
        time.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    main()
