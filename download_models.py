import os
import requests

def download_file(url, output_path):
    if os.path.exists(output_path):
        print(f"{output_path} already exists. Skipping download.")
        return
    print(f"Downloading {url} to {output_path}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print("Download complete.")
    except Exception as e:
        print(f"Failed to download: {e}")
        print("Please ensure you have internet access and the URL is reachable.")

if __name__ == "__main__":
    # Standard YOLOv8 model for general object detection (including animals)
    yolov8n_url = "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt"
    
    # Download directly to project root as the detector looks there by default
    download_file(yolov8n_url, "yolov8n.pt")
