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
    # YOLOv8 face model from public community source
    yolo_face_url = "https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolov8n-face.pt"
    os.makedirs("models", exist_ok=True)
    download_file(yolo_face_url, "models/yolov8n-face.pt")
