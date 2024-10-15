import requests
import cv2
import numpy as np
from ultralytics import YOLO
from src.config.config import config

# Load a pretrained YOLO model
model = YOLO(config.MODEL_PATH)

def load_image_from_url(url):
    # Fetch the image from the URL
    response = requests.get(url)
    if response.status_code == 200:
        # Convert the image data to a numpy array
        image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        # Decode the image to an OpenCV format
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        return image
    else:
        print(f"Failed to fetch image from URL: {url}")
        return None

# URL of the image to predict
image_url = "https://www.batamnews.co.id/foto_berita/2023/04/2023-04-10-kenapa-mobil-di-batam-tak-boleh-dibawa-keluar-pulau-batam-atau-mudik.jpeg"  # Replace with your image URL

# Load the image from the URL
image = load_image_from_url(image_url)

if image is not None:
    # Run inference on the loaded image
    results = model.predict(image, save=True, imgsz=320, conf=0.5)
    print(results)
else:
    print("No image to process.")
