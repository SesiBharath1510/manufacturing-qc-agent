import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

print("✅ NumPy Version:", np.__version__)
print("✅ Pandas Version:", pd.__version__)
print("✅ OpenCV Version:", cv2.__version__)

# Download a tiny pre-trained YOLO nano model and run a test prediction
print("⏳ Loading YOLO nano model...")
model = YOLO("yolo11n.pt")

# Run inference on an official sample image from Ultralytics
results = model.predict(source="https://ultralytics.com/images/bus.jpg", save=False)
print(f"✅ Success! Detected {len(results[0].boxes)} objects in sample image.")