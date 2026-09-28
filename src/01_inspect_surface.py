import os
import urllib.request
import cv2
from ultralytics import YOLO

# 1. Define paths
image_dir = os.path.join("data", "images")
output_dir = "output"
os.makedirs(image_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

sample_image_path = os.path.join(image_dir, "metal_weld.jpg")

# 2. Download an industrial sample image (a metallic surface / weld joint)
image_url = "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/bus.jpg"
# We will download a high-contrast industrial component sample
industrial_url = "https://images.unsplash.com/photo-1581092160607-ee22621dd758?w=800&q=80"

print(f"⏳ Downloading sample manufacturing image to {sample_image_path}...")
urllib.request.urlretrieve(industrial_url, sample_image_path)
print("✅ Image downloaded successfully.")

# 3. Load the YOLO model
model = YOLO("yolo11n.pt")

# 4. Run inference on the industrial image
print("⏳ Running computer vision inference...")
results = model(sample_image_path)

# 5. Inspect the raw detection data
first_result = results[0]
boxes = first_result.boxes

print("\n--- INSPECTION TELEMETRY OUTPUT ---")
print(f"Total objects detected: {len(boxes)}")

for idx, box in enumerate(boxes):
    # Coordinates in pixels: [xmin, ymin, xmax, ymax]
    coords = box.xyxy[0].tolist()
    # Confidence score (0.0 to 1.0)
    conf = float(box.conf[0])
    # Predicted class ID and class name
    class_id = int(box.cls[0])
    class_name = model.names[class_id]

    print(f"Object #{idx + 1}: Class='{class_name}', Confidence={conf:.2f}, Box={coords}")

# 6. Save the visual result with boxes drawn on it
annotated_frame = first_result.plot()
output_image_path = os.path.join(output_dir, "inspected_metal_weld.jpg")
cv2.imwrite(output_image_path, annotated_frame)

print(f"\n✅ Visual inspection saved to: {output_image_path}")