import os
import cv2
from ultralytics import YOLO

# 1. Locate the trained weights produced by your training run
weights_candidates = [
    os.path.join("runs", "detect", "models", "defect_yolo11n", "weights", "best.pt"),
    os.path.join("runs", "detect", "models", "defect_yolo11n", "weights", "last.pt"),
    os.path.join("models", "defect_yolo11n", "weights", "best.pt"),
]

weights_path = None
for candidate in weights_candidates:
    if os.path.exists(candidate):
        weights_path = candidate
        break

if weights_path is None:
    raise FileNotFoundError("Could not find trained weights file. Please verify training output path.")

print(f"⏳ Loading custom defect detector from: {weights_path}")
model = YOLO(weights_path)

# 2. Select a validation image from our defect dataset
test_image = os.path.join("data", "defect_dataset", "images", "val", "sample_val_000.jpg")

# 3. Run inference
results = model(test_image)
res = results[0]

# 4. Display detected defects
print("\n--- DEFECT INSPECTION RESULTS ---")
print(f"Inspected Image: {test_image}")
print(f"Defects Identified: {len(res.boxes)}")

for idx, box in enumerate(res.boxes):
    class_id = int(box.cls[0])
    class_name = model.names[class_id]
    conf = float(box.conf[0])
    coords = [round(c, 2) for c in box.xyxy[0].tolist()]
    print(f"  [{idx + 1}] Flaw: '{class_name}' | Confidence: {conf:.2f} | Bounding Box: {coords}")

# 5. Save the visual bounding-box result
output_path = os.path.join("output", "custom_defect_detected.jpg")
cv2.imwrite(output_path, res.plot())
print(f"\n✅ Annotated visual inspection saved to: {output_path}")