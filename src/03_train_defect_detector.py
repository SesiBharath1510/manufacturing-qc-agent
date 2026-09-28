import os
from ultralytics import YOLO

# 1. Path to dataset YAML configuration
dataset_yaml = os.path.abspath(os.path.join("data", "defect_dataset", "defects.yaml"))

# 2. Load the base pre-trained model
print("⏳ Initializing YOLOv11 nano backbone...")
model = YOLO("yolo11n.pt")

# 3. Start fine-tuning for 5 epochs
# On CPU, 5 epochs on this dataset takes approximately 20-30 seconds
print(f"🚀 Starting training on: {dataset_yaml}")
results = model.train(
    data=dataset_yaml,
    epochs=5,
    imgsz=320,
    batch=4,
    workers=2,
    project="models",
    name="defect_yolo11n"
)

print("\n✅ Training complete!")
print("Trained model weights saved inside: models/defect_yolo11n/weights/best.pt")