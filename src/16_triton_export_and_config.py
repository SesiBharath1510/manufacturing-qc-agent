import os
import shutil
from ultralytics import YOLO

# 1. Locate trained model weights
weights_candidates = [
    os.path.join("runs", "detect", "models", "defect_yolo11n", "weights", "best.pt"),
    os.path.join("runs", "detect", "models", "defect_yolo11n", "weights", "last.pt"),
    "yolo11n.pt"
]
weights_path = next((p for p in weights_candidates if os.path.exists(p)), "yolo11n.pt")

print(f"⏳ Loading model from {weights_path} to export for Triton Inference Server...")
model = YOLO(weights_path)

# 2. Export YOLOv11 to ONNX format (320x320)
print("⏳ Compiling YOLOv11 to ONNX graph format...")
exported_onnx_path = model.export(format="onnx", imgsz=320, dynamic=True)
print(f"✅ ONNX model generated at: {exported_onnx_path}")

# 3. Build Triton Model Repository Structure
triton_repo = os.path.abspath("triton_model_repository")
model_version_dir = os.path.join(triton_repo, "yolo11_defect_detector", "1")
os.makedirs(model_version_dir, exist_ok=True)

# Copy exported ONNX model into Triton version folder
dest_onnx = os.path.join(model_version_dir, "model.onnx")
if os.path.exists(exported_onnx_path):
    shutil.copy2(exported_onnx_path, dest_onnx)
    print(f"📦 Stored Triton model at: {dest_onnx}")

# 4. Generate Triton config.pbtxt specification
triton_config = """name: "yolo11_defect_detector"
platform: "onnxruntime_onnx"
max_batch_size: 8

input [
  {
    name: "images"
    data_type: TYPE_FP32
    dims: [ 3, 320, 320 ]
  }
]

output [
  {
    name: "output0"
    data_type: TYPE_FP32
    dims: [ 7, 2100 ]
  }
]

dynamic_batching {
  max_queue_delay_microseconds: 5000
}

instance_group [
  {
    count: 2
    kind: KIND_CPU
  }
]
"""

config_path = os.path.join(triton_repo, "yolo11_defect_detector", "config.pbtxt")
with open(config_path, "w", encoding="utf-8") as f:
    f.write(triton_config)

print(f"✅ Triton Server Configuration generated at:\n   {config_path}")
print("🚀 Ready for Triton containerized deployment.")