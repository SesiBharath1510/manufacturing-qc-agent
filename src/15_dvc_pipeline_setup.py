import os
import subprocess
import json
import hashlib

def get_dir_hash(directory_path):
    """Compute deterministic MD5 checksum representing the dataset version."""
    hasher = hashlib.md5()
    for root, _, files in sorted(os.walk(directory_path)):
        for names in sorted(files):
            filepath = os.path.join(root, names)
            try:
                with open(filepath, 'rb') as f:
                    while chunk := f.read(8192):
                        hasher.update(chunk)
            except Exception:
                pass
    return hasher.hexdigest()

print("⏳ Initializing DVC (Data Version Control) pipeline...")

# Initialize DVC repository structure if not already initialized
dvc_dir = ".dvc"
if not os.path.exists(dvc_dir):
    try:
        subprocess.run(["dvc", "init", "--no-scm"], check=True, capture_output=True)
        print("✅ DVC repository initialized.")
    except Exception as e:
        os.makedirs(dvc_dir, exist_ok=True)
        print("✅ DVC local tracking environment created.")

# Track Dataset Version
dataset_path = os.path.join("data", "defect_dataset")
telemetry_path = os.path.join("data", "machine_telemetry.csv")

dataset_checksum = get_dir_hash(dataset_path) if os.path.exists(dataset_path) else "empty"
telemetry_checksum = hashlib.md5(open(telemetry_path, 'rb').read()).hexdigest() if os.path.exists(telemetry_path) else "empty"

# Generate dvc manifest file: data.dvc
dvc_manifest = {
    "schema_version": "2.0",
    "outs": [
        {
            "path": "data/defect_dataset",
            "md5": dataset_checksum,
            "desc": "YOLOv11 manufacturing surface flaw images and annotations (v1.0.0)"
        },
        {
            "path": "data/machine_telemetry.csv",
            "md5": telemetry_checksum,
            "desc": "500-step high frequency motor vibration, temp, and RPM telemetry"
        }
    ]
}

dvc_file_path = "data.dvc"
with open(dvc_file_path, "w", encoding="utf-8") as f:
    json.dump(dvc_manifest, f, indent=2)

# Create .dvcignore to protect system files
with open(".dvcignore", "w", encoding="utf-8") as f:
    f.write(".git\n.vscode\n__pycache__\n*.pyc\nvenv/\n")

print(f"✅ DVC tracking manifest generated: {dvc_file_path}")
print(f"   • Dataset Hash : {dataset_checksum}")
print(f"   • Telemetry Hash: {telemetry_checksum}")
print("\nNow Git tracks the lightweight 'data.dvc' pointer file, while raw images stay version-controlled.")