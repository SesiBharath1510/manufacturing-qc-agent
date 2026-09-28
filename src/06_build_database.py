import os
import json
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from ultralytics import YOLO

# 1. Database connection path
db_path = os.path.join("data", "factory_quality.db")
csv_telemetry_path = os.path.join("data", "machine_telemetry.csv")

print(f"⏳ Initializing database at: {db_path}")
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# 2. Define Table Schemas using standard SQL
cursor.execute("""
CREATE TABLE IF NOT EXISTS machine_telemetry (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME NOT NULL,
    machine_id TEXT NOT NULL,
    temperature_c REAL NOT NULL,
    vibration_mms REAL NOT NULL,
    rpm REAL NOT NULL,
    hotelling_t2 REAL,
    mspc_alert INTEGER DEFAULT 0
);
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS vision_inspections (
    inspection_id INTEGER PRIMARY KEY AUTOINCREMENT,
    inspected_at DATETIME NOT NULL,
    assembly_line TEXT NOT NULL,
    part_serial_number TEXT NOT NULL UNIQUE,
    defect_detected INTEGER NOT NULL,
    defect_type TEXT,
    confidence REAL,
    bounding_box TEXT
);
""")

# Create indexes for fast timestamp searches
cursor.execute("CREATE INDEX IF NOT EXISTS idx_telemetry_time ON machine_telemetry(timestamp);")
cursor.execute("CREATE INDEX IF NOT EXISTS idx_inspection_time ON vision_inspections(inspected_at);")
conn.commit()
print("✅ SQL tables and indexes created successfully.")

# 3. Ingest Telemetry Data from Step 3
if os.path.exists(csv_telemetry_path):
    df_telemetry = pd.read_csv(csv_telemetry_path)
    df_telemetry["machine_id"] = "CNC-MILL-04"
    # Ensure mspc_alert is stored as 0 or 1
    if "mspc_alert" in df_telemetry.columns:
        df_telemetry["mspc_alert"] = df_telemetry["mspc_alert"].astype(int)
    else:
        df_telemetry["mspc_alert"] = 0

    # Write telemetry dataframe directly into SQL
    df_telemetry.to_sql("machine_telemetry", conn, if_exists="append", index=False)
    print(f"✅ Ingested {len(df_telemetry)} telemetry records into 'machine_telemetry'.")
else:
    print("⚠️ Warning: machine_telemetry.csv not found. Run src/05_sensor_drift_detector.py first.")

# 4. Ingest Simulated Vision Inspection Logs
# Locate trained weights
weights_candidates = [
    os.path.join("runs", "detect", "models", "defect_yolo11n", "weights", "best.pt"),
    os.path.join("runs", "detect", "models", "defect_yolo11n", "weights", "last.pt"),
]
weights_path = next((p for p in weights_candidates if os.path.exists(p)), None)

if weights_path:
    print(f"⏳ Running vision defect logging with model: {weights_path}")
    model = YOLO(weights_path)
else:
    print("⏳ Using fallback base model for inspection logging...")
    model = YOLO("yolo11n.pt")

# Simulate 20 manufactured parts passing the inspection camera
val_dir = os.path.join("data", "defect_dataset", "images", "val")
sample_images = [os.path.join(val_dir, f) for f in os.listdir(val_dir) if f.endswith(".jpg")] if os.path.exists(val_dir) else []

base_time = datetime(2026, 3, 29, 8, 30, 0)
inspection_rows = []

for i in range(20):
    timestamp = base_time + timedelta(minutes=i * 2)
    serial = f"PART-2026-SN{1000 + i}"
    
    # Run model if sample images exist
    defect_detected = 0
    defect_type = "NONE"
    conf = 0.0
    bbox_str = "[]"

    if sample_images:
        img_file = sample_images[i % len(sample_images)]
        res = model(img_file, verbose=False)[0]
        if len(res.boxes) > 0:
            box = res.boxes[0]
            defect_detected = 1
            defect_type = model.names[int(box.cls[0])]
            conf = float(box.conf[0])
            bbox_str = json.dumps([round(c, 2) for c in box.xyxy[0].tolist()])
    
    cursor.execute("""
    INSERT OR REPLACE INTO vision_inspections 
    (inspected_at, assembly_line, part_serial_number, defect_detected, defect_type, confidence, bounding_box)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (timestamp.strftime("%Y-%m-%d %H:%M:%S"), "LINE-A", serial, defect_detected, defect_type, conf, bbox_str))

conn.commit()
conn.close()
print("✅ Inserted 20 inspection logs into 'vision_inspections'.")
print("🎉 Phase 4 Database initialization completed.")