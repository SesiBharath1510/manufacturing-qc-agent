import os
import json
import sqlite3
import random
from datetime import datetime, timedelta

db_path = os.path.join("data", "factory_quality.db")
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Clear existing vision inspections
cursor.execute("DELETE FROM vision_inspections;")

# Generate 60 parts produced every 45 seconds from 08:35:00 to 09:20:00
base_time = datetime(2026, 3, 29, 8, 35, 0)
total_parts = 60

print("⏳ Seeding realistic industrial production run into SQL...")

for i in range(total_parts):
    inspect_time = base_time + timedelta(seconds=i * 45)
    serial = f"PART-2026-SN{5000 + i}"
    time_str = inspect_time.strftime("%Y-%m-%d %H:%M:%S")

    # Check if this timestamp falls inside the machine drift zone (after 09:00:00)
    # If the machine is drifting, defect probability jumps to 85%
    if inspect_time >= datetime(2026, 3, 29, 9, 0, 0):
        if random.random() < 0.85:
            defect_detected = 1
            defect_type = random.choice(["crack", "void", "crack"])
            conf = round(random.uniform(0.88, 0.98), 2)
            bbox = json.dumps([round(random.uniform(50, 100), 1), round(random.uniform(50, 100), 1), 
                              round(random.uniform(150, 250), 1), round(random.uniform(150, 250), 1)])
        else:
            defect_detected = 0
            defect_type = "NONE"
            conf = 0.0
            bbox = "[]"
    else:
        # Healthy machine state before 09:00:00 (baseline 2% scrap)
        if random.random() < 0.02:
            defect_detected = 1
            defect_type = "scratch"
            conf = 0.76
            bbox = json.dumps([20.0, 30.0, 80.0, 90.0])
        else:
            defect_detected = 0
            defect_type = "NONE"
            conf = 0.0
            bbox = "[]"

    cursor.execute("""
    INSERT INTO vision_inspections 
    (inspected_at, assembly_line, part_serial_number, defect_detected, defect_type, confidence, bounding_box)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (time_str, "LINE-A", serial, defect_detected, defect_type, conf, bbox))

conn.commit()

# Print inspection audit
cursor.execute("SELECT defect_type, COUNT(*) FROM vision_inspections GROUP BY defect_type;")
print("\n--- NEW PRODUCTION RUN SEEDED ---")
for row in cursor.fetchall():
    print(f"Defect: {row[0]:<10} | Count: {row[1]}")

conn.close()
print("✅ Production database ready for Autonomous Agent.")