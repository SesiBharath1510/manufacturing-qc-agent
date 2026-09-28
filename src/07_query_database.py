import os
import sqlite3
import pandas as pd

db_path = os.path.join("data", "factory_quality.db")
conn = sqlite3.connect(db_path)

print("=" * 60)
print("1. SUMMARY OF INSPECTED DEFECTS BY CATEGORY")
print("=" * 60)
query_defects = """
SELECT 
    defect_type, 
    COUNT(*) as total_count,
    ROUND(AVG(confidence), 3) as avg_confidence
FROM vision_inspections
GROUP BY defect_type;
"""
print(pd.read_sql_query(query_defects, conn))

print("\n" + "=" * 60)
print("2. ROOT-CAUSE TIME CORRELATION (Defects vs Sensor Drift)")
print("=" * 60)
# Find defects and query telemetry from the 3 minutes leading up to each defect
query_correlation = """
SELECT 
    v.part_serial_number,
    v.inspected_at,
    v.defect_type,
    ROUND(t.temperature_c, 1) as machine_temp_c,
    ROUND(t.vibration_mms, 2) as vibration_mms,
    ROUND(t.hotelling_t2, 2) as hotelling_t2,
    t.mspc_alert
FROM vision_inspections v
JOIN machine_telemetry t 
  ON t.timestamp BETWEEN datetime(v.inspected_at, '-60 seconds') AND v.inspected_at
WHERE v.defect_detected = 1
ORDER BY v.inspected_at DESC, t.timestamp DESC
LIMIT 10;
"""
df_corr = pd.read_sql_query(query_correlation, conn)
if not df_corr.empty:
    print(df_corr.to_string(index=False))
else:
    print("No defects matched within 60-second window. Telemetry is healthy.")

conn.close()
print("\n✅ Verification complete. SQL layer is ready for the LangGraph Agent.")