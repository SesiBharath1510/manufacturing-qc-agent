import os
import sqlite3
import pandas as pd

DB_PATH = os.path.join("data", "factory_quality.db")
conn = sqlite3.connect(DB_PATH)

# 1. Compute Plant Quality KPIs
query_kpi = """
SELECT 
    COUNT(*) AS total_inspected,
    SUM(defect_detected) AS total_defective,
    ROUND(100.0 * (COUNT(*) - SUM(defect_detected)) / COUNT(*), 2) AS yield_percentage,
    ROUND(100.0 * SUM(defect_detected) / COUNT(*), 2) AS scrap_rate_percentage
FROM vision_inspections;
"""
df_kpi = pd.read_sql_query(query_kpi, conn)

# 2. Defect Pareto Breakdown
query_pareto = """
SELECT 
    defect_type,
    COUNT(*) AS defect_count,
    ROUND(100.0 * COUNT(*) / (SELECT SUM(defect_detected) FROM vision_inspections WHERE defect_detected = 1), 1) AS pct_of_defects
FROM vision_inspections
WHERE defect_detected = 1
GROUP BY defect_type
ORDER BY defect_count DESC;
"""
df_pareto = pd.read_sql_query(query_pareto, conn)

# 3. Export flat CSV tables directly usable in Power BI
os.makedirs("output", exist_ok=True)
export_kpi_path = os.path.join("output", "powerbi_kpi_summary.csv")
export_pareto_path = os.path.join("output", "powerbi_pareto_defects.csv")
export_full_path = os.path.join("output", "powerbi_full_plant_data.csv")

df_kpi.to_csv(export_kpi_path, index=False)
df_pareto.to_csv(export_pareto_path, index=False)

# Export complete unified join for Power BI Direct Import
query_unified = """
SELECT 
    v.inspected_at,
    v.part_serial_number,
    v.defect_detected,
    v.defect_type,
    v.confidence,
    t.temperature_c,
    t.vibration_mms,
    t.rpm,
    t.hotelling_t2,
    t.mspc_alert
FROM vision_inspections v
LEFT JOIN machine_telemetry t 
  ON t.timestamp BETWEEN datetime(v.inspected_at, '-30 seconds') AND datetime(v.inspected_at, '+30 seconds');
"""
df_unified = pd.read_sql_query(query_unified, conn)
df_unified.to_csv(export_full_path, index=False)

conn.close()

print("=" * 60)
print("📊 PLANT OPERATIONS EXECUTIVE SUMMARY (Power BI Ready)")
print("=" * 60)
print(f"Total Parts Inspected : {df_kpi.loc[0, 'total_inspected']}")
print(f"Total Defects Found   : {df_kpi.loc[0, 'total_defective']}")
print(f"Overall Plant Yield   : {df_kpi.loc[0, 'yield_percentage']} %")
print(f"Plant Scrap Rate      : {df_kpi.loc[0, 'scrap_rate_percentage']} %")
print("\n--- DEFECT PARETO BREAKDOWN ---")
print(df_pareto.to_string(index=False))
print("\n✅ Power BI CSV exports generated:")
print(f" • {export_kpi_path}")
print(f" • {export_pareto_path}")
print(f" • {export_full_path}")