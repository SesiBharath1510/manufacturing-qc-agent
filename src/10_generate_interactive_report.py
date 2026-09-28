import os
import sqlite3
import webbrowser
import pandas as pd

DB_PATH = os.path.join("data", "factory_quality.db")
conn = sqlite3.connect(DB_PATH)

# 1. Fetch Executive KPIs
kpi = pd.read_sql_query("""
SELECT 
    COUNT(*) as total,
    SUM(defect_detected) as defective,
    ROUND(100.0 * (COUNT(*) - SUM(defect_detected)) / COUNT(*), 1) as yield_pct,
    ROUND(100.0 * SUM(defect_detected) / COUNT(*), 1) as scrap_pct
FROM vision_inspections;
""", conn).iloc[0]

# 2. Fetch Defect Breakdown
pareto = pd.read_sql_query("""
SELECT defect_type, COUNT(*) as qty
FROM vision_inspections
WHERE defect_detected = 1
GROUP BY defect_type
ORDER BY qty DESC;
""", conn)

# 3. Read RCA Incident Report Ticket
rca_path = os.path.join("output", "incident_rca_report.txt")
rca_text = ""
if os.path.exists(rca_path):
    with open(rca_path, "r", encoding="utf-8") as f:
        rca_text = f.read()

conn.close()

# 4. Generate HTML Dashboard
html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Autonomous Quality Control & RCA Dashboard</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 0; background: #0f172a; color: #f8fafc; padding: 24px; }}
        .header {{ border-bottom: 1px solid #334155; padding-bottom: 16px; margin-bottom: 24px; }}
        .header h1 {{ margin: 0 0 6px 0; color: #38bdf8; font-size: 24px; }}
        .header p {{ margin: 0; color: #94a3b8; font-size: 14px; }}
        .grid-kpi {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 24px; }}
        .card {{ background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 18px; }}
        .card h4 {{ margin: 0; color: #94a3b8; font-size: 13px; text-transform: uppercase; }}
        .card .val {{ font-size: 28px; font-weight: bold; margin-top: 8px; }}
        .val.good {{ color: #4ade80; }}
        .val.alert {{ color: #f87171; }}
        .val.blue {{ color: #38bdf8; }}
        .layout {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
        pre {{ background: #090d16; border: 1px solid #1e293b; border-radius: 6px; padding: 14px; color: #a5f3fc; font-size: 12px; line-height: 1.45; overflow-x: auto; white-space: pre-wrap; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 8px; font-size: 14px; }}
        th, td {{ text-align: left; padding: 10px; border-bottom: 1px solid #334155; }}
        th {{ color: #94a3b8; }}
        .badge {{ background: #ef4444; color: white; padding: 3px 8px; border-radius: 4px; font-size: 11px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Autonomous Manufacturing Quality & Root-Cause Operations</h1>
        <p>Production Line-A | Live Telemetry, YOLOv11 Computer Vision & Autonomous Agent RCA</p>
    </div>

    <div class="grid-kpi">
        <div class="card">
            <h4>Total Inspected</h4>
            <div class="val blue">{int(kpi['total'])} units</div>
        </div>
        <div class="card">
            <h4>Defects Detected</h4>
            <div class="val alert">{int(kpi['defective'])} units</div>
        </div>
        <div class="card">
            <h4>First-Pass Yield</h4>
            <div class="val good">{kpi['yield_pct']}%</div>
        </div>
        <div class="card">
            <h4>Scrap Rate</h4>
            <div class="val alert">{kpi['scrap_pct']}%</div>
        </div>
    </div>

    <div class="layout">
        <div class="card">
            <h3>Defect Distribution (Pareto)</h3>
            <table>
                <tr><th>Defect Category</th><th>Detected Count</th><th>Impact Status</th></tr>
"""
for _, r in pareto.iterrows():
    html_content += f"<tr><td><strong>{r['defect_type'].upper()}</strong></td><td>{r['qty']}</td><td><span class='badge'>CRITICAL DRIFT</span></td></tr>"

html_content += f"""
            </table>
            <h3 style="margin-top:24px;">Statistical Process Control Drift Chart</h3>
            <img src="spc_drift_charts.png" style="width:100%; border-radius:6px; border:1px solid #334155;" alt="SPC Control Chart">
        </div>
        <div class="card">
            <h3>Autonomous Agent Incident Ticket</h3>
            <pre>{rca_text}</pre>
        </div>
    </div>
</body>
</html>
"""

dashboard_file = os.path.join("output", "plant_operations_dashboard.html")
with open(dashboard_file, "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"✅ Interactive Dashboard generated at: {dashboard_file}")
webbrowser.open(os.path.abspath(dashboard_file))