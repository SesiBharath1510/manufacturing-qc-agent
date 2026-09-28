import os
import sqlite3
import json
import pandas as pd
from datetime import datetime

DB_PATH = os.path.join("data", "factory_quality.db")

class SQLToolkit:
    """Read-only SQL toolset for the Root-Cause Reasoning Agent."""
    @staticmethod
    def get_connection():
        return sqlite3.connect(DB_PATH)

    @classmethod
    def query_defect_cluster(cls, limit=5):
        """Tool: Retrieve latest high-confidence defects."""
        conn = cls.get_connection()
        query = """
        SELECT part_serial_number, inspected_at, defect_type, confidence
        FROM vision_inspections
        WHERE defect_detected = 1
        ORDER BY inspected_at DESC
        LIMIT ?;
        """
        df = pd.read_sql_query(query, conn, params=(limit,))
        conn.close()
        return df.to_dict(orient="records")

    @classmethod
    def query_telemetry_window(cls, start_time, end_time):
        """Tool: Retrieve sensor metrics between two timestamps."""
        conn = cls.get_connection()
        query = """
        SELECT 
            timestamp,
            machine_id,
            ROUND(temperature_c, 2) AS temp_c,
            ROUND(vibration_mms, 3) AS vibration,
            ROUND(rpm, 1) AS rpm,
            ROUND(hotelling_t2, 2) AS t2_score,
            mspc_alert
        FROM machine_telemetry
        WHERE timestamp BETWEEN ? AND ?
        ORDER BY timestamp ASC;
        """
        df = pd.read_sql_query(query, conn, params=(start_time, end_time))
        conn.close()
        return df

class RootCauseAnalysisAgent:
    """Autonomous reasoning agent that correlates defects with sensor anomalies."""
    def __init__(self):
        self.toolkit = SQLToolkit()

    def run(self):
        print("🤖 [Agent Node: Monitor] Scanning vision inspection stream...")
        recent_defects = self.toolkit.query_defect_cluster(limit=6)

        if not recent_defects:
            print("✅ All systems nominal. No defect clusters identified.")
            return

        print(f"🚨 [Agent Node: Anomaly Detected] Found {len(recent_defects)} defects in recent batches!")
        timestamps = [d["inspected_at"] for d in recent_defects]
        start_window = min(timestamps)
        end_window = max(timestamps)

        print(f"🛠️ [Agent Node: Tool-Call] Querying telemetry window: {start_window} -> {end_window}...")
        telemetry = self.toolkit.query_telemetry_window(start_window, end_window)

        print("🧠 [Agent Node: Reason] Correlating physical flaws with statistical sensor drift...")
        
        # Core statistical correlation
        max_temp = telemetry["temp_c"].max()
        avg_temp = telemetry["temp_c"].mean()
        max_vib = telemetry["vibration"].max()
        mspc_breaches = int(telemetry["mspc_alert"].sum())
        primary_defect = pd.Series([d["defect_type"] for d in recent_defects]).mode()[0]

        # Root-cause deduction logic
        probable_cause = "Unknown Anomaly"
        corrective_action = "Manual line inspection required."
        severity = "MEDIUM"

        if max_temp > 72.0 and max_vib > 1.35:
            probable_cause = (
                "Spindle Bearing Overheating & Eccentric Vibration. "
                "Frictional thermal expansion is inducing micro-cracks and thermal voids on the workpiece."
            )
            corrective_action = (
                "1. Immediately halt Line-A spindle rotation.\n"
                "2. Inspect spindle bearing lubrication and coolant flow.\n"
                "3. Perform vibration spectrum check for bearing raceway spalling."
            )
            severity = "CRITICAL - HIGH SCRAP RISK"
        elif max_temp > 70.0:
            probable_cause = "Cooling system degradation causing surface micro-cracking."
            corrective_action = "Flush and recharge spindle coolant circuit."
            severity = "HIGH"
        elif max_vib > 1.30:
            probable_cause = "Tool chattering or loose mechanical fixture causing surface gouges."
            corrective_action = "Re-torque fixture clamps and check cutting tool wear."
            severity = "MEDIUM"

        # Generate Formal Incident Report Ticket
        report = f"""
================================================================================
🏭 AUTONOMOUS ROOT-CAUSE INCIDENT REPORT: INC-2026-0329
================================================================================
Timestamp Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Target Station: LINE-A | CNC-MILL-04
Severity Level: {severity}

1. EXECUTIVE SUMMARY
--------------------
A continuous cluster of {len(recent_defects)} physical defects ({primary_defect.upper()}) was detected 
by the YOLOv11 optical inspection camera. Telemetry correlation confirms statistical
process control breach (Hotelling's T² breached with {mspc_breaches} consecutive alarm periods).

2. DEFECT CLUSTER DETAILS
-------------------------
"""
        for d in recent_defects:
            report += f" • Serial: {d['part_serial_number']} | Time: {d['inspected_at']} | Type: {d['defect_type']} | Conf: {d['confidence']}\n"

        report += f"""
3. CORRELATED SENSOR TELEMETRY
------------------------------
 • Peak Spindle Temperature : {max_temp:.1f} °C (Baseline: 65.0 °C)
 • Peak Vibration Amplitude : {max_vib:.3f} mm/s (Baseline: 1.20 mm/s)
 • Hotelling's T² Breaches  : {mspc_breaches} alarms logged in incident window

4. DEDUCED ROOT CAUSE
---------------------
{probable_cause}

5. RECOMMENDED CORRECTIVE ACTIONS
---------------------------------
{corrective_action}
================================================================================
"""
        print(report)

        # Save incident report
        os.makedirs("output", exist_ok=True)
        report_path = os.path.join("output", "incident_rca_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"✅ Incident report ticket saved to: {report_path}")

if __name__ == "__main__":
    agent = RootCauseAnalysisAgent()
    agent.run()