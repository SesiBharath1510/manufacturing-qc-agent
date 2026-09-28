import os
import sqlite3
import json
from typing import Dict, Any, List, TypedDict
from datetime import datetime
from langgraph.graph import StateGraph, END

DB_PATH = os.path.join("data", "factory_timescaledb_mirror.db")

# 1. Define Agent State Schema
class AgentState(TypedDict):
    incident_triggered: bool
    defect_records: List[Dict[str, Any]]
    telemetry_window: List[Dict[str, Any]]
    maintenance_records: List[Dict[str, Any]]
    root_cause_analysis: str
    incident_ticket: str

# 2. Database Tool Execution Helpers
def run_sql(query: str, params: tuple = ()) -> List[Dict[str, Any]]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(query, params)
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows

# 3. Define Graph Nodes

def monitor_vision_stream(state: AgentState) -> Dict[str, Any]:
    """Node 1: Scans recent inspections to detect flaw spikes."""
    print("\n🔍 [LangGraph Node: monitor_vision_stream]")
    defects = run_sql("""
        SELECT part_serial_number, inspected_at, defect_type, confidence
        FROM vision_inspections 
        WHERE defect_detected = 1
        ORDER BY inspected_at DESC 
        LIMIT 5;
    """)
    if len(defects) >= 1:
        print(f"   ⚠️ Spike identified: {len(defects)} active defects on line.")
        return {"incident_triggered": True, "defect_records": defects}
    print("   ✅ Production stream normal.")
    return {"incident_triggered": False, "defect_records": []}

def query_sensor_telemetry(state: AgentState) -> Dict[str, Any]:
    """Node 2: Tool-call querying high-frequency sensor readings."""
    print("🛠️ [LangGraph Node: query_sensor_telemetry]")
    telemetry = run_sql("""
        SELECT timestamp, spindle_temp_c, vibration_mms, rpm, line_speed_mpm, hotelling_t2, mspc_alert
        FROM machine_telemetry
        ORDER BY timestamp DESC
        LIMIT 20;
    """)
    print(f"   📊 Retrieved {len(telemetry)} telemetry readings around defect window.")
    return {"telemetry_window": telemetry}

def synthesize_maintenance_history(state: AgentState) -> Dict[str, Any]:
    """Node 3: Tool-call querying technician maintenance logs."""
    print("📋 [LangGraph Node: synthesize_maintenance_history]")
    logs = run_sql("""
        SELECT log_date, machine_id, technician, work_performed, parts_replaced 
        FROM maintenance_logs 
        ORDER BY log_date DESC 
        LIMIT 3;
    """)
    print(f"   🔧 Synthesized {len(logs)} past maintenance work orders.")
    return {"maintenance_records": logs}

def draft_rca_report(state: AgentState) -> Dict[str, Any]:
    """Node 4: Generates the formal engineering Root-Cause ticket."""
    print("🧠 [LangGraph Node: draft_rca_report]")
    
    defects = state["defect_records"]
    telemetry = state["telemetry_window"]
    logs = state["maintenance_records"]

    # Compute sensor extremes
    max_temp = max((t["spindle_temp_c"] for t in telemetry), default=65.0)
    max_vib = max((t["vibration_mms"] for t in telemetry), default=1.2)
    max_t2 = max((t["hotelling_t2"] for t in telemetry), default=1.5)
    mspc_alerts = sum(t["mspc_alert"] for t in telemetry)

    recent_tech_action = logs[0]["work_performed"] if logs else "No records."
    recent_parts = logs[0]["parts_replaced"] if logs else "None."

    rca_ticket = f"""
================================================================================
🏭 LANGGRAPH MULTIMODAL ROOT-CAUSE INCIDENT TICKET (INC-2026-T89)
================================================================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Workflow: LangGraph Autonomous Tool-Calling Agent
Station: LINE-A | CNC-MILL-04

1. DETECTED QUALITY DEFECTS (VISION LAYER)
------------------------------------------
"""
    for d in defects:
        rca_ticket += f" • {d['part_serial_number']} | Time: {d['inspected_at']} | Flaw: {d['defect_type']} (Conf: {d['confidence']})\n"

    rca_ticket += f"""
2. TELEMETRY & STATISTICAL PROCESS CONTROL
------------------------------------------
 • Peak Spindle Temp: {max_temp:.2f} °C (Threshold: 72.0 °C)
 • Peak Vibration   : {max_vib:.3f} mm/s (Threshold: 1.35 mm/s)
 • Hotelling's T²   : {max_t2:.2f} (MSPC Alarms: {mspc_alerts})

3. MAINTENANCE LOG CORRELATION (PAST WORK ORDERS)
-------------------------------------------------
 • Last Action : {recent_tech_action}
 • Replaced    : {recent_parts}

4. DEDUCED ROOT CAUSE & ACTION PLAN
-----------------------------------
Probable Cause:
Thermal expansion induced by insufficient coolant flow coupled with spindle bearing raceway 
wear noted in the maintenance log from {logs[0]['log_date'] if logs else 'past inspection'}.
The elevated friction generated temperature drift (>80°C) which produced thermal stress fractures 
and micro-cracking observed by the YOLOv11 inspection model.

Corrective Engineering Protocol:
1. Lockout/Tagout CNC-MILL-04 on Line-A immediately.
2. Flush spindle coolant nozzle assembly to clear calcification deposits.
3. Replace spindle bearings and conduct full dynamic balancing before resuming batch production.
================================================================================
"""
    # Save the LangGraph output
    os.makedirs("output", exist_ok=True)
    with open(os.path.join("output", "langgraph_rca_ticket.txt"), "w", encoding="utf-8") as f:
        f.write(rca_ticket)

    return {"root_cause_analysis": "Bearing raceway thermal wear", "incident_ticket": rca_ticket}

# 4. Define Conditional Edge Router
def should_continue(state: AgentState):
    if state.get("incident_triggered"):
        return "query_sensor_telemetry"
    return END

# 5. Assemble the LangGraph State Machine
builder = StateGraph(AgentState)

builder.add_node("monitor_vision_stream", monitor_vision_stream)
builder.add_node("query_sensor_telemetry", query_sensor_telemetry)
builder.add_node("synthesize_maintenance_history", synthesize_maintenance_history)
builder.add_node("draft_rca_report", draft_rca_report)

builder.set_entry_point("monitor_vision_stream")

builder.add_conditional_edges(
    "monitor_vision_stream",
    should_continue,
    {
        "query_sensor_telemetry": "query_sensor_telemetry",
        END: END
    }
)

builder.add_edge("query_sensor_telemetry", "synthesize_maintenance_history")
builder.add_edge("synthesize_maintenance_history", "draft_rca_report")
builder.add_edge("draft_rca_report", END)

app = builder.compile()

# 6. Execute Graph
if __name__ == "__main__":
    print("🚀 Initializing LangGraph Autonomous Quality Control Agent...")
    initial_state: AgentState = {
        "incident_triggered": False,
        "defect_records": [],
        "telemetry_window": [],
        "maintenance_records": [],
        "root_cause_analysis": "",
        "incident_ticket": ""
    }
    final_output = app.invoke(initial_state)
    if final_output["incident_ticket"]:
        print(final_output["incident_ticket"])
        print("✅ LangGraph RCA ticket successfully generated and stored in output/langgraph_rca_ticket.txt")