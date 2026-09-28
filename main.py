import sys
import subprocess
import time

enterprise_pipeline = [
    ("Stage 01: Dataset Preparation", "src/02_prepare_defect_dataset.py"),
    ("Stage 02: MSPC & EWMA Sensor Drift Modeling", "src/05_sensor_drift_detector.py"),
    ("Stage 03: TimescaleDB / Enterprise SQL Engine", "src/11_timescaledb_engine.py"),
    ("Stage 04: Industrial Line Production Seeding", "src/06b_seed_production_run.py"),
    ("Stage 05: YOLOv11-seg Instance Segmentation", "src/12_instance_segmentation.py"),
    ("Stage 06: LangGraph Autonomous Root-Cause Agent", "src/13_langgraph_rca_agent.py"),
    ("Stage 07: MLflow Experiment Tracking", "src/14_mlflow_tracker.py"),
    ("Stage 08: DVC Data Versioning Manifest", "src/15_dvc_pipeline_setup.py"),
    ("Stage 09: Triton Inference ONNX Compilation", "src/16_triton_export_and_config.py"),
    ("Stage 10: Evidently AI Telemetry Drift Monitor", "src/17_evidently_drift_monitor.py"),
    ("Stage 11: Power BI KPI Export", "src/09_operations_dashboard.py"),
    ("Stage 12: Interactive Web Operations Dashboard", "src/10_generate_interactive_report.py")
]

print("=" * 80)
print("🏭 EXECUTING FULL MULTIMODAL AUTONOMOUS MANUFACTURING QC PIPELINE")
print("=" * 80)

total_start = time.time()

for step_name, script_path in enterprise_pipeline:
    print(f"\n▶ [{step_name}] -> Running {script_path}...")
    step_start = time.time()
    res = subprocess.run([sys.executable, script_path])
    if res.returncode != 0:
        print(f"\n❌ Pipeline failed during: {step_name}")
        sys.exit(res.returncode)
    print(f"✔ Completed in {time.time() - step_start:.2f}s")

print("\n" + "=" * 80)
print(f"🎉 FULL SYSTEM PIPELINE COMPLETED IN {time.time() - total_start:.2f} SECONDS")
print("=" * 80)