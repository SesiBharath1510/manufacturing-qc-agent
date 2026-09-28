import os
import sqlite3
import numpy as np
import pytest

def test_database_integrity():
    """Verify database schema, table presence, and indexing."""
    db_path = os.path.join("data", "factory_timescaledb_mirror.db")
    assert os.path.exists(db_path), "Database mirror does not exist."
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [r[0] for r in cursor.fetchall()]
    assert "machine_telemetry" in tables, "Table machine_telemetry missing."
    assert "vision_inspections" in tables, "Table vision_inspections missing."
    assert "maintenance_logs" in tables, "Table maintenance_logs missing."
    conn.close()

def test_hotelling_t2_calculation():
    """Verify multivariate Hotelling's T² distance math."""
    # Synthetic healthy covariance
    X_healthy = np.array([
        [65.0, 1.20, 3000.0],
        [65.2, 1.18, 3005.0],
        [64.8, 1.22, 2995.0],
        [65.1, 1.19, 3002.0],
        [64.9, 1.21, 2998.0]
    ])
    mu = np.mean(X_healthy, axis=0)
    cov = np.cov(X_healthy, rowvar=False)
    inv_cov = np.linalg.pinv(cov)

    # An anomalous observation: high temp, high vibration, low rpm
    x_anomaly = np.array([85.0, 2.5, 2700.0])
    diff = x_anomaly - mu
    t2 = np.dot(np.dot(diff, inv_cov), diff.T)

    # T² distance for extreme anomaly must exceed baseline
    assert t2 > 10.0, f"Hotelling's T² calculation failed to detect severe anomaly (score: {t2})"

def test_triton_config_exists():
    """Verify Triton inference server config is valid and formatted."""
    config_path = os.path.join("triton_model_repository", "yolo11_defect_detector", "config.pbtxt")
    assert os.path.exists(config_path), "Triton config.pbtxt missing."
    
    with open(config_path, "r", encoding="utf-8") as f:
        content = f.read()
        assert 'name: "yolo11_defect_detector"' in content
        assert "dynamic_batching" in content