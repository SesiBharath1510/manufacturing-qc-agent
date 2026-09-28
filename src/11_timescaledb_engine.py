import os
import json
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

# PostgreSQL / TimescaleDB Connection Config
PG_HOST = "localhost"
PG_PORT = 5432
PG_USER = "factory_admin"
PG_PASSWORD = "factory_secure_password_2026"
PG_DB = "factory_telemetry"

def get_connection():
    """Attempt connecting to TimescaleDB (PostgreSQL); fallback to local SQLite if Docker is inactive."""
    try:
        conn = psycopg2.connect(
            host=PG_HOST,
            port=PG_PORT,
            user=PG_USER,
            password=PG_PASSWORD,
            dbname=PG_DB
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        print("Connected to TimescaleDB (PostgreSQL 16) successfully.")
        return conn, "POSTGRES"
    except Exception as e:
        print(f"TimescaleDB connection failed ({e}).")
        print("Operating in SQLite-Compatible Enterprise Fallback Mode...")
        sqlite_path = os.path.join("data", "factory_timescaledb_mirror.db")
        conn = sqlite3.connect(sqlite_path)
        return conn, "SQLITE"

def initialize_schema():
    conn, engine_type = get_connection()
    cursor = conn.cursor()

    print("\n--- INITIALIZING INDUSTRIAL ENTERPRISE SCHEMA ---")

    if engine_type == "POSTGRES":
        # Enable TimescaleDB extension
        cursor.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;")
        
        # 1. Telemetry Table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS machine_telemetry (
            timestamp TIMESTAMPTZ NOT NULL,
            machine_id VARCHAR(50) NOT NULL,
            line_speed_mpm DOUBLE PRECISION NOT NULL,
            spindle_temp_c DOUBLE PRECISION NOT NULL,
            vibration_mms DOUBLE PRECISION NOT NULL,
            rpm DOUBLE PRECISION NOT NULL,
            hotelling_t2 DOUBLE PRECISION NOT NULL,
            mspc_alert INT DEFAULT 0
        );
        """)
        # Convert to TimescaleDB Hypertable partitioned by 1-hour chunks
        try:
            cursor.execute("SELECT create_hypertable('machine_telemetry', 'timestamp', if_not_exists => TRUE);")
            print("Hypertable 'machine_telemetry' enabled with time partitioning.")
        except Exception as e:
            print(f"Hypertable notice: {e}")

        # 2. Vision Inspections Table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS vision_inspections (
            inspection_id SERIAL PRIMARY KEY,
            inspected_at TIMESTAMPTZ NOT NULL,
            assembly_line VARCHAR(50) NOT NULL,
            part_serial_number VARCHAR(100) UNIQUE NOT NULL,
            defect_detected INT NOT NULL,
            defect_type VARCHAR(50),
            confidence DOUBLE PRECISION,
            bounding_box JSONB,
            segmentation_mask_polygon JSONB
        );
        """)

        # 3. Maintenance Work Orders Log Table (for LLM Agent reasoning)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS maintenance_logs (
            log_id SERIAL PRIMARY KEY,
            log_date DATE NOT NULL,
            machine_id VARCHAR(50) NOT NULL,
            technician VARCHAR(100) NOT NULL,
            work_performed TEXT NOT NULL,
            parts_replaced TEXT NOT NULL,
            status VARCHAR(50) NOT NULL
        );
        """)

    else:
        # SQLite Mirror Mode
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS machine_telemetry (
            timestamp DATETIME NOT NULL,
            machine_id TEXT NOT NULL,
            line_speed_mpm REAL NOT NULL,
            spindle_temp_c REAL NOT NULL,
            vibration_mms REAL NOT NULL,
            rpm REAL NOT NULL,
            hotelling_t2 REAL NOT NULL,
            mspc_alert INTEGER DEFAULT 0
        );
        """)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS vision_inspections (
            inspection_id INTEGER PRIMARY KEY AUTOINCREMENT,
            inspected_at DATETIME NOT NULL,
            assembly_line TEXT NOT NULL,
            part_serial_number TEXT UNIQUE NOT NULL,
            defect_detected INTEGER NOT NULL,
            defect_type TEXT,
            confidence REAL,
            bounding_box TEXT,
            segmentation_mask_polygon TEXT
        );
        """)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS maintenance_logs (
            log_id INTEGER PRIMARY KEY AUTOINCREMENT,
            log_date DATE NOT NULL,
            machine_id TEXT NOT NULL,
            technician TEXT NOT NULL,
            work_performed TEXT NOT NULL,
            parts_replaced TEXT NOT NULL,
            status TEXT NOT NULL
        );
        """)
        conn.commit()

    print("Schemas for Telemetry, Vision, and Maintenance Logs verified.")

    # Populate Initial Maintenance Logs (Historical Context for LLM Agent)
    cursor.execute("SELECT COUNT(*) FROM maintenance_logs;")
    if cursor.fetchone()[0] == 0:
        print("Seeding baseline maintenance work orders...")
        m_logs = [
            ("2026-03-15", "CNC-MILL-04", "Marcus Vance", 
             "Replaced worn spindle belt. Bearings showed minor frictional wear but approved for 200 more operating hours.", 
             "Spindle Belt #B-402", "COMPLETED"),
            ("2026-03-22", "CNC-MILL-04", "Marcus Vance", 
             "Topped off cutting fluid reservoir. Noticed slight coolant nozzle calcification; partially cleaned.", 
             "Coolant Additive Synth-5", "COMPLETED"),
            ("2026-03-27", "LINE-A-CONVEYOR", "Sarah Chen", 
             "Recalibrated optical inspection strobe illumination and tensioned conveyor motor drive.", 
             "Drive Roller Tensioner", "COMPLETED")
        ]
        for log in m_logs:
            if engine_type == "POSTGRES":
                cursor.execute("""
                INSERT INTO maintenance_logs (log_date, machine_id, technician, work_performed, parts_replaced, status)
                VALUES (%s, %s, %s, %s, %s, %s);
                """, log)
            else:
                cursor.execute("""
                INSERT INTO maintenance_logs (log_date, machine_id, technician, work_performed, parts_replaced, status)
                VALUES (?, ?, ?, ?, ?, ?);
                """, log)
        if engine_type == "SQLITE":
            conn.commit()
        print("Maintenance history populated.")

    # Populate High-Frequency Telemetry with Line Speeds
    cursor.execute("SELECT COUNT(*) FROM machine_telemetry;")
    if cursor.fetchone()[0] == 0:
        print("Migrating high-frequency telemetry with line speed indicators...")
        source_csv = os.path.join("data", "machine_telemetry.csv")
        if os.path.exists(source_csv):
            df = pd.read_csv(source_csv)
            for idx, row in df.iterrows():
                # Line speed in meters per minute (normally 45 m/min; drops if line slows down)
                line_speed = 45.0 + (float(row["rpm"]) - 3000.0) * 0.02
                t2_val = float(row.get("hotelling_t2", 1.5))
                mspc_val = int(row.get("mspc_alert", 0))

                vals = (
                    str(row["timestamp"]), "CNC-MILL-04", round(line_speed, 2),
                    round(float(row["temperature_c"]), 2), round(float(row["vibration_mms"]), 3),
                    round(float(row["rpm"]), 1), round(t2_val, 2), mspc_val
                )
                if engine_type == "POSTGRES":
                    cursor.execute("""
                    INSERT INTO machine_telemetry 
                    (timestamp, machine_id, line_speed_mpm, spindle_temp_c, vibration_mms, rpm, hotelling_t2, mspc_alert)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
                    """, vals)
                else:
                    cursor.execute("""
                    INSERT INTO machine_telemetry 
                    (timestamp, machine_id, line_speed_mpm, spindle_temp_c, vibration_mms, rpm, hotelling_t2, mspc_alert)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                    """, vals)
            if engine_type == "SQLITE":
                conn.commit()
            print(f"Migrated {len(df)} telemetry records with line speeds.")

    conn.close()
    print("Database Layer (Sprint 1) fully operational.\n")

if __name__ == "__main__":
    initialize_schema()