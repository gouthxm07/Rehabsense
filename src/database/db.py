"""
SQLite persistence layer — matches the schema in blueprint section 28
(patients, exercises, sessions, alerts).
"""

import sqlite3
from pathlib import Path
from datetime import datetime

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "sessions" / "rehabsense.db"


def get_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS patients (
            patient_id TEXT PRIMARY KEY,
            name_or_alias TEXT,
            age_group TEXT,
            created_at TEXT
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT,
            exercise_name TEXT,
            date TEXT,
            duration REAL,
            total_reps INTEGER,
            correct_reps INTEGER,
            incorrect_reps INTEGER,
            rom REAL,
            movement_score REAL,
            rehab_score REAL,
            FOREIGN KEY (patient_id) REFERENCES patients (patient_id)
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            alert_id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER,
            alert_type TEXT,
            description TEXT,
            severity TEXT,
            review_status TEXT DEFAULT 'pending',
            FOREIGN KEY (session_id) REFERENCES sessions (session_id)
        )
    """)

    conn.commit()
    conn.close()


def ensure_patient(patient_id, name_or_alias=None, age_group=None):
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT 1 FROM patients WHERE patient_id = ?", (patient_id,))
    if not c.fetchone():
        c.execute(
            "INSERT INTO patients (patient_id, name_or_alias, age_group, created_at) VALUES (?, ?, ?, ?)",
            (patient_id, name_or_alias or patient_id, age_group or "unspecified", datetime.now().isoformat()),
        )
        conn.commit()
    conn.close()


def save_session(patient_id, exercise_name, duration, total_reps, correct_reps,
                  incorrect_reps, rom, movement_score, rehab_score):
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        INSERT INTO sessions
        (patient_id, exercise_name, date, duration, total_reps, correct_reps,
         incorrect_reps, rom, movement_score, rehab_score)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (patient_id, exercise_name, datetime.now().isoformat(), duration,
          total_reps, correct_reps, incorrect_reps, rom, movement_score, rehab_score))
    session_id = c.lastrowid
    conn.commit()
    conn.close()
    return session_id


def add_alert(session_id, alert_type, description, severity="info"):
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        INSERT INTO alerts (session_id, alert_type, description, severity)
        VALUES (?, ?, ?, ?)
    """, (session_id, alert_type, description, severity))
    conn.commit()
    conn.close()


def get_sessions_for_patient(patient_id):
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM sessions WHERE patient_id = ? ORDER BY date ASC", (patient_id,))
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows


def get_all_patients():
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM patients ORDER BY created_at DESC")
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows


def get_alerts_for_patient(patient_id):
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        SELECT alerts.*, sessions.exercise_name, sessions.date
        FROM alerts
        JOIN sessions ON alerts.session_id = sessions.session_id
        WHERE sessions.patient_id = ?
        ORDER BY sessions.date DESC
    """, (patient_id,))
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows


def check_for_alerts(patient_id, session_id, current_rom, current_score):
    """Very simple significant-change detector, per blueprint section 17."""
    history = get_sessions_for_patient(patient_id)
    previous_sessions = [s for s in history if s["session_id"] != session_id]
    if not previous_sessions:
        return
    last = previous_sessions[-1]

    if last["rom"] and current_rom < last["rom"] * 0.7:
        add_alert(
            session_id, "rom_drop",
            f"ROM dropped from {last['rom']}° to {current_rom}° vs previous session.",
            severity="review",
        )
    if last["rehab_score"] and current_score < last["rehab_score"] - 15:
        add_alert(
            session_id, "score_drop",
            f"Rehabilitation score dropped from {last['rehab_score']} to {current_score}.",
            severity="review",
        )
