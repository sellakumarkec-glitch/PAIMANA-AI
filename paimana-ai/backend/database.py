"""SQLite persistence for authentication and project progress reports."""

import hashlib
import hmac
import secrets
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "paimana.db"

_USERS = {
    ("mospi", "mospi"): "MOSPI@2026",
    ("reporting_manager", "reporting_manager"): "REPORT@2026",
}


def _hash_password(password, salt):
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 120_000).hex()


def _connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def initialize():
    with _connection() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                role TEXT NOT NULL CHECK (role IN ('mospi', 'reporting_manager')),
                password_salt TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS progress_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id TEXT NOT NULL,
                project_name TEXT NOT NULL,
                submitted_by TEXT NOT NULL,
                role TEXT NOT NULL,
                report_month TEXT NOT NULL,
                physical_progress_pct REAL NOT NULL,
                financial_progress_pct REAL NOT NULL,
                notes TEXT NOT NULL DEFAULT '',
                submitted_at TEXT NOT NULL
            );
            """
        )
        for (username, role), password in _USERS.items():
            existing = connection.execute(
                "SELECT id FROM users WHERE username = ? AND role = ?",
                (username, role),
            ).fetchone()
            if existing:
                continue
            salt = secrets.token_bytes(16)
            connection.execute(
                "INSERT INTO users (username, role, password_salt, password_hash, created_at) VALUES (?, ?, ?, ?, ?)",
                (username, role, salt.hex(), _hash_password(password, salt), datetime.now(timezone.utc).isoformat()),
            )


def verify_user(username, password, role):
    with _connection() as connection:
        user = connection.execute(
            "SELECT username, role, password_salt, password_hash FROM users WHERE username = ? AND role = ?",
            (username.strip().lower(), role),
        ).fetchone()
    if not user:
        return False
    expected = _hash_password(password, bytes.fromhex(user["password_salt"]))
    return hmac.compare_digest(expected, user["password_hash"])


def create_progress_report(report):
    submitted_at = datetime.now(timezone.utc).isoformat()
    with _connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO progress_reports (
                project_id, project_name, submitted_by, role, report_month,
                physical_progress_pct, financial_progress_pct, notes, submitted_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                report["project_id"], report["project_name"], report["submitted_by"],
                report["role"], report["month"], report["physical_progress_pct"],
                report["financial_progress_pct"], report.get("notes", ""), submitted_at,
            ),
        )
        return {"id": cursor.lastrowid, "submitted_at": submitted_at}


def list_progress_reports(project_id):
    with _connection() as connection:
        rows = connection.execute(
            "SELECT * FROM progress_reports WHERE project_id = ? ORDER BY submitted_at DESC",
            (project_id,),
        ).fetchall()
    return [dict(row) for row in rows]
