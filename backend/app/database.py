"""
Database configuration, SQLite schema migration helper, and session management
for Cotton Leaf AI Backend.

Supports local SQLite (default) and PostgreSQL via DATABASE_URL environment variable.
Ensures CVAT's internal Postgres database remains completely separate.
"""

import os
from pathlib import Path
from typing import Dict, Tuple
from sqlalchemy import create_engine, text
from sqlalchemy.orm import declarative_base, sessionmaker

from backend.app.config import PROJECT_ROOT, get_settings

DEFAULT_DB_PATH = PROJECT_ROOT / "cotton_leaf_ai.db"

DATABASE_URL = os.getenv("DATABASE_URL") or get_settings().resolved_database_url

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(
    DATABASE_URL,
    connect_args=connect_args,
    echo=False,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """Dependency that yields a database session and closes it cleanly afterwards."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _migrate_sqlite_columns() -> None:
    """
    Safely adds new columns to existing SQLite tables without touching or
    dropping existing user and prediction rows.
    """
    if not DATABASE_URL.startswith("sqlite"):
        return

    expected_columns: Dict[str, Tuple[Tuple[str, str], ...]] = {
        "users": (
            ("is_active", "BOOLEAN NOT NULL DEFAULT 1"),
        ),
        "prediction_results": (
            ("gradcam_heatmap_path", "VARCHAR(500)"),
            ("gradcam_overlay_path", "VARCHAR(500)"),
            ("gradcam_comparison_path", "VARCHAR(500)"),
            ("gradcam_metadata_path", "VARCHAR(500)"),
            ("all_probabilities_json", "TEXT"),
            ("latency_ms", "FLOAT"),
        ),
    }

    with engine.begin() as conn:
        for table_name, cols in expected_columns.items():
            table_check = conn.execute(
                text("SELECT name FROM sqlite_master WHERE type='table' AND name=:t"),
                {"t": table_name},
            ).fetchone()
            if not table_check:
                continue

            existing = {
                row[1]
                for row in conn.execute(text(f"PRAGMA table_info({table_name})")).fetchall()
            }
            for col_name, col_type in cols:
                if col_name not in existing:
                    conn.execute(text(f"ALTER TABLE {table_name} ADD COLUMN {col_name} {col_type}"))


def init_db() -> None:
    """Initializes database tables and applies additive schema migrations safely."""
    import backend.app.models  # noqa: F401 - Register ORM models
    Base.metadata.create_all(bind=engine)
    _migrate_sqlite_columns()
