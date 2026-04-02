"""
Database connection setup for Smart Parking System
Uses SQLAlchemy + psycopg2 to connect to PostgreSQL
"""

import os
import sys

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

# Add project root to path so config can be imported from any entry point
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from config import DATABASE_URL

# Create engine with connection pool
engine = create_engine(
    DATABASE_URL,
    pool_size=5,          # Max persistent connections
    max_overflow=10,      # Extra connections when pool full
    pool_pre_ping=True,   # Check connection health before use
    echo=False,           # Set True for SQL debug logging
)

# Session factory
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


def get_db() -> Session:
    """
    Get a database session.
    Usage:
        db = get_db()
        try:
            # ... do work ...
            db.commit()
        except:
            db.rollback()
            raise
        finally:
            db.close()
    """
    return SessionLocal()


def init_db():
    """
    Create all tables defined in models.py if they don't exist.
    Safe to call multiple times (uses CREATE TABLE IF NOT EXISTS).
    """
    from database.models import Base
    Base.metadata.create_all(bind=engine)
    print("[DB] All tables created/verified successfully.")


# ── Pool health monitoring ────────────────────────────────────────────────────────

_pool_warning_count = 0


def check_pool_health() -> bool:
    """
    Check PostgreSQL connection pool health.
    Returns True if pool is healthy (enough available connections),
    False if pool is near exhaustion.
    Logs a warning every ~60 calls when repeatedly low.
    """
    global _pool_warning_count
    size = engine.pool.size()          # pool_size (5)
    checked_in = engine.pool.checkedin()  # currently in use
    overflow = engine.pool.overflow()      # extra connections over pool_size

    # available = persistent slots not in use + max_overflow slack
    available = size + 10 - checked_in
    if available < 3:
        _pool_warning_count += 1
        if _pool_warning_count % 60 == 1:
            print(
                f"[DB POOL WARNING] Low connections: available={available}, "
                f"pool_size={size}, checked_in={checked_in}, overflow={overflow}"
            )
        return False
    return True


def get_pool_stats() -> dict:
    """Return current pool statistics for health endpoint."""
    return {
        "pool_size": engine.pool.size(),
        "checked_in": engine.pool.checkedin(),
        "overflow": engine.pool.overflow(),
        "available": engine.pool.size() + 10 - engine.pool.checkedin(),
        "healthy": check_pool_health(),
    }
