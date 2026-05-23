"""
Cleanup Service for Smart Parking System
==========================================
Automatically removes old database records and orphaned image files.

Tables cleaned:
    - parking_sessions: completed sessions older than N days
    - gate_logs: entries older than N days
    - improper_parking_logs: entries older than N days
    - tracking_events: events belonging to deleted sessions

Image directories cleaned:
    - static/gate_captures/
    - static/parking_captures/

Usage:
    # Python import
    from services.cleanup_service import cleanup_old_records
    cleanup_old_records(days=7, dry_run=False)

    # CLI (after integrating into main.py)
    python main.py cleanup --days 7 --dry-run
    python main.py cleanup --days 7
"""

import os
import sys
import shutil
import logging
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from typing import Set, Optional

# ── Resolve base directory ────────────────────────────────────────────────────
_services_dir = os.path.dirname(os.path.abspath(__file__))
_BASE_DIR = os.path.dirname(os.path.dirname(_services_dir))
if _BASE_DIR not in sys.path:
    sys.path.insert(0, _BASE_DIR)

from config import BASE_DIR
from database.db import get_db
from database.models import ParkingSession, GateLog, ImproperParkingLog, TrackingEvent

# ── Paths ─────────────────────────────────────────────────────────────────────
GATE_CAPTURE_DIR = os.path.join(BASE_DIR, 'static', 'gate_captures')
PARKING_CAPTURE_DIR = os.path.join(BASE_DIR, 'static', 'parking_captures')

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
_log = logging.getLogger('cleanup')


# ── Results dataclass ──────────────────────────────────────────────────────────
@dataclass
class CleanupResult:
    """Holds the outcome of a single cleanup run."""
    dry_run: bool
    cutoff_date: datetime
    days: int

    # DB records
    sessions_deleted: int = 0
    gate_logs_deleted: int = 0
    improper_logs_deleted: int = 0
    tracking_events_deleted: int = 0

    # Image files
    images_deleted: int = 0
    images_freed_bytes: int = 0
    orphan_images_checked: int = 0

    # Active sessions skipped
    active_sessions_skipped: int = 0

    def _bytes_fmt(self, b: int) -> str:
        for unit in ('B', 'KB', 'MB', 'GB'):
            if b < 1024:
                return f"{b:.1f} {unit}"
            b /= 1024
        return f"{b:.1f} GB"

    def print_summary(self):
        mode = "DRY-RUN (no data was deleted)" if self.dry_run else "COMMITTED"
        print()
        print("=" * 60)
        print(f"  CLEANUP SUMMARY [{mode}]")
        print("=" * 60)
        print(f"  Cutoff date:  {self.cutoff_date.strftime('%Y-%m-%d %H:%M:%S')} UTC")
        print(f"  Retention:    {self.days} days")
        print()
        print("  [Database Records]")
        print(f"    parking_sessions:      {self.sessions_deleted:>6} deleted")
        print(f"    gate_logs:             {self.gate_logs_deleted:>6} deleted")
        print(f"    improper_parking_logs:  {self.improper_logs_deleted:>6} deleted")
        print(f"    tracking_events:        {self.tracking_events_deleted:>6} deleted")
        print(f"    active sessions kept:   {self.active_sessions_skipped:>6} skipped")
        print()
        print("  [Image Files]")
        print(f"    orphan files checked:  {self.orphan_images_checked:>6}")
        print(f"    files deleted:        {self.images_deleted:>6}")
        print(f"    space freed:          {self._bytes_fmt(self.images_freed_bytes):>10}")
        print("=" * 60)

    def has_work(self) -> bool:
        return (
            self.sessions_deleted > 0
            or self.gate_logs_deleted > 0
            or self.improper_logs_deleted > 0
            or self.tracking_events_deleted > 0
            or self.images_deleted > 0
        )


# ── Core cleanup logic ─────────────────────────────────────────────────────────
def cleanup_old_records(days: int = 7, dry_run: bool = False) -> CleanupResult:
    """
    Remove old parking records and orphaned image files.

    Args:
        days:      How many days of records to keep (older records are deleted).
        dry_run:   If True, only query and report — no actual deletion.

    Returns:
        CleanupResult with counts and statistics.
    """
    result = CleanupResult(dry_run=dry_run, cutoff_date=datetime.now(timezone.utc), days=days)
    cutoff = result.cutoff_date - timedelta(days=days)

    _log.info("Starting cleanup — retention=%d days, cutoff=%s, dry_run=%s",
              days, cutoff.strftime('%Y-%m-%d %H:%M:%S'), dry_run)

    # ── 1. Collect image paths that are still referenced in DB ────────────────
    referenced_paths: Set[str] = set()
    db_paths_collected = 0

    db = get_db()
    try:
        # parking_sessions — entry and exit images
        for row in db.query(
            ParkingSession.entry_gate_image_path,
            ParkingSession.exit_gate_image_path,
        ).all():
            if row.entry_gate_image_path:
                referenced_paths.add(row.entry_gate_image_path.lstrip('/'))
                db_paths_collected += 1
            if row.exit_gate_image_path:
                referenced_paths.add(row.exit_gate_image_path.lstrip('/'))
                db_paths_collected += 1

        # gate_logs
        for row in db.query(GateLog.image_path).filter(GateLog.image_path.isnot(None)).all():
            referenced_paths.add(row.image_path.lstrip('/'))
            db_paths_collected += 1

        # improper_parking_logs
        for row in db.query(ImproperParkingLog.image_path).filter(
            ImproperParkingLog.image_path.isnot(None)
        ).all():
            referenced_paths.add(row.image_path.lstrip('/'))
            db_paths_collected += 1

        _log.info("Collected %d referenced image paths from DB", db_paths_collected)

        # ── 2. Delete parking_sessions (completed/expired only) ───────────────
        old_sessions = db.query(ParkingSession).filter(
            ParkingSession.entry_time < cutoff,
            ParkingSession.status != 'active',
        ).all()
        result.sessions_deleted = len(old_sessions)

        if old_sessions:
            _log.info("Found %d parking_sessions to delete", result.sessions_deleted)
            if not dry_run:
                session_ids = [s.session_id for s in old_sessions]
                # Delete tracking_events first (FK)
                te_deleted = db.query(TrackingEvent).filter(
                    TrackingEvent.session_id.in_(session_ids)
                ).delete(synchronize_session=False)
                result.tracking_events_deleted = te_deleted
                _log.info("Deleted %d tracking_events", te_deleted)

                db.query(ParkingSession).filter(
                    ParkingSession.session_id.in_(session_ids)
                ).delete(synchronize_session=False)
                db.commit()

        # Active sessions older than cutoff (overnight) — keep them
        active_old = db.query(ParkingSession).filter(
            ParkingSession.entry_time < cutoff,
            ParkingSession.status == 'active',
        ).count()
        result.active_sessions_skipped = active_old
        if active_old:
            _log.info("Skipped %d active/overnight sessions", active_old)

        # ── 3. Delete old gate_logs ───────────────────────────────────────────
        gl_deleted = db.query(GateLog).filter(GateLog.timestamp < cutoff).delete(
            synchronize_session=False
        )
        result.gate_logs_deleted = gl_deleted
        _log.info("Marked %d gate_logs for deletion", gl_deleted)

        # ── 4. Delete old improper_parking_logs ─────────────────────────────────
        ipl_deleted = db.query(ImproperParkingLog).filter(
            ImproperParkingLog.timestamp < cutoff
        ).delete(synchronize_session=False)
        result.improper_logs_deleted = ipl_deleted
        _log.info("Marked %d improper_parking_logs for deletion", ipl_deleted)

        if not dry_run:
            db.commit()
            _log.info("DB deletions committed.")

        # ── 5. Delete orphaned image files ─────────────────────────────────────
        for capture_dir, dir_label in [
            (GATE_CAPTURE_DIR, 'gate_captures'),
            (PARKING_CAPTURE_DIR, 'parking_captures'),
        ]:
            if not os.path.isdir(capture_dir):
                _log.info("Directory %s does not exist — skipping", capture_dir)
                continue

            files_removed, bytes_freed = _delete_orphan_images(
                capture_dir, referenced_paths, dry_run
            )
            result.images_deleted += files_removed
            result.images_freed_bytes += bytes_freed
            result.orphan_images_checked += len([
                f for f in os.listdir(capture_dir)
                if os.path.isfile(os.path.join(capture_dir, f))
            ])

    except Exception as e:
        _log.error("Cleanup error — rolling back: %s", e)
        db.rollback()
        raise
    finally:
        db.close()

    result.print_summary()
    return result


def _delete_orphan_images(
    capture_dir: str,
    referenced_paths: Set[str],
    dry_run: bool,
) -> tuple[int, int]:
    """
    Delete image files in `capture_dir` that have no corresponding DB record.
    Returns (files_deleted, bytes_freed).
    """
    total_deleted = 0
    total_bytes = 0

    # Also add the full absolute paths (some records may be stored as /static/...)
    abs_referenced = {os.path.join(_BASE_DIR, p) for p in referenced_paths}
    rel_referenced = referenced_paths  # already stripped

    for filename in os.listdir(capture_dir):
        filepath = os.path.join(capture_dir, filename)
        if not os.path.isfile(filepath):
            continue

        # Skip non-image files
        if not any(filename.lower().endswith(ext) for ext in ('.jpg', '.jpeg', '.png', '.bmp')):
            continue

        # Check if this file is referenced
        # Try both relative and absolute forms
        is_orphan = (
            filename not in rel_referenced
            and filepath not in abs_referenced
            and f"/static/{capture_dir.split('static/')[-1]}/{filename}" not in referenced_paths
        )

        if is_orphan:
            fsize = os.path.getsize(filepath)
            if dry_run:
                _log.info("[DRY-RUN] Would delete: %s (%.1f KB)", filepath, fsize / 1024)
            else:
                os.remove(filepath)
                _log.info("Deleted orphan: %s (%.1f KB)", filepath, fsize / 1024)
            total_deleted += 1
            total_bytes += fsize

    _log.info("%s — removed %d orphan files (%.1f MB)",
              capture_dir, total_deleted, total_bytes / (1024 * 1024))
    return total_deleted, total_bytes


# ── Scheduled wrapper ─────────────────────────────────────────────────────────
def run_scheduled_cleanup(days: int = 7):
    """
    Called by the background scheduler.
    Silently logs errors so the scheduler thread doesn't crash.
    """
    try:
        _log.info("Scheduled cleanup triggered.")
        cleanup_old_records(days=days, dry_run=False)
    except Exception as e:
        _log.error("Scheduled cleanup failed: %s", e)


# ── CLI entry point (standalone script) ──────────────────────────────────────
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Cleanup old parking records and orphaned images.'
    )
    parser.add_argument(
        '--days', type=int, default=7,
        help='Number of days to retain (default: 7)'
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Show what would be deleted without actually deleting anything'
    )
    args = parser.parse_args()

    cleanup_old_records(days=args.days, dry_run=args.dry_run)
