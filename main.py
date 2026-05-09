"""
Smart Parking System - Main Launcher
====================================
Central entry point for the Smart Parking monitoring system.

This file initializes the Flask application and starts all services:
- Parking detection camera
- Gate camera with OCR

- Real-time dashboard

Usage:
    python main.py
    python main.py cleanup --days 30 --dry-run
"""

import os
import threading
import click
from datetime import datetime, timedelta, timezone

# Set environment variable to allow duplicated libraries (must be before imports)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from flask import Flask
from flask_cors import CORS
from flask_socketio import SocketIO

from config import SERVER_HOST, SERVER_PORT, PARKING_VIDEO_URL, GATE_VIDEO_URL, PARKING_USE_HALF_PRECISION
from shared.models import check_gpu, initialize_model
from shared.state import current_parking_status, gate_ocr_results
from api.routes import register_routes
from database.db import init_db
from services.parking_detection import process_video_stream
from services.gate_camera import process_gate_video_stream
from services.notification_scheduler import _run_notification_scheduler
from services.reservation_scheduler import start_scheduler as start_reservation_scheduler
from services.cleanup_service import cleanup_old_records, run_scheduled_cleanup


# Ensure all DB tables exist (safe to call multiple times)
init_db()

# Initialize Flask app and SocketIO
app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'smart-parking-dev-secret-change-in-production')
# Session persistence: cookie survives browser restart, 7 days
app.config['PERMANENT_SESSION_LIFETIME'] = 60 * 60 * 24 * 7  # 7 days
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_COOKIE_HTTPONLY'] = True

# Allow all hosts / origins
CORS(app, resources={r"/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading', engineio_logger=False, logger=False)

# Register all routes
register_routes(app, socketio)


# ── Cleanup CLI command ────────────────────────────────────────────────────────
@app.cli.command('cleanup')
@click.option('--days', default=30, help='Number of days to retain (default: 30)')
@click.option('--dry-run', is_flag=True, help='Preview what would be deleted without deleting anything')
def cleanup_cmd(days, dry_run):
    """Remove database records older than N days and orphaned image files."""
    from flask import current_app
    with current_app.app_context():
        cleanup_old_records(days=days, dry_run=dry_run)


# Lazy model — initialized in background so camera sync isn't blocked by YOLO load
_parking_model = None
_parking_model_ready = threading.Event()
_parking_model_lock = threading.Lock()

def _init_parking_model(app, use_half):
    """Background thread: load YOLO model once, then unblocks camera startup."""
    with app.app_context():
        global _parking_model
        model = initialize_model(use_half=use_half)
        with _parking_model_lock:
            _parking_model = model
        _parking_model_ready.set()
        print("[BG] Parking YOLO model loaded.")


def _run_parking_camera(app, socketio):
    """Background thread: parking detection + slot status updates."""
    with app.app_context():
        _parking_model_ready.wait()  # Wait for model to finish loading
        with _parking_model_lock:
            model = _parking_model
        print("[BG] Parking camera processing started.")
        for _ in process_video_stream(model, PARKING_VIDEO_URL, socketio, current_parking_status):
            pass

def _run_gate_camera(app, socketio):
    """Background thread: gate camera OCR + gate log updates."""
    with app.app_context():
        print("[BG] Gate camera processing started.")
        for _ in process_gate_video_stream(GATE_VIDEO_URL, socketio, gate_ocr_results):
            pass


def _run_cleanup_scheduler(app, interval_hours=24, run_hour_utc=3):
    """
    Background thread: runs data cleanup once per day at a fixed UTC hour.

    On startup, computes how long to wait until the next run time,
    then re-schedules itself on every subsequent run.
    """
    import time
    while True:
        now_utc = datetime.now(timezone.utc)
        # Next run: today at run_hour_utc, or tomorrow if already passed
        next_run = now_utc.replace(hour=run_hour_utc, minute=0, second=0, microsecond=0)
        if next_run <= now_utc:
            next_run += timedelta(days=1)
        wait_seconds = (next_run - now_utc).total_seconds()
        print(f"[CLEANUP] Next scheduled run at {next_run.strftime('%Y-%m-%d %H:%M')} UTC (in {wait_seconds/3600:.1f}h)")
        time.sleep(wait_seconds)
        with app.app_context():
            run_scheduled_cleanup(days=30)
        # Loop immediately computes next time after each run


# ── Background service management ────────────────────────────────────────────────

_stop_events = []


def start_background_services():
    """Start all background threads. Call once after Flask/SocketIO init."""
    check_gpu()

    # Load YOLO model in background BEFORE camera threads start.
    # This ensures camera sync (60s timeout) isn't wasted waiting for model load.
    model_init_thread = threading.Thread(
        target=_init_parking_model,
        args=(app, PARKING_USE_HALF_PRECISION),
        daemon=True,
        name="ParkingModelInit",
    )
    model_init_thread.start()

    # Camera threads now start AFTER model thread; they wait for model via _parking_model_ready
    t1 = threading.Thread(target=_run_parking_camera, args=(app, socketio), daemon=True, name="ParkingBG")
    t2 = threading.Thread(target=_run_gate_camera, args=(app, socketio), daemon=True, name="GateBG")
    t3 = threading.Thread(target=_run_notification_scheduler, args=(app,), daemon=True, name="NotifyBG")
    t4 = threading.Thread(target=_run_cleanup_scheduler, args=(app,), daemon=True, name="CleanupBG")
    t1.start()
    t2.start()
    t3.start()
    t4.start()
    start_reservation_scheduler(socketio)


def stop_background_services():
    """Stop all background services gracefully. Call on shutdown."""
    from services.vehicle_tracking.tracker import get_tracker
    get_tracker().stop()
    for evt in _stop_events:
        evt.set()


if __name__ == "__main__":
    print("=" * 60)
    print("  Smart Parking System - Starting...")
    print("=" * 60)

    # Start all background services
    start_background_services()

    print(f"\n  Dashboard:         http://{SERVER_HOST}:{SERVER_PORT}")
    print(f"  Manager Dashboard: http://{SERVER_HOST}:{SERVER_PORT}/manager")
    print("=" * 60)

    # Start the server (cameras are already processing in background)
    # Luồng: (1) Camera detect+OCR chạy trong 2 daemon threads riêng
    #        (2) Flask xử lý API/HTTP trên main thread - không block bởi camera
    socketio.run(
        app,
        host=SERVER_HOST,
        port=SERVER_PORT,
        allow_unsafe_werkzeug=True,
        debug=False,
    )
