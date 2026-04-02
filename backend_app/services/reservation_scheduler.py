"""
Unified Reservation & Notification Scheduler
Gom notification_scheduler.py va reservation_scheduler.py thanh mot file.
Chay background thread, kiem tra moi 30s:
- 15 phut / 5 phut / tai arrival -> thong bao + Socket.IO emit
- Qua arrival + 30 phut -> auto-cancel + thong bao
- VIP slot dat -> broadcast notification + Socket.IO
"""
import threading
import time
from datetime import datetime, timezone, timedelta, date
from database.db import get_db
from database.models import SlotReservation, Notification, ParkingSlot, GateLog

# Module-level socketio instance (set by backend_app/app.py)
_scheduler_socketio = None
_scheduler_thread = None
_scheduler_running = False


# ============================================================
# Helpers
# ============================================================

def _user_has_entered(plate_text: str, booking_date: date, arrival_time) -> bool:
    """Kiem tra xe da vao cong (IN) trong khoang 2h sau arrival."""
    db = get_db()
    try:
        start_dt = datetime.combine(booking_date, arrival_time) if arrival_time else datetime.combine(booking_date, datetime.min.time())
        end_dt = start_dt + timedelta(hours=2)
        found = (
            db.query(GateLog)
            .filter(GateLog.plate_text == plate_text)
            .filter(GateLog.direction == 'IN')
            .filter(GateLog.timestamp >= start_dt)
            .filter(GateLog.timestamp <= end_dt)
            .first()
        )
        return found is not None
    finally:
        db.close()


def _emit(socketio, event: str, data: dict):
    """Emit Socket.IO event neu socketio duoc cau hinh."""
    if socketio is not None:
        try:
            socketio.emit(event, data)
        except Exception as e:
            print(f"[Scheduler] Socket.IO emit error ({event}): {e}")


# ============================================================
# Notification: 15 phut / 5 phut / arrival time reminder
# ============================================================

def _check_booking_reminders(socketio=None):
    """Sắp đến giờ: gui notification + Socket.IO event."""
    db = get_db()
    try:
        now = datetime.now()
        today = now.date()
        now_time = now.time()
        window_end = (datetime.combine(today, now_time) + timedelta(minutes=15)).time()

        rows = (
            db.query(SlotReservation, ParkingSlot)
            .join(ParkingSlot, SlotReservation.slot_id == ParkingSlot.slot_id)
            .filter(SlotReservation.booking_date == today)
            .filter(SlotReservation.status == 'confirmed')
            .all()
        )

        for r, slot in rows:
            arr = r.arrival_time or r.time_from
            if not (now_time <= arr <= window_end):
                continue
            if _user_has_entered(r.plate_text, today, arr):
                continue

            # 15 min reminder
            if not db.query(Notification).filter_by(
                type='booking_reminder',
                related_id=r.reservation_id,
            ).first():
                n = Notification(
                    user_id=r.user_id,
                    title='Sắp đến giờ đặt chỗ',
                    body=f'Bạn đặt Slot {slot.slot_number} lúc {arr.strftime("%H:%M")}. Vui lòng đến bãi xe sớm.',
                    type='booking_reminder',
                    related_id=r.reservation_id,
                )
                db.add(n)
                db.commit()
                _emit(socketio, 'booking_reminder', {
                    'reservation_id': r.reservation_id,
                    'slot_number': slot.slot_number,
                    'time_from': r.time_from.strftime('%H:%M') if r.time_from else None,
                    'arrival_time': arr.strftime('%H:%M'),
                })
                print(f"[Scheduler] 15min reminder: reservation {r.reservation_id}")

    except Exception as e:
        db.rollback()
        print(f"[Scheduler] _check_booking_reminders error: {e}")
    finally:
        db.close()


def _check_booking_arrival(socketio=None):
    """Tai arrival_time: notification + Socket.IO."""
    db = get_db()
    try:
        now = datetime.now()
        today = now.date()
        now_time = now.time()
        window_start = (datetime.combine(today, now_time) - timedelta(minutes=2)).time()

        rows = (
            db.query(SlotReservation, ParkingSlot)
            .join(ParkingSlot, SlotReservation.slot_id == ParkingSlot.slot_id)
            .filter(SlotReservation.booking_date == today)
            .filter(SlotReservation.status == 'confirmed')
            .all()
        )

        for r, slot in rows:
            arr = r.arrival_time or r.time_from
            if not (window_start <= arr <= now_time):
                continue
            if not db.query(Notification).filter_by(
                type='booking_started',
                related_id=r.reservation_id,
            ).first():
                n = Notification(
                    user_id=r.user_id,
                    title='Đã đến giờ đặt chỗ',
                    body=f'Đến giờ đặt Slot {slot.slot_number} ({arr.strftime("%H:%M")}). Vui lòng nhanh chóng đến bãi xe.',
                    type='booking_started',
                    related_id=r.reservation_id,
                )
                db.add(n)
                db.commit()
                _emit(socketio, 'booking_started', {
                    'reservation_id': r.reservation_id,
                    'slot_number': slot.slot_number,
                    'arrival_time': arr.strftime('%H:%M'),
                })
                print(f"[Scheduler] Arrival: reservation {r.reservation_id}")

    except Exception as e:
        db.rollback()
        print(f"[Scheduler] _check_booking_arrival error: {e}")
    finally:
        db.close()


def _check_booking_overdue(socketio=None):
    """Qua arrival + 30 phut: notification + Socket.IO."""
    db = get_db()
    try:
        now = datetime.now()
        today = now.date()
        now_time = now.time()

        rows = (
            db.query(SlotReservation, ParkingSlot)
            .join(ParkingSlot, SlotReservation.slot_id == ParkingSlot.slot_id)
            .filter(SlotReservation.booking_date == today)
            .filter(SlotReservation.status == 'confirmed')
            .all()
        )

        for r, slot in rows:
            arr = r.arrival_time or r.time_from
            if arr > now_time:
                continue
            if _user_has_entered(r.plate_text, today, arr):
                continue
            if not db.query(Notification).filter_by(
                type='booking_overdue',
                related_id=r.reservation_id,
            ).first():
                n = Notification(
                    user_id=r.user_id,
                    title='Đã quá giờ đặt chỗ',
                    body=f'Bạn đặt Slot {slot.slot_number} lúc {arr.strftime("%H:%M")} nhưng chưa vào bãi. Vui lòng đến sớm hoặc hủy đặt chỗ.',
                    type='booking_overdue',
                    related_id=r.reservation_id,
                )
                db.add(n)
                db.commit()
                _emit(socketio, 'booking_overdue', {
                    'reservation_id': r.reservation_id,
                    'slot_number': slot.slot_number,
                    'arrival_time': arr.strftime('%H:%M'),
                })
                print(f"[Scheduler] Overdue: reservation {r.reservation_id}")

    except Exception as e:
        db.rollback()
        print(f"[Scheduler] _check_booking_overdue error: {e}")
    finally:
        db.close()


def _auto_cancel_expired(socketio=None):
    """Auto-cancel reservations past time_to + 30 min grace period."""
    db = get_db()
    try:
        now = datetime.now(timezone.utc)
        today = now.date()
        grace = timedelta(minutes=30)

        expired = db.query(SlotReservation).filter(
            SlotReservation.booking_date == today,
            SlotReservation.status == 'confirmed',
        ).all()

        cancelled = []
        for r in expired:
            if not r.time_to:
                continue
            end_dt = datetime.combine(today, r.time_to, tzinfo=timezone.utc)
            grace_end = end_dt + grace
            if now >= grace_end:
                r.status = 'cancelled'
                r.updated_at = datetime.now(timezone.utc)
                cancelled.append(r.reservation_id)
                _emit(socketio, 'reservation_auto_cancelled', {
                    'reservation_id': r.reservation_id,
                    'slot_id': r.slot_id,
                })

        if cancelled:
            db.commit()
            print(f"[Scheduler] Auto-cancelled: {cancelled}")

    except Exception as e:
        db.rollback()
        print(f"[Scheduler] _auto_cancel_expired error: {e}")
    finally:
        db.close()


# ============================================================
# Scheduler Loop
# ============================================================

def _scheduler_loop(socketio=None):
    """Main loop - runs every 30 seconds."""
    global _scheduler_running
    print("[Scheduler] Unified reservation/notification scheduler started")

    while _scheduler_running:
        try:
            # Run all checks
            _check_booking_reminders(socketio)
            _check_booking_arrival(socketio)
            _check_booking_overdue(socketio)
            _auto_cancel_expired(socketio)
        except Exception as e:
            print(f"[Scheduler] Loop error: {e}")

        # Sleep 30 seconds
        time.sleep(30)

    print("[Scheduler] Unified scheduler stopped")


def start_scheduler(socketio=None):
    """Start the unified scheduler in a background thread."""
    global _scheduler_thread, _scheduler_running, _scheduler_socketio

    if _scheduler_running:
        print("[Scheduler] Already running")
        return

    _scheduler_socketio = socketio
    _scheduler_running = True
    _scheduler_thread = threading.Thread(
        target=_scheduler_loop,
        args=(socketio,),
        daemon=True,
        name="ReservationScheduler",
    )
    _scheduler_thread.start()
    print("[Scheduler] Unified scheduler thread started")


def stop_scheduler():
    """Stop the unified scheduler."""
    global _scheduler_running

    if not _scheduler_running:
        return

    print("[Scheduler] Stopping...")
    _scheduler_running = False
    if _scheduler_thread:
        _scheduler_thread.join(timeout=5)
    print("[Scheduler] Stopped")
