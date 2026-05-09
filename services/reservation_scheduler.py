"""
DEPRECATED: This file is replaced by backend_app/services/reservation_scheduler.py.
The unified version uses DB-based deduplication instead of in-memory sets
and includes both notification scheduling and auto-cancellation.
DO NOT start this scheduler -- use the backend_app version instead.
"""

import threading
import time
from datetime import datetime, timezone, timedelta
from database.db import get_db
from database.models import SlotReservation, Notification, User

# Track sent notifications to avoid duplicates
_sent_notifications = set()  # Set of (reservation_id, notification_type)
_scheduler_thread = None
_scheduler_running = False


def _send_reminder_notification(db, reservation, user, minutes_before):
    """Send reminder notification X minutes before arrival time"""
    notif_type = f'reminder_{minutes_before}min'
    key = (reservation.reservation_id, notif_type)
    
    if key in _sent_notifications:
        return  # Already sent
    
    slot_name = f"Slot {reservation.slot.slot_number}"
    arrival_str = reservation.arrival_time.strftime('%H:%M') if reservation.arrival_time else ''
    
    notif = Notification(
        user_id=user.user_id,
        title=f'Nhắc nhở: {minutes_before} phút nữa đến giờ đặt chỗ',
        body=f'Bạn đã đặt {slot_name} lúc {arrival_str}. Vui lòng chuẩn bị đến bãi xe.',
        type='reminder',
        related_id=reservation.reservation_id,
    )
    db.add(notif)
    db.commit()
    _sent_notifications.add(key)
    print(f"[Scheduler] Sent {minutes_before}min reminder for reservation {reservation.reservation_id}")


def _send_overdue_notification(db, reservation, user):
    """Send notification when reservation is overdue (past time_to)"""
    notif_type = 'overdue'
    key = (reservation.reservation_id, notif_type)
    
    if key in _sent_notifications:
        return  # Already sent
    
    slot_name = f"Slot {reservation.slot.slot_number}"
    time_to_str = reservation.time_to.strftime('%H:%M') if reservation.time_to else ''
    
    notif = Notification(
        user_id=user.user_id,
        title='Đặt chỗ đã quá hạn',
        body=f'{slot_name} đã hết hạn lúc {time_to_str}. Vui lòng rời khỏi bãi xe.',
        type='overdue',
        related_id=reservation.reservation_id,
    )
    db.add(notif)
    db.commit()
    _sent_notifications.add(key)
    print(f"[Scheduler] Sent overdue notification for reservation {reservation.reservation_id}")


def _auto_cancel_expired_reservations(db):
    """Auto-cancel reservations that are past time_to + grace period (30 min)"""
    now = datetime.now(timezone.utc)
    today = now.date()
    now_time = now.time()
    grace_period = timedelta(minutes=30)
    
    # Find reservations that ended more than 30 minutes ago
    expired = db.query(SlotReservation).filter(
        SlotReservation.booking_date == today,
        SlotReservation.status.in_(['pending', 'confirmed']),
    ).all()
    
    for r in expired:
        if not r.time_to:
            continue
        
        # Calculate end time + grace period
        end_dt = datetime.combine(today, r.time_to, tzinfo=timezone.utc)
        grace_end = end_dt + grace_period
        
        if now >= grace_end:
            r.status = 'cancelled'
            print(f"[Scheduler] Auto-cancelled expired reservation {r.reservation_id}")
    
    db.commit()


def _scheduler_loop(socketio=None):
    """Main scheduler loop - runs every 30 seconds"""
    global _scheduler_running
    
    print("[Scheduler] Reservation scheduler started")
    
    while _scheduler_running:
        try:
            db = get_db()
            try:
                now = datetime.now(timezone.utc)
                today = now.date()
                now_time = now.time()
                
                # Get active reservations for today
                active_reservations = db.query(SlotReservation, User).join(
                    User, SlotReservation.user_id == User.user_id
                ).filter(
                    SlotReservation.booking_date == today,
                    SlotReservation.status.in_(['pending', 'confirmed']),
                ).all()
                
                for reservation, user in active_reservations:
                    if not reservation.arrival_time or not reservation.time_to:
                        continue
                    
                    # Calculate time until arrival
                    arrival_dt = datetime.combine(today, reservation.arrival_time, tzinfo=timezone.utc)
                    time_until_arrival = (arrival_dt - now).total_seconds() / 60  # minutes
                    
                    # Calculate time until end
                    end_dt = datetime.combine(today, reservation.time_to, tzinfo=timezone.utc)
                    time_until_end = (end_dt - now).total_seconds() / 60  # minutes
                    
                    # Send 15-minute reminder
                    if 14 <= time_until_arrival <= 16:
                        _send_reminder_notification(db, reservation, user, 15)
                    
                    # Send 5-minute reminder
                    elif 4 <= time_until_arrival <= 6:
                        _send_reminder_notification(db, reservation, user, 5)
                    
                    # Send overdue notification (5 minutes after time_to)
                    elif time_until_end < -4:
                        _send_overdue_notification(db, reservation, user)
                
                # Auto-cancel expired reservations
                _auto_cancel_expired_reservations(db)
                
                # Emit Socket.IO event with updated reservations (if socketio available)
                if socketio:
                    socketio.emit('reservations_updated', {'timestamp': now.isoformat()})
                
            finally:
                db.close()
        
        except Exception as e:
            print(f"[Scheduler] Error in scheduler loop: {e}")
        
        # Sleep for 30 seconds
        time.sleep(30)
    
    print("[Scheduler] Reservation scheduler stopped")


def start_scheduler(socketio=None):
    """Start the reservation scheduler in a background thread"""
    global _scheduler_thread, _scheduler_running
    
    if _scheduler_running:
        print("[Scheduler] Scheduler already running")
        return
    
    _scheduler_running = True
    _scheduler_thread = threading.Thread(
        target=_scheduler_loop,
        args=(socketio,),
        daemon=True,
        name="ReservationScheduler"
    )
    _scheduler_thread.start()
    print("[Scheduler] Reservation scheduler thread started")


def stop_scheduler():
    """Stop the reservation scheduler"""
    global _scheduler_running
    
    if not _scheduler_running:
        return
    
    print("[Scheduler] Stopping reservation scheduler...")
    _scheduler_running = False
    
    if _scheduler_thread:
        _scheduler_thread.join(timeout=5)
    
    print("[Scheduler] Reservation scheduler stopped")


def get_remaining_time(reservation):
    """
    Calculate remaining time for a reservation
    Returns dict with:
    - seconds_until_arrival: seconds until arrival_time
    - seconds_until_end: seconds until time_to
    - is_active: True if within arrival_time and time_to
    - is_overdue: True if past time_to
    """
    now = datetime.now(timezone.utc)
    today = now.date()
    
    if reservation.booking_date != today:
        return None
    
    if not reservation.arrival_time or not reservation.time_to:
        return None
    
    arrival_dt = datetime.combine(today, reservation.arrival_time, tzinfo=timezone.utc)
    end_dt = datetime.combine(today, reservation.time_to, tzinfo=timezone.utc)
    
    seconds_until_arrival = (arrival_dt - now).total_seconds()
    seconds_until_end = (end_dt - now).total_seconds()
    
    return {
        'seconds_until_arrival': int(seconds_until_arrival),
        'seconds_until_end': int(seconds_until_end),
        'is_active': seconds_until_arrival <= 0 and seconds_until_end > 0,
        'is_overdue': seconds_until_end < 0,
    }
