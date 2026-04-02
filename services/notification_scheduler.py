"""
Scheduler: tạo thông báo "sắp đến giờ" và "quá giờ" cho đặt chỗ.
Chạy mỗi phút trong background thread.
"""

import time
from datetime import datetime, date, timedelta

from database.db import get_db
from database.models import SlotReservation, Notification, ParkingSlot, GateLog


def _run_notification_scheduler(app, interval_sec=60):
    """Chạy vòng lặp kiểm tra và tạo notification mỗi interval_sec giây."""
    with app.app_context():
        while True:
            try:
                _check_booking_reminders()
                _check_booking_overdue()
            except Exception as e:
                print(f"[NotificationScheduler] Error: {e}")
            time.sleep(interval_sec)


def _user_has_entered(plate_text: str, booking_date: date, arrival_time) -> bool:
    """Kiểm tra xe đã vào cổng (IN) trong ngày đặt chỗ chưa."""
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


def _check_booking_reminders():
    """Sắp đến giờ (15 phút): tạo notification nếu chưa vào bãi."""
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
            if db.query(Notification).filter_by(
                type='booking_reminder',
                related_id=r.reservation_id,
            ).first():
                continue

            n = Notification(
                user_id=r.user_id,
                title='Sắp đến giờ đặt chỗ',
                body=f'Bạn đặt Slot {slot.slot_number} lúc {arr.strftime("%H:%M")}. Vui lòng đến bãi xe sớm.',
                type='booking_reminder',
                related_id=r.reservation_id,
            )
            db.add(n)
            db.commit()
            print(f"[NotificationScheduler] Reminder: reservation {r.reservation_id}")
    except Exception as e:
        db.rollback()
        raise
    finally:
        db.close()


def _check_booking_overdue():
    """Quá giờ: tạo notification nếu chưa vào bãi."""
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
            if db.query(Notification).filter_by(
                type='booking_overdue',
                related_id=r.reservation_id,
            ).first():
                continue

            n = Notification(
                user_id=r.user_id,
                title='Đã quá giờ đặt chỗ',
                body=f'Bạn đặt Slot {slot.slot_number} lúc {arr.strftime("%H:%M")} nhưng chưa vào bãi. Vui lòng đến sớm hoặc hủy đặt chỗ.',
                type='booking_overdue',
                related_id=r.reservation_id,
            )
            db.add(n)
            db.commit()
            print(f"[NotificationScheduler] Overdue: reservation {r.reservation_id}")
    except Exception as e:
        db.rollback()
        raise
    finally:
        db.close()
