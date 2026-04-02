"""
Database operations for Smart Parking System.
Thread-safe helper functions for inserting/updating records.
Each function opens its own session and commits independently.
"""

from datetime import datetime, timezone, timedelta

from database.db import get_db
from database.models import Vehicle, GateLog, ParkingSession, ParkingSlot, ImproperParkingLog
from database.models import User, UserVehicle, Notification, SlotReservation


def log_gate_entry(plate_text: str, confidence: float, image_path: str = None, result_store: dict = None):
    """
    Log vehicle ENTERING the parking lot through the gate.
    Creates: GateLog(IN) + Vehicle (if new) + ParkingSession(active).

    Args:
        plate_text: License plate string (may be None if OCR failed)
        confidence: OCR confidence score
        image_path: Optional path to gate capture image
        result_store: Optional dict to store {'session_id', 'gate_log_id'} after commit
                      (used to update plate later if OCR improves)

    Returns:
        session_id (UUID) of the new ParkingSession, or None on failure
    """
    db = get_db()
    try:
        now = datetime.now(timezone.utc)

        # 1. Upsert Vehicle (only if plate known)
        vehicle = None
        if plate_text:
            vehicle = db.query(Vehicle).filter_by(plate_text=plate_text).first()
            if vehicle is None:
                vehicle = Vehicle(plate_text=plate_text, first_seen=now, last_seen=now)
                db.add(vehicle)
                db.flush()  # Get vehicle_uuid before using it
                print(f"[DB] New vehicle created: {plate_text}")
            else:
                vehicle.last_seen = now

        # 2. Insert GateLog (append-only)
        gate_log = GateLog(
            timestamp=now,
            plate_text=plate_text or None,
            direction='IN',
            confidence=confidence,
            image_path=image_path,
        )
        db.add(gate_log)

        # 3. Create ParkingSession
        session = ParkingSession(
            vehicle_uuid=vehicle.vehicle_uuid if vehicle else None,
            entry_time=now,
            entry_plate_conf=confidence,
            entry_gate_image_path=image_path,
            status='active',
        )
        db.add(session)

        db.commit()
        print(f"[DB] Gate IN logged: {plate_text or '(no plate)'}")

        # Populate result_store so caller can update plate later if OCR improves
        if result_store is not None:
            result_store['gate_log_id'] = gate_log.log_id
            result_store['session_id'] = str(session.session_id)

        return session.session_id

    except Exception as e:
        db.rollback()
        print(f"[DB ERROR] log_gate_entry failed: {e}")
        return None
    finally:
        db.close()


def update_gate_entry_media(gate_log_id: int, session_id: str, image_path: str):
    """
    Update IN gate log / session images when plate text is not yet available (OCR backfill).
    """
    if not gate_log_id or not image_path:
        return
    db = get_db()
    try:
        gate_log = db.query(GateLog).filter_by(log_id=gate_log_id).first()
        if gate_log and gate_log.direction == 'IN':
            gate_log.image_path = image_path
        if session_id:
            import uuid as _uuid
            try:
                sid = _uuid.UUID(session_id)
            except ValueError:
                sid = None
            if sid:
                session = db.query(ParkingSession).filter_by(session_id=sid).first()
                if session:
                    session.entry_gate_image_path = image_path
        db.commit()
        print(f"[DB] Gate entry image updated (log_id={gate_log_id})")
    except Exception as e:
        db.rollback()
        print(f"[DB ERROR] update_gate_entry_media failed: {e}")
    finally:
        db.close()


def update_gate_entry_plate(gate_log_id: int, session_id: str, new_plate: str, new_conf: float, image_path: str = None):
    """
    Update plate_text on a GateLog(IN) and its linked ParkingSession when OCR
    result improves after the initial entry was logged.

    Args:
        gate_log_id: GateLog.log_id to update
        session_id:  ParkingSession.session_id (UUID string) to update
        new_plate:   Improved license plate text
        new_conf:    Improved OCR confidence
        image_path:  Improved plate image crop path
    """
    if not gate_log_id:
        return
    if not new_plate and not image_path:
        return
    if not new_plate:
        update_gate_entry_media(gate_log_id, session_id, image_path)
        return
    db = get_db()
    try:
        # Update GateLog
        gate_log = db.query(GateLog).filter_by(log_id=gate_log_id).first()
        if gate_log and gate_log.direction == 'IN':
            old_plate = gate_log.plate_text
            gate_log.plate_text = new_plate
            gate_log.confidence = new_conf
            if image_path:
                gate_log.image_path = image_path

        # Update/move ParkingSession vehicle link
        if session_id:
            import uuid as _uuid
            try:
                sid = _uuid.UUID(session_id)
            except ValueError:
                sid = None
            if sid:
                session = db.query(ParkingSession).filter_by(session_id=sid).first()
                if session:
                    # Upsert vehicle with correct plate
                    now = datetime.now(timezone.utc)
                    vehicle = db.query(Vehicle).filter_by(plate_text=new_plate).first()
                    if vehicle is None:
                        vehicle = Vehicle(plate_text=new_plate, first_seen=now, last_seen=now)
                        db.add(vehicle)
                        db.flush()
                        print(f"[DB] New vehicle created (OCR update): {new_plate}")
                    else:
                        vehicle.last_seen = now
                    session.vehicle_uuid = vehicle.vehicle_uuid
                    session.entry_plate_conf = new_conf
                    if image_path:
                        session.entry_gate_image_path = image_path

        db.commit()
        print(f"[DB] Gate entry plate updated: {old_plate!r} → {new_plate!r} (log_id={gate_log_id})")

    except Exception as e:
        db.rollback()
        print(f"[DB ERROR] update_gate_entry_plate failed: {e}")
    finally:
        db.close()


def log_gate_exit(plate_text: str, confidence: float, image_path: str = None, result_store: dict = None):
    """
    Log vehicle EXITING the parking lot through the gate.
    Creates: GateLog(OUT) + completes matching ParkingSession.

    Args:
        plate_text: License plate string (may be None if OCR failed)
        confidence: OCR confidence score
        image_path: Optional path to gate capture image
        result_store: Optional dict to store {'gate_log_id'} after commit

    Returns:
        True if session was completed, False otherwise
    """
    db = get_db()
    try:
        now = datetime.now(timezone.utc)

        # --- Gate OUT de-dup (plate-based, short window) ---
        # Tracking/crossing jitter có thể tạo nhiều lệnh EXIT cho cùng một xe
        # trong thời gian ngắn (thường do track_id bị đổi).
        # Nếu DB đã có GateLog(OUT) cho cùng plate trong một cửa sổ nhỏ,
        # bỏ qua lần ghi tiếp theo để tránh "ghi record liên tục".
        dedup_exit_window_seconds = 10
        if plate_text:
            cutoff_ts = now - timedelta(seconds=dedup_exit_window_seconds)
            recent_out = (
                db.query(GateLog)
                .filter(GateLog.direction == 'OUT')
                .filter(GateLog.plate_text == plate_text)
                .filter(GateLog.timestamp >= cutoff_ts)
                .order_by(GateLog.timestamp.desc())
                .first()
            )
            if recent_out is not None:
                print(
                    f"[DB] log_gate_exit dedup skipped for {plate_text}: "
                    f"recent OUT at {recent_out.timestamp}"
                )
                db.commit()
                return False

        # 1. Update Vehicle last_seen
        vehicle = None
        if plate_text:
            vehicle = db.query(Vehicle).filter_by(plate_text=plate_text).first()
            if vehicle:
                vehicle.last_seen = now

        # 2. Find active ParkingSession and decide if EXIT is plausible
        completed = False
        active_session = None
        min_exit_delay_seconds = 30  # Guard: xe vừa vào không thể ra ngay trong vài giây

        if vehicle:
            active_session = (
                db.query(ParkingSession)
                .filter_by(vehicle_uuid=vehicle.vehicle_uuid, status='active')
                .order_by(ParkingSession.entry_time.desc())
                .first()
            )

        # If there is an active session, enforce a minimum dwell time before allowing EXIT.
        if active_session and active_session.entry_time:
            delta_seconds = (now - active_session.entry_time).total_seconds()
            if delta_seconds < min_exit_delay_seconds:
                print(
                    f"[DB] log_gate_exit skipped for {plate_text}: "
                    f"only {delta_seconds:.1f}s since entry (< {min_exit_delay_seconds}s)"
                )
                db.commit()
                return False

        # 3. Insert GateLog (append-only) for plausible EXIT
        gate_log = GateLog(
            timestamp=now,
            plate_text=plate_text or None,
            direction='OUT',
            confidence=confidence,
            image_path=image_path,
        )
        db.add(gate_log)

        # 4. Complete active session if present
        if active_session:
            active_session.exit_time = now
            active_session.exit_plate_conf = confidence
            active_session.exit_gate_image_path = image_path
            active_session.status = 'completed'
            # Calculate duration
            if active_session.entry_time:
                delta = now - active_session.entry_time
                active_session.duration_minutes = int(delta.total_seconds() / 60)
            completed = True
            print(f"[DB] Session completed: {plate_text} (duration={active_session.duration_minutes}min)")

        db.commit()
        print(f"[DB] Gate OUT logged: {plate_text or '(no plate)'} (conf={confidence:.2f})")
        
        if result_store is not None:
            result_store['gate_log_id'] = gate_log.log_id

        return completed

    except Exception as e:
        db.rollback()
        print(f"[DB ERROR] log_gate_exit failed: {e}")
        return False
    finally:
        db.close()


def update_gate_exit_plate(gate_log_id: int, new_plate: str, new_conf: float, image_path: str = None):
    """
    Update plate_text on a GateLog(OUT) and complete its linked ParkingSession when OCR
    result finishes/improves after the initial exit was logged.
    """
    if not gate_log_id:
        return
    db = get_db()
    try:
        now = datetime.now(timezone.utc)
        # Update GateLog
        gate_log = db.query(GateLog).filter_by(log_id=gate_log_id).first()
        old_plate = None
        if gate_log and gate_log.direction == 'OUT':
            old_plate = gate_log.plate_text
            if new_plate:
                gate_log.plate_text = new_plate
                gate_log.confidence = new_conf
            if image_path:
                gate_log.image_path = image_path

        final_plate = new_plate or old_plate
        if not final_plate:
            db.commit()
            print(f"[DB] Gate OUT image updated (No Plate): gate_log_id={gate_log_id}")
            return

        # Find or create vehicle
        vehicle = db.query(Vehicle).filter_by(plate_text=final_plate).first()
        if vehicle is None:
            vehicle = Vehicle(plate_text=new_plate, first_seen=now, last_seen=now)
            db.add(vehicle)
            db.flush()
        else:
            vehicle.last_seen = now

        # Check for active session to complete
        active_session = (
            db.query(ParkingSession)
            .filter_by(vehicle_uuid=vehicle.vehicle_uuid, status='active')
            .order_by(ParkingSession.entry_time.desc())
            .first()
        )
        if active_session and gate_log:
            min_exit_delay_seconds = 30
            delta_seconds = (gate_log.timestamp - active_session.entry_time).total_seconds() if active_session.entry_time else 999
            if delta_seconds >= min_exit_delay_seconds:
                active_session.exit_time = gate_log.timestamp
                active_session.exit_plate_conf = new_conf
                active_session.exit_gate_image_path = image_path or gate_log.image_path
                active_session.status = 'completed'
                if active_session.entry_time:
                    active_session.duration_minutes = int((gate_log.timestamp - active_session.entry_time).total_seconds() / 60)
                print(f"[DB] Session completed via OCR update: {new_plate} (duration={active_session.duration_minutes}min)")

        db.commit()
        print(f"[DB] Gate exit plate updated: {old_plate!r} → {new_plate!r} (log_id={gate_log_id})")

    except Exception as e:
        db.rollback()
        print(f"[DB ERROR] update_gate_exit_plate failed: {e}")
    finally:
        db.close()


def update_vehicle_slot(plate_text: str, slot_number: int):
    """
    Record that a vehicle has parked in a specific slot.
    Updates: ParkingSession.assigned_slot_id + ParkingSlot.status='occupied'.

    Args:
        plate_text: License plate of the vehicle
        slot_number: Slot number (1-19)
    """
    if not plate_text:
        return

    db = get_db()
    try:
        now = datetime.now(timezone.utc)

        # Find vehicle
        vehicle = db.query(Vehicle).filter_by(plate_text=plate_text).first()
        if not vehicle:
            print(f"[DB] update_vehicle_slot: vehicle {plate_text} not found")
            return

        # Find active session
        active_session = (
            db.query(ParkingSession)
            .filter_by(vehicle_uuid=vehicle.vehicle_uuid, status='active')
            .order_by(ParkingSession.entry_time.desc())
            .first()
        )
        if active_session:
            # Find slot by slot_number
            slot = db.query(ParkingSlot).filter_by(slot_number=slot_number).first()
            if slot:
                active_session.assigned_slot_id = slot.slot_id
                active_session.parked_time = now
                slot.status = 'occupied'
                slot.last_updated = now
                db.commit()
                print(f"[DB] Vehicle {plate_text} parked in Slot {slot_number}")
            else:
                print(f"[DB] Slot {slot_number} not found in database")
        else:
            print(f"[DB] No active session for {plate_text}")

    except Exception as e:
        db.rollback()
        print(f"[DB ERROR] update_vehicle_slot failed: {e}")
    finally:
        db.close()


def update_vehicle_parked_outside(plate_text: str):
    """
    Mark a vehicle as parked outside any defined slot.
    Updates: ParkingSession.status = 'parked_outside'.

    Args:
        plate_text: License plate of the vehicle
    """
    if not plate_text:
        return

    db = get_db()
    try:
        now = datetime.now(timezone.utc)

        vehicle = db.query(Vehicle).filter_by(plate_text=plate_text).first()
        if not vehicle:
            return

        active_session = (
            db.query(ParkingSession)
            .filter_by(vehicle_uuid=vehicle.vehicle_uuid, status='active')
            .order_by(ParkingSession.entry_time.desc())
            .first()
        )
        if active_session:
            active_session.status = 'parked_outside'
            active_session.parked_time = now
            db.commit()
            print(f"[DB] Vehicle {plate_text} marked as parked_outside")

    except Exception as e:
        db.rollback()
        print(f"[DB ERROR] update_vehicle_parked_outside failed: {e}")
    finally:
        db.close()


def release_slot(slot_number: int):
    """
    Release a parking slot (set status back to 'free').

    Args:
        slot_number: Slot number (1-19)
    """
    db = get_db()
    try:
        now = datetime.now(timezone.utc)
        slot = db.query(ParkingSlot).filter_by(slot_number=slot_number).first()
        if slot and slot.status != 'free':
            slot.status = 'free'
            slot.last_updated = now
            db.commit()
            print(f"[DB] Slot {slot_number} released (free)")
    except Exception as e:
        db.rollback()
        print(f"[DB ERROR] release_slot failed: {e}")
    finally:
        db.close()


def log_improper_parking(plate_text: str, event_type: str, image_path: str = None, slot_number: int = None):
    """
    Log a vehicle parked improperly (outside any slot, or overlapping a slot boundary).

    Args:
        plate_text: License plate string (may be None if unknown)
        event_type: 'outside' or 'overlapping'
        image_path: Optional path to cropped vehicle image
        slot_number: Slot number when overlapping (1-19)
    """
    db = get_db()
    try:
        entry = ImproperParkingLog(
            plate_text=plate_text or None,
            event_type=event_type,
            image_path=image_path,
            slot_number=slot_number,
        )
        db.add(entry)
        db.commit()
        print(f"[DB] Improper parking logged: {event_type}, plate={plate_text or '(no plate)'}, slot={slot_number}")
    except Exception as e:
        db.rollback()
        print(f"[DB ERROR] log_improper_parking failed: {e}")
    finally:
        db.close()


def check_and_notify_slot_hijacked(slots_info: list):
    """
    For each occupied slot with plate: if there is an active reservation and plate != reserver's plate,
    create slot_hijacked notification for reserver and wrong_slot for the wrong parker.
    Avoids spam by checking for recent notification (15 min) with same related_id.
    """
    if not slots_info:
        return
    now = datetime.now(timezone.utc)
    today = now.date()
    now_time = now.time()
    spam_window = timedelta(minutes=15)
    db = get_db()
    try:
        for s in slots_info:
            if s.get('status') != 'occupied':
                continue
            plate_in_slot = (s.get('plate') or '').strip().upper()
            if not plate_in_slot:
                continue
            slot_number = s.get('slot_number')
            if not slot_number:
                continue
            slot = db.query(ParkingSlot).filter_by(slot_number=slot_number).first()
            if not slot:
                continue
            res = (
                db.query(SlotReservation)
                .filter_by(slot_id=slot.slot_id)
                .filter(SlotReservation.booking_date == today)
                .filter(SlotReservation.status.in_(['pending', 'confirmed']))
                .filter(SlotReservation.time_from <= now_time)
                .filter(SlotReservation.time_to >= now_time)
                .order_by(SlotReservation.created_at.desc())
                .first()
            )
            if not res:
                continue
            res_plate = (res.plate_text or '').strip().upper()
            if not res_plate or res_plate == plate_in_slot:
                continue
            recent = (
                db.query(Notification)
                .filter_by(type='slot_hijacked', related_id=res.reservation_id)
                .filter(Notification.created_at >= now - spam_window)
                .first()
            )
            if recent:
                continue
            n1 = Notification(
                user_id=res.user_id,
                title='Slot bị chiếm',
                body=f'Slot {slot_number} đã bị xe khác đậu trước khi bạn đến. Biển số trong slot: {plate_in_slot}.',
                type='slot_hijacked',
                related_id=res.reservation_id,
            )
            db.add(n1)
            wrong_user = db.query(User).filter(User.plate == plate_in_slot).first()
            if not wrong_user:
                wrong_user = (
                    db.query(User)
                    .join(UserVehicle, User.user_id == UserVehicle.user_id)
                    .filter(UserVehicle.plate_text == plate_in_slot)
                    .first()
                )
            if wrong_user:
                n2 = Notification(
                    user_id=wrong_user.user_id,
                    title='Bạn đã đậu sai slot',
                    body=f'Slot {slot_number} đã được người khác đặt trước. Vui lòng di chuyển xe.',
                    type='wrong_slot',
                    related_id=res.reservation_id,
                )
                db.add(n2)
            db.commit()
            print(f"[DB] Slot hijack notified: slot {slot_number}, reserver={res.user_id}, wrong_plate={plate_in_slot}")
    except Exception as e:
        db.rollback()
        print(f"[DB ERROR] check_and_notify_slot_hijacked failed: {e}")
    finally:
        db.close()
