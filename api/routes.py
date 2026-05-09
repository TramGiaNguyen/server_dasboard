# API Routes for Smart Parking System
import os
import sys
import csv
import io
import time
from datetime import datetime, timezone, timedelta
from collections import defaultdict
from flask import Response, render_template, jsonify, redirect, url_for, request, session, stream_with_context
from werkzeug.security import check_password_hash, generate_password_hash
from sqlalchemy import func, extract, cast, case, Date, desc

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import shared.state as shared_state
from shared.state import current_parking_status, gate_ocr_results, vehicle_tracking_state
from services.vehicle_tracking.tracker import get_tracker
from database.db import get_db, check_pool_health, get_pool_stats
from database.models import GateLog, ParkingSlot, ParkingSession, Vehicle, ImproperParkingLog, User, SlotReservation
from auth.session import login_required, role_required, login_user, logout_user


MJPEG_SLEEP_SEC = 0.033  # ~30 FPS relay cadence


def _no_cache_headers():
    return {
        'Cache-Control': 'no-store, no-cache, must-revalidate, max-age=0',
        'Pragma': 'no-cache',
        'Expires': '0',
        'X-Accel-Buffering': 'no',  # useful behind nginx
    }


def _safe_attr(name, default=None):
    return getattr(shared_state, name, default)


def _build_stream_meta(kind='parking'):
    """
    Return lightweight metadata so frontend can correlate displayed MJPEG
    with backend render sequence / timestamp if needed.
    """
    if kind == 'parking':
        return {
            'render_seq': _safe_attr('parking_render_seq', current_parking_status.get('render_seq')),
            'render_ts': _safe_attr('parking_render_ts'),
            'last_update': current_parking_status.get('last_update'),
        }
    return {
        'render_seq': _safe_attr('gate_render_seq', gate_ocr_results.get('render_seq')),
        'render_ts': _safe_attr('gate_render_ts'),
        'last_update': gate_ocr_results.get('last_detection_time'),
    }


def _mjpeg_relay(frame_getter, sleep_sec=MJPEG_SLEEP_SEC):
    """
    Generic MJPEG relay that always sends the latest JPEG from shared state.
    """
    @stream_with_context
    def _relay():
        while True:
            frame = frame_getter()
            if frame:
                yield (
                    b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n'
                    b'Cache-Control: no-store\r\n\r\n' + frame + b'\r\n'
                )
            time.sleep(sleep_sec)
    return _relay()


def register_routes(app, socketio):
    """Register all Flask routes and SocketIO handlers"""

    @app.route('/login', methods=['GET', 'POST'])
    def login_page():
        if request.method == 'GET':
            if session.get('user_id'):
                if session.get('role') == 'manager':
                    return redirect(url_for('manager_dashboard'))
                return redirect(url_for('index'))  # guard, staff
            return render_template('login.html')

        username = (request.form.get('username') or '').strip()
        password = request.form.get('password') or ''
        next_url = request.form.get('next') or url_for('index')

        if not username or not password:
            return render_template('login.html', error='Vui lòng nhập username và mật khẩu')

        db = get_db()
        try:
            user = db.query(User).filter_by(username=username).first()
            if not user or not check_password_hash(user.password_hash, password):
                return render_template('login.html', error='Tên đăng nhập hoặc mật khẩu không đúng')
            if user.role == 'student':
                return render_template('login.html', error='Tài khoản sinh viên không được truy cập web dashboard. Dùng app mobile.')
            login_user(user)
            if user.role == 'manager':
                return redirect(url_for('manager_dashboard'))
            return redirect(url_for('index'))
        finally:
            db.close()

    @app.route('/logout')
    def auth_logout():
        logout_user()
        return redirect(url_for('login_page'))

    @app.route('/')
    @login_required
    @role_required('guard', 'manager', 'staff')
    def index():
        return render_template('Dashboard.html')

    @app.route('/manager')
    @login_required
    @role_required('manager')
    def manager_dashboard():
        return render_template('Manager_dashboard.html')

    @app.route('/video_feed')
    def video_feed():
        """MJPEG relay: reads latest frame from background parking processing thread."""

        def _get_frame():
            with shared_state.parking_jpeg_lock:
                return shared_state.parking_latest_jpeg

        return Response(
            _mjpeg_relay(_get_frame),
            mimetype='multipart/x-mixed-replace; boundary=frame',
            headers=_no_cache_headers()
        )

    @app.route('/video_feed_gate')
    def video_feed_gate():
        """MJPEG relay: reads latest frame from background gate processing thread."""

        def _get_frame():
            with shared_state.gate_jpeg_lock:
                return shared_state.gate_latest_jpeg

        return Response(
            _mjpeg_relay(_get_frame),
            mimetype='multipart/x-mixed-replace; boundary=frame',
            headers=_no_cache_headers()
        )

    @app.route('/api/stream/parking/meta')
    def get_parking_stream_meta():
        return jsonify(_build_stream_meta('parking'))

    @app.route('/api/stream/gate/meta')
    def get_gate_stream_meta():
        return jsonify(_build_stream_meta('gate'))

    @app.route('/api/parking/status')
    def get_parking_status():
        payload = dict(current_parking_status)
        payload['render_seq'] = _safe_attr('parking_render_seq', payload.get('render_seq'))
        payload['render_ts'] = _safe_attr('parking_render_ts')
        return jsonify(payload)

    def _merge_slots_with_reservation_vip():
        """Merge live slots (from detection) with reservation + VIP from DB."""
        now = datetime.now(timezone.utc)
        today = now.date()
        now_time = now.time()
        raw_slots = current_parking_status.get('slots', [])

        db = get_db()
        try:
            slots_db = db.query(ParkingSlot).order_by(ParkingSlot.slot_number).all()
            vip_slot_numbers = {s.slot_number for s in slots_db if getattr(s, 'is_vip', False)}

            active_reservations = (
                db.query(SlotReservation, User.full_name)
                .join(User, SlotReservation.user_id == User.user_id)
                .join(ParkingSlot, SlotReservation.slot_id == ParkingSlot.slot_id)
                .filter(SlotReservation.booking_date == today)
                .filter(SlotReservation.status.in_(['pending', 'confirmed']))
                .filter(SlotReservation.time_from <= now_time)
                .filter(SlotReservation.time_to >= now_time)
                .all()
            )

            res_by_slot = {}
            for r, full_name in active_reservations:
                res_by_slot[r.slot.slot_number] = {
                    'user_full_name': full_name or '',
                    'plate_text': r.plate_text,
                    'time_from': r.time_from.strftime('%H:%M') if r.time_from else None,
                    'time_to': r.time_to.strftime('%H:%M') if r.time_to else None,
                    'arrival_time': r.arrival_time.strftime('%H:%M') if r.arrival_time else None,
                    'reservation_id': r.reservation_id,
                }
        finally:
            db.close()

        merged = []
        for s in raw_slots:
            slot_num = s.get('slot_number')
            status = s.get('status', 'available')
            plate = s.get('plate')
            reservation = res_by_slot.get(slot_num)
            is_vip = slot_num in vip_slot_numbers
            is_hijacked = False

            if reservation and status == 'occupied' and plate:
                res_plate = (reservation.get('plate_text') or '').strip().upper()
                slot_plate = (plate or '').strip().upper()
                if res_plate and slot_plate and res_plate != slot_plate:
                    is_hijacked = True

            merged.append({
                **s,
                'is_vip': is_vip,
                'reservation': reservation,
                'is_hijacked': is_hijacked,
            })
        return merged

    @app.route('/api/parking/slots')
    def get_all_slots():
        merged = _merge_slots_with_reservation_vip()
        return jsonify({
            'slots': merged,
            'last_update': current_parking_status.get('last_update'),
            'render_seq': _safe_attr('parking_render_seq', current_parking_status.get('render_seq')),
            'render_ts': _safe_attr('parking_render_ts'),
        })

    @app.route('/api/parking/slots/1-18')
    def get_slots_1_to_18():
        merged = _merge_slots_with_reservation_vip()
        slots_1_to_18 = [slot for slot in merged if slot['slot_number'] <= 18]
        occupied_count = sum(1 for slot in slots_1_to_18 if slot['status'] == 'occupied')
        available_count = len(slots_1_to_18) - occupied_count

        return jsonify({
            'total_slots': len(slots_1_to_18),
            'occupied': occupied_count,
            'available': available_count,
            'slots': slots_1_to_18,
            'last_update': current_parking_status.get('last_update'),
            'render_seq': _safe_attr('parking_render_seq', current_parking_status.get('render_seq')),
            'render_ts': _safe_attr('parking_render_ts'),
        })

    @app.route('/api/parking/slot/<int:slot_number>')
    def get_single_slot(slot_number):
        merged = _merge_slots_with_reservation_vip()
        if slot_number < 1 or slot_number > len(merged):
            return jsonify({'error': 'Invalid slot number'}), 404

        slot = next((s for s in merged if s['slot_number'] == slot_number), None)
        if slot:
            return jsonify({
                'slot': slot,
                'last_update': current_parking_status.get('last_update'),
                'render_seq': _safe_attr('parking_render_seq', current_parking_status.get('render_seq')),
                'render_ts': _safe_attr('parking_render_ts'),
            })
        return jsonify({'error': 'Slot not found'}), 404

    @app.route('/api/gate/ocr')
    def get_gate_ocr():
        """Get latest OCR results from gate camera"""
        payload = dict(gate_ocr_results)
        payload['render_seq'] = _safe_attr('gate_render_seq', payload.get('render_seq'))
        payload['render_ts'] = _safe_attr('gate_render_ts')
        return jsonify(payload)

    # ============ Vehicle Tracking API Endpoints ============

    @app.route('/api/tracking/status')
    def get_tracking_status():
        """Get current tracking status and statistics"""
        tracker = get_tracker()
        stats = tracker.get_stats()
        return jsonify({
            **stats,
            **vehicle_tracking_state
        })

    @app.route('/api/tracking/pending')
    def get_tracking_pending():
        """Get list of vehicles waiting to be matched at parking"""
        tracker = get_tracker()
        return jsonify({
            'pending_vehicles': tracker.get_pending_vehicles(),
            'count': tracker.get_pending_count()
        })

    @app.route('/api/tracking/matched')
    def get_tracking_matched():
        """Get recent matched vehicles"""
        tracker = get_tracker()
        return jsonify({
            'matched_vehicles': tracker.get_matched_vehicles(limit=50),
            'total': tracker.get_stats()['total_matched']
        })

    @app.route('/api/tracking/stats')
    def get_tracking_stats():
        """Get tracking statistics"""
        tracker = get_tracker()
        return jsonify(tracker.get_stats())

    @socketio.on('connect')
    def handle_connect():
        print('Client connected')

    @socketio.on('request_tracking_status')
    def handle_tracking_status_request():
        """Handle request for tracking status via WebSocket"""
        tracker = get_tracker()
        socketio.emit('tracking_status', tracker.get_stats())

    # ============ Database Query API Endpoints ============

    @app.route('/api/db/gate-logs')
    def get_db_gate_logs():
        """Get recent gate logs from PostgreSQL"""
        db = get_db()
        try:
            logs = (
                db.query(GateLog)
                .order_by(GateLog.timestamp.desc())
                .limit(10)
                .all()
            )
            return jsonify([{
                'log_id': log.log_id,
                'timestamp': log.timestamp.isoformat() if log.timestamp else None,
                'plate_text': log.plate_text or '---',
                'direction': log.direction,
                'image_path': log.image_path or None,
            } for log in logs])
        except Exception as e:
            return jsonify({'error': str(e)}), 500
        finally:
            db.close()

    @app.route('/api/db/improper-parking-logs')
    def get_improper_parking_logs():
        """Get recent improper parking logs from PostgreSQL"""
        db = get_db()
        try:
            logs = (
                db.query(ImproperParkingLog)
                .order_by(ImproperParkingLog.timestamp.desc())
                .limit(20)
                .all()
            )
            return jsonify([{
                'log_id': l.log_id,
                'timestamp': l.timestamp.isoformat() if l.timestamp else None,
                'plate_text': l.plate_text or '---',
                'event_type': l.event_type,
                'image_path': l.image_path or None,
            } for l in logs])
        except Exception as e:
            return jsonify({'error': str(e)}), 500
        finally:
            db.close()

    @app.route('/api/db/slot-status')
    def get_db_slot_status():
        """Get all 19 parking slots with current vehicle info from PostgreSQL"""
        db = get_db()
        try:
            slots = (
                db.query(ParkingSlot)
                .order_by(ParkingSlot.slot_number)
                .all()
            )
            result = []
            for slot in slots:
                vehicle_plate = None
                session_status = None
                active_session = (
                    db.query(ParkingSession)
                    .filter_by(assigned_slot_id=slot.slot_id)
                    .filter(ParkingSession.status.in_(['active', 'parked_outside']))
                    .order_by(ParkingSession.entry_time.desc())
                    .first()
                )
                if active_session and active_session.vehicle:
                    vehicle_plate = active_session.vehicle.plate_text
                    session_status = active_session.status

                result.append({
                    'slot_number': slot.slot_number,
                    'slot_name': slot.slot_name,
                    'status': slot.status,
                    'vehicle_plate': vehicle_plate,
                    'session_status': session_status,
                })
            return jsonify(result)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
        finally:
            db.close()

    # ============ Manager API Endpoints (full data, no LIMIT) ============

    @app.route('/api/manager/gate-logs')
    def get_manager_gate_logs():
        """All gate logs for Manager dashboard (no limit)"""
        db = get_db()
        try:
            logs = (
                db.query(GateLog)
                .order_by(GateLog.timestamp.desc())
                .all()
            )
            return jsonify([{
                'log_id': log.log_id,
                'timestamp': log.timestamp.isoformat() if log.timestamp else None,
                'plate_text': log.plate_text or '---',
                'direction': log.direction,
                'image_path': log.image_path or None,
            } for log in logs])
        except Exception as e:
            return jsonify({'error': str(e)}), 500
        finally:
            db.close()

    @app.route('/api/manager/vip-slots', methods=['GET'])
    @login_required
    @role_required('manager')
    def get_manager_vip_slots():
        """Get list of slot_numbers marked as VIP."""
        db = get_db()
        try:
            slots = db.query(ParkingSlot).filter(ParkingSlot.is_vip == True).order_by(ParkingSlot.slot_number).all()
            return jsonify({'slot_numbers': [s.slot_number for s in slots]})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
        finally:
            db.close()
    @app.route('/api/manager/improper-parking-logs')
    def get_manager_improper_parking_logs():
        """All improper parking logs for Manager dashboard (no limit)"""
        db = get_db()
        try:
            logs = (
                db.query(ImproperParkingLog)
                .order_by(ImproperParkingLog.timestamp.desc())
                .all()
            )
            return jsonify([{
                'log_id': l.log_id,
                'timestamp': l.timestamp.isoformat() if l.timestamp else None,
                'plate_text': l.plate_text or '---',
                'event_type': l.event_type,
                'image_path': l.image_path or None,
                'slot_number': l.slot_number,
            } for l in logs])
        except Exception as e:
            return jsonify({'error': str(e)}), 500
        finally:
            db.close()

    @app.route('/api/manager/hourly-traffic')
    def get_hourly_traffic():
        """Count IN/OUT gate events per hour for today"""
        db = get_db()
        try:
            today = datetime.now().date()
            rows = (
                db.query(
                    extract('hour', GateLog.timestamp).label('hour'),
                    GateLog.direction,
                    func.count().label('cnt')
                )
                .filter(cast(GateLog.timestamp, Date) == today)
                .group_by('hour', GateLog.direction)
                .all()
            )
            hours_in = [0] * 24
            hours_out = [0] * 24
            for hour, direction, cnt in rows:
                h = int(hour)
                if direction == 'IN':
                    hours_in[h] = cnt
                else:
                    hours_out[h] = cnt
            return jsonify({
                'labels': [f'{h:02d}:00' for h in range(24)],
                'in': hours_in,
                'out': hours_out
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
        finally:
            db.close()

    # ============ Manager Statistics Endpoints ============

    @app.route('/api/manager/slot-frequency')
    def get_slot_frequency():
        """Slot usage frequency for current week and month"""
        db = get_db()
        try:
            today = datetime.now().date()
            week_start = today - timedelta(days=today.weekday())
            month_start = today.replace(day=1)

            def _aggregate(start_date):
                rows = (
                    db.query(ParkingSlot.slot_number, Vehicle.plate_text)
                    .join(ParkingSession, ParkingSession.assigned_slot_id == ParkingSlot.slot_id)
                    .outerjoin(Vehicle, Vehicle.vehicle_uuid == ParkingSession.vehicle_uuid)
                    .filter(ParkingSession.assigned_slot_id.isnot(None))
                    .filter(cast(ParkingSession.entry_time, Date) >= start_date)
                    .all()
                )
                buckets = defaultdict(lambda: {'count': 0, 'plates': set(), 'no_plate_count': 0})
                for slot_num, plate in rows:
                    buckets[slot_num]['count'] += 1
                    if plate and plate != '---':
                        buckets[slot_num]['plates'].add(plate)
                    else:
                        buckets[slot_num]['no_plate_count'] += 1
                return sorted([
                    {
                        'slot_number': sn,
                        'count': info['count'],
                        'plates': list(info['plates']),
                        'no_plate_count': info['no_plate_count'],
                    }
                    for sn, info in buckets.items()
                ], key=lambda x: x['count'], reverse=True)

            return jsonify({
                'week': _aggregate(week_start),
                'month': _aggregate(month_start)
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
        finally:
            db.close()

    @app.route('/api/manager/frequent-violators')
    def get_frequent_violators():
        """Vehicles with >5 improper parking events (including unrecognised plates)"""
        db = get_db()
        try:
            plate_label = case(
                (ImproperParkingLog.plate_text.is_(None), 'Không nhận dạng'),
                (ImproperParkingLog.plate_text == '', 'Không nhận dạng'),
                (ImproperParkingLog.plate_text == '---', 'Không nhận dạng'),
                else_=ImproperParkingLog.plate_text,
            )
            plate_counts = (
                db.query(
                    plate_label.label('plate_label'),
                    func.count().label('cnt'),
                )
                .group_by(plate_label)
                .having(func.count() > 5)
                .order_by(desc('cnt'))
                .all()
            )
            result = []
            for label, count in plate_counts:
                if label == 'Không nhận dạng':
                    events_q = db.query(ImproperParkingLog).filter(
                        (ImproperParkingLog.plate_text.is_(None))
                        | (ImproperParkingLog.plate_text == '')
                        | (ImproperParkingLog.plate_text == '---')
                    )
                else:
                    events_q = db.query(ImproperParkingLog).filter(
                        ImproperParkingLog.plate_text == label
                    )
                events = (
                    events_q
                    .order_by(ImproperParkingLog.timestamp.desc())
                    .limit(3)
                    .all()
                )
                result.append({
                    'plate_text': label,
                    'count': count,
                    'events': [{
                        'timestamp': e.timestamp.isoformat() if e.timestamp else None,
                        'event_type': e.event_type,
                        'image_path': e.image_path,
                    } for e in events],
                })
            return jsonify(result)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
        finally:
            db.close()

    @app.route('/api/manager/low-availability-hours')
    def get_low_availability_hours():
        """Average available slots per hour over the current week and month."""
        db = get_db()
        try:
            today = datetime.now().date()
            week_start = today - timedelta(days=today.weekday())
            month_start = today.replace(day=1)
            total = current_parking_status.get('total_spaces', 19) or 19

            def _compute(start_date):
                rows = (
                    db.query(
                        cast(GateLog.timestamp, Date).label('day'),
                        extract('hour', GateLog.timestamp).label('hour'),
                        GateLog.direction,
                        func.count().label('cnt')
                    )
                    .filter(cast(GateLog.timestamp, Date) >= start_date)
                    .filter(cast(GateLog.timestamp, Date) <= today)
                    .group_by('day', 'hour', GateLog.direction)
                    .all()
                )

                day_data = defaultdict(lambda: defaultdict(lambda: {'in': 0, 'out': 0}))
                for day, hour, direction, cnt in rows:
                    h = int(hour)
                    if direction == 'IN':
                        day_data[day][h]['in'] = cnt
                    else:
                        day_data[day][h]['out'] = cnt

                if not day_data:
                    return [{'hour': f'{h:02d}:00', 'avg_available': total} for h in range(24)], []

                sum_available = [0.0] * 24
                num_days = len(day_data)
                for day, hourly in day_data.items():
                    cumulative = 0
                    for h in range(24):
                        cumulative += hourly[h]['in'] - hourly[h]['out']
                        cumulative = max(0, min(cumulative, total))
                        sum_available[h] += max(0, total - cumulative)

                all_hours = []
                flagged = []
                for h in range(24):
                    avg = round(sum_available[h] / num_days, 1)
                    entry = {'hour': f'{h:02d}:00', 'avg_available': avg}
                    all_hours.append(entry)
                    if avg < 3:
                        flagged.append(entry)
                return all_hours, flagged

            week_hours, week_flagged = _compute(week_start)
            month_hours, month_flagged = _compute(month_start)

            return jsonify({
                'total_spaces': total,
                'week': {'hours': week_hours, 'flagged_hours': week_flagged},
                'month': {'hours': month_hours, 'flagged_hours': month_flagged}
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
        finally:
            db.close()

    # ============ CSV Export Endpoints ============

    def _make_csv_response(header, rows, filename):
        buf = io.StringIO()
        buf.write('\ufeff')
        writer = csv.writer(buf)
        writer.writerow(header)
        writer.writerows(rows)
        return Response(
            buf.getvalue(),
            mimetype='text/csv; charset=utf-8',
            headers={'Content-Disposition': f'attachment; filename="{filename}"'}
        )

    @app.route('/api/export/gate-logs.csv')
    def export_gate_logs_csv():
        db = get_db()
        try:
            logs = db.query(GateLog).order_by(GateLog.timestamp.desc()).all()
            header = ['Thoi_gian', 'Bien_so', 'Huong', 'Anh']
            rows = []
            for log in logs:
                ts = log.timestamp.strftime('%d/%m/%Y %H:%M:%S') if log.timestamp else ''
                direction = 'VÃ o' if log.direction == 'IN' else 'Ra'
                rows.append([ts, log.plate_text or '', direction, log.image_path or ''])
            return _make_csv_response(header, rows, 'gate_logs.csv')
        except Exception as e:
            return jsonify({'error': str(e)}), 500
        finally:
            db.close()

    @app.route('/api/export/improper-parking-logs.csv')
    def export_improper_parking_logs_csv():
        db = get_db()
        try:
            logs = db.query(ImproperParkingLog).order_by(ImproperParkingLog.timestamp.desc()).all()
            header = ['Thoi_gian', 'Bien_so', 'Loai', 'Anh']
            rows = []
            for l in logs:
                ts = l.timestamp.strftime('%d/%m/%Y %H:%M:%S') if l.timestamp else ''
                event = 'NgoÃ i slot' if l.event_type == 'outside' else 'Lấn ô'
                rows.append([ts, l.plate_text or '', event, l.image_path or ''])
            return _make_csv_response(header, rows, 'improper_parking_logs.csv')
        except Exception as e:
            return jsonify({'error': str(e)}), 500
        finally:
            db.close()

    @app.route('/api/export/slot-status.csv')
    def export_slot_status_csv():
        slots = current_parking_status.get('slots', [])
        header = ['Slot', 'Trang_thai', 'Bien_so']
        rows = []
        for s in slots:
            status = 'Có xe' if s.get('status') == 'occupied' else 'Trá»‘ng'
            plate = s.get('plate') or ''
            rows.append([f"Slot {s.get('slot_number', '')}", status, plate])
        return _make_csv_response(header, rows, 'slot_status.csv')

    @app.route('/api/export/frequent-violators.csv')
    def export_frequent_violators_csv():
        db = get_db()
        try:
            plate_label = case(
                (ImproperParkingLog.plate_text.is_(None), 'Không nhận dạng'),
                (ImproperParkingLog.plate_text == '', 'Không nhận dạng'),
                (ImproperParkingLog.plate_text == '---', 'Không nhận dạng'),
                else_=ImproperParkingLog.plate_text,
            )
            plate_counts = (
                db.query(
                    plate_label.label('plate_label'),
                    func.count().label('cnt'),
                )
                .group_by(plate_label)
                .having(func.count() > 5)
                .order_by(desc('cnt'))
                .all()
            )
            header = ['Bien_so', 'So_lan', 'Loai_gan_nhat', 'Anh_gan_nhat']
            rows = []
            for label, count in plate_counts:
                if label == 'Không nhận dạng':
                    latest = (
                        db.query(ImproperParkingLog)
                        .filter(
                            (ImproperParkingLog.plate_text.is_(None))
                            | (ImproperParkingLog.plate_text == '')
                            | (ImproperParkingLog.plate_text == '---')
                        )
                        .order_by(ImproperParkingLog.timestamp.desc())
                        .first()
                    )
                else:
                    latest = (
                        db.query(ImproperParkingLog)
                        .filter(ImproperParkingLog.plate_text == label)
                        .order_by(ImproperParkingLog.timestamp.desc())
                        .first()
                    )
                event = ''
                img = ''
                if latest:
                    event = 'NgoÃ i slot' if latest.event_type == 'outside' else 'Lấn ô'
                    img = latest.image_path or ''
                rows.append([label, count, event, img])
            return _make_csv_response(header, rows, 'frequent_violators.csv')
        except Exception as e:
            return jsonify({'error': str(e)}), 500
        finally:
            db.close()

    # â”€â”€ System Health â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


    # ============================================================
    # Manager: User Management
    # ============================================================

    @app.route('/api/manager/users', methods=['GET'])
    @login_required
    @role_required('manager')
    def get_manager_users():
        """Get all users with pagination and search."""
        page = max(1, int(request.args.get('page', 1)))
        limit = min(50, max(1, int(request.args.get('limit', 20))))
        offset = (page - 1) * limit
        search = (request.args.get('search') or '').strip()
        role_filter = request.args.get('role')

        db = get_db()
        try:
            q = db.query(User)
            if search:
                q = q.filter(
                    User.full_name.ilike(f'%{search}%') |
                    User.username.ilike(f'%{search}%') |
                    User.plate.ilike(f'%{search}%') |
                    User.email.ilike(f'%{search}%')
                )
            if role_filter:
                q = q.filter(User.role == role_filter)
            total = q.count()
            users = q.order_by(User.created_at.desc()).offset(offset).limit(limit).all()
            return jsonify({
                'items': [{
                    'user_id': u.user_id,
                    'username': u.username,
                    'full_name': u.full_name or '',
                    'email': u.email or '',
                    'phone': u.phone or '',
                    'plate': u.plate or '',
                    'role': u.role,
                    'created_at': u.created_at.isoformat() if u.created_at else None,
                } for u in users],
                'total': total, 'page': page, 'limit': limit,
            })
        finally:
            db.close()

    @app.route('/api/manager/users', methods=['POST'])
    @login_required
    @role_required('manager')
    def create_manager_user():
        """Create a new user (manual entry)."""
        data = request.get_json() or {}
        required = ['username', 'password', 'role']
        for f in required:
            if not data.get(f):
                return jsonify({'error': f'{f} is required'}), 400
        if data['role'] not in ('student', 'guard', 'staff', 'manager'):
            return jsonify({'error': 'Invalid role'}), 400
        db = get_db()
        try:
            existing = db.query(User).filter_by(username=data['username']).first()
            if existing:
                return jsonify({'error': 'Username da ton tai'}), 409
            u = User(
                username=data['username'],
                password_hash=generate_password_hash(data['password']),
                role=data['role'],
                full_name=data.get('full_name') or None,
                email=data.get('email') or None,
                phone=data.get('phone') or None,
                plate=data.get('plate', '').strip().upper()[:20] or None,
            )
            db.add(u)
            db.commit()
            return jsonify({
                'user_id': u.user_id,
                'username': u.username,
                'full_name': u.full_name or '',
                'email': u.email or '',
                'phone': u.phone or '',
                'plate': u.plate or '',
                'role': u.role,
                'created_at': u.created_at.isoformat() if u.created_at else None,
            }), 201
        except Exception as e:
            db.rollback()
            return jsonify({'error': str(e)}), 400
        finally:
            db.close()

    @app.route('/api/manager/users/import-csv', methods=['POST'])
    @login_required
    @role_required('manager')
    def import_users_csv():
        """Bulk import users from CSV file."""
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        file = request.files['file']
        if not file.filename.endswith('.csv'):
            return jsonify({'error': 'File phai la .csv'}), 400
        stream = io.StringIO(file.stream.read().decode('utf-8-sig'))
        reader = csv.reader(stream)
        header = next(reader, None)
        if not header:
            return jsonify({'error': 'Empty CSV file'}), 400
        required_cols = {'username', 'password', 'role'}
        col_names = {h.lower().strip(): i for i, h in enumerate(header)}
        if not required_cols.issubset(col_names.keys()):
            return jsonify({'error': 'CSV must have columns: username,password,role[,full_name,email,phone,plate]'}), 400
        db = get_db()
        created, skipped = [], []
        try:
            for row_num, row in enumerate(reader, start=2):
                if not any(c.strip() for c in row):
                    continue
                try:
                    username = row[col_names['username']].strip()
                    password = row[col_names['password']].strip()
                    role = row[col_names['role']].strip()
                    if role not in ('student', 'guard', 'staff', 'manager'):
                        skipped.append({'row': row_num, 'reason': f'Invalid role: {role}'})
                        continue
                    if db.query(User).filter_by(username=username).first():
                        skipped.append({'row': row_num, 'reason': f'Username "{username}" da ton tai'})
                        continue
                    u = User(
                        username=username,
                        password_hash=generate_password_hash(password),
                        role=role,
                        full_name=row[col_names.get('full_name', -1)].strip() or None if col_names.get('full_name', -1) >= 0 else None,
                        email=row[col_names.get('email', -1)].strip() or None if col_names.get('email', -1) >= 0 else None,
                        phone=row[col_names.get('phone', -1)].strip() or None if col_names.get('phone', -1) >= 0 else None,
                        plate=row[col_names.get('plate', -1)].strip().upper()[:20] or None if col_names.get('plate', -1) >= 0 else None,
                    )
                    db.add(u)
                    created.append(username)
                except Exception as e:
                    skipped.append({'row': row_num, 'reason': str(e)})
            db.commit()
            return jsonify({'created': len(created), 'usernames': created, 'skipped': skipped})
        except Exception as e:
            db.rollback()
            return jsonify({'error': str(e)}), 400
        finally:
            db.close()

    @app.route('/api/manager/users/<int:uid>', methods=['PUT'])
    @login_required
    @role_required('manager')
    def update_manager_user(uid):
        """Update an existing user."""
        data = request.get_json() or {}
        db = get_db()
        try:
            u = db.query(User).filter_by(user_id=uid).first()
            if not u:
                return jsonify({'error': 'User not found'}), 404
            if 'full_name' in data:
                u.full_name = data['full_name'][:100] or None
            if 'email' in data:
                u.email = data['email'][:100] or None
            if 'phone' in data:
                u.phone = data['phone'][:20] or None
            if 'plate' in data:
                u.plate = data['plate'].strip().upper()[:20] or None
            if 'role' in data and data['role'] in ('student', 'guard', 'staff', 'manager'):
                u.role = data['role']
            if data.get('password'):
                u.password_hash = generate_password_hash(data['password'])
            u.updated_at = datetime.now(timezone.utc)
            db.commit()
            return jsonify({'ok': True})
        except Exception as e:
            db.rollback()
            return jsonify({'error': str(e)}), 400
        finally:
            db.close()

    @app.route('/api/manager/users/<int:uid>', methods=['DELETE'])
    @login_required
    @role_required('manager')
    def delete_manager_user(uid):
        """Delete a user."""
        if uid == session.get('user_id'):
            return jsonify({'error': 'Khong the xoa chinh ban'}), 400
        db = get_db()
        try:
            u = db.query(User).filter_by(user_id=uid).first()
            if not u:
                return jsonify({'error': 'User not found'}), 404
            db.delete(u)
            db.commit()
            return jsonify({'ok': True})
        except Exception as e:
            db.rollback()
            return jsonify({'error': str(e)}), 400
        finally:
            db.close()

    @app.route('/api/manager/users/export-csv')
    @login_required
    @role_required('manager')
    def export_users_csv():
        """Export all users to CSV."""
        db = get_db()
        try:
            users = db.query(User).order_by(User.created_at.desc()).all()
            header = ['username', 'full_name', 'email', 'phone', 'plate', 'role', 'created_at']
            rows = [[
                u.username, u.full_name or '', u.email or '',
                u.phone or '', u.plate or '', u.role,
                u.created_at.strftime('%d/%m/%Y') if u.created_at else '',
            ] for u in users]
            return _make_csv_response(header, rows, 'users.csv')
        finally:
            db.close()

    @app.route('/api/manager/users/csv-template')
    @login_required
    @role_required('manager')
    def users_csv_template():
        """Download CSV template for bulk user import."""
        header = ['username', 'password', 'role', 'full_name', 'email', 'phone', 'plate']
        sample = [
            ['sv001', 'password123', 'student', 'Nguyen Van A', 'sv001@bdu.edu.vn', '0901234567', '61A12345'],
            ['bv001', 'password123', 'guard', 'Tran Thi B', '', '', ''],
        ]
        buf = io.StringIO()
        buf.write('﻿')
        w = csv.writer(buf)
        w.writerow(header)
        w.writerows(sample)
        return Response(
            buf.getvalue(),
            mimetype='text/csv; charset=utf-8',
            headers={'Content-Disposition': 'attachment; filename="users_template.csv"'}
        )

    # ============================================================
    # Manager: VIP Slots Change -> notify backend app (port 5002)
    # ============================================================

    @app.route('/api/manager/vip-slots', methods=['PUT'])
    @login_required
    @role_required('manager')
    def put_manager_vip_slots():
        """Set VIP slots + notify backend app so it broadcasts to mobile clients."""
        data = request.get_json() or {}
        slot_ids = data.get('slot_ids') or []
        slot_numbers = data.get('slot_numbers') or []

        db = get_db()
        try:
            all_slots = db.query(ParkingSlot).all()
            target_slot_ids = set()
            if slot_ids:
                target_slot_ids = set(int(x) for x in slot_ids)
            elif slot_numbers:
                for sn in slot_numbers:
                    s = next((x for x in all_slots if x.slot_number == int(sn)), None)
                    if s:
                        target_slot_ids.add(s.slot_id)

            for s in all_slots:
                s.is_vip = s.slot_id in target_slot_ids
            db.commit()

            vip_nums = sorted([x.slot_number for x in all_slots if x.is_vip])

            # Emit via Socket.IO on main app (web dashboard)
            if socketio:
                socketio.emit('vip_slots_updated', {'slot_numbers': vip_nums})

            # Notify backend app (port 5002) to broadcast to mobile clients
            try:
                import urllib.request
                import json as _json
                backend_url = os.getenv('BACKEND_APP_URL', 'http://localhost:5002')
                req = urllib.request.Request(
                    f'{backend_url}/api/app/internal/vip-slots-changed',
                    data=_json.dumps({'slot_numbers': vip_nums}).encode(),
                    headers={'Content-Type': 'application/json'},
                    method='POST'
                )
                urllib.request.urlopen(req, timeout=3)
            except Exception as _e:
                print(f"[Manager] Could not notify backend app: {_e}")

            return jsonify({'slot_numbers': vip_nums})
        except Exception as e:
            db.rollback()
            return jsonify({'error': str(e)}), 400
        finally:
            db.close()

    @app.route('/health')
    def health_check():
        """Simple health check for Docker healthcheck and load balancers."""
        return jsonify({'status': 'ok', 'service': 'parking-main'}), 200

    @app.route('/api/health/detailed')
    def get_detailed_health():
        """
        Detailed system health check including:
        - DB pool availability
        - Pipeline FPS metrics (parking + gate)
        """
        pool_stats = get_pool_stats()

        # Extract FPS from pipeline runtimes if available
        parking_fps = None
        gate_fps = None
        try:
            from services.parking_detection.pipeline import ParkingPipelineRuntime
            rt = ParkingPipelineRuntime()
            avg_ms = rt.metrics._loop_ms[-1] if rt.metrics._loop_ms else None
            if avg_ms and avg_ms > 0:
                parking_fps = round(1000.0 / avg_ms, 1)
        except Exception:
            pass

        try:
            from services.gate_camera.pipeline import GatePipelineRuntime
            rt = GatePipelineRuntime()
            avg_ms = rt.metrics._loop_ms[-1] if rt.metrics._loop_ms else None
            if avg_ms and avg_ms > 0:
                gate_fps = round(1000.0 / avg_ms, 1)
        except Exception:
            pass

        return jsonify({
            'db_pool_available': pool_stats['healthy'],
            'db_pool': {
                'pool_size': pool_stats['pool_size'],
                'checked_in': pool_stats['checked_in'],
                'overflow': pool_stats['overflow'],
                'available': pool_stats['available'],
            },
            'parking_pipeline_fps': parking_fps,
            'gate_pipeline_fps': gate_fps,
            'timestamp': datetime.now().isoformat(),
        })