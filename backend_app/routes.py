"""
API routes for Flutter app: login, profile, vehicles, reservations, notifications.
Uses Bearer token in Authorization header.
"""

import secrets
import time
from datetime import datetime, timezone, date
from functools import wraps

from flask import request, jsonify, Response
from werkzeug.security import check_password_hash, generate_password_hash

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.db import get_db
from database.models import User, UserVehicle, SlotReservation, Notification, ParkingSlot
from utils import get_remaining_time

# In-memory token store: token -> user_id (for production use Redis/DB)
_app_tokens = {}


def _get_user_from_token():
    """Extract user from Authorization: Bearer <token>. Returns (user, None) or (None, error_response)."""
    auth = request.headers.get('Authorization')
    if not auth or not auth.startswith('Bearer '):
        return None, (jsonify({'error': 'Missing or invalid Authorization header'}), 401)
    token = auth[7:].strip()
    user_id = _app_tokens.get(token)
    if not user_id:
        return None, (jsonify({'error': 'Invalid or expired token'}), 401)
    db = get_db()
    try:
        user = db.query(User).filter_by(user_id=user_id).first()
        if not user:
            return None, (jsonify({'error': 'User not found'}), 401)
        return user, None
    finally:
        db.close()


def app_auth_required(f):
    """Decorator: require valid app token."""
    @wraps(f)
    def decorated(*args, **kwargs):
        user, err = _get_user_from_token()
        if err:
            return err
        return f(user=user, *args, **kwargs)
    return decorated


def register_app_routes(app, socketio=None):
    """Register /api/app/* routes."""

    @app.route('/api/app/camera-stream')
    def app_camera_stream():
        """
        Camera stream endpoint - returns 503 Service Unavailable
        Mobile app should connect directly to main app (port 5001) for camera streams
        """
        return jsonify({
            'error': 'Camera streams not available on backend API',
            'message': 'Please connect to main app at port 5001 for camera streams'
        }), 503

    @app.route('/api/app/login', methods=['POST'])
    def app_login():
        data = request.get_json() or {}
        username = (data.get('username') or '').strip()
        password = data.get('password') or ''
        if not username or not password:
            return jsonify({'error': 'username and password required'}), 400
        db = get_db()
        try:
            user = db.query(User).filter_by(username=username).first()
            if not user or not check_password_hash(user.password_hash, password):
                return jsonify({'error': 'Invalid username or password'}), 401
            # Cho phép student, guard, manager đều dùng app (đặt chỗ, xe, thông báo)
            token = secrets.token_urlsafe(32)
            _app_tokens[token] = user.user_id
            return jsonify({
                'token': token,
                'user': {
                    'user_id': user.user_id,
                    'username': user.username,
                    'full_name': user.full_name or user.username,
                    'email': user.email or '',
                    'phone': user.phone or '',
                    'plate': user.plate or '',
                }
            })
        finally:
            db.close()

    @app.route('/api/app/profile', methods=['GET'])
    @app_auth_required
    def app_profile_get(user):
        return jsonify({
            'user_id': user.user_id,
            'username': user.username,
            'full_name': user.full_name or user.username,
            'email': user.email or '',
            'phone': user.phone or '',
            'plate': user.plate or '',
        })

    @app.route('/api/app/profile', methods=['PUT'])
    @app_auth_required
    def app_profile_put(user):
        data = request.get_json() or {}
        db = get_db()
        try:
            if 'full_name' in data:
                user.full_name = str(data['full_name'])[:100] if data['full_name'] else None
            if 'email' in data:
                user.email = str(data['email'])[:100] if data['email'] else None
            if 'phone' in data:
                user.phone = str(data['phone'])[:20] if data['phone'] else None
            if 'plate' in data:
                val = (data.get('plate') or '').strip().upper()
                user.plate = val[:20] if val else None
            user.updated_at = datetime.now(timezone.utc)
            db.commit()
            return jsonify({
                'user_id': user.user_id,
                'username': user.username,
                'full_name': user.full_name or user.username,
                'email': user.email or '',
                'phone': user.phone or '',
                'plate': user.plate or '',
            })
        except Exception as e:
            db.rollback()
            return jsonify({'error': str(e)}), 400
        finally:
            db.close()

    @app.route('/api/app/vehicles', methods=['GET'])
    @app_auth_required
    def app_vehicles_get(user):
        db = get_db()
        try:
            vehicles = db.query(UserVehicle).filter_by(user_id=user.user_id).all()
            return jsonify([{
                'id': v.id,
                'plate_text': v.plate_text,
                'is_primary': v.is_primary,
            } for v in vehicles])
        finally:
            db.close()

    @app.route('/api/app/vehicles', methods=['POST'])
    @app_auth_required
    def app_vehicles_post(user):
        data = request.get_json() or {}
        plate = (data.get('plate_text') or data.get('plate') or '').strip().upper()
        if not plate:
            return jsonify({'error': 'plate_text required'}), 400
        db = get_db()
        try:
            existing = db.query(UserVehicle).filter_by(user_id=user.user_id, plate_text=plate).first()
            if existing:
                return jsonify({'error': 'Biển số đã tồn tại'}), 400
            is_primary = not db.query(UserVehicle).filter_by(user_id=user.user_id).first()
            v = UserVehicle(user_id=user.user_id, plate_text=plate, is_primary=is_primary)
            db.add(v)
            db.commit()
            return jsonify({'id': v.id, 'plate_text': v.plate_text, 'is_primary': v.is_primary}), 201
        except Exception as e:
            db.rollback()
            return jsonify({'error': str(e)}), 400
        finally:
            db.close()

    @app.route('/api/app/vehicles/<int:vid>', methods=['DELETE'])
    @app_auth_required
    def app_vehicles_delete(user, vid):
        db = get_db()
        try:
            v = db.query(UserVehicle).filter_by(id=vid, user_id=user.user_id).first()
            if not v:
                return jsonify({'error': 'Not found'}), 404
            db.delete(v)
            db.commit()
            return jsonify({'success': True})
        except Exception as e:
            db.rollback()
            return jsonify({'error': str(e)}), 400
        finally:
            db.close()

    @app.route('/api/app/slots/available', methods=['GET'])
    @app_auth_required
    def app_slots_available(user):
        """Get slots available for given date and time range.
        Returns slots that are not reserved and not VIP.
        Note: Real-time occupancy status is available on main app (port 5001).
        """
        date_str = request.args.get('date')  # YYYY-MM-DD
        time_from = request.args.get('time_from')  # HH:MM
        time_to = request.args.get('time_to')  # HH:MM
        if not date_str or not time_from or not time_to:
            return jsonify({'error': 'date, time_from, time_to required'}), 400
        try:
            booking_date = datetime.strptime(date_str, '%Y-%m-%d').date()
        except ValueError:
            return jsonify({'error': 'Invalid date format (use YYYY-MM-DD)'}), 400
        try:
            tfrom = datetime.strptime(time_from, '%H:%M').time()
            tto = datetime.strptime(time_to, '%H:%M').time()
        except ValueError:
            return jsonify({'error': 'Invalid time format (use HH:MM)'}), 400
        db = get_db()
        try:
            conflicting = db.query(SlotReservation.slot_id).filter(
                SlotReservation.booking_date == booking_date,
                SlotReservation.status.in_(['pending', 'confirmed']),
                ((SlotReservation.time_from < tto) & (SlotReservation.time_to > tfrom)),
            ).distinct().all()
            busy_slot_ids = {r[0] for r in conflicting}
            slots = db.query(ParkingSlot).order_by(ParkingSlot.slot_number).all()
            vip_slot_ids = {s.slot_id for s in slots if getattr(s, 'is_vip', False)}
            busy_slot_ids |= vip_slot_ids
            
            available = []
            for s in slots:
                if s.slot_id not in busy_slot_ids:
                    available.append({
                        'slot_id': s.slot_id,
                        'slot_number': s.slot_number,
                        'slot_name': s.slot_name,
                    })
            vip_numbers = [s.slot_number for s in slots if getattr(s, 'is_vip', False)]
            return jsonify({'slots': available, 'vip_slot_numbers': vip_numbers})
        finally:
            db.close()

    @app.route('/api/app/reservations', methods=['POST'])
    @app_auth_required
    def app_reservations_post(user):
        data = request.get_json() or {}
        slot_id = data.get('slot_id')
        booking_date = data.get('booking_date')  # YYYY-MM-DD
        time_from = data.get('time_from')  # HH:MM
        time_to = data.get('time_to')
        arrival_time = data.get('arrival_time')
        plate_text = (data.get('plate_text') or '').strip()
        if not all([slot_id, booking_date, time_from, time_to, plate_text]):
            return jsonify({'error': 'slot_id, booking_date, time_from, time_to, plate_text required'}), 400
        try:
            bdate = datetime.strptime(booking_date, '%Y-%m-%d').date()
            tfrom = datetime.strptime(time_from, '%H:%M').time()
            tto = datetime.strptime(time_to, '%H:%M').time()
            atime = datetime.strptime(arrival_time, '%H:%M').time() if arrival_time else tfrom
        except (ValueError, TypeError):
            return jsonify({'error': 'Invalid date/time format'}), 400
        db = get_db()
        try:
            slot = db.query(ParkingSlot).filter_by(slot_id=slot_id).first()
            if not slot:
                return jsonify({'error': 'Slot not found'}), 404
            
            # Check if slot is VIP - regular users cannot book VIP slots
            if getattr(slot, 'is_vip', False):
                return jsonify({'error': 'Slot này dành riêng cho VIP. Vui lòng chọn slot khác.'}), 403
            
            # Check for conflicting reservations
            conflict = db.query(SlotReservation).filter(
                SlotReservation.slot_id == slot_id,
                SlotReservation.booking_date == bdate,
                SlotReservation.status.in_(['pending', 'confirmed']),
                ((SlotReservation.time_from < tto) & (SlotReservation.time_to > tfrom)),
            ).first()
            if conflict:
                return jsonify({'error': 'Slot đã được đặt trong khung giờ này. Vui lòng chọn slot hoặc giờ khác.'}), 409
            
            r = SlotReservation(
                user_id=user.user_id,
                slot_id=slot_id,
                booking_date=bdate,
                time_from=tfrom,
                time_to=tto,
                arrival_time=atime,
                plate_text=plate_text,
                status='confirmed',
            )
            db.add(r)
            db.flush()
            # Thông báo đặt chỗ thành công
            slot_name = f"Slot {slot.slot_number}"
            arr_str = arrival_time or time_from or ''
            notif = Notification(
                user_id=user.user_id,
                title='Đặt chỗ thành công',
                body=f'Bạn đã đặt {slot_name} ngày {booking_date} lúc {arr_str} - {plate_text}. Vui lòng đến đúng giờ.',
                type='reservation',
                related_id=r.reservation_id,
            )
            db.add(notif)
            broadcast = Notification(
                user_id=None,
                title='Slot đã được đặt',
                body=f'Slot {slot.slot_number} đã được đặt từ {time_from}–{time_to} ngày {booking_date}. Vui lòng chọn slot khác.',
                type='slot_reserved',
                related_id=r.reservation_id,
            )
            db.add(broadcast)
            db.commit()
            
            # Emit Socket.IO event for real-time slot update
            if socketio:
                socketio.emit('reservation_created', {
                    'slot_id': r.slot_id,
                    'slot_number': slot.slot_number,
                    'booking_date': booking_date,
                    'time_from': time_from,
                    'time_to': time_to,
                    'user_name': user.full_name or user.username,
                })

                # VIP slot booked -> broadcast to all connected clients
                if getattr(slot, 'is_vip', False):
                    socketio.emit('vip_slot_booked', {
                        'slot_number': slot.slot_number,
                        'user_name': user.full_name or user.username,
                        'time_from': time_from,
                        'time_to': time_to,
                        'booking_date': booking_date,
                    })

            return jsonify({
                'reservation_id': r.reservation_id,
                'slot_id': r.slot_id,
                'slot_number': slot.slot_number,
                'booking_date': booking_date,
                'time_from': time_from,
                'time_to': time_to,
                'arrival_time': arrival_time or time_from,
                'plate_text': r.plate_text,
                'status': r.status,
            }), 201
        except Exception as e:
            db.rollback()
            return jsonify({'error': str(e)}), 400
        finally:
            db.close()

    @app.route('/api/app/reservations', methods=['GET'])
    @app_auth_required
    def app_reservations_get(user):
        db = get_db()
        try:
            rows = db.query(SlotReservation, ParkingSlot).join(
                ParkingSlot, SlotReservation.slot_id == ParkingSlot.slot_id
            ).filter(SlotReservation.user_id == user.user_id).order_by(
                SlotReservation.booking_date.desc(), SlotReservation.created_at.desc()
            ).all()
            out = []
            for r, slot in rows:
                # Calculate remaining time from server
                remaining = get_remaining_time(r)
                
                out.append({
                    'reservation_id': r.reservation_id,
                    'slot_id': r.slot_id,
                    'slot_number': slot.slot_number,
                    'slot_name': slot.slot_name,
                    'booking_date': r.booking_date.isoformat() if r.booking_date else None,
                    'time_from': r.time_from.strftime('%H:%M') if r.time_from else None,
                    'time_to': r.time_to.strftime('%H:%M') if r.time_to else None,
                    'arrival_time': r.arrival_time.strftime('%H:%M') if r.arrival_time else None,
                    'plate_text': r.plate_text,
                    'status': r.status,
                    'created_at': r.created_at.isoformat() if r.created_at else None,
                    # Server-calculated remaining time
                    'remaining_time': remaining,
                })
            return jsonify(out)
        finally:
            db.close()

    @app.route('/api/app/reservations/<int:rid>/cancel', methods=['POST'])
    @app_auth_required
    def app_reservations_cancel(user, rid):
        db = get_db()
        try:
            r = db.query(SlotReservation).filter_by(reservation_id=rid, user_id=user.user_id).first()
            if not r:
                return jsonify({'error': 'Not found'}), 404
            if r.status in ('completed', 'cancelled'):
                return jsonify({'error': 'Cannot cancel'}), 400
            r.status = 'cancelled'
            r.updated_at = datetime.now(timezone.utc)
            db.commit()

            # Emit Socket.IO cancellation event
            if socketio:
                socketio.emit('reservation_cancelled', {
                    'reservation_id': rid,
                    'slot_id': r.slot_id,
                })

            return jsonify({'success': True})
        except Exception as e:
            db.rollback()
            return jsonify({'error': str(e)}), 400
        finally:
            db.close()

    @app.route('/api/app/notifications', methods=['GET'])
    @app_auth_required
    def app_notifications_get(user):
        page = max(1, int(request.args.get('page', 1)))
        limit = min(50, max(1, int(request.args.get('limit', 20))))
        offset = (page - 1) * limit
        db = get_db()
        try:
            q = db.query(Notification).filter(
                (Notification.user_id == user.user_id) | (Notification.user_id.is_(None))
            ).order_by(Notification.created_at.desc())
            total = q.count()
            rows = q.offset(offset).limit(limit).all()
            return jsonify({
                'items': [{
                    'notification_id': n.notification_id,
                    'title': n.title,
                    'body': n.body or '',
                    'type': n.type or 'system',
                    'related_id': n.related_id,
                    'created_at': n.created_at.isoformat() if n.created_at else None,
                    'read_at': n.read_at.isoformat() if n.read_at else None,
                } for n in rows],
                'total': total,
                'page': page,
                'limit': limit,
            })
        finally:
            db.close()

    @app.route('/api/app/notifications/<int:nid>/read', methods=['POST'])
    @app_auth_required
    def app_notifications_read(user, nid):
        db = get_db()
        try:
            n = db.query(Notification).filter_by(notification_id=nid).first()
            if not n:
                return jsonify({'error': 'Not found'}), 404
            if n.user_id is not None and n.user_id != user.user_id:
                return jsonify({'error': 'Forbidden'}), 403
            n.read_at = datetime.now(timezone.utc)
            db.commit()
            return jsonify({'success': True})
        except Exception as e:
            db.rollback()
            return jsonify({'error': str(e)}), 400
        finally:
            db.close()

    @app.route('/api/app/internal/vip-slots-changed', methods=['POST'])
    def internal_vip_slots_changed():
        """
        Internal endpoint: main app (port 5001) goi khi VIP slots thay doi.
        Backend app se broadcast sang tat ca mobile app qua Socket.IO.
        """
        data = request.get_json() or {}
        slot_numbers = data.get('slot_numbers', [])
        if socketio:
            socketio.emit('vip_slots_updated', {'slot_numbers': slot_numbers})
        return jsonify({'ok': True})

    @app.route('/health')
    def app_health_check():
        return {'status': 'ok', 'service': 'parking-app-backend'}, 200
