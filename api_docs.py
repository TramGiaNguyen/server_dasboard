"""
Flask-Smorest API documentation for Smart Parking System.
Generates Swagger UI at /api/docs
Note: These are DOCUMENTATION endpoints only. The actual API logic
resides in api/routes.py. This file provides OpenAPI/Swagger spec.
"""

from flask import Blueprint, current_app
from flask_restx import Api, Resource, fields, Namespace

api_bp = Blueprint('api_docs', __name__, url_prefix='/api')

authorizations = {
    'Bearer': {
        'type': 'apiKey',
        'in': 'header',
        'name': 'Authorization',
        'description': 'Bearer token for mobile app. Format: "Bearer <token>"',
    }
}

api = Api(
    api_bp,
    version='1.0',
    title='BDU Smart Parking API',
    description='REST API for Binh Duong University Smart Parking System',
    authorizations=authorizations,
    security='Bearer',
    doc='/docs',
)

# Namespaces
parking_ns = Namespace('parking', description='Parking slot operations')
tracking_ns = Namespace('tracking', description='Vehicle tracking')
manager_ns = Namespace('manager', description='Manager operations')
gate_ns = Namespace('gate', description='Gate and OCR operations')
health_ns = Namespace('health', description='System health checks')
db_ns = Namespace('db', description='Database query endpoints')

api.add_namespace(parking_ns, path='/parking')
api.add_namespace(tracking_ns, path='/tracking')
api.add_namespace(manager_ns, path='/manager')
api.add_namespace(gate_ns, path='/gate')
api.add_namespace(health_ns, path='/health')
api.add_namespace(db_ns, path='/db')

# Shared models
slot_number = fields.Integer(description='Slot number (1-19)')
status = fields.String(description='Status: free, occupied, reserved')
plate = fields.String(description='Vehicle license plate')
is_vip = fields.Boolean(description='VIP slot flag')
is_hijacked = fields.Boolean(description='True if wrong plate occupies reserved slot')


# ── Parking Namespace ─────────────────────────────────────────────────────────

@parking_ns.route('/status')
class ParkingStatus(Resource):
    @parking_ns.doc(
        description='Get real-time parking KPIs from shared state. '
                    'Returns total, occupied, available, overlapping, outside counts.'
    )
    def get(self):
        """Get current parking status."""
        return {'payload': 'see api/routes.py: get_parking_status()'}


@parking_ns.route('/slots')
class ParkingSlots(Resource):
    @parking_ns.doc(description='Get all 19 parking slots with occupancy, reservation, and VIP status.')
    def get(self):
        """Get all slots."""
        return {'payload': 'see api/routes.py: get_all_slots()'}


@parking_ns.route('/slots/1-18')
class ParkingSlotsZone(Resource):
    @parking_ns.doc(description='Slots 1-18 with occupancy counts and zone breakdown.')
    def get(self):
        """Get slots 1-18."""
        return {'payload': 'see api/routes.py: get_slots_1_to_18()'}


@parking_ns.route('/slot/<int:slot_number>')
class ParkingSlot(Resource):
    @parking_ns.param('slot_number', 'Slot number (1-19)', type=int)
    @parking_ns.doc(description='Get detailed info for a specific slot including reservation and VIP status.')
    def get(self, slot_number):
        """Get single slot."""
        return {'payload': 'see api/routes.py: get_single_slot(slot_number)'}


# ── Gate Namespace ─────────────────────────────────────────────────────────────

@gate_ns.route('/ocr')
class GateOCR(Resource):
    @gate_ns.doc(
        description='Get latest OCR results from gate camera stream. '
                    'Returns plate_text, confidence, direction (IN/OUT), render_ts.'
    )
    def get(self):
        """Get gate OCR results."""
        return {'payload': 'see api/routes.py: get_gate_ocr()'}


# ── Tracking Namespace ─────────────────────────────────────────────────────────

@tracking_ns.route('/status')
class TrackingStatus(Resource):
    @tracking_ns.doc(description='Get current tracking status: pending_count, matched_count, total_matched, active_vehicles.')
    def get(self):
        """Get tracking status."""
        return {'payload': 'see api/routes.py: get_tracking_status()'}


@tracking_ns.route('/pending')
class TrackingPending(Resource):
    @tracking_ns.doc(description='Vehicles waiting to be matched at parking area. FIFO queue.')
    def get(self):
        """Get pending vehicles."""
        return {'payload': 'see api/routes.py: get_tracking_pending()'}


@tracking_ns.route('/matched')
class TrackingMatched(Resource):
    @tracking_ns.doc(description='Recent matched vehicles (last 50).')
    def get(self):
        """Get matched vehicles."""
        return {'payload': 'see api/routes.py: get_tracking_matched()'}


@tracking_ns.route('/stats')
class TrackingStats(Resource):
    @tracking_ns.doc(description='Tracking statistics.')
    def get(self):
        """Get tracking stats."""
        return {'payload': 'see api/routes.py: get_tracking_stats()'}


# ── DB Namespace ────────────────────────────────────────────────────────────────

@db_ns.route('/gate-logs')
class DBGateLogs(Resource):
    @db_ns.doc(description='Last 10 gate logs (IN/OUT events). Available to guard/staff.')
    def get(self):
        """Get recent gate logs."""
        return {'payload': 'see api/routes.py: get_db_gate_logs()'}


@db_ns.route('/improper-parking-logs')
class DBImproperLogs(Resource):
    @db_ns.doc(description='Last 20 improper parking events. Available to guard/staff.')
    def get(self):
        """Get improper parking logs."""
        return {'payload': 'see api/routes.py: get_improper_parking_logs()'}


@db_ns.route('/slot-status')
class DBSlotStatus(Resource):
    @db_ns.doc(description='Current slot status from database (DB-backed, not real-time).')
    def get(self):
        """Get DB slot status."""
        return {'payload': 'see api/routes.py: get_db_slot_status()'}


# ── Manager Namespace ─────────────────────────────────────────────────────────

@manager_ns.route('/gate-logs')
class ManagerGateLogs(Resource):
    @manager_ns.doc(description='All gate logs (no limit). Requires manager role.')
    def get(self):
        """Get all gate logs."""
        return {'payload': 'see api/routes.py: get_manager_gate_logs()'}


@manager_ns.route('/improper-parking-logs')
class ManagerImproperLogs(Resource):
    @manager_ns.doc(description='All improper parking logs. Requires manager role.')
    def get(self):
        """Get all improper logs."""
        return {'payload': 'see api/routes.py: get_manager_improper_parking_logs()'}


@manager_ns.route('/vip-slots')
class ManagerVipSlots(Resource):
    @manager_ns.doc(description='Get or set VIP slot numbers. Requires manager role.')
    def get(self):
        """Get VIP slots."""
        return {'payload': 'see api/routes.py: get_manager_vip_slots()'}

    @manager_ns.doc(description='Set VIP slots. Body: {"slot_numbers": [1, 5, 10]}')
    def put(self):
        """Update VIP slots."""
        return {'payload': 'see api/routes.py: put_manager_vip_slots()'}


@manager_ns.route('/hourly-traffic')
class ManagerHourlyTraffic(Resource):
    @manager_ns.doc(description='IN/OUT counts per hour for today.')
    def get(self):
        """Get hourly traffic."""
        return {'payload': 'see api/routes.py: get_hourly_traffic()'}


@manager_ns.route('/slot-frequency')
class ManagerSlotFrequency(Resource):
    @manager_ns.doc(description='Slot usage frequency for current week and month.')
    def get(self):
        """Get slot frequency."""
        return {'payload': 'see api/routes.py: get_slot_frequency()'}


@manager_ns.route('/frequent-violators')
class ManagerFrequentViolators(Resource):
    @manager_ns.doc(description='Vehicles with more than 5 improper parking events.')
    def get(self):
        """Get frequent violators."""
        return {'payload': 'see api/routes.py: get_frequent_violators()'}


@manager_ns.route('/low-availability-hours')
class ManagerLowAvailability(Resource):
    @manager_ns.doc(description='Average available slots per hour over week/month.')
    def get(self):
        """Get low availability hours."""
        return {'payload': 'see api/routes.py: get_low_availability_hours()'}


@manager_ns.route('/users')
class ManagerUsers(Resource):
    @manager_ns.param('page', 'Page number', type=int, default=1)
    @manager_ns.param('limit', 'Items per page', type=int, default=20)
    @manager_ns.param('search', 'Search by username/full_name', type=str)
    @manager_ns.param('role', 'Filter by role (student/guard/staff/manager)', type=str)
    @manager_ns.doc(description='Paginated user list. Requires manager role.')
    def get(self):
        """Get users."""
        return {'payload': 'see api/routes.py: get_manager_users()'}

    @manager_ns.doc(description='Create user. Body: {username, password, role, full_name, email, phone, plate}')
    def post(self):
        """Create user."""
        return {'payload': 'see api/routes.py: create_manager_user()'}


@manager_ns.route('/users/import-csv')
class ManagerUsersImport(Resource):
    @manager_ns.doc(description='Bulk import users from CSV file. Requires manager role. Multipart form: file=.csv')
    def post(self):
        """Import users CSV."""
        return {'payload': 'see api/routes.py: import_users_csv()'}


@manager_ns.route('/users/export-csv')
class ManagerUsersExport(Resource):
    @manager_ns.doc(description='Export all users as CSV. Requires manager role.')
    def get(self):
        """Export users CSV."""
        return {'payload': 'see api/routes.py: export_users_csv()'}


@manager_ns.route('/users/csv-template')
class ManagerUsersCsvTemplate(Resource):
    @manager_ns.doc(description='Download CSV template for bulk user import. Requires manager role.')
    def get(self):
        """Get CSV template."""
        return {'payload': 'see api/routes.py: users_csv_template()'}


@manager_ns.route('/users/<int:uid>')
class ManagerUser(Resource):
    @manager_ns.param('uid', 'User ID', type=int)
    @manager_ns.doc(description='Update user. Body: {full_name, email, phone, plate, role, password}')
    def put(self, uid):
        """Update user."""
        return {'payload': 'see api/routes.py: update_manager_user(uid)'}

    @manager_ns.param('uid', 'User ID', type=int)
    @manager_ns.doc(description='Delete user. Cannot delete self. Requires manager role.')
    def delete(self, uid):
        """Delete user."""
        return {'payload': 'see api/routes.py: delete_manager_user(uid)'}


# ── Health Namespace ───────────────────────────────────────────────────────────

@health_ns.route('')
class Health(Resource):
    @health_ns.doc(description='Basic health check. Returns {"status": "ok", "service": "parking-main"}')
    def get(self):
        """Simple health check."""
        return {'payload': 'see api/routes.py: health_check()'}


@health_ns.route('/detailed')
class DetailedHealth(Resource):
    @health_ns.doc(description='Detailed health: DB pool stats, pipeline FPS for parking and gate cameras.')
    def get(self):
        """Detailed health check."""
        return {'payload': 'see api/routes.py: get_detailed_health()'}


# ── Export Namespace ───────────────────────────────────────────────────────────

export_ns = Namespace('export', description='CSV export endpoints')
api.add_namespace(export_ns, path='/export')

@export_ns.route('/gate-logs.csv')
class ExportGateLogs(Resource):
    @export_ns.doc(description='Download gate logs as CSV. Requires manager role.')
    def get(self):
        """Export gate logs CSV."""
        return {'payload': 'see api/routes.py: export_gate_logs_csv()'}


@export_ns.route('/improper-parking-logs.csv')
class ExportImproperLogs(Resource):
    @export_ns.doc(description='Download improper parking logs as CSV. Requires manager role.')
    def get(self):
        """Export improper logs CSV."""
        return {'payload': 'see api/routes.py: export_improper_parking_logs_csv()'}


@export_ns.route('/slot-status.csv')
class ExportSlotStatus(Resource):
    @export_ns.doc(description='Download current slot status as CSV.')
    def get(self):
        """Export slot status CSV."""
        return {'payload': 'see api/routes.py: export_slot_status_csv()'}


@export_ns.route('/frequent-violators.csv')
class ExportFrequentViolators(Resource):
    @export_ns.doc(description='Download frequent violators as CSV. Requires manager role.')
    def get(self):
        """Export frequent violators CSV."""
        return {'payload': 'see api/routes.py: export_frequent_violators_csv()'}
