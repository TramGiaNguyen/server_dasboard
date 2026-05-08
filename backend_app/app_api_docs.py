"""
Flask-Smorest API documentation for Mobile Backend (port 5002).
Generates Swagger UI at /app/docs
"""

from flask import Blueprint
from flask_restx import Api, Resource, fields, Namespace

app_docs_bp = Blueprint('app_docs', __name__, url_prefix='/app')

authorizations = {
    'Bearer': {
        'type': 'apiKey',
        'in': 'header',
        'name': 'Authorization',
        'description': 'Mobile app Bearer token. Format: "Bearer <token>"',
    }
}

api = Api(
    app_docs_bp,
    version='1.0',
    title='BDU Smart Parking - Mobile API',
    description='REST API for Binh Duong University Smart Parking Mobile App',
    authorizations=authorizations,
    security='Bearer',
    doc='/docs',
)

# Namespaces
auth_ns = Namespace('auth', description='Authentication')
profile_ns = Namespace('profile', description='User profile')
vehicles_ns = Namespace('vehicles', description='Vehicle management')
reservations_ns = Namespace('reservations', description='Slot reservations')
notifications_ns = Namespace('notifications', description='User notifications')
slots_ns = Namespace('slots', description='Parking slots')
health_ns = Namespace('health', description='System health')

api.add_namespace(auth_ns, path='')
api.add_namespace(profile_ns, path='/profile')
api.add_namespace(vehicles_ns, path='/vehicles')
api.add_namespace(reservations_ns, path='/reservations')
api.add_namespace(notifications_ns, path='/notifications')
api.add_namespace(slots_ns, path='/slots')
api.add_namespace(health_ns, path='/health')

# Models
login_model = api.model('Login', {
    'username': fields.String(required=True, description='Username'),
    'password': fields.String(required=True, description='Password'),
})

token_model = api.model('Token', {
    'token': fields.String(description='Bearer token for subsequent requests'),
    'user': fields.Raw(description='User profile'),
})

profile_model = api.model('Profile', {
    'user_id': fields.Integer,
    'username': fields.String,
    'full_name': fields.String,
    'email': fields.String,
    'phone': fields.String,
    'plate': fields.String,
})

vehicle_model = api.model('Vehicle', {
    'id': fields.Integer,
    'plate_text': fields.String,
    'is_primary': fields.Boolean,
})

vehicle_create_model = api.model('VehicleCreate', {
    'plate_text': fields.String(required=True, description='License plate number'),
})

slot_model = api.model('Slot', {
    'slot_id': fields.Integer,
    'slot_number': fields.Integer,
    'slot_name': fields.String,
})

reservation_model = api.model('Reservation', {
    'reservation_id': fields.Integer,
    'slot_id': fields.Integer,
    'slot_number': fields.Integer,
    'booking_date': fields.String,
    'time_from': fields.String,
    'time_to': fields.String,
    'arrival_time': fields.String,
    'plate_text': fields.String,
    'status': fields.String,
    'remaining_time': fields.String,
    'created_at': fields.String,
})

reservation_create_model = api.model('ReservationCreate', {
    'slot_id': fields.Integer(required=True),
    'booking_date': fields.String(required=True, description='YYYY-MM-DD'),
    'time_from': fields.String(required=True, description='HH:MM'),
    'time_to': fields.String(required=True, description='HH:MM'),
    'arrival_time': fields.String(description='HH:MM (optional)'),
    'plate_text': fields.String(required=True),
})

notification_model = api.model('Notification', {
    'notification_id': fields.Integer,
    'title': fields.String,
    'body': fields.String,
    'type': fields.String,
    'related_id': fields.Integer,
    'created_at': fields.String,
    'read_at': fields.String,
})

notification_page_model = api.model('NotificationPage', {
    'items': fields.List(fields.Nested(notification_model)),
    'total': fields.Integer,
    'page': fields.Integer,
    'limit': fields.Integer,
})

# Auth
@auth_ns.route('/login')
class Login(Resource):
    @auth_ns.doc(description='Login with username and password. Returns Bearer token.', model=login_model)
    @auth_ns.expect(login_model)
    @auth_ns.marshal_with(token_model, code=200)
    def post(self):
        """Login and get Bearer token."""
        from flask import current_app
        return current_app.view_functions['app_login']()


# Profile
@profile_ns.route('')
class Profile(Resource):
    @auth_ns.doc(security='Bearer', description='Get current user profile')
    @auth_ns.marshal_with(profile_model)
    def get(self):
        """Get profile."""
        from flask import current_app
        return current_app.view_functions['app_profile_get']()

    @auth_ns.doc(security='Bearer', description='Update user profile')
    def put(self):
        """Update profile."""
        from flask import current_app
        return current_app.view_functions['app_profile_put']()


# Vehicles
@vehicles_ns.route('')
class Vehicles(Resource):
    @auth_ns.doc(security='Bearer', description='List user vehicles')
    @auth_ns.marshal_list_with(vehicle_model)
    def get(self):
        """Get vehicles."""
        from flask import current_app
        return current_app.view_functions['app_vehicles_get']()

    @auth_ns.doc(security='Bearer', description='Add a vehicle')
    @auth_ns.expect(vehicle_create_model)
    def post(self):
        """Add vehicle."""
        from flask import current_app
        return current_app.view_functions['app_vehicles_post']()


@vehicles_ns.route('/<int:vid>')
@vehicles_ns.param('vid', 'Vehicle ID')
class Vehicle(Resource):
    @auth_ns.doc(security='Bearer', description='Delete a vehicle')
    def delete(self, vid):
        """Delete vehicle."""
        from flask import current_app
        return current_app.view_functions['app_vehicles_delete'](vid)


# Slots
@slots_ns.route('/available')
class AvailableSlots(Resource):
    @auth_ns.doc(security='Bearer', description='Get available slots for date/time range')
    @slots_ns.param('date', 'Booking date (YYYY-MM-DD)')
    @slots_ns.param('time_from', 'Start time (HH:MM)')
    @slots_ns.param('time_to', 'End time (HH:MM)')
    def get(self):
        """Get available slots."""
        from flask import current_app
        return current_app.view_functions['app_slots_available']()


# Reservations
@reservations_ns.route('')
class Reservations(Resource):
    @auth_ns.doc(security='Bearer', description='List user reservations')
    @auth_ns.marshal_list_with(reservation_model)
    def get(self):
        """Get reservations."""
        from flask import current_app
        return current_app.view_functions['app_reservations_get']()

    @auth_ns.doc(security='Bearer', description='Create a new reservation')
    @auth_ns.expect(reservation_create_model)
    @reservations_ns.marshal_with(reservation_model, code=201)
    def post(self):
        """Create reservation."""
        from flask import current_app
        return current_app.view_functions['app_reservations_post']()


@reservations_ns.route('/<int:rid>/cancel')
@reservations_ns.param('rid', 'Reservation ID')
class CancelReservation(Resource):
    @auth_ns.doc(security='Bearer', description='Cancel a reservation')
    def post(self, rid):
        """Cancel reservation."""
        from flask import current_app
        return current_app.view_functions['app_reservations_cancel'](rid)


# Notifications
@notifications_ns.route('')
class Notifications(Resource):
    @auth_ns.doc(security='Bearer', description='Get paginated notifications')
    @auth_ns.marshal_with(notification_page_model)
    def get(self):
        """Get notifications."""
        from flask import current_app
        return current_app.view_functions['app_notifications_get']()


@notifications_ns.route('/<int:nid>/read')
@notifications_ns.param('nid', 'Notification ID')
class ReadNotification(Resource):
    @auth_ns.doc(security='Bearer', description='Mark notification as read')
    def post(self, nid):
        """Mark as read."""
        from flask import current_app
        return current_app.view_functions['app_notifications_read'](nid)


# Health
@health_ns.route('')
class Health(Resource):
    @auth_ns.doc(description='Basic health check')
    def get(self):
        """Health check."""
        from flask import current_app
        return current_app.view_functions['health_check']()
