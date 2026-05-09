"""
Backend API Server for Smart Parking Mobile App
Standalone Flask server serving mobile app endpoints only.
"""
import os
import sys
from datetime import datetime
from flask import Flask
from flask_cors import CORS
from flask_socketio import SocketIO

# Add parent directory to path for shared imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from routes import register_app_routes

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')

# Enable CORS for mobile app
CORS(app, resources={
    r"/api/*": {
        "origins": "*",
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# Initialize SocketIO
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Register mobile app routes
register_app_routes(app, socketio)

# Start the unified reservation/notification scheduler in a background thread
try:
    from backend_app.services.reservation_scheduler import start_scheduler as _start_res_scheduler
    _start_res_scheduler(socketio)
    print("[App] Reservation scheduler started")
except Exception as e:
    print(f"[App] Failed to start scheduler: {e}")


# ============================================================
# Socket.IO Event Handlers
# ============================================================

@socketio.on('connect')
def handle_connect():
    print('[Socket.IO] Mobile app connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('[Socket.IO] Mobile app disconnected')

@socketio.on('ping')
def handle_ping():
    socketio.emit('pong', {'ts': datetime.now().isoformat()})


# ============================================================
# Health check endpoint
# ============================================================
@app.route('/health')
def health_check():
    return {'status': 'ok', 'service': 'parking-app-backend'}, 200

if __name__ == '__main__':
    port = int(os.getenv('APP_PORT', 5002))
    debug = os.getenv('FLASK_ENV', 'production') == 'development'

    print(f"Starting Mobile App Backend Server on port {port}...")
    socketio.run(app, host='0.0.0.0', port=port, debug=debug, allow_unsafe_werkzeug=True)
