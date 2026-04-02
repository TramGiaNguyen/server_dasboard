"""
Session-based auth for web dashboard.
Uses Flask session. Roles: guard (Dashboard), manager (Dashboard + Manager).
"""

from functools import wraps
from flask import session, redirect, url_for, request

from database.db import get_db
from database.models import User


def login_user(user: User):
    """Store user in session after successful login."""
    session.permanent = True
    session['user_id'] = user.user_id
    session['username'] = user.username
    session['role'] = user.role
    session['full_name'] = user.full_name or user.username


def logout_user():
    """Clear session."""
    session.clear()


def get_current_user():
    """Get current user from session, or None."""
    user_id = session.get('user_id')
    if not user_id:
        return None
    db = get_db()
    try:
        return db.query(User).filter_by(user_id=user_id).first()
    finally:
        db.close()


def login_required(f):
    """Decorator: require login. Redirect to /login if not logged in."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get('user_id'):
            return redirect(url_for('login_page', next=request.url))
        return f(*args, **kwargs)
    return decorated


def role_required(*allowed_roles):
    """
    Decorator: require one of allowed_roles.
    guard: only / (Dashboard)
    manager: / and /manager
    """
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            if not session.get('user_id'):
                return redirect(url_for('login_page', next=request.url))
            role = session.get('role')
            if role not in allowed_roles:
                return redirect(url_for('index'))
            return f(*args, **kwargs)
        return decorated
    return decorator
