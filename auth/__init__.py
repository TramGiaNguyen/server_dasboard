"""
Auth module for Smart Parking - Web dashboard login and role-based access.
"""

from .session import (
    login_required,
    role_required,
    get_current_user,
    login_user,
    logout_user,
)

__all__ = [
    'login_required',
    'role_required',
    'get_current_user',
    'login_user',
    'logout_user',
]
