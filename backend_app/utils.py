"""
Utility functions for backend app
"""
from datetime import datetime, timezone


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
