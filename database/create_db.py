"""
Database Initialization Script
Run once to create all tables and seed initial parking slot data.

Usage:
    python create_db.py
    python database/create_db.py  # From project root
"""

import sys
import os

# Add project root to path (support both direct run and from root)
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_current_dir) if 'database' in _current_dir else _current_dir
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from werkzeug.security import generate_password_hash
from sqlalchemy import text
from database.db import init_db, get_db, engine
from database.models import ParkingSlot, User


# 19 parking slots matching init.sql seed data
INITIAL_SLOTS = [
    (1, 'Slot 1 - A', False),
    (2, 'Slot 2 - A', False),
    (3, 'Slot 3 - A', False),
    (4, 'Slot 4 - A', False),
    (5, 'Slot 5 - A', False),
    (6, 'Slot 6 - A', False),
    (7, 'Slot 7 - B', False),
    (8, 'Slot 8 - B', False),
    (9, 'Slot 9 - B', False),
    (10, 'Slot 10 - B', False),
    (11, 'Slot 11 - C', False),
    (12, 'Slot 12 - C', False),
    (13, 'Slot 13 - C', False),
    (14, 'Slot 14 - C', False),
    (15, 'Slot 15 - D', False),
    (16, 'Slot 16 - D', False),
    (17, 'Slot 17 - D', False),
    (18, 'Slot 18 - E', False),
    (19, 'Slot 19 - E', False),
]


def seed_slots():
    """Insert 19 parking slots if they don't already exist."""
    db = get_db()
    try:
        inserted = 0
        for slot_number, slot_name, is_occluded in INITIAL_SLOTS:
            existing = db.query(ParkingSlot).filter_by(slot_number=slot_number).first()
            if existing is None:
                slot = ParkingSlot(
                    slot_number=slot_number,
                    slot_name=slot_name,
                    is_occluded=is_occluded,
                    status='free',
                )
                db.add(slot)
                inserted += 1
        db.commit()
        print(f"[DB] Seeded {inserted} new parking slots (total: {len(INITIAL_SLOTS)}).")
    except Exception as e:
        db.rollback()
        print(f"[ERROR] Failed to seed slots: {e}")
        raise
    finally:
        db.close()


def migrate_users_plate():
    """Thêm cột plate vào users nếu chưa có."""
    try:
        with engine.connect() as conn:
            conn.execute(text(
                "ALTER TABLE users ADD COLUMN IF NOT EXISTS plate VARCHAR(20)"
            ))
            conn.commit()
        print("[DB] Migration: users.plate added (if missing).")
    except Exception as e:
        print(f"[DB] Migration plate: {e}")


def migrate_improper_parking_logs():
    """Thêm cột slot_number vào improper_parking_logs nếu chưa có."""
    try:
        with engine.connect() as conn:
            conn.execute(text(
                "ALTER TABLE improper_parking_logs ADD COLUMN IF NOT EXISTS slot_number INTEGER"
            ))
            conn.commit()
        print("[DB] Migration: improper_parking_logs.slot_number added (if missing).")
    except Exception as e:
        print(f"[DB] Migration slot_number: {e}")


def migrate_parking_slots_is_vip():
    """Thêm cột is_vip vào parking_slots nếu chưa có."""
    try:
        with engine.connect() as conn:
            conn.execute(text(
                "ALTER TABLE parking_slots ADD COLUMN IF NOT EXISTS is_vip BOOLEAN DEFAULT FALSE"
            ))
            conn.commit()
        print("[DB] Migration: parking_slots.is_vip added (if missing).")
    except Exception as e:
        print(f"[DB] Migration is_vip: {e}")


def seed_users():
    """Seed guard và manager. Nếu user đã tồn tại thì cập nhật password để đảm bảo đăng nhập được."""
    db = get_db()
    try:
        users_to_create = [
            ('guard', 'guard123', 'guard', 'Bảo vệ', None, None),
            ('manager', 'manager123', 'manager', 'Quản lý', None, None),
            ('staff', 'staff123', 'staff', 'Nhân viên', None, None),
            ('22050026', '12345609876', 'student', 'Nguyễn Văn A', 'vana.nguyen@example.com', '090 123 4567'),
        ]
        inserted = 0
        updated = 0
        for username, password, role, full_name, email, phone in users_to_create:
            existing = db.query(User).filter_by(username=username).first()
            if existing is None:
                user = User(
                    username=username,
                    password_hash=generate_password_hash(password),
                    role=role,
                    full_name=full_name,
                    email=email,
                    phone=phone,
                )
                db.add(user)
                inserted += 1
            else:
                existing.password_hash = generate_password_hash(password)
                existing.full_name = full_name or existing.full_name
                existing.email = email or existing.email
                existing.phone = phone or existing.phone
                updated += 1
        db.commit()
        print(f"[DB] Seeded {inserted} new users, updated {updated} existing.")
    except Exception as e:
        db.rollback()
        print(f"[ERROR] Failed to seed users: {e}")
        raise
    finally:
        db.close()


if __name__ == '__main__':
    from config import DATABASE_URL
    # Mask password in display
    _url = DATABASE_URL
    if '@' in _url and ':' in _url.split('@')[0]:
        _parts = _url.split('@')
        _url = _parts[0].rsplit(':', 1)[0] + ':****@' + _parts[1]
    print("=" * 50)
    print("  Smart Parking - Database Initialization")
    print("=" * 50)
    print(f"\n  Database: {_url}")

    # Step 1: Create all tables
    print("\n[Step 1] Creating tables...")
    init_db()

    # Step 2: Migrations
    print("\n[Step 2] Migration: users.plate...")
    migrate_users_plate()
    print("  Migration: improper_parking_logs.slot_number...")
    migrate_improper_parking_logs()
    print("  Migration: parking_slots.is_vip...")
    migrate_parking_slots_is_vip()

    # Step 3: Seed parking slots
    print("\n[Step 3] Seeding parking slots...")
    seed_slots()

    # Step 4: Seed users (guard, manager)
    print("\n[Step 4] Seeding users...")
    seed_users()

    # Verify
    db = get_db()
    try:
        count = db.query(User).count()
        print(f"\n[OK] Database initialization complete! Users in DB: {count}")
    finally:
        db.close()
    print("  Check pgAdmin4 - Database: PARKING_PLATE to verify tables.")
