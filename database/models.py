"""
SQLAlchemy ORM Models for Smart Parking System
Maps to schema defined in init.sql
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Column, Integer, String, Float, Boolean, Text, DateTime, Date, Time, ForeignKey,
    Index, UniqueConstraint
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship, declarative_base

Base = declarative_base()


# =============================================
# 1. Parking Slots (19 vị trí đỗ)
# =============================================
class ParkingSlot(Base):
    __tablename__ = 'parking_slots'

    slot_id = Column(Integer, primary_key=True, autoincrement=True)
    slot_number = Column(Integer, unique=True, nullable=False)
    slot_name = Column(String(50))
    status = Column(String(20), default='free')          # free | occupied | reserved
    is_occluded = Column(Boolean, default=False)
    is_vip = Column(Boolean, default=False)
    coordinates = Column(JSONB, nullable=True)
    last_updated = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    # Relationship
    sessions = relationship('ParkingSession', back_populates='slot')

    def __repr__(self):
        return f"<ParkingSlot {self.slot_number} ({self.status})>"


# =============================================
# 2. Vehicles (Kho thông tin xe)
# =============================================
class Vehicle(Base):
    __tablename__ = 'vehicles'

    vehicle_uuid = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    plate_text = Column(String(20), unique=True)
    vehicle_type = Column(String(20))                    # car | bus | truck
    first_seen = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    last_seen = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    trust_score = Column(Float, default=0.0)

    # Relationships
    sessions = relationship('ParkingSession', back_populates='vehicle')

    def __repr__(self):
        return f"<Vehicle {self.plate_text}>"


# =============================================
# 3. Parking Sessions (Vòng đời gửi xe)
# =============================================
class ParkingSession(Base):
    __tablename__ = 'parking_sessions'

    session_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    vehicle_uuid = Column(UUID(as_uuid=True), ForeignKey('vehicles.vehicle_uuid'))

    # Entry
    entry_time = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    entry_gate_image_path = Column(Text)
    entry_plate_conf = Column(Float)

    # Parking
    assigned_slot_id = Column(Integer, ForeignKey('parking_slots.slot_id'), nullable=True)
    parked_time = Column(DateTime(timezone=True), nullable=True)

    # Exit
    exit_time = Column(DateTime(timezone=True), nullable=True)
    exit_gate_image_path = Column(Text, nullable=True)
    exit_plate_conf = Column(Float, nullable=True)

    # Stats
    duration_minutes = Column(Integer, nullable=True)
    status = Column(String(20), default='active')        # active | completed | overnight

    # Relationships
    vehicle = relationship('Vehicle', back_populates='sessions')
    slot = relationship('ParkingSlot', back_populates='sessions')
    tracking_events = relationship('TrackingEvent', back_populates='session')

    # Indexes (defined via __table_args__)
    __table_args__ = (
        Index('idx_session_active', 'status', postgresql_where=(status == 'active')),
        Index('idx_session_entry', 'entry_time'),
    )

    def __repr__(self):
        return f"<ParkingSession {self.session_id} ({self.status})>"


# =============================================
# 4. Tracking Events (Debug log di chuyển)
# =============================================
class TrackingEvent(Base):
    __tablename__ = 'tracking_events'

    event_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey('parking_sessions.session_id'))
    event_type = Column(String(50))
    timestamp = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    details = Column(JSONB, nullable=True)

    # Relationship
    session = relationship('ParkingSession', back_populates='tracking_events')

    def __repr__(self):
        return f"<TrackingEvent {self.event_type}>"


# =============================================
# 5. Gate Logs (Lịch sử cổng - append-only)
# =============================================
class GateLog(Base):
    __tablename__ = 'gate_logs'

    log_id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    plate_text = Column(String(20))
    direction = Column(String(10))                       # IN | OUT
    confidence = Column(Float)
    image_path = Column(Text, nullable=True)

    def __repr__(self):
        return f"<GateLog {self.direction} {self.plate_text}>"


# =============================================
# 6. Improper Parking Logs (Xe đậu sai vị trí)
# =============================================
class ImproperParkingLog(Base):
    __tablename__ = 'improper_parking_logs'

    log_id      = Column(Integer, primary_key=True, autoincrement=True)
    timestamp   = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    plate_text  = Column(String(20), nullable=True)
    event_type  = Column(String(20))                       # outside | overlapping
    image_path  = Column(Text, nullable=True)
    slot_number = Column(Integer, nullable=True)           # Slot khi overlapping

    def __repr__(self):
        return f"<ImproperParkingLog {self.event_type} {self.plate_text}>"


# =============================================
# 7. Users (App user + Web staff)
# =============================================
class User(Base):
    __tablename__ = 'users'

    user_id       = Column(Integer, primary_key=True, autoincrement=True)
    username      = Column(String(50), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    role          = Column(String(20), nullable=False)      # student | guard | manager | staff
    full_name     = Column(String(100), nullable=True)
    email         = Column(String(100), nullable=True)
    phone         = Column(String(20), nullable=True)
    plate         = Column(String(20), nullable=True)   # Biển số xe mặc định
    created_at    = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at    = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    vehicles      = relationship('UserVehicle', back_populates='user', cascade='all, delete-orphan')
    reservations  = relationship('SlotReservation', back_populates='user')
    notifications = relationship('Notification', back_populates='user', foreign_keys='Notification.user_id')

    def __repr__(self):
        return f"<User {self.username} ({self.role})>"


# =============================================
# 8. User Vehicles (Biển số xe đăng ký)
# =============================================
class UserVehicle(Base):
    __tablename__ = 'user_vehicles'

    id         = Column(Integer, primary_key=True, autoincrement=True)
    user_id    = Column(Integer, ForeignKey('users.user_id', ondelete='CASCADE'), nullable=False)
    plate_text = Column(String(20), nullable=False)
    is_primary = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    user = relationship('User', back_populates='vehicles')

    __table_args__ = (UniqueConstraint('user_id', 'plate_text', name='uq_user_vehicle'),)

    def __repr__(self):
        return f"<UserVehicle {self.plate_text}>"


# =============================================
# 9. Slot Reservations (Đặt trước slot)
# =============================================
class SlotReservation(Base):
    __tablename__ = 'slot_reservations'

    reservation_id = Column(Integer, primary_key=True, autoincrement=True)
    user_id        = Column(Integer, ForeignKey('users.user_id'), nullable=False)
    slot_id        = Column(Integer, ForeignKey('parking_slots.slot_id'), nullable=False)
    booking_date   = Column(Date, nullable=False)
    time_from      = Column(Time, nullable=False)
    time_to        = Column(Time, nullable=False)
    arrival_time   = Column(Time, nullable=True)
    plate_text     = Column(String(20), nullable=False)
    status         = Column(String(20), default='pending')  # pending | confirmed | completed | cancelled
    created_at     = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at     = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    user = relationship('User', back_populates='reservations')
    slot = relationship('ParkingSlot', back_populates='reservations')

    def __repr__(self):
        return f"<SlotReservation {self.reservation_id} ({self.status})>"


# =============================================
# 10. Notifications (Thông báo cho app)
# =============================================
class Notification(Base):
    __tablename__ = 'notifications'

    notification_id = Column(Integer, primary_key=True, autoincrement=True)
    user_id         = Column(Integer, ForeignKey('users.user_id', ondelete='CASCADE'), nullable=True)  # NULL = broadcast
    title           = Column(String(200), nullable=False)
    body            = Column(Text, nullable=True)
    type            = Column(String(30), nullable=True)     # reservation | violation | announcement | system
    related_id      = Column(Integer, nullable=True)
    created_at      = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    read_at         = Column(DateTime(timezone=True), nullable=True)

    user = relationship('User', back_populates='notifications', foreign_keys=[user_id])

    def __repr__(self):
        return f"<Notification {self.title}>"


# Add relationships to ParkingSlot
ParkingSlot.reservations = relationship('SlotReservation', back_populates='slot')

# Index for vehicles.plate_text (handled by unique constraint)
Index('idx_vehicle_plate', Vehicle.plate_text)
