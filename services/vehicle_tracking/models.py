"""
Data Models for Vehicle Tracking System
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List
import numpy as np
import uuid


@dataclass
class VehicleTicket:
    """
    Ticket assigned to vehicle when entering through gate.
    Contains all identifying information for cross-camera matching.
    """
    ticket_id: str                      # UUID unique identifier
    plate_text: str                     # License plate text (e.g., "29A-12345")
    plate_conf: float                   # OCR confidence (0.0-1.0)
    entry_time: datetime                # Timestamp when vehicle entered gate
    vehicle_type: str                   # COCO class: "car", "bus", "truck"
    vehicle_bbox: List[int]             # [x1, y1, x2, y2] at gate camera
    
    @classmethod
    def create(cls, 
               plate_text: str = "",
               plate_conf: float = 0.0,
               vehicle_type: str = "car",
               vehicle_bbox: List[int] = None) -> 'VehicleTicket':
        """Factory method to create ticket with auto-generated ID and timestamp"""
        return cls(
            ticket_id=str(uuid.uuid4()),
            plate_text=plate_text,
            plate_conf=plate_conf,
            entry_time=datetime.now(),
            vehicle_type=vehicle_type,
            vehicle_bbox=vehicle_bbox or [0, 0, 0, 0]
        )
    
    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict (for API responses)"""
        return {
            'ticket_id': self.ticket_id,
            'plate_text': self.plate_text,
            'plate_conf': self.plate_conf,
            'entry_time': self.entry_time.isoformat(),
            'vehicle_type': self.vehicle_type
        }


@dataclass
class PendingVehicle:
    """
    Vehicle waiting in queue to be matched at parking camera.
    Created when vehicle crosses gate Line 2 (direction='in').
    """
    ticket: VehicleTicket               # Full ticket info
    timestamp_registered: datetime      # When added to pending queue
    match_attempts: int = 0             # Number of match attempts
    
    @property
    def age_seconds(self) -> float:
        """How long vehicle has been in pending queue (seconds)"""
        return (datetime.now() - self.timestamp_registered).total_seconds()
    
    @property
    def ticket_id(self) -> str:
        return self.ticket.ticket_id
    
    @property
    def plate_text(self) -> str:
        return self.ticket.plate_text


@dataclass 
class MatchedVehicle:
    """
    Successfully matched vehicle across gate and parking cameras.
    """
    ticket: VehicleTicket               # Original ticket from gate
    parking_slot: Optional[int]         # Parking slot number (1-19) or None
    timestamp_matched: datetime         # When matched at parking
    match_score: float                  # Matching score (0-100)
    transit_time: float                 # Seconds between gate and parking
    
    @classmethod
    def from_pending(cls, 
                     pending: PendingVehicle,
                     parking_slot: Optional[int],
                     match_score: float) -> 'MatchedVehicle':
        """Create MatchedVehicle from PendingVehicle"""
        now = datetime.now()
        return cls(
            ticket=pending.ticket,
            parking_slot=parking_slot,
            timestamp_matched=now,
            match_score=match_score,
            transit_time=(now - pending.ticket.entry_time).total_seconds()
        )
    
    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict"""
        return {
            **self.ticket.to_dict(),
            'parking_slot': self.parking_slot,
            'timestamp_matched': self.timestamp_matched.isoformat(),
            'match_score': self.match_score,
            'transit_time': round(self.transit_time, 1)
        }

