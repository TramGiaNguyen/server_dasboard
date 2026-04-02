"""
Vehicle Tracker - Simplified Ticket Manager
Refactored to support FIFO Handoff flow (no visual matching).
"""
import threading
from datetime import datetime
from typing import Optional, List, Dict
import time

from .models import VehicleTicket, PendingVehicle, MatchedVehicle

class VehicleTracker:
    """
    Ticket Manager for Vehicle System.
    Manages vehicle lifecycle: Entry -> Pending -> Matched (Parked) -> Exit.
    """
    
    def __init__(self, max_wait_time: int = 300, cleanup_interval: int = 60):
        """
        Initialize tracker.
        
        Args:
            max_wait_time: Maximum seconds to keep pending vehicle (default: 300s = 5 mins)
            cleanup_interval: Seconds between cleanup runs
        """
        self.max_wait_time = max_wait_time
        self.cleanup_interval = cleanup_interval
        
        # Thread-safe storage
        self._lock = threading.RLock()
        self._pending_vehicles: List[PendingVehicle] = []
        self._matched_vehicles: List[MatchedVehicle] = []
        self._active_tickets: Dict[str, VehicleTicket] = {}  # ticket_id -> ticket
        
        # Statistics
        self._total_registered = 0
        self._total_matched = 0
        self._total_expired = 0
        self._total_transit_time = 0.0
        
        # Start cleanup thread
        self._cleanup_thread = None
        self._running = False
        self._start_cleanup_thread()
    
    def _start_cleanup_thread(self):
        """Start background cleanup thread"""
        self._running = True
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            daemon=True,
            name="VehicleTrackerCleanup"
        )
        self._cleanup_thread.start()
    
    def _cleanup_loop(self):
        """Background loop to clean up expired vehicles"""
        while self._running:
            time.sleep(self.cleanup_interval)
            self.cleanup_expired()
    
    def stop(self):
        """Stop the cleanup thread gracefully."""
        self._running = False
        if self._cleanup_thread is not None:
            self._cleanup_thread.join(timeout=5.0)
            self._cleanup_thread = None
        print("[TRACKER] Stopped gracefully.")
    
    # ========== Gate Camera Methods ==========
    
    def register_vehicle(self, ticket: VehicleTicket) -> str:
        """
        Register vehicle from gate camera.
        Called when vehicle ticket is finalized at gate.
        """
        with self._lock:
            pending = PendingVehicle(
                ticket=ticket,
                timestamp_registered=datetime.now()
            )
            self._pending_vehicles.append(pending)
            self._active_tickets[ticket.ticket_id] = ticket
            self._total_registered += 1
            
            print(f"[TRACKER] Registered: {ticket.plate_text}")
            
        return ticket.ticket_id
    
    # ========== Parking Camera Methods ==========
    
    def match_by_plate(self, plate_text: str) -> Optional[PendingVehicle]:
        """
        Find pending vehicle by plate text.
        Used when parking camera identifies a vehicle via FIFO queue or other means.
        """
        if not plate_text:
            return None
            
        with self._lock:
            # Normalize plate for comparison
            target_norm = plate_text.replace("-", "").replace(" ", "").upper()
            
            for pending in self._pending_vehicles:
                if pending.ticket.plate_text:
                    source_norm = pending.ticket.plate_text.replace("-", "").replace(" ", "").upper()
                    if source_norm == target_norm:
                        return pending
            
            return None

    def confirm_match(
        self, 
        pending: PendingVehicle, 
        parking_slot: Optional[int]
    ) -> MatchedVehicle:
        """
        Confirm a match and move from pending to matched.
        """
        with self._lock:
            # Create matched record
            # Score is practically 100/100 since we matched by reliable FIFO/Plate
            matched = MatchedVehicle.from_pending(pending, parking_slot, match_score=100.0)
            
            # Remove from pending, add to matched
            if pending in self._pending_vehicles:
                self._pending_vehicles.remove(pending)
            
            self._matched_vehicles.append(matched)
            self._total_matched += 1
            self._total_transit_time += matched.transit_time
            
            print(f"[TRACKER] Vehicle Parked: {pending.ticket.plate_text} -> Slot {parking_slot} "
                  f"(Transit: {matched.transit_time:.1f}s)")
            
            return matched
    
    # ========== Cleanup & Maintenance ==========
    
    def cleanup_expired(self) -> int:
        """Remove vehicles that have been pending too long."""
        with self._lock:
            now = datetime.now()
            expired = []
            
            for pending in self._pending_vehicles:
                if pending.age_seconds > self.max_wait_time:
                    expired.append(pending)
            
            for pending in expired:
                self._pending_vehicles.remove(pending)
                if pending.ticket_id in self._active_tickets:
                    del self._active_tickets[pending.ticket_id]
                self._total_expired += 1
                
                print(f"[TRACKER] Expired ticket: {pending.ticket.plate_text} (Age: {pending.age_seconds:.0f}s)")
            
            return len(expired)
    
    # ========== Query Methods ==========
    
    def get_pending_count(self) -> int:
        with self._lock:
            return len(self._pending_vehicles)
            
    def get_pending_vehicles(self) -> List[dict]:
        with self._lock:
            return [
                {
                    **p.ticket.to_dict(),
                    'age_seconds': round(p.age_seconds, 1),
                    'match_attempts': p.match_attempts
                }
                for p in self._pending_vehicles
            ]

    def get_matched_vehicles(self, limit: int = 50) -> List[dict]:
        with self._lock:
            return [m.to_dict() for m in self._matched_vehicles[-limit:]]
            
    def get_ticket(self, ticket_id: str) -> Optional[VehicleTicket]:
        with self._lock:
            return self._active_tickets.get(ticket_id)

    def get_stats(self) -> dict:
        with self._lock:
            avg_transit = (
                self._total_transit_time / self._total_matched 
                if self._total_matched > 0 else 0
            )
            return {
                'pending': len(self._pending_vehicles),
                'total_registered': self._total_registered,
                'total_matched': self._total_matched,
                'total_expired': self._total_expired,
                'average_transit_time': round(avg_transit, 1)
            }

# Global singleton instance
_tracker_instance: Optional[VehicleTracker] = None


def get_tracker() -> VehicleTracker:
    """Get or create global tracker instance (singleton)"""
    global _tracker_instance
    if _tracker_instance is None:
        _tracker_instance = VehicleTracker()
    return _tracker_instance
