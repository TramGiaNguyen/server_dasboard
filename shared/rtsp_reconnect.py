"""
Robust RTSP Capture with Auto-Reconnect
Handles network failures and automatically reconnects to RTSP stream
"""
import time
import cv2


class RobustRTSPCapture:
    """RTSP capture with auto-reconnect on failure"""
    
    def __init__(self, url, buffer_size=2, reconnect_delay=2.0):
        """
        Initialize robust RTSP capture.
        
        Args:
            url: RTSP stream URL
            buffer_size: Number of frames to buffer (smaller = lower latency)
            reconnect_delay: Seconds to wait before reconnecting
        """
        self.url = url
        self.buffer_size = buffer_size
        self.reconnect_delay = reconnect_delay
        self.cap = None
        self.consecutive_failures = 0
        self.max_failures = 10
        self.total_reconnects = 0
    
    def open(self):
        """Open RTSP stream with retry"""
        from shared.rtsp_capture import RTSPCapture
        
        for attempt in range(3):
            try:
                self.cap = RTSPCapture(self.url, buffer_size=self.buffer_size)
                if self.cap.open():
                    self.consecutive_failures = 0
                    print(f"✅ [RTSP] Connected: {self.url}")
                    return True
            except Exception as e:
                print(f"❌ [RTSP] Connection failed (attempt {attempt+1}/3): {e}")
                time.sleep(self.reconnect_delay)
        
        return False
    
    def read(self, timeout=0.1):
        """Read frame with auto-reconnect on failure"""
        if self.cap is None:
            if not self.open():
                return False, None
        
        ret, frame = self.cap.read(timeout=timeout)
        
        if not ret or frame is None:
            self.consecutive_failures += 1
            
            if self.consecutive_failures >= self.max_failures:
                print(f"⚠️  [RTSP] Too many failures ({self.consecutive_failures}), reconnecting...")
                self.total_reconnects += 1
                self.cap = None
                if self.open():
                    self.cap.flush(wait_seconds=1.0)
                    return self.cap.read(timeout=timeout)
            
            return False, None
        
        self.consecutive_failures = 0
        return True, frame
    
    def flush(self, wait_seconds=1.0):
        """Flush buffer to discard old frames"""
        if self.cap:
            self.cap.flush(wait_seconds=wait_seconds)
    
    def get_stats(self):
        """Get connection statistics"""
        return {
            'consecutive_failures': self.consecutive_failures,
            'total_reconnects': self.total_reconnects,
            'is_connected': self.cap is not None
        }
