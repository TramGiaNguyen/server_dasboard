"""
RTSPCapture – Non-blocking RTSP/stream reader with background grab thread.

Problem solved:
  With a standard cv2.VideoCapture, cap.read() both grabs AND decodes the next
  buffered frame.  When the processing loop is slower than the camera FPS (e.g.
  YOLO takes 150 ms but camera pushes at 30 fps), frames pile up in the internal
  FFmpeg buffer.  The next cap.read() then returns a *stale* frame, causing the
  "freeze → jump" artefact seen during live demo.

Solution:
  A daemon thread continuously calls cap.read() and keeps only the last
  `buffer_size` frames in a Queue.  Old frames are dropped when the queue is
  full.  The processing loop calls RTSPCapture.read() which just pops from the
  queue – it always gets a recent frame and is never blocked by the network.

Usage:
    cap = RTSPCapture(rtsp_url, buffer_size=2)
    if not cap.open():
        return
    ...
    while True:
        ret, frame = cap.read()
        if not ret:
            continue          # brief timeout, try again
        ...process frame...
    cap.release()

Buffer size guidance:
  - Parking camera (static occupancy):  buffer_size=2  (only latest needed)
  - Gate camera (line-crossing events):  buffer_size=3  (small window preserved)

Note on Flask-SocketIO + eventlet/gevent:
  RTSPCapture uses stdlib threading.Thread.  If Flask-SocketIO monkey-patches
  threading (eventlet/gevent mode), the daemon thread still works correctly for
  I/O-bound RTSP reads.  No special adjustments are needed.
"""

import cv2
import time
import threading
from queue import Queue, Empty


class RTSPCapture:
    """Non-blocking RTSP / HTTP stream capture with a background reader thread."""

    def __init__(self, url: str, buffer_size: int = 2, reconnect_delay: float = 3.0):
        """
        Args:
            url:              RTSP or HTTP stream URL.
            buffer_size:      Maximum frames kept in the queue.  When full, the
                              oldest frame is discarded to make room for the latest.
            reconnect_delay:  Seconds to wait before reconnecting after stream loss.
        """
        self.url = url
        self.buffer_size = buffer_size
        self.reconnect_delay = reconnect_delay

        self._queue = Queue(maxsize=buffer_size)
        self._running = False
        self._thread = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def open(self) -> bool:
        """
        Open the stream and start the background reader thread.

        Returns:
            True  – stream opened successfully.
            False – could not open; caller should abort.
        """
        cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            cap.release()
            return False

        self._running = True
        self._thread = threading.Thread(
            target=self._reader_loop,
            args=(cap,),
            daemon=True,
            name="RTSPCapture-reader",
        )
        self._thread.start()
        return True

    def read(self, timeout: float = 2.0):
        """
        Retrieve the next buffered frame (non-blocking from the caller's view).

        Args:
            timeout: Seconds to wait when the queue is empty before giving up.

        Returns:
            (True,  frame)  – a recent frame is available.
            (False, None)   – timeout or stream not yet ready; caller should retry.
        """
        try:
            return self._queue.get(timeout=timeout)
        except Empty:
            return False, None

    def flush(self, wait_seconds: float = 1.0) -> None:
        """
        Discard all buffered frames and wait briefly for fresh ones.

        Call this after a startup synchronisation wait so the processing loop
        starts from a recent frame rather than stale buffered content.

        Args:
            wait_seconds: How long to wait for the background thread to fill
                          the queue with fresh frames.
        """
        self.drain()
        time.sleep(wait_seconds)

    def drain(self) -> int:
        """
        Drop every frame currently buffered, do NOT sleep. Returns the number
        of frames that were discarded.

        Use this when the worker has just finished a slow iteration (e.g. a
        GPU stall) and wants to read the *freshest* frame next, instead of a
        frame that became stale while the worker was busy. Cheap to call —
        non-blocking, no I/O.
        """
        dropped = 0
        while True:
            try:
                self._queue.get_nowait()
                dropped += 1
            except Empty:
                break
        return dropped

    def qsize(self) -> int:
        """Current number of buffered frames (approximate)."""
        return self._queue.qsize()

    def isOpened(self) -> bool:
        """True if the background reader thread is alive."""
        return self._running and self._thread is not None and self._thread.is_alive()

    def release(self) -> None:
        """Stop the background reader thread and release all resources."""
        self._running = False
        # Drain the queue so the reader thread is not blocked on put()
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except Empty:
                break
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _reader_loop(self, cap: cv2.VideoCapture) -> None:
        """
        Background thread body.

        Continuously reads frames from the stream.  On stream loss the thread
        waits `reconnect_delay` seconds and reopens the connection, so the
        caller's read() will automatically start receiving frames again once
        the camera recovers.
        """
        while self._running:
            ret, frame = cap.read()

            if not ret:
                print(
                    f"[RTSPCapture] Stream lost ({self.url}), "
                    f"reconnecting in {self.reconnect_delay:.0f}s ..."
                )
                cap.release()
                time.sleep(self.reconnect_delay)
                cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                continue

            # Drop the oldest frame when the queue is full to keep the latest
            if self._queue.full():
                try:
                    self._queue.get_nowait()
                except Empty:
                    pass
            try:
                self._queue.put_nowait((True, frame))
            except Exception:
                pass

        cap.release()
