import time
import threading
from collections import deque
from typing import Deque, Dict, Optional


# Process-wide registry so HTTP handlers can pull metrics by name without having
# to grab the pipeline runtime object directly.
_REGISTRY: Dict[str, "PipelineMetrics"] = {}
_REGISTRY_LOCK = threading.Lock()


def get_registered(name: str) -> Optional["PipelineMetrics"]:
    """Lookup a metrics instance by pipeline name (e.g. 'gate_pipeline')."""
    with _REGISTRY_LOCK:
        return _REGISTRY.get(name)


def all_registered() -> Dict[str, "PipelineMetrics"]:
    with _REGISTRY_LOCK:
        return dict(_REGISTRY)


class PipelineMetrics:
    """Lightweight moving-window metrics logger for pipeline loops."""

    def __init__(self, name: str, report_interval_sec: float = 5.0) -> None:
        self.name = name
        self.report_interval_sec = report_interval_sec
        self._created_ts = time.time()
        self._last_report = self._created_ts
        self._loop_ms: Deque[float] = deque(maxlen=300)
        self._emit_count = 0
        self._frame_count = 0
        self._last_queue_depth: Dict[str, int] = {}
        # Per-call extra timings (e.g. OCR latency). Keyed by event name.
        self._timings: Dict[str, Deque[float]] = {}
        self._counters: Dict[str, int] = {}
        self._lock = threading.Lock()
        with _REGISTRY_LOCK:
            _REGISTRY[name] = self

    def mark_loop(self, loop_ms: float) -> None:
        with self._lock:
            self._loop_ms.append(loop_ms)
            self._frame_count += 1

    def mark_emit(self) -> None:
        with self._lock:
            self._emit_count += 1

    def mark_timing(self, event: str, elapsed_ms: float) -> None:
        with self._lock:
            buf = self._timings.get(event)
            if buf is None:
                buf = deque(maxlen=200)
                self._timings[event] = buf
            buf.append(float(elapsed_ms))

    def inc_counter(self, name: str, by: int = 1) -> None:
        with self._lock:
            self._counters[name] = self._counters.get(name, 0) + int(by)

    def set_queue_depth(self, queue_name: str, depth: int) -> None:
        with self._lock:
            self._last_queue_depth[queue_name] = max(0, int(depth))

    def maybe_report(self, logger=print) -> None:
        now = time.time()
        if now - self._last_report < self.report_interval_sec:
            return
        self._last_report = now
        snap = self.get_snapshot()
        queues = ", ".join(f"{k}={v}" for k, v in sorted(snap['queues'].items())) or "none"
        logger(
            f"[PERF][{self.name}] avg_loop_ms={snap['avg_loop_ms']:.1f} "
            f"fps={snap['loop_fps']:.1f} frames={snap['frames']} "
            f"emits={snap['emits']} queues({queues})"
        )

    def get_snapshot(self) -> dict:
        """Return a JSON-serialisable dict of current metrics."""
        with self._lock:
            n = len(self._loop_ms)
            if n:
                avg = sum(self._loop_ms) / n
                lo = min(self._loop_ms)
                hi = max(self._loop_ms)
                # Approx p95 — sort copy of small window
                xs = sorted(self._loop_ms)
                p95 = xs[max(0, int(0.95 * (n - 1)))]
            else:
                avg = lo = hi = p95 = 0.0
            loop_fps = (1000.0 / avg) if avg > 0 else 0.0
            uptime = max(0.001, time.time() - self._created_ts)
            timings_out = {}
            for ev, buf in self._timings.items():
                if not buf:
                    continue
                timings_out[ev] = {
                    "avg_ms": sum(buf) / len(buf),
                    "samples": len(buf),
                }
            return {
                "name": self.name,
                "uptime_sec": uptime,
                "frames": self._frame_count,
                "emits": self._emit_count,
                "avg_loop_ms": avg,
                "min_loop_ms": lo,
                "max_loop_ms": hi,
                "p95_loop_ms": p95,
                "loop_fps": loop_fps,
                "queues": dict(self._last_queue_depth),
                "timings": timings_out,
                "counters": dict(self._counters),
            }
