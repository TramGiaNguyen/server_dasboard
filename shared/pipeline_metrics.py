import time
from collections import deque
from typing import Deque, Dict


class PipelineMetrics:
    """Lightweight moving-window metrics logger for pipeline loops."""

    def __init__(self, name: str, report_interval_sec: float = 5.0) -> None:
        self.name = name
        self.report_interval_sec = report_interval_sec
        self._last_report = time.time()
        self._loop_ms: Deque[float] = deque(maxlen=150)
        self._emit_count = 0
        self._frame_count = 0
        self._last_queue_depth: Dict[str, int] = {}

    def mark_loop(self, loop_ms: float) -> None:
        self._loop_ms.append(loop_ms)
        self._frame_count += 1

    def mark_emit(self) -> None:
        self._emit_count += 1

    def set_queue_depth(self, queue_name: str, depth: int) -> None:
        self._last_queue_depth[queue_name] = max(0, int(depth))

    def maybe_report(self, logger=print) -> None:
        now = time.time()
        if now - self._last_report < self.report_interval_sec:
            return
        self._last_report = now
        avg_loop = (sum(self._loop_ms) / len(self._loop_ms)) if self._loop_ms else 0.0
        queues = ", ".join(f"{k}={v}" for k, v in sorted(self._last_queue_depth.items())) or "none"
        logger(
            f"[PERF][{self.name}] avg_loop_ms={avg_loop:.1f} "
            f"frames={self._frame_count} emits={self._emit_count} queues({queues})"
        )
