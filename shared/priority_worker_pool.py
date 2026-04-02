"""
PriorityWorkerPool — tách IO pool thành 2 queue theo priority.

High priority: gate OCR backfill, slot updates, entry/exit plate writes
Low priority:  logging, improper parking, notifications, reservations

Mỗi queue hỗ trợ task coalescing: nếu task cùng (group, key) đã đang chạy,
submit tiếp theo sẽ bị bỏ qua thay vì tạo duplicate.
"""
from concurrent.futures import ThreadPoolExecutor
import threading
from typing import Any, Callable, Dict, Optional, Tuple


class PriorityWorkerPool:
    def __init__(self, high_workers: int = 4, low_workers: int = 4) -> None:
        self._high = ThreadPoolExecutor(
            max_workers=high_workers, thread_name_prefix="io-high"
        )
        self._low = ThreadPoolExecutor(
            max_workers=low_workers, thread_name_prefix="io-low"
        )
        self._high_lock = threading.Lock()
        self._low_lock = threading.Lock()
        # {(group, key): True} — task đang chạy hoặc đã submit, chưa xong
        self._high_inflight: Dict[Tuple[str, str], bool] = {}
        self._low_inflight: Dict[Tuple[str, str], bool] = {}

    # ── Internal coalescing helper ────────────────────────────────────────────

    def _submit_coalesced(
        self,
        executor: ThreadPoolExecutor,
        fn: Callable[..., Any],
        *args: Any,
        coalesce_group: Optional[str] = None,
        coalesce_key: Optional[str] = None,
        inflight: Optional[Dict[Tuple[str, str], bool]] = None,
        lock: Optional[threading.Lock] = None,
        **kwargs: Any,
    ):
        """Submit fn with optional coalescing. Returns None if coalesced (duplicate)."""
        key: Optional[Tuple[str, str]] = None
        if coalesce_group and coalesce_key and inflight is not None and lock is not None:
            key = (str(coalesce_group), str(coalesce_key))
            with lock:
                if inflight.get(key):
                    return None  # Coalesced: already in-flight
                inflight[key] = True

        def _run():
            try:
                return fn(*args, **kwargs)
            finally:
                if key is not None and inflight is not None and lock is not None:
                    with lock:
                        inflight.pop(key, None)

        return executor.submit(_run)

    # ── High priority ─────────────────────────────────────────────────────────

    def submit_high(
        self,
        fn: Callable[..., Any],
        *args: Any,
        coalesce_group: Optional[str] = None,
        coalesce_key: Optional[str] = None,
        **kwargs: Any,
    ):
        """Gate OCR backfill, slot updates, entry/exit plate writes.

        coalesce_group / coalesce_key: nếu cùng cặp đã đang chạy → skip (coalesced).
        """
        return self._submit_coalesced(
            self._high, fn, *args,
            coalesce_group=coalesce_group,
            coalesce_key=coalesce_key,
            inflight=self._high_inflight,
            lock=self._high_lock,
            **kwargs,
        )

    # ── Low priority ──────────────────────────────────────────────────────────

    def submit_low(
        self,
        fn: Callable[..., Any],
        *args: Any,
        coalesce_group: Optional[str] = None,
        coalesce_key: Optional[str] = None,
        **kwargs: Any,
    ):
        """Logging, improper parking, notifications, reservations."""
        return self._submit_coalesced(
            self._low, fn, *args,
            coalesce_group=coalesce_group,
            coalesce_key=coalesce_key,
            inflight=self._low_inflight,
            lock=self._low_lock,
            **kwargs,
        )

    # ── Backward-compat alias (trỏ vào high để không break code cũ) ─────────

    def submit(self, fn: Callable[..., Any], *args: Any, **kwargs: Any):
        """Alias cho submit_high — giữ backward compatibility."""
        return self.submit_high(fn, *args, **kwargs)

    # ── Lifecycle ───────────────────────────────────────────────────────────

    def shutdown(self, wait: bool = True) -> None:
        self._high.shutdown(wait=wait)
        self._low.shutdown(wait=wait)


# ── Singleton ──────────────────────────────────────────────────────────────────

_priority_pool: Optional[PriorityWorkerPool] = None
_pool_lock = threading.Lock()


def get_priority_worker_pool() -> PriorityWorkerPool:
    global _priority_pool
    if _priority_pool is None:
        with _pool_lock:
            if _priority_pool is None:
                _priority_pool = PriorityWorkerPool()
    return _priority_pool
