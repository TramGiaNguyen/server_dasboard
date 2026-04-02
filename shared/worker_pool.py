import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, Optional, Tuple


class SharedWorkerPool:
    """
    Shared IO worker pool with optional task coalescing.

    Coalescing key prevents scheduling duplicate work bursts for the same entity
    (for example repeated update on the same gate log in a short window).
    """

    def __init__(self, max_workers: int = 8) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="io-worker")
        self._lock = threading.Lock()
        self._inflight: Dict[Tuple[str, str], bool] = {}

    def submit(
        self,
        fn: Callable[..., Any],
        *args: Any,
        coalesce_group: Optional[str] = None,
        coalesce_key: Optional[str] = None,
        **kwargs: Any,
    ):
        key: Optional[Tuple[str, str]] = None
        if coalesce_group and coalesce_key:
            key = (coalesce_group, coalesce_key)
            with self._lock:
                if self._inflight.get(key):
                    return None
                self._inflight[key] = True

        def _run():
            try:
                return fn(*args, **kwargs)
            finally:
                if key is not None:
                    with self._lock:
                        self._inflight.pop(key, None)

        return self._executor.submit(_run)


_pool: Optional[SharedWorkerPool] = None
_pool_lock = threading.Lock()


def get_shared_worker_pool() -> SharedWorkerPool:
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                _pool = SharedWorkerPool(max_workers=8)
    return _pool
