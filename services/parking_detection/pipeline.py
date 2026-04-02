from shared.pipeline_metrics import PipelineMetrics
from shared.priority_worker_pool import get_priority_worker_pool


class ParkingPipelineRuntime:
    """Runtime helpers for parking pipeline orchestration."""

    def __init__(self) -> None:
        self._pool = get_priority_worker_pool()
        self.metrics = PipelineMetrics("parking_pipeline")

    def submit_high(self, fn, *args, coalesce_group=None, coalesce_key=None, **kwargs):
        """High priority: slot updates, plate writes."""
        self._pool.submit_high(
            fn,
            *args,
            coalesce_group=coalesce_group,
            coalesce_key=coalesce_key,
            **kwargs,
        )

    def submit_low(self, fn, *args, coalesce_group=None, coalesce_key=None, **kwargs):
        """Low priority: logging, notifications, cleanup."""
        self._pool.submit_low(
            fn,
            *args,
            coalesce_group=coalesce_group,
            coalesce_key=coalesce_key,
            **kwargs,
        )

    def submit_io(self, fn, *args, coalesce_group=None, coalesce_key=None, **kwargs):
        """Backward-compat alias: route to high priority pool."""
        return self.submit_high(fn, *args, coalesce_group=coalesce_group, coalesce_key=coalesce_key, **kwargs)

    def mark_emit(self) -> None:
        self.metrics.mark_emit()

    def mark_loop(self, elapsed_ms: float, plate_fifo_depth: int, parking_trigger_depth: int, matched_depth: int) -> None:
        self.metrics.mark_loop(elapsed_ms)
        self.metrics.set_queue_depth("plate_fifo", plate_fifo_depth)
        self.metrics.set_queue_depth("parking_trigger", parking_trigger_depth)
        self.metrics.set_queue_depth("matched", matched_depth)
        self.metrics.maybe_report()
