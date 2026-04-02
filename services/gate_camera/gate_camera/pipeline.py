from shared.pipeline_metrics import PipelineMetrics
from shared.priority_worker_pool import get_priority_worker_pool


class GatePipelineRuntime:
    """Runtime helpers for gate pipeline orchestration."""

    def __init__(self) -> None:
        self._pool = get_priority_worker_pool()
        self.metrics = PipelineMetrics("gate_pipeline")
        self.queue_index_by_track = {}

    def submit_high(self, fn, *args, coalesce_group=None, coalesce_key=None, **kwargs):
        self._pool.submit_high(
            fn, *args,
            coalesce_group=coalesce_group, coalesce_key=coalesce_key, **kwargs,
        )

    def submit_low(self, fn, *args, coalesce_group=None, coalesce_key=None, **kwargs):
        self._pool.submit_low(
            fn, *args,
            coalesce_group=coalesce_group, coalesce_key=coalesce_key, **kwargs,
        )

    def submit_io(self, fn, *args, coalesce_group=None, coalesce_key=None, **kwargs):
        return self.submit_high(fn, *args, coalesce_group=coalesce_group, coalesce_key=coalesce_key, **kwargs)

    def mark_emit(self) -> None:
        self.metrics.mark_emit()

    def mark_loop(self, elapsed_ms: float, mailbox_depth: int, render_queue_depth: int, plate_fifo_depth: int) -> None:
        self.metrics.mark_loop(elapsed_ms)
        self.metrics.set_queue_depth("gate_ocr_scheduler", mailbox_depth)
        self.metrics.set_queue_depth("gate_render_queue", render_queue_depth)
        self.metrics.set_queue_depth("plate_fifo", plate_fifo_depth)
        self.metrics.maybe_report()
