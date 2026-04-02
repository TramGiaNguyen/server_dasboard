"""Regression tests for gate OCR scheduler / DB backfill helpers."""
import unittest

import shared.state as s


class TestGateOcrMergeCtx(unittest.TestCase):
    def tearDown(self):
        with s.gate_ocr_scheduler_lock:
            s.gate_ocr_track_db_ctx.clear()
            s.gate_ocr_latest_jobs.clear()
            s.gate_ocr_pending_queue.clear()
            s.gate_ocr_pending_enqueued.clear()
        with s.plate_fifo_lock:
            s.plate_fifo_queue.clear()

    def test_merge_in_top_level_ids(self):
        s.gate_ocr_merge_ctx_from_handoff(
            "track_a",
            {"gate_log_id": 42, "session_id": "abc", "direction": "in"},
        )
        self.assertEqual(s.gate_ocr_track_db_ctx["track_a"]["gate_log_id"], 42)
        self.assertEqual(s.gate_ocr_track_db_ctx["track_a"]["session_id"], "abc")

    def test_merge_out_nested_result_store(self):
        s.gate_ocr_merge_ctx_from_handoff(
            "track_b",
            {"direction": "out", "result_store": {"gate_log_id": 99}},
        )
        self.assertEqual(s.gate_ocr_track_db_ctx["track_b"]["gate_log_id"], 99)

    def test_enqueue_and_depth(self):
        s.gate_ocr_enqueue_job("t1", {"crop_frame": None, "frame_count": 1})
        s.gate_ocr_enqueue_job("t2", {"crop_frame": None, "frame_count": 2})
        pend, latest = s.gate_ocr_scheduler_depth()
        self.assertEqual(pend, 2)
        self.assertEqual(latest, 2)

    def test_update_plate_fifo_entry_assigned_but_reserved(self):
        # Simulate: parking side consumed the gate fifo entry for pairing
        # (assigned=True + reserved_ingress_seq), but OCR must still be able
        # to backfill plate/conf into that same FIFO entry.
        entry = {
            "gate_track_id": "t1",
            "ingress_seq": 1,
            "plate": None,
            "conf": 0.0,
            "assigned": True,
            "reserved_ingress_seq": 1,
            "timestamp": None,
        }
        with s.plate_fifo_lock:
            s.plate_fifo_queue.append(entry)

        updated = s.update_plate_fifo_entry("t1", "ABC123", 0.92)
        self.assertTrue(updated)
        self.assertEqual(entry["plate"], "ABC123")
        self.assertAlmostEqual(entry["conf"], 0.92)


if __name__ == "__main__":
    unittest.main()
