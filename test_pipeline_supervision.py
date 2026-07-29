import queue
import threading
import unittest
from unittest.mock import patch

from main_pipe import MultiprocessPipeline


class PipelineSupervisionTests(unittest.TestCase):
    def test_child_exception_reports_role_and_sets_global_stop(self):
        pipeline = MultiprocessPipeline.__new__(MultiprocessPipeline)
        pipeline.q_failures = queue.Queue()
        pipeline.stop_event = threading.Event()

        def fail():
            raise ValueError("invalid pose")

        pipeline.reader_process = lambda: None
        pipeline.inference_process = lambda: None
        pipeline.analyzer_process = fail

        with patch("main_pipe.ignore_child_interrupts"):
            with self.assertRaisesRegex(ValueError, "invalid pose"):
                pipeline._process_entry("Analyzer")

        report = pipeline.q_failures.get_nowait()
        self.assertEqual(report["role"], "Analyzer")
        self.assertEqual(report["error_type"], "ValueError")
        self.assertIn("invalid pose", report["traceback"])
        self.assertTrue(pipeline.stop_event.is_set())


if __name__ == "__main__":
    unittest.main()
