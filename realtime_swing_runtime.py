"""Asynchronous runtime for frame journals, Swing events, and Coach sidecars.

The Analyzer process only submits versioned frame records and rendered frames.
Swing segmentation, local coaching, persistence, and DeepSeek callbacks execute
behind this module's small interface.
"""

from __future__ import annotations

import queue
import threading
from typing import Callable, Dict, Optional


class RealtimeSwingRuntime:
    """Run the non-visual realtime data path independently from OSD rendering."""

    def __init__(
        self,
        *,
        engine,
        output,
        frame_journal=None,
        deepseek_sidecar=None,
        stop_event=None,
        queue_size: int = 512,
        logger: Callable[[str], None] = print,
    ):
        self.engine = engine
        self.output = output
        self.frame_journal = frame_journal
        self.deepseek_sidecar = deepseek_sidecar
        self.stop_event = stop_event
        self.logger = logger
        self._queue: queue.Queue = queue.Queue(maxsize=max(1, int(queue_size)))
        self._sentinel = object()
        self._closed = False
        self._worker_error: Optional[BaseException] = None
        self._thread = threading.Thread(
            target=self._run,
            name="realtime-swing-analysis",
            daemon=True,
        )
        self._thread.start()

    def submit_frame(self, frame_record: Dict) -> None:
        """Submit one completed FrameRecord without running Swing analysis inline."""
        self._raise_if_failed()
        if self._closed:
            raise RuntimeError("Cannot submit a frame after realtime Swing runtime closes")
        self._queue.put(frame_record)

    def record_rendered_frame(self, frame_id: int, frame) -> None:
        """Submit an OSD frame to the already-asynchronous clip/preview buffer."""
        self._raise_if_failed()
        if self._closed:
            raise RuntimeError("Cannot record a frame after realtime Swing runtime closes")
        if self.output is not None:
            self.output.record_frame(frame_id, frame)

    def close(self) -> None:
        """Drain records, finalize events and sidecars, then surface any failure."""
        if self._closed:
            self._raise_if_failed()
            return
        self._closed = True
        self._queue.join()
        self._queue.put(self._sentinel)
        self._queue.join()
        self._thread.join()

        close_errors = []
        if self._worker_error is None and self.engine is not None and self.output is not None:
            try:
                self._publish(self.engine.flush(), final=True)
            except BaseException as exc:
                self._fail(exc)
        if self.deepseek_sidecar is not None:
            try:
                self.deepseek_sidecar.close()
            except BaseException as exc:
                close_errors.append(exc)
        if self.output is not None:
            try:
                self.output.close()
            except BaseException as exc:
                close_errors.append(exc)
        if self.frame_journal is not None:
            try:
                self.frame_journal.close()
            except BaseException as exc:
                close_errors.append(exc)
        if self._worker_error is None and close_errors:
            self._worker_error = close_errors[0]
        self._raise_if_failed()

    def snapshot(self) -> Dict:
        if self.engine is None:
            return {"summary": {}, "events": [], "frame_trace": []}
        return self.engine.snapshot()

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is self._sentinel:
                    return
                if self._worker_error is not None:
                    continue
                if self.frame_journal is not None:
                    self.frame_journal.record(item)
                if self.engine is not None and self.output is not None:
                    self._publish(self.engine.push_frame(item))
            except BaseException as exc:
                self._fail(exc)
            finally:
                self._queue.task_done()

    def _publish(self, events, final: bool = False) -> None:
        if self.deepseek_sidecar is not None:
            for event in events:
                event["deepseek_advice"] = {
                    "status": "pending",
                    "model": self.deepseek_sidecar.model,
                    "source": "deepseek_sidecar",
                }
        if events or final:
            self.output.publish_events(events, self.engine.snapshot())
        for event in events:
            prefix = "Final Event" if final else "Event"
            self.logger(
                f"🎾 [Swing-Live] {prefix} #{event['event_id']}"
                f" | {event['stroke_type']}"
                f" | frames {event['start_frame']}-{event['end_frame']}"
                f" | contact {event['contact_frame']}"
                f" | latency {event['latency_frames']}F"
            )
            advices = event.get("coach_advices") or []
            if not advices and event.get("coach_advice"):
                advices = [event["coach_advice"]]
            for index, advice in enumerate(advices[:3], start=1):
                if not advice.get("message"):
                    continue
                self.logger(
                    f"🎯 [Coach] Swing #{event['event_id']}"
                    f" | {index}/{len(advices[:3])}"
                    f" | {advice['message']}"
                    f" | {float(advice.get('confidence') or 0.0):.0%}"
                )
            if self.deepseek_sidecar is not None:
                event_id = int(event["event_id"])
                frame_records = self.engine.frame_records_for_event(event)
                self.deepseek_sidecar.submit(
                    event,
                    lambda result, target_event_id=event_id: self._publish_deepseek(
                        target_event_id,
                        result,
                    ),
                    frame_records=frame_records,
                )

    def _publish_deepseek(self, event_id: int, result: Dict) -> None:
        self.output.update_event(event_id, {"deepseek_advice": result})
        if result.get("status") == "ready":
            self.logger(
                f"🧠 [DeepSeek] Swing #{event_id}"
                f" | {result['message']}"
                f" | {int(result.get('latency_ms') or 0)}ms"
            )
        elif result.get("status") in {"failed", "unavailable"}:
            self.logger(
                f"⚠️ [DeepSeek] Swing #{event_id}"
                f" | {result.get('status')}"
                " | 本地建议继续生效"
            )

    def _fail(self, exc: BaseException) -> None:
        if self._worker_error is None:
            self._worker_error = exc
            self.logger(f"❌ [Swing-Live] 实时数据线程失败: {exc}")
            if self.stop_event is not None:
                self.stop_event.set()

    def _raise_if_failed(self) -> None:
        if self._worker_error is not None:
            raise RuntimeError("Realtime Swing data runtime failed") from self._worker_error
