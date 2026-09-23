"""Asynchronous runtime for frame journals, Swing events, and Coach sidecars.

The Analyzer process only submits versioned frame records and rendered frames.
Swing segmentation, local coaching, persistence, and DeepSeek callbacks execute
behind this module's small interface.
"""

from __future__ import annotations

import queue
import threading
from typing import Callable, Dict, Optional


def build_impact_telemetry_card(event_dict: Dict) -> Dict:
    """Build telemetry card dictionary for DualViewRenderer overlay."""
    bio = event_dict.get("biomechanics") or {}
    ext = bio.get("extended_biomechanics") or event_dict.get("extended_biomechanics") or {}
    score = ext.get("swing_quality_score") or {}
    rkt = ext.get("racket_head_speed") or {}
    brush = ext.get("brush_angle") or {}
    stc = ext.get("stance") or {}
    leg = ext.get("leg_drive") or {}
    seq = ext.get("kinematic_sequence") or {}
    return {
        "stroke_type": event_dict.get("stroke_type", "FOREHAND"),
        "swing_score": score.get("overall_score", 0.0),
        "swing_grade": score.get("grade", "N/A"),
        "racket_speed_kmh": rkt.get("contact_kmh", 0.0),
        "racket_max_speed_kmh": rkt.get("max_kmh", 0.0),
        "brush_angle_deg": brush.get("low_to_high_angle_deg", 0.0),
        "drop_depth_ratio": brush.get("drop_depth_ratio"),
        "stance_type": stc.get("stance_type", "Semi-Open Stance"),
        "leg_drive_ratio": leg.get("drive_ratio"),
        "kinematic_sequence_text": f"腿 -> 髋 -> 肩 -> 拍 ({seq.get('sequence_quality', 'OPTIMAL')})",
    }


class RealtimeSwingRuntime:
    """Run the non-visual realtime data path independently from OSD rendering."""

    def __init__(
        self,
        *,
        engine,
        output,
        frame_journal=None,
        deepseek_sidecar=None,
        coach_tts_sidecar=None,
        stop_event=None,
        queue_size: int = 512,
        logger: Callable[[str], None] = print,
    ):
        self.engine = engine
        self.output = output
        self.frame_journal = frame_journal
        self.deepseek_sidecar = deepseek_sidecar
        self.coach_tts_sidecar = coach_tts_sidecar
        self.stop_event = stop_event
        self.logger = logger
        self._queue: queue.Queue = queue.Queue(maxsize=max(1, int(queue_size)))
        self._sentinel = object()
        self._closed = False
        self._worker_error: Optional[BaseException] = None
        self._active_display_lock = threading.Lock()
        fps_val = float(getattr(engine, "fps", 25.0) or 25.0)
        self._active_hold_frames = max(10, int(round(fps_val * 1.0)))
        self._active_display = {
            "event_label": "READY STANCE",
            "coaching_text": "",
            "telemetry_card": None,
            "expire_at_frame": -1,
            "event_id": -1,
        }
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
        if self.coach_tts_sidecar is not None:
            try:
                self.coach_tts_sidecar.close()
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
        if self.coach_tts_sidecar is not None:
            for event in events:
                event["coach_tts"] = self.coach_tts_sidecar.pending_payload(event)
        if events or final:
            self.output.publish_events(events, self.engine.snapshot())
        for event in events:
            prefix = "Final Event" if final else "Event"
            self.logger(
                f"🎾 [Swing-Live] {prefix} #{event['event_id']}"
                f" | {event['stroke_type']}"
                f" | frames {event['start_frame']}-{event['end_frame']}"
                f" | contact {event['contact_frame']}"
                f" | latency {event.get('latency_frames', 0)}F"
            )
            advices = event.get("coach_advices") or []
            if not advices and event.get("coach_advice"):
                advices = [event["coach_advice"]]
            coach_msg = ""
            for index, advice in enumerate(advices[:3], start=1):
                if not advice.get("message"):
                    continue
                if not coach_msg:
                    coach_msg = advice["message"]
                self.logger(
                    f"🎯 [Coach] Swing #{event['event_id']}"
                    f" | {index}/{len(advices[:3])}"
                    f" | {advice['message']}"
                    f" | {float(advice.get('confidence') or 0.0):.0%}"
                )

            # 更新用于实时双视角渲染的 Active Display 状态
            conf_val = float(event.get("confidence", 0.9) or 0.9)
            stroke_name = str(event.get("stroke_type", "SWING")).upper()
            display_label = f"{stroke_name} ({conf_val * 100:.0f}%)"
            card_info = build_impact_telemetry_card(event)
            contact_f = int(event.get("contact_frame", 0))
            emitted_at = int(event.get("emitted_at_frame", contact_f))
            expire_f = max(contact_f, emitted_at) + self._active_hold_frames

            with self._active_display_lock:
                self._active_display = {
                    "event_label": display_label,
                    "coaching_text": coach_msg,
                    "telemetry_card": card_info,
                    "expire_at_frame": expire_f,
                    "event_id": int(event.get("event_id", -1)),
                }

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
            if self.coach_tts_sidecar is not None:
                event_id = int(event["event_id"])
                self.coach_tts_sidecar.submit(
                    event,
                    lambda result, target_event_id=event_id: self._publish_coach_tts(
                        target_event_id,
                        result,
                    ),
                )

    def get_active_display_state(self, current_frame_id: int) -> Dict:
        """Return real-time display HUD information (event label, coaching advice, telemetry card)."""
        with self._active_display_lock:
            if current_frame_id <= self._active_display.get("expire_at_frame", -1):
                return {
                    "event_label": self._active_display.get("event_label", "READY STANCE"),
                    "coaching_text": self._active_display.get("coaching_text", ""),
                    "telemetry_card": self._active_display.get("telemetry_card"),
                    "event_id": self._active_display.get("event_id", -1),
                    "active": True,
                }
            return {
                "event_label": "READY STANCE",
                "coaching_text": "",
                "telemetry_card": None,
                "event_id": -1,
                "active": False,
            }

    def _publish_deepseek(self, event_id: int, result: Dict) -> None:
        self.output.update_event(event_id, {"deepseek_advice": result})
        if result.get("status") == "ready":
            self.logger(
                f"🧠 [DeepSeek] Swing #{event_id}"
                f" | {result['message']}"
                f" | {int(result.get('latency_ms') or 0)}ms"
            )
        elif result.get("status") in {"failed", "unavailable", "skipped"}:
            self.logger(
                f"⚠️ [DeepSeek] Swing #{event_id}"
                f" | {result.get('status')}"
                " | 本地建议继续生效"
            )

    def _publish_coach_tts(self, event_id: int, result: Dict) -> None:
        """Persist optional local speech state without changing Coach text."""
        self.output.update_event(event_id, {"coach_tts": result})

    def _fail(self, exc: BaseException) -> None:
        if self._worker_error is None:
            self._worker_error = exc
            self.logger(f"❌ [Swing-Live] 实时数据线程失败: {exc}")
            if self.stop_event is not None:
                self.stop_event.set()

    def _raise_if_failed(self) -> None:
        if self.output is not None and hasattr(self.output, "check_health"):
            self.output.check_health()
        if self._worker_error is not None:
            raise RuntimeError("Realtime Swing data runtime failed") from self._worker_error
