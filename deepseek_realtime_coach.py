"""Asynchronous DeepSeek sidecar guidance for completed Swing events."""

from __future__ import annotations

import json
import time
import urllib.request
from concurrent.futures import Future, ThreadPoolExecutor
from copy import deepcopy
from statistics import median
from typing import Callable, Dict, List, Optional

from deepseek_evidence_view import build_deepseek_evidence_view
from swing_coach_data_collector import build_coach_dataset
from swing_motion_features import extract_motion_features
from swing_quality_policy import is_ball_capture_advice


class EvidencePolicyError(ValueError):
    """Raised when remote advice exceeds the evidence packet's claim policy."""


class UrllibDeepSeekTransport:
    """HTTP adapter for DeepSeek's OpenAI-compatible ChatCompletions endpoint."""

    def complete(
        self,
        base_url: str,
        api_key: str,
        payload: Dict,
        timeout: float,
    ) -> Dict:
        endpoint = base_url.rstrip("/")
        if not endpoint.endswith("/chat/completions"):
            endpoint = f"{endpoint}/chat/completions"
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))


class DeepSeekCoachSidecar:
    """Generate optional remote advice without delaying local Coach output."""

    HARD_MAX_CHARS = 15

    def __init__(
        self,
        api_key: str,
        model: str = "deepseek-v4-flash",
        base_url: str = "https://api.deepseek.com",
        timeout_seconds: float = 3.0,
        max_chars: int = HARD_MAX_CHARS,
        workers: int = 2,
        transport=None,
    ):
        self.api_key = str(api_key or "")
        self.model = str(model or "deepseek-v4-flash")
        self.base_url = str(base_url or "https://api.deepseek.com")
        self.timeout_seconds = max(0.2, float(timeout_seconds))
        self.max_chars = min(self.HARD_MAX_CHARS, max(1, int(max_chars)))
        self.transport = transport or UrllibDeepSeekTransport()
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, int(workers)),
            thread_name_prefix="deepseek-coach",
        )
        self._futures = []
        self._closed = False

    def submit(
        self,
        event: Dict,
        callback: Callable[[Dict], None],
        frame_records: Optional[List[Dict]] = None,
        evidence_packet: Optional[Dict] = None,
    ) -> Future:
        """Queue one completed event and deliver a status document to callback."""
        if self._closed:
            raise RuntimeError("Cannot submit after the DeepSeek Coach sidecar is closed")
        future = self._executor.submit(
            self._generate,
            deepcopy(event),
            deepcopy(frame_records or []),
            deepcopy(evidence_packet),
        )
        future.add_done_callback(
            lambda completed: self._deliver(completed, callback)
        )
        self._futures.append(future)
        return future

    def close(self) -> None:
        """Wait only for bounded in-flight HTTP calls."""
        if self._closed:
            return
        self._closed = True
        self._executor.shutdown(wait=True)

    def _generate(
        self,
        event: Dict,
        frame_records: List[Dict],
        evidence_packet: Optional[Dict],
    ) -> Dict:
        started = time.perf_counter()
        if not self.api_key:
            return {
                "status": "unavailable",
                "reason": "missing_api_key",
                "model": self.model,
                "source": "deepseek_sidecar",
                "latency_ms": 0,
            }
        try:
            model_view = self._model_view(
                event,
                frame_records=frame_records,
                evidence_packet=evidence_packet,
            )
            response = self.transport.complete(
                base_url=self.base_url,
                api_key=self.api_key,
                payload=self._request_payload_from_view(model_view),
                timeout=self.timeout_seconds,
            )
            content = response["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            message = self._normalize_message(parsed.get("message"))
            if not message:
                raise ValueError("DeepSeek returned an empty advice message")
            category = str(parsed.get("category") or "").strip().lower()
            self._validate_advice(parsed, message, category, model_view)
            evidence_frames = [
                int(frame_id)
                for frame_id in parsed.get("evidence_frames") or []
            ]
            confidence = max(
                0.0,
                min(1.0, float(parsed.get("confidence") or 0.0)),
            )
            return {
                "status": "ready",
                "message": message[: self.max_chars],
                "focus": str(parsed.get("focus") or "general")[:40],
                "category": category,
                "evidence_frames": evidence_frames[:8],
                "confidence": round(confidence, 4),
                "model": self.model,
                "source": "deepseek_sidecar",
                "latency_ms": self._elapsed_ms(started),
            }
        except Exception as exc:
            return {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc)[:200],
                "model": self.model,
                "source": "deepseek_sidecar",
                "latency_ms": self._elapsed_ms(started),
            }

    def _request_payload(
        self,
        event: Dict,
        frame_records: Optional[List[Dict]] = None,
        evidence_packet: Optional[Dict] = None,
    ) -> Dict:
        model_view = self._model_view(
            event,
            frame_records=frame_records or [],
            evidence_packet=evidence_packet,
        )
        return self._request_payload_from_view(model_view)

    def _model_view(
        self,
        event: Dict,
        frame_records: List[Dict],
        evidence_packet: Optional[Dict],
    ) -> Dict:
        packet = evidence_packet or self._legacy_evidence_packet(
            event,
            frame_records,
        )
        model_view = build_deepseek_evidence_view(packet)
        model_view.setdefault("decision_policy", {})[
            "max_advice_chars"
        ] = self.max_chars
        return model_view

    def _request_payload_from_view(self, model_view: Dict) -> Dict:
        system_prompt = (
            "你是谨慎、简洁的实时网球教练。输入是按证据优先级整理的挥拍JSON。"
            "必须遵守decision_policy：只能使用allowed_advice_categories，绝不讨论"
            "prohibited_claims；null、缺失或低置信度都代表证据不足，不能自行补全。"
            "只把decision_policy.effective_warnings当作可行动警告；"
            "tolerated_conditions中的间歇漏检不要作为拍摄问题。"
            "coaching_allowed为true且advice_candidates非空时，必须优先从最高优先级"
            "候选中给出技术或正向建议，focus必须使用候选focus；"
            "review_required只表示部分证据域需复核，不能因此拒绝其他可靠证据域的技术评价。"
            "只有coaching_allowed为false时才只给拍摄改善或复核建议。"
            "像素和每处理帧速度只可在本视频内部作相对判断，不能对标职业标准。"
            "不得讨论blocked_advice_topics、上旋、拍面、落点等被禁止结论。"
            "综合整个frame_sequence，只选一个最重要且立刻可执行的动作。"
            "message必须是自然简短中文，不超过decision_policy.max_advice_chars；"
            "拍摄画面使用“入镜”，不要使用“导入”。focus使用snake_case英文标签。"
            "evidence_frames只能引用输入中真实存在的frame_id。只输出JSON对象："
            '{"message":"建议","focus":"标签","category":"capture|review|technique|positive",'
            '"evidence_frames":[帧号],"confidence":0到1}。'
        )
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": json.dumps(
                        model_view,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                },
            ],
            "thinking": {"type": "disabled"},
            "response_format": {"type": "json_object"},
            "temperature": 0.1,
            "max_tokens": 120,
            "stream": False,
        }

    @staticmethod
    def _legacy_evidence_packet(event: Dict, frame_records: List[Dict]) -> Dict:
        records = deepcopy(frame_records)
        event_copy = deepcopy(event)
        frame_ids = [
            int(row["frame_id"])
            for row in records
            if row.get("frame_id") is not None
        ]
        if frame_ids:
            event_copy.setdefault("start_frame", min(frame_ids))
            event_copy.setdefault("contact_frame", event_copy["start_frame"])
            event_copy.setdefault("peak_frame", event_copy["contact_frame"])
            event_copy.setdefault("end_frame", max(frame_ids))
            event_copy.setdefault(
                "duration_frames",
                event_copy["end_frame"] - event_copy["start_frame"] + 1,
            )
        features = extract_motion_features(records)
        fps = DeepSeekCoachSidecar._estimate_fps(records)
        coach_metrics = {}
        if records and frame_ids:
            coach_document = build_coach_dataset(
                {
                    "video_info": {"fps": fps},
                    "frames": records,
                },
                {
                    "summary": {"thresholds": {"dominant_hand": "right"}},
                    "events": [event_copy],
                    "features": features,
                    "frame_trace": [],
                },
            )
            if coach_document.get("events"):
                coach_metrics = coach_document["events"][0]
                coach_metrics.pop("frame_trace", None)
        return {
            "schema_version": "realtime_swing_evidence_v1",
            "event_id": int(event_copy.get("event_id", -1)),
            "video_context": {
                "fps": fps,
                "coordinate_space": "screen_pixels",
                "timestamp_unit": "seconds",
                "distance_unit": "pixels",
                "motion_feature_speed_unit": "pixels_per_processed_record",
            },
            "player_context": {},
            "event_summary": event_copy,
            "coach_metrics": coach_metrics,
            "event_frame_records": records,
            "motion_features": features,
            "frame_trace": [],
            "recent_swings": [],
            "integrity": {
                "frame_record_count": len(records),
                "motion_feature_count": len(records),
                "frame_trace_count": 0,
                "aligned_to_event_range": True,
            },
        }

    @staticmethod
    def _estimate_fps(frame_records: List[Dict]) -> float:
        timestamps = [
            float(row["timestamp"])
            for row in frame_records
            if row.get("timestamp") is not None
        ]
        deltas = [
            current - previous
            for previous, current in zip(timestamps, timestamps[1:])
            if current > previous
        ]
        if not deltas:
            return 25.0
        frame_delta = median(deltas)
        return round(1.0 / frame_delta, 4) if frame_delta > 0 else 25.0

    @staticmethod
    def _normalize_message(value) -> str:
        message = "".join(str(value or "").split())
        replacements = {
            "导入": "入镜",
            "输入画面": "入镜",
        }
        for source, target in replacements.items():
            message = message.replace(source, target)
        return message

    @staticmethod
    def _validate_advice(
        parsed: Dict,
        message: str,
        category: str,
        model_view: Dict,
    ) -> None:
        policy = model_view.get("decision_policy") or {}
        allowed = set(policy.get("allowed_advice_categories") or [])
        if not category or category not in allowed:
            raise EvidencePolicyError(
                f"advice category {category or 'missing'} is not allowed"
            )
        candidates = policy.get("advice_candidates") or []
        if (
            policy.get("coaching_allowed")
            and candidates
            and category not in {"technique", "positive"}
        ):
            raise EvidencePolicyError(
                "review or capture advice cannot replace available body coaching"
            )
        candidate_focuses = {
            str(candidate.get("focus") or "")
            for candidate in candidates
        }
        response_focus = str(parsed.get("focus") or "")
        if (
            category in {"technique", "positive"}
            and candidate_focuses
            and response_focus not in candidate_focuses
        ):
            raise EvidencePolicyError(
                f"advice focus {response_focus or 'missing'} is not an evidence candidate"
            )
        valid_frames = {
            int(row["frame_id"])
            for row in model_view.get("frame_sequence") or []
        }
        evidence_frames = parsed.get("evidence_frames")
        if not isinstance(evidence_frames, list):
            raise EvidencePolicyError("evidence_frames must be a list")
        if any(int(frame_id) not in valid_frames for frame_id in evidence_frames):
            raise EvidencePolicyError("advice cites a frame outside the event")
        claim_text = f"{parsed.get('focus') or ''} {message}".lower()
        if (
            "intermittent_ball_detection"
            in set(policy.get("tolerated_conditions") or [])
            and is_ball_capture_advice(
                {
                    "focus": parsed.get("focus"),
                    "message": message,
                }
            )
        ):
            raise EvidencePolicyError(
                "advice treats tolerated intermittent ball detection as a capture issue"
            )
        claim_terms = {
            "spin": ("spin", "topspin", "slice", "上旋", "下旋", "旋转"),
            "landing": ("landing", "bounce", "落点", "弹跳"),
            "racket_face": ("racket_face", "拍面"),
            "net_clearance": ("net_clearance", "过网高度"),
        }
        for claim in policy.get("prohibited_claims") or []:
            if any(term in claim_text for term in claim_terms.get(claim, ())):
                raise EvidencePolicyError(
                    f"advice uses prohibited claim {claim}"
                )

    @staticmethod
    def _deliver(future: Future, callback: Callable[[Dict], None]) -> None:
        try:
            callback(future.result())
        except Exception:
            # Callback failures must not escape the sidecar worker thread.
            return

    @staticmethod
    def _elapsed_ms(started: float) -> int:
        return max(0, int(round((time.perf_counter() - started) * 1000)))
