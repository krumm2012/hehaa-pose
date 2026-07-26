import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from deepseek_realtime_coach import DeepSeekCoachSidecar, UrllibDeepSeekTransport


class FakeDeepSeekTransport:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.requests = []

    def complete(self, base_url, api_key, payload, timeout):
        self.requests.append(
            {
                "base_url": base_url,
                "api_key": api_key,
                "payload": payload,
                "timeout": timeout,
            }
        )
        if self.error is not None:
            raise self.error
        return self.response


class DeepSeekCoachSidecarTests(unittest.TestCase):
    def test_returns_short_structured_advice_without_blocking_local_result(self):
        transport = FakeDeepSeekTransport(
            response={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "message": "提前转肩充分引拍",
                                    "focus": "preparation",
                                    "category": "technique",
                                    "evidence_frames": [31],
                                    "confidence": 0.78,
                                },
                                ensure_ascii=False,
                            )
                        }
                    }
                ]
            }
        )
        sidecar = DeepSeekCoachSidecar(
            api_key="test-key",
            model="deepseek-v4-flash",
            base_url="https://api.deepseek.com",
            timeout_seconds=2.5,
            max_chars=15,
            transport=transport,
        )
        results = []
        event = {
            "event_id": 3,
            "stroke_type": "Forehand",
            "confidence": 0.87,
            "phase_counts": {"backswing": 2, "follow_through": 7},
            "quality_flags": {"warnings": []},
            "coach_advice": {"message": "提前准备充分引拍"},
        }
        event_frame_records = [
            {
                "frame_id": 31,
                "timestamp": 1.24,
                "swing_type": "Forehand",
                "ball": [320, 180],
                "rackets": [{"box": [100, 90, 140, 130], "confidence": 0.9}],
                "pose": {"right_wrist": [120, 110]},
                "metrics": {"swing_motion": {"arm_ext": "145.0deg"}},
                "detection_diagnostics": {"final_decision": "selected"},
            },
            {
                "frame_id": 32,
                "timestamp": 1.28,
                "swing_type": "Forehand",
                "ball": None,
                "rackets": [],
                "pose": {"right_wrist": [130, 112]},
                "metrics": {"swing_motion": {"arm_ext": "148.0deg"}},
                "detection_diagnostics": {"final_decision": "no_candidates"},
            },
        ]

        sidecar.submit(
            event,
            results.append,
            frame_records=event_frame_records,
        )
        sidecar.close()

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["status"], "ready")
        self.assertEqual(results[0]["message"], "提前转肩充分引拍")
        self.assertEqual(results[0]["category"], "technique")
        self.assertEqual(results[0]["evidence_frames"], [31])
        self.assertLessEqual(len(results[0]["message"]), 15)
        request = transport.requests[0]
        self.assertEqual(request["payload"]["model"], "deepseek-v4-flash")
        self.assertEqual(request["payload"]["thinking"], {"type": "disabled"})
        self.assertEqual(
            request["payload"]["response_format"],
            {"type": "json_object"},
        )
        self.assertEqual(request["payload"]["temperature"], 0.1)
        evidence = json.loads(request["payload"]["messages"][1]["content"])
        self.assertEqual(evidence["schema_version"], "deepseek_swing_evidence_v1")
        self.assertEqual(
            [row["frame_id"] for row in evidence["frame_sequence"]],
            [31, 32],
        )
        system_prompt = request["payload"]["messages"][0]["content"]
        self.assertIn("证据不足", system_prompt)
        self.assertIn("入镜", system_prompt)
        self.assertIn("evidence_frames", system_prompt)

    def test_evidence_gate_rejects_unsupported_technique_advice(self):
        transport = FakeDeepSeekTransport(
            response={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "message": "加强上旋拉拍质量",
                                    "focus": "topspin",
                                    "category": "technique",
                                    "evidence_frames": [10],
                                    "confidence": 0.9,
                                },
                                ensure_ascii=False,
                            )
                        }
                    }
                ]
            }
        )
        sidecar = DeepSeekCoachSidecar(api_key="test-key", transport=transport)
        results = []
        event = {
            "event_id": 4,
            "start_frame": 10,
            "contact_frame": 10,
            "peak_frame": 10,
            "end_frame": 10,
            "stroke_type": "Forehand",
            "quality_flags": {
                "warnings": ["ball_track_gaps"],
                "review_recommended": True,
            },
            "coach_advice": {
                "message": "确保来球完整入镜",
                "category": "capture",
            },
        }

        sidecar.submit(
            event,
            results.append,
            frame_records=[{"frame_id": 10, "timestamp": 0.4}],
        )
        sidecar.close()

        self.assertEqual(results[0]["status"], "failed")
        self.assertEqual(results[0]["error_type"], "EvidencePolicyError")

    def test_evidence_gate_rejects_ball_capture_advice_for_tolerated_gaps(self):
        transport = FakeDeepSeekTransport(
            response={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "message": "确保来球完整入镜",
                                    "focus": "ball_track_gaps",
                                    "category": "capture",
                                    "evidence_frames": [10],
                                    "confidence": 0.7,
                                },
                                ensure_ascii=False,
                            )
                        }
                    }
                ]
            }
        )
        sidecar = DeepSeekCoachSidecar(api_key="test-key", transport=transport)
        results = []
        event = {
            "event_id": 5,
            "start_frame": 10,
            "contact_frame": 10,
            "peak_frame": 10,
            "end_frame": 10,
            "stroke_type": "Forehand",
            "quality_flags": {
                "warnings": ["ball_track_gaps", "racket_track_gaps"],
                "review_recommended": True,
                "ball_frame_ratio": 0.42,
                "racket_frame_ratio": 0.5,
            },
        }

        sidecar.submit(
            event,
            results.append,
            frame_records=[{"frame_id": 10, "timestamp": 0.4}],
        )
        sidecar.close()

        self.assertEqual(results[0]["status"], "failed")
        self.assertEqual(results[0]["error_type"], "EvidencePolicyError")

    def test_evidence_gate_rejects_review_when_body_coaching_is_available(self):
        transport = FakeDeepSeekTransport(
            response={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "message": "请复核击球点",
                                    "focus": "contact_point_review",
                                    "category": "review",
                                    "evidence_frames": [12],
                                    "confidence": 0.6,
                                },
                                ensure_ascii=False,
                            )
                        }
                    }
                ]
            }
        )
        sidecar = DeepSeekCoachSidecar(api_key="test-key", transport=transport)
        event = {
            "event_id": 9,
            "start_frame": 12,
            "contact_frame": 12,
            "peak_frame": 12,
            "end_frame": 12,
            "stroke_type": "Forehand",
            "confidence": 0.8,
            "quality_flags": {
                "warnings": ["racket_track_gaps", "static_ball_mask_in_event"],
                "review_recommended": True,
                "pose_frame_ratio": 1.0,
            },
        }
        packet = {
            "event_id": 9,
            "event_summary": event,
            "coach_metrics": {
                "confidence": 0.8,
                "body": {
                    "hip_shoulder_separation_at_contact": 8.0,
                    "unit_turn_quality": "adequate",
                },
                "scores": {"power_transfer_score": 0.1},
                "data_quality": {
                    "pose_frame_ratio": 1.0,
                    "missing_fields": [
                        "ball.estimated_spin",
                        "racket.racket_face_angle_deg",
                    ],
                },
            },
            "event_frame_records": [
                {"frame_id": 12, "timestamp": 0.48, "pose": {}}
            ],
            "motion_features": [],
            "frame_trace": [],
            "integrity": {"aligned_to_event_range": True},
        }
        results = []

        sidecar.submit(
            event,
            results.append,
            evidence_packet=packet,
        )
        sidecar.close()

        self.assertEqual(results[0]["status"], "failed")
        self.assertEqual(results[0]["error_type"], "EvidencePolicyError")

    def test_normalizes_awkward_camera_wording(self):
        transport = FakeDeepSeekTransport(
            response={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "message": "确保球拍完整导入",
                                    "focus": "capture",
                                    "category": "capture",
                                    "evidence_frames": [10],
                                    "confidence": 0.7,
                                },
                                ensure_ascii=False,
                            )
                        }
                    }
                ]
            }
        )
        sidecar = DeepSeekCoachSidecar(api_key="test-key", transport=transport)
        results = []
        event = {
            "event_id": 5,
            "start_frame": 10,
            "contact_frame": 10,
            "peak_frame": 10,
            "end_frame": 10,
            "stroke_type": "Forehand",
            "quality_flags": {
                "warnings": ["racket_track_gaps"],
                "review_recommended": True,
            },
        }

        sidecar.submit(
            event,
            results.append,
            frame_records=[{"frame_id": 10, "timestamp": 0.4}],
        )
        sidecar.close()

        self.assertEqual(results[0]["status"], "ready")
        self.assertEqual(results[0]["message"], "确保球拍完整入镜")

    def test_accepts_full_swing_evidence_packet(self):
        transport = FakeDeepSeekTransport(
            response={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "message": "击球后完成随挥",
                                    "focus": "follow_through",
                                    "category": "technique",
                                    "evidence_frames": [12],
                                    "confidence": 0.82,
                                },
                                ensure_ascii=False,
                            )
                        }
                    }
                ]
            }
        )
        sidecar = DeepSeekCoachSidecar(api_key="test-key", transport=transport)
        results = []
        event = {
            "event_id": 6,
            "start_frame": 12,
            "contact_frame": 12,
            "peak_frame": 12,
            "end_frame": 12,
            "stroke_type": "Forehand",
            "quality_flags": {"warnings": [], "review_recommended": False},
        }
        packet = {
            "schema_version": "swing_evidence_packet_v1",
            "event_id": 6,
            "video_context": {"fps": 25.0},
            "player_context": {"level": "intermediate"},
            "event_summary": event,
            "coach_metrics": {
                "timing": {"duration_seconds": 1.8},
                "scores": {"follow_through_score": 0.3},
                "data_quality": {"missing_fields": []},
            },
            "event_frame_records": [
                {"frame_id": 12, "timestamp": 0.48, "pose": {}}
            ],
            "motion_features": [
                {"frame_id": 12, "timestamp": 0.48, "wrist_speed": 18.0}
            ],
            "frame_trace": [
                {"frame": 12, "phase": "follow_through", "motion_energy": 18.0}
            ],
            "recent_swings": [],
            "integrity": {"aligned_to_event_range": True},
        }

        sidecar.submit(
            event,
            results.append,
            evidence_packet=packet,
        )
        sidecar.close()

        self.assertEqual(results[0]["status"], "ready")
        evidence = json.loads(
            transport.requests[0]["payload"]["messages"][1]["content"]
        )
        self.assertEqual(evidence["coach_metrics"]["scores"], {
            "follow_through_score": 0.3
        })
        self.assertEqual(evidence["player_context"]["level"], "intermediate")
        self.assertEqual(evidence["frame_sequence"][0]["phase"], "follow_through")

    def test_remote_failure_returns_failed_status_instead_of_raising(self):
        sidecar = DeepSeekCoachSidecar(
            api_key="test-key",
            timeout_seconds=1.0,
            transport=FakeDeepSeekTransport(error=TimeoutError("slow endpoint")),
        )
        results = []

        sidecar.submit({"event_id": 7}, results.append)
        sidecar.close()

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["status"], "failed")
        self.assertEqual(results[0]["error_type"], "TimeoutError")

    def test_missing_api_key_reports_unavailable_without_calling_transport(self):
        transport = FakeDeepSeekTransport()
        sidecar = DeepSeekCoachSidecar(api_key="", transport=transport)
        results = []

        sidecar.submit({"event_id": 8}, results.append)
        sidecar.close()

        self.assertEqual(results[0]["status"], "unavailable")
        self.assertEqual(results[0]["reason"], "missing_api_key")
        self.assertEqual(transport.requests, [])


class UrllibDeepSeekTransportTests(unittest.TestCase):
    def test_posts_openai_compatible_chat_completion_request(self):
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                content_length = int(self.headers["Content-Length"])
                self.server.received_path = self.path
                self.server.received_auth = self.headers["Authorization"]
                self.server.received_payload = json.loads(
                    self.rfile.read(content_length).decode("utf-8")
                )
                response = {
                    "choices": [
                        {
                            "message": {
                                "content": '{"message":"完成随挥","focus":"follow"}'
                            }
                        }
                    ]
                }
                body = json.dumps(response).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format, *args):
                return

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            response = UrllibDeepSeekTransport().complete(
                base_url=f"http://127.0.0.1:{server.server_port}/v1",
                api_key="secret",
                payload={"model": "deepseek-v4-flash", "messages": []},
                timeout=1.0,
            )
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

        self.assertEqual(server.received_path, "/v1/chat/completions")
        self.assertEqual(server.received_auth, "Bearer secret")
        self.assertEqual(
            server.received_payload["model"],
            "deepseek-v4-flash",
        )
        self.assertEqual(
            json.loads(response["choices"][0]["message"]["content"])["message"],
            "完成随挥",
        )


if __name__ == "__main__":
    unittest.main()
