"""HTTP and SSE request handler for the local control panel."""

from __future__ import annotations

import json
import mimetypes
import shutil
import sys
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional
from urllib.parse import unquote, urlsplit

from control_panel.settings import MAX_REQUEST_BYTES, MAX_VIDEO_BYTES, VIDEO_SUFFIXES
from manual_review_workflow import MAX_ANNOTATION_BYTES

if TYPE_CHECKING:
    from local_control_panel import LocalPipelineController


def create_handler(controller: LocalPipelineController):
    class ControlPanelHandler(BaseHTTPRequestHandler):
        server_version = "TennisControlPanel/1.0"

        def log_message(self, format_string, *args):
            try:
                sys.stdout.write(
                    "[web] "
                    + (format_string % args)
                    + "\n"
                )
            except OSError:
                pass

        def do_HEAD(self):
            self._do_request(head_only=True)

        def do_GET(self):
            self._do_request(head_only=False)

        def _handle_sse(self):
            self.close_connection = True
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache, no-transform")
            self.send_header("Connection", "close")
            self.send_header("X-Accel-Buffering", "no")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            def send_sse_event(event_name: str, data_dict: Any) -> None:
                payload = json.dumps(data_dict, ensure_ascii=False, separators=(",", ":"))
                msg = f"event: {event_name}\ndata: {payload}\n\n".encode("utf-8")
                self.wfile.write(msg)
                self.wfile.flush()

            try:
                curr_status = controller.status()
                send_sse_event("status", curr_status)
                curr_swings = controller.session_events()
                send_sse_event("swings", curr_swings)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                return

            last_status_state = curr_status.get("state")
            last_pid = curr_status.get("pid")
            last_returncode = curr_status.get("returncode")
            last_log_count = len(curr_status.get("logs") or [])
            last_total_events = curr_swings.get("total_events", 0)
            last_ping_time = time.time()
            last_status_time = time.time()

            while not controller.is_server_stopped():
                with controller._change_condition:
                    controller._change_condition.wait(timeout=0.3)

                if controller.is_server_stopped():
                    break

                now = time.time()
                try:
                    if now - last_ping_time >= 10.0:
                        self.wfile.write(b": ping\n\n")
                        self.wfile.flush()
                        last_ping_time = now

                    curr_status = controller.status()
                    state_changed = (
                        curr_status.get("state") != last_status_state
                        or curr_status.get("pid") != last_pid
                        or curr_status.get("returncode") != last_returncode
                    )
                    curr_logs = curr_status.get("logs") or []
                    logs_changed = len(curr_logs) != last_log_count
                    elapsed_tick = (curr_status.get("state") == "running" and (now - last_status_time >= 1.0))

                    if state_changed or elapsed_tick or logs_changed:
                        send_sse_event("status", curr_status)
                        last_status_state = curr_status.get("state")
                        last_pid = curr_status.get("pid")
                        last_returncode = curr_status.get("returncode")
                        last_status_time = now

                    if logs_changed:
                        new_logs = curr_logs[last_log_count:] if len(curr_logs) > last_log_count else curr_logs
                        send_sse_event("logs", {"logs": new_logs, "total": len(curr_logs)})
                        last_log_count = len(curr_logs)

                    curr_swings = controller.session_events()
                    if curr_swings.get("total_events", 0) != last_total_events:
                        send_sse_event("swings", curr_swings)
                        last_total_events = curr_swings.get("total_events", 0)

                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                    break
                except Exception:
                    break

        def _do_request(self, head_only: bool = False):
            path = self.path.split("?", 1)[0]
            try:
                if path in ("/api/events", "/api/sse"):
                    if head_only:
                        self.send_response(HTTPStatus.OK)
                        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                        self.end_headers()
                    else:
                        self._handle_sse()
                elif path == "/":
                    self._send_file(controller.frontend_path, "text/html", head_only=head_only)
                elif path == "/ground-calibration":
                    self._send_file(controller.workspace / 'ground_calibration.html', 'text/html', head_only=head_only)
                elif path == '/api/ground/lane2-reference':
                    self._send_file(controller.workspace / 'validation/lane2_ground_reference_20261005.json', 'application/json', head_only=head_only)
                elif path in ("/mirror-calibration", "/mirror_calibration.html"):
                    calib_html = controller.workspace / "mirror_calibration.html"
                    if calib_html.exists():
                        self._send_file(calib_html, "text/html", head_only=head_only)
                    else:
                        self.send_error(HTTPStatus.NOT_FOUND, "mirror_calibration.html not found")
                elif path in ("/player-compare", "/player_compare.html"):
                    compare_html = controller.workspace / "player_compare.html"
                    if compare_html.exists():
                        self._send_file(compare_html, "text/html", head_only=head_only)
                    else:
                        self.send_error(HTTPStatus.NOT_FOUND, "player_compare.html not found")
                elif path in ("/api/mirror/info", "/api/info"):
                    from urllib.parse import parse_qs, urlsplit
                    query = parse_qs(urlsplit(self.path).query)
                    s_id = query.get("stream_id", [None])[0]
                    self._send_json(controller.mirror_info(stream_id=s_id), head_only=head_only)
                elif path == "/api/mirror/depth-geometry":
                    from urllib.parse import parse_qs, urlsplit
                    from mirror_depth_visualization import compute_mirror_depth_geometry
                    query = parse_qs(urlsplit(self.path).query)
                    p_y = float(query.get("player_y", ["4.68"])[0])
                    w_y = float(query.get("wall_y", ["6.20"])[0])
                    cam_h = float(query.get("cam_h", ["3.30"])[0])
                    pitch = float(query.get("pitch", ["35.0"])[0])
                    geom = compute_mirror_depth_geometry(player_y=p_y, wall_y=w_y, cam_h=cam_h, cam_pitch_deg=pitch)
                    self._send_json(geom, head_only=head_only)
                elif path in ("/api/mirror/frame", "/api/frame"):
                    from urllib.parse import parse_qs, urlsplit
                    query = parse_qs(urlsplit(self.path).query)
                    f_idx = int(query.get("index", ["0"])[0])
                    jpeg_bytes = controller.mirror_frame(f_idx)
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(jpeg_bytes)))
                    self.send_header("Cache-Control", "public, max-age=60")
                    self.end_headers()
                    if not head_only:
                        self.wfile.write(jpeg_bytes)
                elif path == "/api/config":
                    self._send_json(controller.public_config(), head_only=head_only)
                elif path == "/api/roi/config":
                    from urllib.parse import parse_qs, urlsplit
                    query = parse_qs(urlsplit(self.path).query)
                    s_id = query.get("stream_id", [None])[0]
                    self._send_json(controller.roi_info(stream_id=s_id), head_only=head_only)
                elif path == "/api/status":
                    self._send_json(controller.status(), head_only=head_only)
                elif path == "/api/sessions":
                    self._send_json(controller.list_sessions(), head_only=head_only)
                elif path == "/api/sessions/duplicates":
                    from urllib.parse import parse_qs, urlsplit
                    query = parse_qs(urlsplit(self.path).query)
                    p_id = query.get("player_id", [None])[0]
                    self._send_json(controller.list_duplicate_sessions(player_id=p_id), head_only=head_only)
                elif path == "/api/players":
                    self._send_json({"players": controller.list_players()}, head_only=head_only)
                elif path == "/api/players/compare":
                    from urllib.parse import parse_qs, urlsplit
                    query = parse_qs(urlsplit(self.path).query)
                    p_id = query.get("player_id", [None])[0]
                    s_type = query.get("stroke_type", [None])[0]
                    self._send_json(
                        controller.compare_player_sessions(player_id=p_id, stroke_type=s_type),
                        head_only=head_only,
                    )
                elif path == "/api/players/global-profile":
                    from urllib.parse import parse_qs, urlsplit
                    query = parse_qs(urlsplit(self.path).query)
                    p_id = query.get("player_id", [None])[0]
                    days_raw = query.get("days", [None])[0]
                    days_val = int(days_raw) if days_raw and str(days_raw).isdigit() else None
                    dedup_raw = str(query.get("dedup", ["1"])[0] or "1").strip().lower()
                    dedup_val = dedup_raw not in ("0", "false", "no", "off")
                    self._send_json(
                        controller.get_player_global_profile(
                            player_id=p_id,
                            days=days_val,
                            dedup=dedup_val,
                        ),
                        head_only=head_only,
                    )
                elif path == "/api/players/coach-reviews":
                    from urllib.parse import parse_qs, urlsplit
                    query = parse_qs(urlsplit(self.path).query)
                    p_id = query.get("player_id", ["default"])[0]
                    side = query.get("stroke_side", [None])[0]
                    limit_raw = query.get("limit", ["20"])[0]
                    limit = int(limit_raw) if str(limit_raw).isdigit() else 20
                    self._send_json(
                        controller.list_coach_reviews(
                            player_id=p_id,
                            stroke_side=side,
                            limit=limit,
                        ),
                        head_only=head_only,
                    )
                elif path == "/api/players/trajectory-summary":
                    from urllib.parse import parse_qs, urlsplit
                    query = parse_qs(urlsplit(self.path).query)
                    p_id = query.get("player_id", [None])[0]
                    side = query.get("stroke_side", ["forehand"])[0]
                    days_raw = query.get("days", [None])[0]
                    days_val = int(days_raw) if days_raw and str(days_raw).isdigit() else None
                    dedup_raw = str(query.get("dedup", ["1"])[0] or "1").strip().lower()
                    dedup_val = dedup_raw not in ("0", "false", "no", "off")
                    self._send_json(
                        controller.get_player_trajectory_summary(
                            player_id=p_id,
                            stroke_side=side,
                            days=days_val,
                            dedup=dedup_val,
                        ),
                        head_only=head_only,
                    )
                elif path == "/api/session/swing-trajectory":
                    from urllib.parse import parse_qs, urlsplit
                    query = parse_qs(urlsplit(self.path).query)
                    s_id = query.get("session_id", [""])[0]
                    ev_raw = query.get("event_id", [""])[0]
                    if not s_id or not str(ev_raw).lstrip("-").isdigit():
                        self._send_json({"error": "缺少合法的 session_id 或 event_id"}, status=HTTPStatus.BAD_REQUEST)
                        return
                    self._send_json(
                        controller.get_swing_trajectory(session_id=s_id, event_id=int(ev_raw)),
                        head_only=head_only,
                    )
                elif path == "/api/session/events":
                    from urllib.parse import parse_qs, urlsplit
                    query = parse_qs(urlsplit(self.path).query)
                    s_id = query.get("session_id", [None])[0]
                    self._send_json(controller.session_events(session_id=s_id), head_only=head_only)
                elif path == "/api/session/summary":
                    from urllib.parse import parse_qs, urlsplit
                    query = parse_qs(urlsplit(self.path).query)
                    s_id = query.get("session_id", [None])[0]
                    self._send_json(controller.session_summary(session_id=s_id), head_only=head_only)
                elif path == "/api/manual-review/state":
                    self._send_json(
                        controller.manual_review_state(
                            self.headers.get("X-Manual-Review-Report")
                        ),
                        head_only=head_only,
                    )
                elif path == "/api/live-preview":
                    preview = controller.live_preview_path()
                    if preview is None or not preview.exists():
                        self.send_error(
                            HTTPStatus.NOT_FOUND,
                            "Live preview not ready",
                        )
                    else:
                        self._send_file(preview, "image/jpeg", no_cache=True, head_only=head_only)
                elif path.startswith("/static/"):
                    relative_str = path[len("/static/") :].lstrip("/")
                    static_root = controller.workspace / "static"
                    try:
                        resolved_file = (static_root / relative_str).resolve()
                        if not resolved_file.is_relative_to(static_root.resolve()):
                            self.send_error(HTTPStatus.FORBIDDEN)
                        elif not resolved_file.is_file():
                            self.send_error(HTTPStatus.NOT_FOUND)
                        else:
                            self._send_file(resolved_file, head_only=head_only)
                    except Exception:
                        self.send_error(HTTPStatus.NOT_FOUND)
                elif path.startswith("/artifacts/"):
                    artifact = controller.artifact_path(
                        path[len("/artifacts/") :]
                    )
                    self._send_file(artifact, head_only=head_only)
                else:
                    self.send_error(HTTPStatus.NOT_FOUND)
            except (ValueError, RuntimeError, OSError) as exc:
                self._send_json(
                    {"error": str(exc)},
                    status=HTTPStatus.BAD_REQUEST,
                    head_only=head_only,
                )

        def do_POST(self):
            path = self.path.split("?", 1)[0]
            if path in ("/api/save_config", "/api/mirror/save_config"):
                try:
                    payload = self._read_json(max_bytes=MAX_REQUEST_BYTES)
                    self._send_json(controller.mirror_save_config(payload))
                except Exception as exc:
                    self._send_json({"success": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return
            if path in ("/api/preview_backview", "/api/mirror/preview_backview"):
                try:
                    payload = self._read_json(max_bytes=MAX_REQUEST_BYTES)
                    jpeg_bytes = controller.mirror_preview_backview(payload)
                    self._send_bytes(jpeg_bytes, "image/jpeg", no_cache=True)
                except Exception as exc:
                    self._send_json({"success": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return

            allowed_paths = {
                "/api/preview",
                "/api/start",
                "/api/video/upload",
                "/api/stop",
                "/api/manual-review/evaluate",
                "/api/manual-review/auto-calibrate",
                "/api/roi/config",
                "/api/ground/preview",
                "/api/ground/info",
                "/api/ground/save",
                "/api/ground/profile/import",
                "/api/players",
                "/api/players/global-review",
                "/api/sessions/assign-player",
                "/api/sessions/assign-player-batch",
                "/api/sessions/dedup-cleanup",
            }
            if path not in allowed_paths:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            if self.headers.get("X-Control-Token") != controller.token:
                self._send_json(
                    {"error": "控制令牌无效，请刷新页面"},
                    status=HTTPStatus.FORBIDDEN,
                )
                return
            try:
                if path == "/api/video/upload":
                    self.connection.settimeout(120)
                    self._send_json(controller.upload_video(
                        self.rfile, int(self.headers.get("Content-Length", "0")),
                        self.headers.get("X-Video-Suffix", ""),
                    ))
                    return
                payload = self._read_json(
                    max_bytes=(
                        MAX_ANNOTATION_BYTES
                        if path in ("/api/manual-review/evaluate", "/api/manual-review/auto-calibrate")
                        else MAX_REQUEST_BYTES
                    )
                )
                if path == "/api/preview":
                    image = controller.preview(payload)
                    self._send_bytes(
                        image,
                        "image/jpeg",
                        no_cache=True,
                    )
                elif path == "/api/start":
                    self._send_json(controller.start(payload))
                elif path == "/api/stop":
                    self._send_json(controller.stop())
                elif path == "/api/roi/config":
                    self._send_json(controller.save_roi_boundary(payload))
                elif path == '/api/ground/preview':
                    self._send_json(controller.ground_preview(payload))
                elif path == '/api/ground/info':
                    self._send_json(controller.ground_info(payload))
                elif path == '/api/ground/save':
                    self._send_json(controller.save_ground_calibration(payload))
                elif path == '/api/ground/profile/import':
                    self._send_json(controller.import_ground_camera_profile(payload))
                elif path == "/api/players":
                    self._send_json(controller.create_or_update_player(payload))
                elif path == "/api/players/global-review":
                    self._send_json(controller.generate_player_global_review(payload))
                elif path == "/api/sessions/assign-player":
                    self._send_json(controller.assign_session_player(payload))
                elif path == "/api/sessions/assign-player-batch":
                    self._send_json(controller.assign_sessions_player_batch(payload))
                elif path == "/api/sessions/dedup-cleanup":
                    self._send_json(controller.cleanup_duplicate_sessions(payload))
                elif path == "/api/manual-review/evaluate":
                    self._send_json(
                        controller.evaluate_manual_review(
                            payload,
                            self.headers.get("X-Manual-Review-Report"),
                        )
                    )
                elif path == "/api/manual-review/auto-calibrate":
                    self._send_json(
                        controller.auto_calibrate_manual_review(
                            payload,
                            self.headers.get("X-Manual-Review-Report"),
                        )
                    )
            except FileNotFoundError as exc:
                self._send_json(
                    {
                        "error": "manual_review_unavailable",
                        "message": str(exc),
                    },
                    status=HTTPStatus.NOT_FOUND,
                )
            except (json.JSONDecodeError, ValueError) as exc:
                self._send_json(
                    {
                        "error": (
                            "invalid_annotations"
                            if path in ("/api/manual-review/evaluate", "/api/manual-review/auto-calibrate")
                            else str(exc)
                        ),
                        "message": str(exc),
                    },
                    status=(
                        HTTPStatus.UNPROCESSABLE_ENTITY
                        if path in ("/api/manual-review/evaluate", "/api/manual-review/auto-calibrate")
                        else HTTPStatus.BAD_REQUEST
                    ),
                )
            except (RuntimeError, OSError) as exc:
                self._send_json(
                    {"error": str(exc)},
                    status=HTTPStatus.BAD_REQUEST,
                )
            except Exception as exc:
                if path not in ("/api/manual-review/evaluate", "/api/manual-review/auto-calibrate"):
                    raise
                self._send_json(
                    {"error": "workflow_failed", "message": str(exc)},
                    status=HTTPStatus.INTERNAL_SERVER_ERROR,
                )

        def _read_json(self, max_bytes: int = MAX_REQUEST_BYTES) -> Dict[str, Any]:
            if "application/json" not in self.headers.get(
                "Content-Type",
                "",
            ):
                raise ValueError("请求必须使用 application/json")
            length = int(self.headers.get("Content-Length") or 0)
            if length < 0 or length > max_bytes:
                raise ValueError("请求内容过大")
            raw = self.rfile.read(length)
            value = json.loads(raw.decode("utf-8")) if raw else {}
            if not isinstance(value, dict):
                raise ValueError("请求 JSON 必须是对象")
            return value

        def _send_json(
            self,
            payload: Dict[str, Any],
            status: HTTPStatus = HTTPStatus.OK,
            head_only: bool = False,
        ):
            self._send_bytes(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8"),
                "application/json; charset=utf-8",
                status=status,
                no_cache=True,
                head_only=head_only,
            )

        def _send_file(
            self,
            path: Path,
            content_type: Optional[str] = None,
            no_cache: bool = False,
            head_only: bool = False,
        ):
            if not path.exists() or not path.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            resolved_type = (
                content_type
                or mimetypes.guess_type(str(path))[0]
                or "application/octet-stream"
            )
            if resolved_type in ("text/css", "text/javascript", "text/html"):
                resolved_type = f"{resolved_type}; charset=utf-8"
            file_size = path.stat().st_size
            range_header = self.headers.get("Range")
            if range_header and range_header.startswith("bytes="):
                try:
                    ranges = range_header[6:].split("-", 1)
                    start = int(ranges[0]) if ranges[0] else 0
                    end = int(ranges[1]) if ranges[1] else file_size - 1
                    end = min(end, file_size - 1)
                    if start > end or start >= file_size:
                        self.send_response(int(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE))
                        self.send_header("Content-Range", f"bytes */{file_size}")
                        self.end_headers()
                        return
                    length = end - start + 1
                    chunk = b""
                    if not head_only:
                        with path.open("rb") as f:
                            f.seek(start)
                            chunk = f.read(length)
                    self.send_response(int(HTTPStatus.PARTIAL_CONTENT))
                    self.send_header("Content-Type", resolved_type)
                    self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
                    self.send_header("Content-Length", str(length))
                    self.send_header("Accept-Ranges", "bytes")
                    self.send_header("X-Content-Type-Options", "nosniff")
                    if no_cache:
                        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
                    self.end_headers()
                    if not head_only:
                        self.wfile.write(chunk)
                    return
                except Exception:
                    pass

            self.send_response(int(HTTPStatus.OK))
            self.send_header("Content-Type", resolved_type)
            self.send_header("Content-Length", str(file_size))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "SAMEORIGIN")
            if no_cache:
                self.send_header(
                    "Cache-Control",
                    "no-store, no-cache, must-revalidate",
                )
            self.end_headers()
            if not head_only:
                with path.open("rb") as f:
                    shutil.copyfileobj(f, self.wfile)


        def _send_bytes(
            self,
            data: bytes,
            content_type: str,
            status: HTTPStatus = HTTPStatus.OK,
            no_cache: bool = False,
            head_only: bool = False,
        ):
            self.send_response(int(status))
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "SAMEORIGIN")
            if no_cache:
                self.send_header(
                    "Cache-Control",
                    "no-store, no-cache, must-revalidate",
                )
            self.end_headers()
            if not head_only:
                self.wfile.write(data)

    return ControlPanelHandler

