"""Session discovery, event caching, artifact resolution, and manual review storage."""

from __future__ import annotations

import json
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import quote, unquote

from analysis_metric_delivery import normalize_swing_event_for_delivery
from control_panel.db_store import DEFAULT_PLAYER_ID, DEFAULT_PLAYER_NAME, SessionDatabase
from control_panel.settings import _session_capture_time
from manual_review_workflow import (
    discover_session_paths,
    load_review_state,
    process_manual_review,
)
from swing_session_summary import build_session_coaching_summary


class SessionStore:
    """Manages historical session listing, event caching, and artifact paths."""

    def __init__(
        self,
        workspace: Path,
        state_lock: threading.RLock,
        artifacts_provider: Callable[[], Dict[str, str]],
    ):
        self.workspace = workspace
        self._state_lock = state_lock
        self._artifacts_provider = artifacts_provider
        self._cache_lock = threading.Lock()
        self._session_events_cache: Dict[str, Tuple[Tuple[int, int], Dict[str, Any]]] = {}
        self._session_meta_cache: Dict[str, Dict[str, Any]] = {}
        self.db = SessionDatabase(self.workspace / "data" / "tennis_sessions.db")

    @property
    def _artifacts(self) -> Dict[str, str]:
        return self._artifacts_provider()

    def artifact_path(self, request_path: str) -> Path:
        relative = unquote(request_path).lstrip("/")
        candidate = (self.workspace / relative).resolve()
        if self.workspace not in candidate.parents:
            raise ValueError("非法文件路径")
        return candidate

    def live_preview_path(self) -> Optional[Path]:
        value = self._artifacts.get("preview_path")
        if not value:
            return None
        return self.artifact_path(value)

    def manual_review_paths(
        self,
        report_path: Optional[str] = None,
    ) -> Dict[str, Path]:
        if report_path:
            report = self.artifact_path(report_path)
            if report.suffix.lower() != ".html":
                raise ValueError("人工校准报告路径无效")
            paths = discover_session_paths(report.parent)
            if paths["report"].resolve() != report.resolve():
                raise ValueError("人工校准报告与会话不匹配")
            return paths
        with self._state_lock:
            session_dir = self._artifacts.get("session_dir")
        if not session_dir:
            raise RuntimeError("当前没有可供人工校准的分析会话")
        return discover_session_paths(self.artifact_path(session_dir))

    def manual_review_state(
        self,
        report_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        return load_review_state(self.manual_review_paths(report_path))

    def session_events(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        with self._state_lock:
            active_session_id = self._artifacts.get("session_id")
            event_json_rel = self._artifacts.get("event_json")
            session_dir_rel = self._artifacts.get("session_dir")

        target_file = None
        session_dir = None
        target_session_id = session_id

        if session_id:
            clean_id = re.sub(r"[^A-Za-z0-9_-]", "", session_id)
            if active_session_id and clean_id == active_session_id and event_json_rel:
                candidate = self.workspace / event_json_rel
                if candidate.exists():
                    target_file = candidate
                    session_dir = (self.workspace / session_dir_rel) if session_dir_rel else candidate.parent
                    target_session_id = active_session_id
            if target_file is None:
                base_dir = self.workspace / "data" / "analysis_results" / "control_panel"
                candidate_dir = base_dir / clean_id
                if candidate_dir.is_dir() and candidate_dir.resolve().parent == base_dir.resolve():
                    json_candidates = list(candidate_dir.glob("*_swing_events.json"))
                    if json_candidates:
                        target_file = json_candidates[0]
                        session_dir = candidate_dir
                        target_session_id = clean_id
        else:
            if event_json_rel:
                candidate = self.workspace / event_json_rel
                if candidate.exists():
                    target_file = candidate
                    session_dir = (self.workspace / session_dir_rel) if session_dir_rel else candidate.parent
                    target_session_id = active_session_id

            if target_file is None:
                base_dir = self.workspace / "data" / "analysis_results" / "control_panel"
                if base_dir.exists():
                    sessions = sorted(
                        [d for d in base_dir.iterdir() if d.is_dir() and not d.name.startswith(".")],
                        key=_session_capture_time,
                        reverse=True,
                    )
                    for s in sessions:
                        json_candidates = list(s.glob("*_swing_events.json"))
                        if json_candidates:
                            target_file = json_candidates[0]
                            session_dir = s
                            target_session_id = s.name
                            break

        if target_file is None or not target_file.exists():
            return {
                "session_id": target_session_id,
                "events": [],
                "total_events": 0,
                "summary": build_session_coaching_summary([]),
            }

        resolved_sid = target_session_id or (session_dir.name if session_dir else None)
        try:
            st = target_file.stat()
            file_sig = (int(st.st_mtime_ns), int(st.st_size))
        except OSError:
            file_sig = None

        cache_key = f"{target_file}::{resolved_sid or ''}"
        if file_sig is not None:
            with self._cache_lock:
                cached = self._session_events_cache.get(cache_key)
                if cached is not None and cached[0] == file_sig:
                    return dict(cached[1])

        try:
            with target_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return {
                "session_id": target_session_id,
                "events": [],
                "total_events": 0,
                "summary": build_session_coaching_summary([]),
            }

        raw_events = data.get("events", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
        processed_events = []
        for ev in raw_events:
            event_item = normalize_swing_event_for_delivery(ev)
            clip_rel = ev.get("clip_path")
            if clip_rel and session_dir:
                clip_path = (session_dir / clip_rel) if not Path(clip_rel).is_absolute() else Path(clip_rel)
                if clip_path.exists():
                    try:
                        event_item["clip_url"] = f"/artifacts/{quote(str(clip_path.resolve().relative_to(self.workspace)))}"
                    except ValueError:
                        pass
            
            freeze_rel = ev.get("impact_freeze_path")
            if freeze_rel and session_dir:
                freeze_path = (session_dir / freeze_rel) if not Path(freeze_rel).is_absolute() else Path(freeze_rel)
                if freeze_path.exists():
                    try:
                        event_item["impact_freeze_url"] = f"/artifacts/{quote(str(freeze_path.resolve().relative_to(self.workspace)))}"
                    except ValueError:
                        pass

            processed_events.append(event_item)

        summary = build_session_coaching_summary(processed_events)
        result = {
            "session_id": resolved_sid,
            "events": processed_events,
            "total_events": len(processed_events),
            "summary": summary,
        }
        if file_sig is not None:
            with self._cache_lock:
                if len(self._session_events_cache) >= 32:
                    self._session_events_cache.pop(next(iter(self._session_events_cache)))
                self._session_events_cache[cache_key] = (file_sig, result)
            if resolved_sid:
                try:
                    ts = _session_capture_time(session_dir) if session_dir else target_file.stat().st_mtime
                    disp = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
                    self.db.upsert_session(
                        session_id=resolved_sid,
                        display_time=disp,
                        timestamp=ts,
                        events=processed_events,
                        json_mtime_ns=file_sig[0],
                        json_size=file_sig[1],
                        session_dir=str(session_dir) if session_dir else None,
                    )
                except Exception:
                    pass
        return dict(result)

    def session_summary(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        events_res = self.session_events(session_id=session_id)
        return {
            "session_id": events_res.get("session_id"),
            "summary": events_res.get("summary") or {},
        }

    def list_sessions(self) -> List[Dict[str, Any]]:
        base_dir = self.workspace / "data" / "analysis_results" / "control_panel"
        if not base_dir.exists():
            return []
        with self._state_lock:
            active_session_id = self._artifacts.get("session_id")

        db_sigs = self.db.get_session_signatures()
        sessions = []
        seen_names = set()
        for d in base_dir.iterdir():
            if not d.is_dir() or d.name.startswith("."):
                continue
            seen_names.add(d.name)
            events_files = list(d.glob("*_swing_events.json"))
            manifest_files = list(d.glob("*_evidence_manifest.json"))

            ev_sig = None
            if events_files:
                try:
                    ev_st = events_files[0].stat()
                    ev_sig = (int(ev_st.st_mtime_ns), int(ev_st.st_size))
                except OSError:
                    ev_sig = None

            mf_sig = None
            if manifest_files:
                try:
                    mf_st = manifest_files[0].stat()
                    mf_sig = (int(mf_st.st_mtime_ns), int(mf_st.st_size))
                except OSError:
                    mf_sig = None

            meta_sig = (ev_sig, mf_sig)
            with self._cache_lock:
                cached_meta = self._session_meta_cache.get(d.name)
            if cached_meta is not None and cached_meta.get("sig") == meta_sig:
                entry = dict(cached_meta["data"])
                entry["is_active"] = (d.name == active_session_id)
                sessions.append(entry)
                continue

            ev_list: Optional[List[Dict[str, Any]]] = None
            event_count = 0
            if events_files:
                try:
                    with open(events_files[0], "r", encoding="utf-8") as f:
                        ev_data = json.load(f)
                        ev_list = ev_data.get("events", []) if isinstance(ev_data, dict) else (ev_data if isinstance(ev_data, list) else [])
                        event_count = len(ev_list)
                except Exception:
                    pass

            source_name = None
            stream_id_val = None
            processing_mode_val = None
            if manifest_files:
                try:
                    with open(manifest_files[0], "r", encoding="utf-8") as f:
                        m_data = json.load(f)
                        cap = m_data.get("capture") or {}
                        sess_m = m_data.get("session") or {}
                        src = cap.get("source") or sess_m.get("source")
                        if src:
                            source_name = Path(src).name
                        stream_id_val = sess_m.get("stream_id")
                        processing_mode_val = cap.get("processing_mode")
                except Exception:
                    pass

            m = re.search(r"_(\d{8}T\d{6}Z)_", d.name)
            display_time = ""
            ts = d.stat().st_mtime
            if m:
                try:
                    dt = datetime.strptime(m.group(1), "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
                    display_time = dt.strftime("%Y-%m-%d %H:%M:%S")
                    ts = dt.timestamp()
                except ValueError:
                    pass
            if not display_time:
                display_time = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")

            if ev_list is not None and ev_sig is not None and db_sigs.get(d.name) != ev_sig:
                try:
                    self.db.upsert_session(
                        session_id=d.name,
                        display_time=display_time,
                        timestamp=ts,
                        events=ev_list,
                        source_name=source_name,
                        stream_id=stream_id_val,
                        processing_mode=processing_mode_val,
                        json_mtime_ns=ev_sig[0],
                        json_size=ev_sig[1],
                        session_dir=str(d),
                    )
                except Exception:
                    pass

            meta_data = {
                "session_id": d.name,
                "display_time": display_time,
                "timestamp": ts,
                "event_count": event_count,
                "source_name": source_name,
            }
            with self._cache_lock:
                self._session_meta_cache[d.name] = {"sig": meta_sig, "data": meta_data}

            entry = dict(meta_data)
            entry["is_active"] = (d.name == active_session_id)
            sessions.append(entry)

        player_map = self.db.get_session_player_map()
        for entry in sessions:
            p_info = player_map.get(entry["session_id"])
            entry["player_id"] = p_info["player_id"] if p_info else DEFAULT_PLAYER_ID
            entry["player_name"] = p_info["player_name"] if p_info else DEFAULT_PLAYER_NAME

        with self._cache_lock:
            stale_keys = [k for k in self._session_meta_cache if k not in seen_names]
            for k in stale_keys:
                del self._session_meta_cache[k]

        sessions.sort(key=lambda s: s["timestamp"], reverse=True)
        return sessions

    def list_players(self) -> List[Dict[str, Any]]:
        return self.db.list_players()

    def create_or_update_player(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self.db.create_or_update_player(
            display_name=payload.get("display_name") or "",
            player_id=payload.get("player_id"),
            dominant_hand=payload.get("dominant_hand") or "right",
            notes=payload.get("notes") or "",
        )

    def assign_session_player(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self.db.assign_session_player(
            session_id=str(payload.get("session_id") or ""),
            player_id=str(payload.get("player_id") or DEFAULT_PLAYER_ID),
        )

    def compare_player_sessions(
        self,
        player_id: Optional[str] = None,
        stroke_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        return self.db.compare_player_sessions(
            player_id=player_id or DEFAULT_PLAYER_ID,
            stroke_type=stroke_type,
        )


    def evaluate_manual_review(
        self,
        payload: Dict[str, Any],
        report_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        return process_manual_review(
            self.manual_review_paths(report_path),
            payload,
        )

