"""SQLite persistence for tennis sessions, players, swing events, and cross-session comparison."""

from __future__ import annotations

import json
import re
import sqlite3
import statistics
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

from analysis_metric_delivery import normalize_swing_event_for_delivery


DEFAULT_PLAYER_ID = "default"
DEFAULT_PLAYER_NAME = "默认球员"
_PLAYER_ID_PATTERN = re.compile(r"[^A-Za-z0-9_-]+")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _safe_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    return num if num == num else None


def _safe_int(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _compute_iqr(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    lower = ordered[:mid]
    upper = ordered[mid:] if n % 2 == 0 else ordered[mid + 1 :]
    q1 = statistics.median(lower) if lower else ordered[0]
    q3 = statistics.median(upper) if upper else ordered[-1]
    return round(float(q3 - q1), 2)


class SessionDatabase:
    """Thread-safe SQLite store for session traceability and player comparisons."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_schema(self) -> None:
        with self._lock, self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS players (
                    player_id TEXT PRIMARY KEY,
                    display_name TEXT NOT NULL,
                    dominant_hand TEXT NOT NULL DEFAULT 'right',
                    notes TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    player_id TEXT NOT NULL DEFAULT 'default',
                    display_time TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    source_name TEXT,
                    stream_id TEXT,
                    processing_mode TEXT,
                    event_count INTEGER NOT NULL DEFAULT 0,
                    forehand_count INTEGER NOT NULL DEFAULT 0,
                    backhand_count INTEGER NOT NULL DEFAULT 0,
                    shadow_count INTEGER NOT NULL DEFAULT 0,
                    json_mtime_ns INTEGER NOT NULL DEFAULT 0,
                    json_size INTEGER NOT NULL DEFAULT 0,
                    session_dir TEXT,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (player_id) REFERENCES players(player_id)
                );

                CREATE TABLE IF NOT EXISTS swing_events (
                    session_id TEXT NOT NULL,
                    event_id INTEGER NOT NULL,
                    player_id TEXT NOT NULL DEFAULT 'default',
                    stroke_type TEXT,
                    is_shadow_swing INTEGER NOT NULL DEFAULT 0,
                    swing_score REAL,
                    swing_grade TEXT,
                    start_frame INTEGER,
                    end_frame INTEGER,
                    contact_frame INTEGER,
                    racket_speed_px_s REAL,
                    racket_max_speed_px_s REAL,
                    brush_angle_deg REAL,
                    drop_depth_ratio REAL,
                    leg_drive_px REAL,
                    sequence_quality TEXT,
                    hip_to_shoulder_ms REAL,
                    payload_json TEXT NOT NULL,
                    PRIMARY KEY (session_id, event_id),
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS swing_metrics (
                    session_id TEXT NOT NULL,
                    event_id INTEGER NOT NULL,
                    player_id TEXT NOT NULL DEFAULT 'default',
                    stroke_type TEXT,
                    metric_key TEXT NOT NULL,
                    metric_label TEXT,
                    value_num REAL,
                    value_text TEXT,
                    unit TEXT,
                    confidence REAL,
                    coach_eligible INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (session_id, event_id, metric_key),
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_sessions_player_ts
                    ON sessions(player_id, timestamp DESC);
                CREATE INDEX IF NOT EXISTS idx_events_player_stroke
                    ON swing_events(player_id, stroke_type, is_shadow_swing);
                CREATE INDEX IF NOT EXISTS idx_metrics_player_key
                    ON swing_metrics(player_id, metric_key, stroke_type);
                """
            )
            conn.execute(
                """
                INSERT OR IGNORE INTO players (player_id, display_name, dominant_hand, notes, created_at)
                VALUES (?, ?, 'right', '', ?)
                """,
                (DEFAULT_PLAYER_ID, DEFAULT_PLAYER_NAME, _utc_now_iso()),
            )

    def list_players(self) -> List[Dict[str, Any]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT p.player_id, p.display_name, p.dominant_hand, p.notes, p.created_at,
                       COUNT(s.session_id) AS session_count
                FROM players p
                LEFT JOIN sessions s ON s.player_id = p.player_id
                GROUP BY p.player_id
                ORDER BY CASE WHEN p.player_id = 'default' THEN 0 ELSE 1 END, p.display_name ASC
                """
            ).fetchall()
            return [dict(r) for r in rows]

    def create_or_update_player(
        self,
        display_name: str,
        player_id: Optional[str] = None,
        dominant_hand: str = "right",
        notes: str = "",
    ) -> Dict[str, Any]:
        name = str(display_name or "").strip()
        if not name:
            raise ValueError("球员姓名不能为空")
        hand = str(dominant_hand or "right").strip().lower()
        if hand not in ("right", "left"):
            hand = "right"
        if player_id:
            pid = _PLAYER_ID_PATTERN.sub("_", str(player_id).strip()).strip("_")[:48]
        else:
            slug = _PLAYER_ID_PATTERN.sub("_", name).strip("_")[:24] or "player"
            pid = f"{slug}_{int(datetime.now(timezone.utc).timestamp())}"
        if not pid:
            raise ValueError("无效的球员 ID")

        now = _utc_now_iso()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO players (player_id, display_name, dominant_hand, notes, created_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(player_id) DO UPDATE SET
                    display_name = excluded.display_name,
                    dominant_hand = excluded.dominant_hand,
                    notes = excluded.notes
                """,
                (pid, name, hand, str(notes or "").strip(), now),
            )
            row = conn.execute(
                "SELECT player_id, display_name, dominant_hand, notes, created_at FROM players WHERE player_id = ?",
                (pid,),
            ).fetchone()
            return dict(row)

    def assign_session_player(self, session_id: str, player_id: str) -> Dict[str, Any]:
        sid = str(session_id or "").strip()
        pid = str(player_id or DEFAULT_PLAYER_ID).strip() or DEFAULT_PLAYER_ID
        if not sid:
            raise ValueError("session_id 不能为空")

        with self._lock, self._connect() as conn:
            player = conn.execute(
                "SELECT player_id, display_name FROM players WHERE player_id = ?",
                (pid,),
            ).fetchone()
            if player is None:
                raise ValueError(f"未找到球员: {pid}")

            sess = conn.execute(
                "SELECT session_id FROM sessions WHERE session_id = ?",
                (sid,),
            ).fetchone()
            if sess is None:
                # Create lightweight placeholder row so pre-binding works before first swing arrives.
                conn.execute(
                    """
                    INSERT INTO sessions (
                        session_id, player_id, display_time, timestamp, updated_at
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (sid, pid, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), datetime.now().timestamp(), _utc_now_iso()),
                )
            else:
                conn.execute("UPDATE sessions SET player_id = ?, updated_at = ? WHERE session_id = ?", (pid, _utc_now_iso(), sid))
                conn.execute("UPDATE swing_events SET player_id = ? WHERE session_id = ?", (pid, sid))
                conn.execute("UPDATE swing_metrics SET player_id = ? WHERE session_id = ?", (pid, sid))

            return {
                "session_id": sid,
                "player_id": pid,
                "player_name": player["display_name"],
            }

    def get_session_signatures(self) -> Dict[str, Tuple[int, int]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute("SELECT session_id, json_mtime_ns, json_size FROM sessions").fetchall()
            return {r["session_id"]: (int(r["json_mtime_ns"]), int(r["json_size"])) for r in rows}

    def get_session_player_map(self) -> Dict[str, Dict[str, str]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT s.session_id, s.player_id, p.display_name AS player_name
                FROM sessions s
                LEFT JOIN players p ON p.player_id = s.player_id
                """
            ).fetchall()
            return {
                r["session_id"]: {
                    "player_id": r["player_id"] or DEFAULT_PLAYER_ID,
                    "player_name": r["player_name"] or DEFAULT_PLAYER_NAME,
                }
                for r in rows
            }

    def upsert_session(
        self,
        *,
        session_id: str,
        display_time: str,
        timestamp: float,
        events: List[Dict[str, Any]],
        source_name: Optional[str] = None,
        stream_id: Optional[str] = None,
        processing_mode: Optional[str] = None,
        json_mtime_ns: int = 0,
        json_size: int = 0,
        session_dir: Optional[str] = None,
        player_id: Optional[str] = None,
    ) -> None:
        if not session_id:
            return
        normalized_events = [normalize_swing_event_for_delivery(ev) for ev in (events or [])]
        forehand_count = sum(
            1 for ev in normalized_events
            if not ev.get("is_shadow_swing") and str(ev.get("stroke_type") or "").lower() == "forehand"
        )
        backhand_count = sum(
            1 for ev in normalized_events
            if not ev.get("is_shadow_swing") and "backhand" in str(ev.get("stroke_type") or "").lower()
        )
        shadow_count = sum(1 for ev in normalized_events if ev.get("is_shadow_swing"))
        now = _utc_now_iso()

        with self._lock, self._connect() as conn:
            existing = conn.execute(
                "SELECT player_id, json_mtime_ns, json_size FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if (
                existing is not None
                and json_mtime_ns > 0
                and int(existing["json_mtime_ns"]) == int(json_mtime_ns)
                and int(existing["json_size"]) == int(json_size)
                and player_id is None
            ):
                return

            resolved_pid = player_id or (existing["player_id"] if existing is not None else DEFAULT_PLAYER_ID)
            # Ensure referenced player exists
            conn.execute(
                "INSERT OR IGNORE INTO players (player_id, display_name, dominant_hand, notes, created_at) VALUES (?, ?, 'right', '', ?)",
                (resolved_pid, resolved_pid if resolved_pid != DEFAULT_PLAYER_ID else DEFAULT_PLAYER_NAME, now),
            )

            conn.execute(
                """
                INSERT INTO sessions (
                    session_id, player_id, display_time, timestamp, source_name,
                    stream_id, processing_mode, event_count, forehand_count,
                    backhand_count, shadow_count, json_mtime_ns, json_size,
                    session_dir, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    player_id = excluded.player_id,
                    display_time = excluded.display_time,
                    timestamp = excluded.timestamp,
                    source_name = COALESCE(excluded.source_name, sessions.source_name),
                    stream_id = COALESCE(excluded.stream_id, sessions.stream_id),
                    processing_mode = COALESCE(excluded.processing_mode, sessions.processing_mode),
                    event_count = excluded.event_count,
                    forehand_count = excluded.forehand_count,
                    backhand_count = excluded.backhand_count,
                    shadow_count = excluded.shadow_count,
                    json_mtime_ns = excluded.json_mtime_ns,
                    json_size = excluded.json_size,
                    session_dir = COALESCE(excluded.session_dir, sessions.session_dir),
                    updated_at = excluded.updated_at
                """,
                (
                    session_id,
                    resolved_pid,
                    display_time,
                    float(timestamp),
                    source_name,
                    stream_id,
                    processing_mode,
                    len(normalized_events),
                    forehand_count,
                    backhand_count,
                    shadow_count,
                    int(json_mtime_ns),
                    int(json_size),
                    session_dir,
                    now,
                ),
            )

            conn.execute("DELETE FROM swing_events WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM swing_metrics WHERE session_id = ?", (session_id,))

            event_rows = []
            metric_rows = []
            for idx, ev in enumerate(normalized_events):
                eid = _safe_int(ev.get("event_id"))
                if eid is None:
                    eid = idx + 1
                stroke = str(ev.get("stroke_type") or "Swing")
                ext = ev.get("extended_biomechanics") or {}
                speed = ext.get("racket_head_speed") or {}
                brush = ext.get("brush_angle") or {}
                leg = ext.get("leg_drive") or {}
                seq = ev.get("kinematic_sequence") or {}

                event_rows.append(
                    (
                        session_id,
                        eid,
                        resolved_pid,
                        stroke,
                        1 if ev.get("is_shadow_swing") else 0,
                        _safe_float(ev.get("swing_score")),
                        str(ev.get("swing_grade") or "REVIEW_REQUIRED"),
                        _safe_int(ev.get("start_frame")),
                        _safe_int(ev.get("end_frame")),
                        _safe_int(ev.get("contact_frame")),
                        _safe_float(speed.get("contact_px_s")),
                        _safe_float(speed.get("max_px_s")),
                        _safe_float(brush.get("low_to_high_angle_deg")),
                        _safe_float(brush.get("drop_depth_ratio")),
                        _safe_float(leg.get("drive_px")),
                        str(seq.get("sequence_quality") or ""),
                        _safe_float(seq.get("latency_hip_to_shoulder_ms")),
                        json.dumps(ev, ensure_ascii=False, separators=(",", ":")),
                    )
                )

                for m in ev.get("analysis_metrics") or []:
                    m_key = str(m.get("key") or "").strip()
                    if not m_key:
                        continue
                    val = m.get("value")
                    val_num = _safe_float(val)
                    val_text = str(val) if val_num is None and val is not None else None
                    metric_rows.append(
                        (
                            session_id,
                            eid,
                            resolved_pid,
                            stroke,
                            m_key,
                            str(m.get("label") or m_key),
                            val_num,
                            val_text,
                            str(m.get("unit") or ""),
                            _safe_float(m.get("confidence")),
                            1 if m.get("coach_eligible") else 0,
                        )
                    )

            if event_rows:
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO swing_events (
                        session_id, event_id, player_id, stroke_type, is_shadow_swing,
                        swing_score, swing_grade, start_frame, end_frame, contact_frame,
                        racket_speed_px_s, racket_max_speed_px_s, brush_angle_deg,
                        drop_depth_ratio, leg_drive_px, sequence_quality,
                        hip_to_shoulder_ms, payload_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    event_rows,
                )
            if metric_rows:
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO swing_metrics (
                        session_id, event_id, player_id, stroke_type, metric_key,
                        metric_label, value_num, value_text, unit, confidence, coach_eligible
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    metric_rows,
                )

    def compare_player_sessions(
        self,
        player_id: str = DEFAULT_PLAYER_ID,
        stroke_type: Optional[str] = None,
        limit: int = 20,
    ) -> Dict[str, Any]:
        pid = str(player_id or DEFAULT_PLAYER_ID).strip() or DEFAULT_PLAYER_ID
        stroke_filter = str(stroke_type or "").strip()
        if stroke_filter.lower() in ("", "all"):
            stroke_filter = ""

        with self._lock, self._connect() as conn:
            player_row = conn.execute(
                "SELECT player_id, display_name, dominant_hand FROM players WHERE player_id = ?",
                (pid,),
            ).fetchone()
            if player_row is None:
                raise ValueError(f"未找到球员: {pid}")

            session_rows = conn.execute(
                """
                SELECT session_id, display_time, timestamp, source_name,
                       event_count, forehand_count, backhand_count, shadow_count, session_dir
                FROM sessions
                WHERE player_id = ? AND event_count > 0
                ORDER BY timestamp ASC
                """,
                (pid,),
            ).fetchall()

            if len(session_rows) > limit:
                session_rows = session_rows[-limit:]

            workspace_root = self.db_path.resolve().parent.parent
            series = []
            for s in session_rows:
                sid = s["session_id"]
                s_dir = Path(s["session_dir"]) if s["session_dir"] else (
                    workspace_root / "data" / "analysis_results" / "control_panel" / sid
                )
                params: List[Any] = [sid]
                stroke_clause = ""
                if stroke_filter:
                    stroke_clause = " AND LOWER(stroke_type) = LOWER(?)"
                    params.append(stroke_filter)

                ev_rows = conn.execute(
                    f"""
                    SELECT event_id, stroke_type, is_shadow_swing, swing_score,
                           start_frame, end_frame, contact_frame,
                           racket_speed_px_s, racket_max_speed_px_s,
                           brush_angle_deg, leg_drive_px, sequence_quality, payload_json
                    FROM swing_events
                    WHERE session_id = ? AND is_shadow_swing = 0{stroke_clause}
                    ORDER BY event_id ASC
                    """,
                    params,
                ).fetchall()

                speeds = [r["racket_speed_px_s"] for r in ev_rows if r["racket_speed_px_s"] is not None]
                max_speeds = [r["racket_max_speed_px_s"] for r in ev_rows if r["racket_max_speed_px_s"] is not None]
                brushes = [r["brush_angle_deg"] for r in ev_rows if r["brush_angle_deg"] is not None]
                legs = [r["leg_drive_px"] for r in ev_rows if r["leg_drive_px"] is not None]
                scores = [r["swing_score"] for r in ev_rows if r["swing_score"] is not None]
                seq_total = sum(1 for r in ev_rows if r["sequence_quality"])
                seq_good = sum(
                    1 for r in ev_rows if r["sequence_quality"] in ("OPTIMAL", "ACCEPTABLE")
                )

                # Extract per-swing media URLs, key metrics, and coach warnings
                swings_summary = []
                warning_counter: Dict[str, int] = {}
                for r in ev_rows:
                    try:
                        payload = json.loads(r["payload_json"]) if r["payload_json"] else {}
                    except Exception:
                        payload = {}
                    clip_url = payload.get("clip_url") or ""
                    if not clip_url and payload.get("clip_path") and s_dir:
                        cp = (s_dir / payload["clip_path"]) if not Path(payload["clip_path"]).is_absolute() else Path(payload["clip_path"])
                        if cp.exists():
                            try:
                                clip_url = f"/artifacts/{quote(str(cp.resolve().relative_to(workspace_root)))}"
                            except ValueError:
                                pass
                    freeze_url = payload.get("impact_freeze_url") or ""
                    if not freeze_url and payload.get("impact_freeze_path") and s_dir:
                        fp = (s_dir / payload["impact_freeze_path"]) if not Path(payload["impact_freeze_path"]).is_absolute() else Path(payload["impact_freeze_path"])
                        if fp.exists():
                            try:
                                freeze_url = f"/artifacts/{quote(str(fp.resolve().relative_to(workspace_root)))}"
                            except ValueError:
                                pass

                    advices = payload.get("coach_advices") or []
                    advice_msgs = []
                    for adv in advices:
                        if isinstance(adv, dict):
                            msg = str(adv.get("message") or adv.get("code") or "").strip()
                            if msg:
                                advice_msgs.append(msg)
                                warning_counter[msg] = warning_counter.get(msg, 0) + 1

                    ev_metrics = {
                        m["key"]: m.get("value")
                        for m in (payload.get("analysis_metrics") or [])
                        if isinstance(m, dict) and m.get("key")
                    }
                    swings_summary.append(
                        {
                            "event_id": r["event_id"],
                            "stroke_type": r["stroke_type"],
                            "start_frame": r["start_frame"],
                            "end_frame": r["end_frame"],
                            "contact_frame": r["contact_frame"],
                            "clip_url": clip_url,
                            "impact_freeze_url": freeze_url,
                            "racket_speed_px_s": r["racket_speed_px_s"],
                            "racket_max_speed_px_s": r["racket_max_speed_px_s"] or ev_metrics.get("racket_max_speed"),
                            "shoulder_turn_change": ev_metrics.get("shoulder_turn_change"),
                            "hip_shoulder_separation": ev_metrics.get("hip_shoulder_separation"),
                            "arm_extension": ev_metrics.get("arm_extension"),
                            "preparation_knee_flexion": ev_metrics.get("preparation_knee_flexion"),
                            "weight_transfer": ev_metrics.get("weight_transfer"),
                            "contact_lateral_distance": ev_metrics.get("contact_lateral_distance"),
                            "coach_messages": advice_msgs,
                        }
                    )

                # Choose representative swing (prefer media + closest to median peak speed)
                rep_swing = None
                if swings_summary:
                    with_media = [sw for sw in swings_summary if sw["clip_url"] or sw["impact_freeze_url"]]
                    candidates = with_media if with_media else swings_summary
                    target_spd = statistics.median(max_speeds) if max_speeds else 0.0
                    rep_swing = min(
                        candidates,
                        key=lambda sw: abs((sw.get("racket_max_speed_px_s") or target_spd) - target_spd),
                    )

                total_valid = max(1, len(ev_rows))
                coach_warnings = [
                    {
                        "message": msg,
                        "count": cnt,
                        "rate_pct": round((cnt / total_valid) * 100.0, 1),
                    }
                    for msg, cnt in sorted(warning_counter.items(), key=lambda kv: kv[1], reverse=True)[:5]
                ]

                # Additional metric medians + IQR from swing_metrics
                m_rows = conn.execute(
                    f"""
                    SELECT metric_key, metric_label, unit, value_num
                    FROM swing_metrics
                    WHERE session_id = ? AND value_num IS NOT NULL{stroke_clause}
                    """,
                    params,
                ).fetchall()
                metric_buckets: Dict[Tuple[str, str, str], List[float]] = {}
                for mr in m_rows:
                    bucket_key = (mr["metric_key"], mr["metric_label"] or mr["metric_key"], mr["unit"] or "")
                    metric_buckets.setdefault(bucket_key, []).append(float(mr["value_num"]))

                metric_medians = {
                    k: {
                        "label": lbl,
                        "unit": unit,
                        "median": round(statistics.median(vals), 2),
                        "iqr": _compute_iqr(vals),
                        "min": round(min(vals), 2),
                        "max": round(max(vals), 2),
                        "count": len(vals),
                    }
                    for (k, lbl, unit), vals in metric_buckets.items()
                }

                series.append(
                    {
                        "session_id": sid,
                        "display_time": s["display_time"],
                        "timestamp": s["timestamp"],
                        "source_name": s["source_name"],
                        "matched_swings": len(ev_rows),
                        "forehand_count": s["forehand_count"],
                        "backhand_count": s["backhand_count"],
                        "shadow_count": s["shadow_count"],
                        "median_score": round(statistics.median(scores), 1) if scores else None,
                        "median_racket_speed_px_s": round(statistics.median(speeds), 1) if speeds else (
                            metric_medians.get("racket_max_speed", {}).get("median")
                        ),
                        "median_brush_angle_deg": round(statistics.median(brushes), 1) if brushes else None,
                        "median_leg_drive_px": round(statistics.median(legs), 1) if legs else None,
                        "kinematic_ok_rate": round((seq_good / seq_total) * 100.0, 1) if seq_total > 0 else None,
                        "metrics": metric_medians,
                        "representative_event": rep_swing,
                        "swings": swings_summary,
                        "coach_warnings": coach_warnings,
                    }
                )

            # Compute deltas between latest and previous session
            deltas: Dict[str, Optional[float]] = {}
            if len(series) >= 2:
                prev, curr = series[-2], series[-1]
                for field in (
                    "median_racket_speed_px_s",
                    "median_brush_angle_deg",
                    "median_leg_drive_px",
                    "kinematic_ok_rate",
                ):
                    if curr.get(field) is not None and prev.get(field) is not None:
                        deltas[field] = round(curr[field] - prev[field], 1)
                    else:
                        deltas[field] = None

            return {
                "player": dict(player_row),
                "stroke_type": stroke_filter or "all",
                "session_count": len(series),
                "sessions": series,
                "latest_delta": deltas,
            }
