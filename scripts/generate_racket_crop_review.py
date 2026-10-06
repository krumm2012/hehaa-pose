"""Generate high-resolution local crops and interactive manual review board for uncertain racket frames.

Produces 640x640 crops directly from raw 2560x1440 video frames, side-by-side comparison images,
and a standalone interactive review dashboard.
"""
from __future__ import annotations

import html
import json
import os
import shutil
from pathlib import Path

import cv2
import numpy as np

VIDEO_PATH = Path("data/control_uploads/304b2cda45f90739ed77c6e3f98f8cd3.mp4")
JOURNAL_PATH = Path(
    "data/analysis_results/control_panel/court02_temporal_racket_20261005T161814Z_0ab2c2/court02_temporal_racket_frames.jsonl"
)
OUTPUT_DIR = Path("data/analysis_results/kinematic_validation/court02_racket_manual_review_20261006_v1")
ARTIFACT_DIR = Path("/Users/krum5539/.gemini/antigravity/brain/5cb7b593-51ca-44d0-b7a2-fe304c1f677b")

TARGET_FRAMES = [
    # Event 1
    {"fid": 20, "event": 1, "category": "motion_blur", "title": "触球前置基准锚点", "desc": "击球前高置信度基准 (0.874)"},
    {"fid": 21, "event": 1, "category": "contact", "title": "触球核心帧", "desc": "触球瞬间严重运动模糊，自愈置信度 0.478，球位于拍框边缘"},
    {"fid": 22, "event": 1, "category": "motion_blur", "title": "击球后自愈地板帧", "desc": "随挥初段置信度降至 0.270（自愈底限 0.25），球在球网回弹"},
    {"fid": 23, "event": 1, "category": "motion_blur", "title": "击球后自愈延续帧", "desc": "弱特征自愈置信度 0.304，拍面加速穿越"},
    {"fid": 25, "event": 1, "category": "boundary", "title": "随挥速度峰值边界帧", "desc": "候选速度边界点 2304 px/s，置信度 0.656"},
    {"fid": 26, "event": 1, "category": "boundary", "title": "身体遮挡截断点", "desc": "拍头绕至背后，模型置信度降为 0，进入断流"},
    # Event 2
    {"fid": 102, "event": 2, "category": "boundary", "title": "VFR 抖动拐点帧", "desc": "时间戳 2.18ms 抖动点，置信度 0.698"},
    {"fid": 103, "event": 2, "category": "boundary", "title": "角速度突变拐点帧", "desc": "速度骤升点 (1628 px/s)，置信度 0.682"},
    {"fid": 105, "event": 2, "category": "gating", "title": "引拍低点规则误杀帧", "desc": "模型检测候选 0.868，因手腕遮挡被门控误杀剔除"},
    {"fid": 106, "event": 2, "category": "gating", "title": "引拍过渡规则误杀帧", "desc": "模型检测候选 0.855，因手腕遮挡被门控误杀剔除"},
    {"fid": 110, "event": 2, "category": "contact", "title": "触球核心帧", "desc": "双手反拍击球瞬间，置信度 0.659，球正中拍面甜点"},
    {"fid": 112, "event": 2, "category": "boundary", "title": "随挥截断前置帧", "desc": "随挥爬升末端，置信度 0.549"},
    {"fid": 113, "event": 2, "category": "boundary", "title": "随挥绕身截断点", "desc": "拍身进入背部盲区，模型置信度归零"},
    # Event 3
    {"fid": 179, "event": 3, "category": "gating", "title": "大引拍手腕遮挡帧", "desc": "模型检测候选 0.783，被手腕门控剔除"},
    {"fid": 180, "event": 3, "category": "gating", "title": "深引拍极限帧", "desc": "模型检测候选 0.515，被手腕门控剔除"},
    {"fid": 187, "event": 3, "category": "gating", "title": "击球前高置信度误杀帧 (重点关注)", "desc": "模型置信度高达 0.897！因侧身手腕遮挡被完全剔除为 0 框"},
    {"fid": 188, "event": 3, "category": "boundary", "title": "击球前重捕获帧", "desc": "向前挥动重获手腕引导，置信度 0.792"},
    {"fid": 190, "event": 3, "category": "boundary", "title": "击球前极限加速帧", "desc": "加速穿越前帧，置信度 0.733"},
    {"fid": 191, "event": 3, "category": "contact", "title": "触球核心帧", "desc": "双手反拍击球瞬间，置信度 0.917，球切入拍面甜点"},
    {"fid": 194, "event": 3, "category": "boundary", "title": "击球后截断点", "desc": "随挥高速外旋，模型置信度归零"},
]


def load_journal_data():
    records = {}
    with open(JOURNAL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            records[data["frame_id"]] = data
    return records


def get_crop_center(record, fallback_w=2560, fallback_h=1440):
    rackets = record.get("rackets") or []
    ball = record.get("ball")
    pose = record.get("pose") or {}

    centers = []
    if rackets and rackets[0].get("box"):
        b = rackets[0]["box"]
        centers.append(((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0))

    if ball:
        centers.append((float(ball[0]), float(ball[1])))

    rw = pose.get("right_wrist")
    re = pose.get("right_elbow")
    if rw:
        centers.append((float(rw[0]), float(rw[1])))
    if re:
        centers.append((float(re[0]), float(re[1])))

    if centers:
        avg_x = sum(c[0] for c in centers) / len(centers)
        avg_y = sum(c[1] for c in centers) / len(centers)
        return int(avg_x), int(avg_y)

    # Event default priors
    fid = record.get("frame_id", 0)
    if fid < 60:
        return 1200, 550
    elif fid < 150:
        return 1600, 500
    else:
        return 1700, 520


def draw_annotations(crop, record, crop_x1, crop_y1):
    annotated = crop.copy()
    h, w, _ = annotated.shape

    rackets = record.get("rackets") or []
    ball = record.get("ball")
    pose = record.get("pose") or {}
    diag = record.get("racket_detection_diagnostics", {})
    t_diag = diag.get("temporal_tracking", {})
    m_diag = diag.get("model_candidates", {})
    rejections = t_diag.get("rejected", {})
    max_c = m_diag.get("max_confidence", 0.0)

    # 1. Racket box
    if rackets and rackets[0].get("box"):
        r = rackets[0]
        box = r["box"]
        bx1 = int(box[0] - crop_x1)
        by1 = int(box[1] - crop_y1)
        bx2 = int(box[2] - crop_x1)
        by2 = int(box[3] - crop_y1)

        is_observed = r.get("observed", False)
        is_rec = r.get("temporal_recovery") is not None
        conf = r.get("confidence", 0.0)

        if is_observed and not is_rec:
            color = (0, 220, 0)  # Bright green
            tag = f"Racket: Model {conf:.3f}"
        elif is_rec:
            color = (0, 165, 255)  # Orange/yellow
            tag = f"Racket: Recovered {conf:.3f}"
        else:
            color = (0, 0, 240)  # Red
            tag = f"Holdover {conf:.3f}"

        cv2.rectangle(annotated, (bx1, by1), (bx2, by2), color, 3)
        cv2.putText(
            annotated,
            tag,
            (bx1, max(22, by1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )
    elif rejections:
        # Rejected frame indicator
        rej_str = ", ".join(f"{k}:{v}" for k, v in rejections.items())
        cv2.putText(
            annotated,
            f"Gating Rejection: max_c={max_c:.3f} [{rej_str}]",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    # 2. Ball
    if ball:
        bx = int(ball[0] - crop_x1)
        by = int(ball[1] - crop_y1)
        if 0 <= bx < w and 0 <= by < h:
            cv2.circle(annotated, (bx, by), 9, (255, 230, 0), -1)  # Cyan dot
            cv2.circle(annotated, (bx, by), 12, (0, 140, 255), 2)
            cv2.putText(
                annotated,
                "Ball",
                (bx + 12, by + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 230, 0),
                2,
                cv2.LINE_AA,
            )

    # 3. Pose Keypoints
    # Wrists: Magenta
    for side in ("right_wrist", "left_wrist"):
        pt = pose.get(side)
        if pt:
            px = int(pt[0] - crop_x1)
            py = int(pt[1] - crop_y1)
            if 0 <= px < w and 0 <= py < h:
                color = (255, 0, 255) if "right" in side else (180, 50, 200)
                cv2.circle(annotated, (px, py), 6, color, -1)
                cv2.putText(
                    annotated,
                    "R-Wrist" if "right" in side else "L-Wrist",
                    (px + 8, py - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1,
                    cv2.LINE_AA,
                )

    # Elbows: Light Blue
    for side in ("right_elbow", "left_elbow"):
        pt = pose.get(side)
        if pt:
            px = int(pt[0] - crop_x1)
            py = int(pt[1] - crop_y1)
            if 0 <= px < w and 0 <= py < h:
                color = (255, 200, 100)
                cv2.circle(annotated, (px, py), 5, color, -1)

    return annotated


def create_comparison_image(clean, annotated, meta, record):
    from PIL import Image, ImageDraw, ImageFont

    h, w, _ = clean.shape
    banner_h = 75
    comp = np.zeros((h + banner_h, w * 2, 3), dtype=np.uint8)

    # Banner background
    comp[0:banner_h, :] = (24, 28, 36)

    # Clean crop on left, Annotated on right
    comp[banner_h:, :w] = clean
    comp[banner_h:, w:] = annotated

    # Divider line
    cv2.line(comp, (w, 0), (w, h + banner_h), (60, 70, 85), 2)

    # Convert to PIL for sharp Unicode/Chinese rendering
    img_pil = Image.fromarray(cv2.cvtColor(comp, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    font_path = "/System/Library/Fonts/Hiragino Sans GB.ttc"
    try:
        font_title = ImageFont.truetype(font_path, 21)
        font_desc = ImageFont.truetype(font_path, 14)
        font_sub = ImageFont.truetype(font_path, 14)
    except Exception:
        font_title = font_desc = font_sub = ImageFont.load_default()

    fid = meta["fid"]
    ev = meta["event"]
    pts = record.get("timestamp", 0.0)
    title = meta["title"]
    desc = meta["desc"]

    rackets = record.get("rackets") or []
    has_racket = bool(rackets)
    is_obs = rackets[0].get("observed", False) if has_racket else False
    is_rec = rackets[0].get("temporal_recovery") is not None if has_racket else False
    conf = rackets[0].get("confidence", 0.0) if has_racket else 0.0
    diag = record.get("racket_detection_diagnostics", {})
    max_c = diag.get("model_candidates", {}).get("max_confidence", 0.0)
    rejections = diag.get("temporal_tracking", {}).get("rejected", {})

    status_tag = f"置信度: {conf:.3f}" if has_racket else f"原始候选: {max_c:.3f}"
    if is_rec:
        status_tag += " [自愈恢复]"
    elif rejections:
        status_tag += f" [门控拦截: {list(rejections.keys())[0]}]"

    # Draw Title & Desc
    draw.text((18, 12), f"挥拍 Event {ev} | 帧 {fid:03d} (PTS {pts:.4f}s) — {title}", font=font_title, fill=(245, 245, 245))
    draw.text((18, 44), f"状态: {status_tag} | 说明: {desc}", font=font_desc, fill=(160, 210, 255))

    # Labels for left/right panels
    draw.text((20, banner_h + 12), "左侧：2560x1440 原画局部裁剪 (无损原图)", font=font_sub, fill=(255, 255, 255))
    draw.text((w + 20, banner_h + 12), "右侧：算法检测/自愈框/关节点叠加标注", font=font_sub, fill=(88, 166, 255))

    # Convert back to BGR numpy array
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def generate_crops():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    records = load_journal_data()
    cap = cv2.VideoCapture(str(VIDEO_PATH))

    crop_metadata = []

    for meta in TARGET_FRAMES:
        fid = meta["fid"]
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ret, frame = cap.read()
        if not ret:
            print(f"Error: Could not read frame {fid}")
            continue

        h, w, _ = frame.shape
        rec = records.get(fid, {})
        cx, cy = get_crop_center(rec, w, h)

        # 640x640 bounds clamped within [0, w] and [0, h]
        x1 = max(0, min(w - 640, cx - 320))
        y1 = max(0, min(h - 640, cy - 320))

        clean_crop = frame[y1 : y1 + 640, x1 : x1 + 640].copy()
        annotated_crop = draw_annotations(clean_crop, rec, x1, y1)
        compare_crop = create_comparison_image(clean_crop, annotated_crop, meta, rec)

        clean_name = f"clean_frame_{fid}.png"
        annotated_name = f"annotated_frame_{fid}.png"
        compare_name = f"compare_frame_{fid}.png"

        cv2.imwrite(str(OUTPUT_DIR / clean_name), clean_crop)
        cv2.imwrite(str(OUTPUT_DIR / annotated_name), annotated_crop)
        cv2.imwrite(str(OUTPUT_DIR / compare_name), compare_crop)

        # Copy to conversation artifact dir for inline chat preview
        shutil.copy2(OUTPUT_DIR / clean_name, ARTIFACT_DIR / clean_name)
        shutil.copy2(OUTPUT_DIR / compare_name, ARTIFACT_DIR / compare_name)

        rackets = rec.get("rackets") or []
        diag = rec.get("racket_detection_diagnostics", {})
        crop_metadata.append(
            {
                "fid": fid,
                "event": meta["event"],
                "category": meta["category"],
                "title": meta["title"],
                "desc": meta["desc"],
                "pts": rec.get("timestamp", 0.0),
                "crop_bounds": [x1, y1, x1 + 640, y1 + 640],
                "rackets": rackets,
                "diagnostics": diag,
                "ball": rec.get("ball"),
                "clean_file": clean_name,
                "annotated_file": annotated_name,
                "compare_file": compare_name,
            }
        )
        print(f"Frame {fid:03d} generated ({meta['title']})")

    cap.release()
    return crop_metadata


def build_review_html(metadata):
    categories = [
        ("all", "全部存疑帧 (20)"),
        ("contact", "触球瞬间核心帧 (3)"),
        ("motion_blur", "运动模糊自愈帧 (4)"),
        ("gating", "规则拦截/遮挡剔除帧 (5)"),
        ("boundary", "动力学拐点与截断边界帧 (8)"),
    ]

    cards_html = []
    for item in metadata:
        fid = item["fid"]
        ev = item["event"]
        cat = item["category"]
        pts = item["pts"]
        title = item["title"]
        desc = item["desc"]
        rackets = item["rackets"]
        diag = item["diagnostics"]
        t_diag = diag.get("temporal_tracking", {})
        m_diag = diag.get("model_candidates", {})
        rejections = t_diag.get("rejected", {})
        max_c = m_diag.get("max_confidence", 0.0)

        has_racket = bool(rackets)
        is_obs = rackets[0].get("observed", False) if has_racket else False
        is_rec = rackets[0].get("temporal_recovery") is not None if has_racket else False
        conf = rackets[0].get("confidence", 0.0) if has_racket else 0.0
        box = rackets[0].get("box") if has_racket else None

        if is_obs and not is_rec:
            badge_color = "#10b981"
            badge_text = f"模型直出 Conf: {conf:.3f}"
        elif is_rec:
            badge_color = "#f59e0b"
            badge_text = f"自愈恢复 Conf: {conf:.3f} (锚点 20)"
        elif rejections:
            badge_color = "#ef4444"
            rej_key = list(rejections.keys())[0]
            badge_text = f"门控拦截: {rej_key} (Max Cand: {max_c:.3f})"
        else:
            badge_color = "#6b7280"
            badge_text = f"断流/未检出 (Max Cand: {max_c:.3f})"

        card = f"""
        <div class="review-card" data-category="{cat}" data-fid="{fid}" data-event="{ev}">
            <div class="card-header">
                <div class="card-title-group">
                    <span class="event-tag">挥拍 Event {ev}</span>
                    <span class="frame-tag">源帧 {fid:03d} (PTS: {pts:.4f}s)</span>
                    <span class="card-title">{html.escape(title)}</span>
                </div>
                <div class="status-badge" style="background: {badge_color};">{badge_text}</div>
            </div>
            <div class="card-desc">{html.escape(desc)}</div>
            
            <div class="image-viewer-container">
                <div class="image-toggle-bar">
                    <button class="toggle-btn active" onclick="switchView({fid}, 'compare')">对比图 (原始 | 算法标注)</button>
                    <button class="toggle-btn" onclick="switchView({fid}, 'clean')">高清原图裁剪 (无标注)</button>
                    <button class="toggle-btn" onclick="switchView({fid}, 'annotated')">算法标注图</button>
                </div>
                <div class="image-display" id="img-display-{fid}">
                    <img src="{item['compare_file']}" alt="Frame {fid} compare" id="img-elem-{fid}" class="zoomable" onclick="zoomImage(this)" />
                </div>
            </div>

            <div class="metadata-grid">
                <div class="meta-item">
                    <span class="meta-label">球拍检测框</span>
                    <span class="meta-val">{box if box else '无 (已被门控剔除或归零)'}</span>
                </div>
                <div class="meta-item">
                    <span class="meta-label">球位置 (Ball)</span>
                    <span class="meta-val">{item['ball'] if item['ball'] else '未检出'}</span>
                </div>
                <div class="meta-item">
                    <span class="meta-label">门控剔除原因</span>
                    <span class="meta-val">{dict(rejections) if rejections else '无拦截'}</span>
                </div>
                <div class="meta-item">
                    <span class="meta-label">模型最大原始候选</span>
                    <span class="meta-val">{max_c:.4f}</span>
                </div>
            </div>

            <div class="action-panel">
                <div class="action-title">人工判定意见：</div>
                <div class="action-options">
                    <label class="radio-label">
                        <input type="radio" name="decision_{fid}" value="valid_racket" onchange="recordDecision({fid}, 'valid_racket')">
                        <span class="opt-text text-green">✅ 确认是真实球拍（检出正确）</span>
                    </label>
                    <label class="radio-label">
                        <input type="radio" name="decision_{fid}" value="false_positive" onchange="recordDecision({fid}, 'false_positive')">
                        <span class="opt-text text-red">❌ 误检（背景/球网/地面虚影）</span>
                    </label>
                    <label class="radio-label">
                        <input type="radio" name="decision_{fid}" value="gating_error" onchange="recordDecision({fid}, 'gating_error')">
                        <span class="opt-text text-amber">⚠️ 门控误杀（确实是球拍，应予放行）</span>
                    </label>
                    <label class="radio-label">
                        <input type="radio" name="decision_{fid}" value="unidentifiable" onchange="recordDecision({fid}, 'unidentifiable')">
                        <span class="opt-text text-gray">❓ 特征过弱 / 重度拖影不可辨认</span>
                    </label>
                </div>
                <div class="note-box">
                    <input type="text" id="note_{fid}" placeholder="填写备注说明（如：拍面在手腕下方、球正在离弦等）..." oninput="recordNote({fid}, this.value)" />
                </div>
            </div>
        </div>
        """
        cards_html.append(card)

    tabs_html = "\n".join(
        [
            f'<button class="tab-btn { "active" if cat == "all" else "" }" onclick="filterCategory(\'{cat}\')">{label}</button>'
            for cat, label in categories
        ]
    )

    full_html = f"""<!doctype html>
<html lang="zh-CN">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Court02 球拍存疑关键帧人工复核面板 (2560x1440 原生裁剪)</title>
    <style>
        :root {{
            --bg: #0d1117;
            --card-bg: #161b22;
            --border: #30363d;
            --text-main: #f0f6fc;
            --text-sub: #8b949e;
            --accent: #58a6ff;
            --green: #238636;
            --green-text: #3fb950;
            --amber: #d29922;
            --amber-text: #e3b341;
            --red: #da3633;
            --red-text: #f85149;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background: var(--bg);
            color: var(--text-main);
            margin: 0;
            padding: 24px;
            line-height: 1.6;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        header {{
            background: linear-gradient(135deg, #1f2937, #111827);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 24px 32px;
            margin-bottom: 24px;
        }}
        h1 {{
            margin: 0 0 10px 0;
            font-size: 26px;
            color: #ffffff;
            display: flex;
            align-items: center;
            gap: 12px;
        }}
        .header-sub {{
            color: var(--text-sub);
            font-size: 15px;
            margin: 0 0 16px 0;
        }}
        .summary-stats {{
            display: flex;
            gap: 16px;
            flex-wrap: wrap;
        }}
        .stat-badge {{
            background: #21262d;
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 8px 16px;
            font-size: 14px;
        }}
        .stat-badge strong {{
            color: var(--accent);
            margin-left: 6px;
        }}
        .tabs-bar {{
            display: flex;
            gap: 8px;
            margin-bottom: 24px;
            flex-wrap: wrap;
            border-bottom: 1px solid var(--border);
            padding-bottom: 12px;
        }}
        .tab-btn {{
            background: #21262d;
            border: 1px solid var(--border);
            color: var(--text-main);
            padding: 8px 18px;
            border-radius: 8px;
            cursor: pointer;
            font-weight: 500;
            font-size: 14px;
            transition: all 0.2s;
        }}
        .tab-btn:hover {{
            background: #30363d;
            border-color: #8b949e;
        }}
        .tab-btn.active {{
            background: var(--accent);
            color: #0d1117;
            font-weight: 600;
            border-color: var(--accent);
        }}
        .review-card {{
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 28px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        }}
        .card-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
            flex-wrap: wrap;
            gap: 12px;
        }}
        .card-title-group {{
            display: flex;
            align-items: center;
            gap: 10px;
            flex-wrap: wrap;
        }}
        .event-tag {{
            background: #388bfd33;
            color: #58a6ff;
            border: 1px solid #388bfd66;
            padding: 3px 10px;
            border-radius: 6px;
            font-size: 13px;
            font-weight: bold;
        }}
        .frame-tag {{
            background: #30363d;
            color: #c9d1d9;
            padding: 3px 10px;
            border-radius: 6px;
            font-size: 13px;
        }}
        .card-title {{
            font-size: 18px;
            font-weight: bold;
            color: #fff;
        }}
        .status-badge {{
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 13px;
            font-weight: bold;
            color: #fff;
        }}
        .card-desc {{
            color: #8b949e;
            font-size: 14px;
            margin-bottom: 16px;
        }}
        .image-viewer-container {{
            background: #090d13;
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 12px;
            margin-bottom: 16px;
        }}
        .image-toggle-bar {{
            display: flex;
            gap: 8px;
            margin-bottom: 10px;
        }}
        .toggle-btn {{
            background: #1f242c;
            border: 1px solid #30363d;
            color: #c9d1d9;
            padding: 6px 14px;
            border-radius: 6px;
            font-size: 13px;
            cursor: pointer;
        }}
        .toggle-btn.active {{
            background: #388bfd;
            color: #fff;
            border-color: #388bfd;
            font-weight: bold;
        }}
        .image-display {{
            display: flex;
            justify-content: center;
            background: #000;
            border-radius: 6px;
            overflow: hidden;
            min-height: 400px;
        }}
        .image-display img {{
            max-width: 100%;
            height: auto;
            object-fit: contain;
            cursor: zoom-in;
            transition: transform 0.2s;
        }}
        .metadata-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 10px;
            background: #0f141c;
            border: 1px solid #21262d;
            border-radius: 8px;
            padding: 12px 16px;
            margin-bottom: 16px;
        }}
        .meta-item {{
            font-size: 13px;
            display: flex;
            flex-direction: column;
        }}
        .meta-label {{
            color: var(--text-sub);
            margin-bottom: 2px;
        }}
        .meta-val {{
            font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
            color: #e6edf3;
            font-weight: 500;
        }}
        .action-panel {{
            background: #1c2128;
            border: 1px solid #30363d;
            border-radius: 8px;
            padding: 14px 18px;
        }}
        .action-title {{
            font-size: 14px;
            font-weight: bold;
            color: #c9d1d9;
            margin-bottom: 10px;
        }}
        .action-options {{
            display: flex;
            gap: 16px;
            flex-wrap: wrap;
            margin-bottom: 12px;
        }}
        .radio-label {{
            display: flex;
            align-items: center;
            gap: 6px;
            cursor: pointer;
            font-size: 14px;
            background: #11151c;
            padding: 6px 12px;
            border-radius: 6px;
            border: 1px solid #30363d;
            transition: all 0.15s;
        }}
        .radio-label:hover {{
            border-color: #58a6ff;
        }}
        .text-green {{ color: var(--green-text); }}
        .text-red {{ color: var(--red-text); }}
        .text-amber {{ color: var(--amber-text); }}
        .text-gray {{ color: #8b949e; }}
        .note-box input {{
            width: 100%;
            background: #0d1117;
            border: 1px solid #30363d;
            border-radius: 6px;
            padding: 8px 12px;
            color: #fff;
            font-size: 13px;
            box-sizing: border-box;
        }}
        .note-box input:focus {{
            outline: none;
            border-color: var(--accent);
        }}
        .footer-bar {{
            position: sticky;
            bottom: 20px;
            background: #161b22ee;
            backdrop-filter: blur(8px);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 14px 24px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            box-shadow: 0 10px 25px rgba(0,0,0,0.5);
            margin-top: 32px;
        }}
        .save-btn {{
            background: var(--green);
            color: #fff;
            border: none;
            padding: 10px 24px;
            border-radius: 8px;
            font-size: 15px;
            font-weight: bold;
            cursor: pointer;
        }}
        .save-btn:hover {{
            background: #2ea043;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🎾 Court02 球拍存疑关键帧局部裁剪与人工复核面板</h1>
            <p class="header-sub">
                本面板针对动力链末端断流、运动模糊自愈底限、以及手腕遮挡门控剔除的 20 个关键帧，提供 2560x1440 原生分辨率局部高保真裁剪（640x640）。
                展示原始画质无损图与算法标注图对比，协助确认球拍真实形态。
            </p>
            <div class="summary-stats">
                <div class="stat-badge">总复核帧数: <strong>20 帧</strong></div>
                <div class="stat-badge">触球核心判定: <strong>3 帧 (21, 110, 191)</strong></div>
                <div class="stat-badge">模糊自愈跟踪: <strong>4 帧 (20, 21, 22, 23)</strong></div>
                <div class="stat-badge">规则误杀待放行: <strong>5 帧 (105, 106, 179, 180, 187)</strong></div>
                <div class="stat-badge">动力学拐点与截断: <strong>8 帧</strong></div>
            </div>
        </header>

        <div class="tabs-bar">
            {tabs_html}
        </div>

        <div id="cards-container">
            {"".join(cards_html)}
        </div>

        <div class="footer-bar">
            <div>
                <strong>复核进度：</strong> <span id="progress-text">已评审 0 / 20 帧</span>
            </div>
            <div>
                <button class="save-btn" onclick="exportReviewJSON()">💾 导出人工复核判定 JSON</button>
            </div>
        </div>
    </div>

    <script>
        const metadata = {json.dumps(metadata, ensure_ascii=False)};
        const decisions = JSON.parse(localStorage.getItem('court02_racket_decisions') || '{{}}');
        const notes = JSON.parse(localStorage.getItem('court02_racket_notes') || '{{}}');

        function init() {{
            for (const [fid, val] of Object.entries(decisions)) {{
                const radio = document.querySelector(`input[name="decision_${{fid}}"][value="${{val}}"]`);
                if (radio) radio.checked = true;
            }}
            for (const [fid, note] of Object.entries(notes)) {{
                const input = document.getElementById(`note_${{fid}}`);
                if (input) input.value = note;
            }}
            updateProgress();
        }}

        function switchView(fid, mode) {{
            const card = document.querySelector(`.review-card[data-fid="${{fid}}"]`);
            if (!card) return;
            const btns = card.querySelectorAll('.toggle-btn');
            btns.forEach(b => b.classList.remove('active'));
            const item = metadata.find(m => m.fid === fid);
            const img = document.getElementById(`img-elem-${{fid}}`);
            if (mode === 'compare') {{
                btns[0].classList.add('active');
                img.src = item.compare_file;
            }} else if (mode === 'clean') {{
                btns[1].classList.add('active');
                img.src = item.clean_file;
            }} else if (mode === 'annotated') {{
                btns[2].classList.add('active');
                img.src = item.annotated_file;
            }}
        }}

        function recordDecision(fid, val) {{
            decisions[fid] = val;
            localStorage.setItem('court02_racket_decisions', JSON.stringify(decisions));
            updateProgress();
        }}

        function recordNote(fid, val) {{
            notes[fid] = val;
            localStorage.setItem('court02_racket_notes', JSON.stringify(notes));
        }}

        function updateProgress() {{
            const count = Object.keys(decisions).length;
            document.getElementById('progress-text').textContent = `已评审 ${{count}} / ${{metadata.length}} 帧`;
        }}

        function filterCategory(cat) {{
            const btns = document.querySelectorAll('.tab-btn');
            btns.forEach(b => b.classList.remove('active'));
            event.target.classList.add('active');

            const cards = document.querySelectorAll('.review-card');
            cards.forEach(c => {{
                if (cat === 'all' || c.getAttribute('data-category') === cat) {{
                    c.style.display = 'block';
                }} else {{
                    c.style.display = 'none';
                }}
            }});
        }}

        function exportReviewJSON() {{
            const result = {{
                session_id: "court02_temporal_racket_20261005T161814Z_0ab2c2",
                schema: "tennis.racket-crop-review.v1",
                exported_at: new Date().toISOString(),
                decisions: decisions,
                notes: notes,
                reviewed_count: Object.keys(decisions).length,
                total_frames: metadata.length
            }};
            const blob = new Blob([JSON.stringify(result, null, 2)], {{type: 'application/json'}});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'court02_racket_manual_review_result.json';
            a.click();
            URL.revokeObjectURL(url);
        }}

        init();
    </script>
</body>
</html>
"""

    (OUTPUT_DIR / "index.html").write_text(full_html, encoding="utf-8")
    print(f"Generated review dashboard at: {OUTPUT_DIR / 'index.html'}")


def main():
    crops = generate_crops()
    build_review_html(crops)
    print("All tasks completed successfully!")


if __name__ == "__main__":
    main()
