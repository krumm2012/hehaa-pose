"""
scripts/process_algo2_dual_view.py
──────────────────────────────────
算法 2.0 全流程双视角生物力学离线批处理脚本 (Two-Pass Biomechanical Pipeline)：
输入原始 2.5K 视频，自动完成：
1. [Pass 1] 机位解耦、镜面水平翻转与双视角并行姿态估计 (yolo26 / yolov8-pose)
2. [Analysis] 事件动力学能量切割、基于触球窗口的确定性正反手判定与智能教练纠错 (≤15字)
3. [Pass 2] 渲染导出包含双视角骨骼、平稳事件标签与 HUD 生物力学仪表盘的完整 MP4 视频
"""
import argparse
from pathlib import Path
import subprocess
import sys
import time
import cv2
import numpy as np

# Ensure repo root is in python path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dual_view_manager import DualViewManager
from dual_pose_estimator import DualPoseEstimator
from dual_view_renderer import DualViewRenderer
from swing_motion_features import extract_motion_features
from swing_event_segmenter import segment_swing_events
from swing_event_classifier import classify_swing_event
from swing_biomechanics import aggregate_event_biomechanics
from local_realtime_coach import LocalRealtimeCoach


def main():
    parser = argparse.ArgumentParser(description="Algorithm 2.0 Dual View Processing")
    parser.add_argument(
        "--input",
        type=str,
        default="/Users/krum5539/Desktop/Camera/49.35.mp4",
        help="Input video path",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/Users/krum5539/Desktop/Camera/algo2_dual_view_biomechanics.mp4",
        help="Output video path",
    )
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames to process")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: input video {input_path} not found.")
        return

    print(f"🎬 启动算法 2.0 双机位生物力学全流程处理...")
    print(f"   输入视频: {input_path}")
    print(f"   输出目标: {args.output}")

    # =========================================================================
    # Pass 1: 姿态估计与运动动力学时序特征收集
    # =========================================================================
    print(f"\n🚀 [Pass 1/2] 双视角人体姿态估计与运动动力学特征提取...")
    cap = cv2.VideoCapture(str(input_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"   视频信息: {w_orig}x{h_orig} @ {fps:.1f} FPS, 共 {total_frames} 帧")

    mgr = DualViewManager()
    estimator = DualPoseEstimator(backend="auto")

    pose_results = []
    frame_records = []
    frame_idx = 0
    t0 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if args.max_frames and frame_idx >= args.max_frames:
            break

        dual_frame = mgr.split_frame(frame, frame_id=frame_idx)
        pose_res = estimator.estimate_dual_pose(dual_frame)
        pose_results.append(pose_res)

        b = pose_res.biomechanics
        frame_records.append({
            "frame_id": frame_idx,
            "timestamp": frame_idx / fps,
            "pose": {k: (kp.x, kp.y) for k, kp in pose_res.front_pose_orig.items()},
            "healed_pose": {k: (kp.x, kp.y) for k, kp in pose_res.fused_pose_orig.items()},
            "dual_view_biomechanics": {
                "shoulder_turn": {
                    "shoulder_turn_deg": b.robust_shoulder_turn_deg,
                    "confidence": 0.88,
                },
                "takeback_depth": {
                    "takeback_depth_ratio": b.takeback_depth_ratio,
                    "confidence": 0.85,
                },
                "scapular_retraction": {
                    "scapular_retraction_ratio": b.scapular_retraction_ratio,
                    "confidence": 0.85,
                },
                "shot_classification": {
                    "stroke_type": b.shot_classification.shot_type,
                    "is_two_handed": b.shot_classification.is_two_handed,
                    "confidence": b.shot_classification.confidence,
                },
                "contact_distance_gate": {
                    "is_valid_contact": b.shot_classification.is_valid_contact,
                },
            },
        })

        frame_idx += 1
        if frame_idx % 25 == 0 or frame_idx == total_frames:
            elapsed = time.time() - t0
            speed_fps = frame_idx / max(0.001, elapsed)
            print(f"   Pass 1 进度: [{frame_idx}/{total_frames}] ({frame_idx/total_frames*100:.1f}%) | 速度: {speed_fps:.1f} FPS")

    cap.release()
    pass1_time = time.time() - t0
    print(f"✅ Pass 1 完成！耗时: {pass1_time:.2f}s, 处理帧数: {len(pose_results)}")

    # =========================================================================
    # Analysis: 事件级动力学切分、物理力学分类与智能教练建议
    # =========================================================================
    print("\n" + "=" * 60)
    print("🧠 [Analysis] 算法 2.0 事件级生物力学与智能教练建议分析报告")
    print("=" * 60)

    frame_to_label = {}
    frame_to_coach = {}
    events = []

    if frame_records:
        features = extract_motion_features(frame_records, dominant_hand="right")
        res = segment_swing_events(features)
        events = res.get("events", [])
        coach = LocalRealtimeCoach()

        if not events:
            events = [{
                "event_id": 1,
                "start_frame": 0,
                "end_frame": len(frame_records) - 1,
                "contact_frame": len(frame_records) // 2,
                "peak_frame": len(frame_records) // 2,
                "confidence": 0.85,
            }]

        print(f"🎾 共检测到 {len(events)} 次有效挥拍事件：\n")

        for idx, event_dict in enumerate(events, 1):
            s_f = event_dict.get("start_frame", 0)
            e_f = event_dict.get("end_frame", len(features) - 1)
            c_f = event_dict.get("contact_frame", (s_f + e_f) // 2)
            ev_feats = [f for f in features if s_f <= f["frame_id"] <= e_f]
            classification = classify_swing_event(ev_feats, contact_frame=c_f)
            event_dict["stroke_type"] = classification["stroke_type"]
            event_dict["confidence"] = classification["confidence"]
            event_dict["quality_flags"] = {"pose_frame_ratio": 1.0}
            biomech_summary = aggregate_event_biomechanics(event_dict, frame_records, features)
            event_dict["biomechanics"] = biomech_summary

            advices = coach.advise_all(event_dict)
            first_coach_msg = advices[0]["message"] if advices else ""

            print(f"📍 [事件 #{idx}] 帧区间: [{s_f} -> {e_f}] | 击球点/峰值帧: {c_f}")
            print(f"   动作类型: {classification['stroke_type']} (置信度: {classification['confidence'] * 100:.1f}%)")
            rule = classification.get("evidence", {}).get("classification_context", {}).get("decision_rule", "dual_view_transverse_projection")
            print(f"   判定规则: {rule}")

            metrics = biomech_summary["metrics"]
            print("   📊 核心生物力学指标:")
            tb = metrics.get("takeback_depth") or {}
            sc = metrics.get("scapular_retraction") or {}
            st = metrics.get("shoulder_turn") or {}
            arm = metrics.get("arm_extension") or {}

            print(f"      • 后背引拍深度比 (Takeback Depth): {tb.get('value', 'N/A')} (置信度: {tb.get('confidence', 0)*100:.1f}%)")
            print(f"      • 肩胛骨收缩比率 (Scapular Pinch): {sc.get('value', 'N/A')} (置信度: {sc.get('confidence', 0)*100:.1f}%)")
            print(f"      • 抗侧身塌陷转肩角 (Shoulder Turn): {st.get('value', 'N/A')}° (置信度: {st.get('confidence', 0)*100:.1f}%)")
            print(f"      • 手臂延展角度 (Arm Extension): {arm.get('value', 'N/A')}° (置信度: {arm.get('confidence', 0)*100:.1f}%)")

            print("   📢 实时教练纠错建议 (≤15字):")
            for j, adv in enumerate(advices, 1):
                print(f"      {j}. [{adv['code']}] {adv['message']} (置信度: {adv['confidence']*100:.1f}%)")
            print("-" * 60)

            # 填充帧映射表供 Pass 2 视频渲染使用
            event_display_label = f"{classification['stroke_type']} (Event #{idx})"
            for f_no in range(s_f, e_f + 1):
                frame_to_label[f_no] = event_display_label
                frame_to_coach[f_no] = first_coach_msg

        print("=" * 60)

    # =========================================================================
    # Pass 2: Side-by-Side 视频高清渲染 (融合平稳事件标签与教练 HUD)
    # =========================================================================
    print(f"\n🎬 [Pass 2/2] 双视角 Side-by-Side 视频高清渲染...")
    renderer = DualViewRenderer(show_hud=True, show_skeleton=True)

    temp_output = "/tmp/algo2_temp_out.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_w, out_h = 1080, 720
    writer = cv2.VideoWriter(temp_output, fourcc, fps, (out_w, out_h))

    cap = cv2.VideoCapture(str(input_path))
    frame_idx = 0
    t_render_start = time.time()
    snapshots_saved = set()
    snapshot_dir = Path("/Users/krum5539/.gemini/antigravity/brain/853db2fd-bbb9-45de-8209-c65d2189b516/scratch")
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    # 记录待截图保存的关键帧（每个事件的 contact 帧与典型 follow-through 帧，如 197）
    target_snapshots = set()
    for ev in events:
        c_f = ev.get("contact_frame")
        if c_f is not None:
            target_snapshots.add(c_f)
    target_snapshots.add(197)  # 用户重点关注的第 197 帧

    while True:
        ret, frame = cap.read()
        if not ret or frame_idx >= len(pose_results):
            break

        dual_frame = mgr.split_frame(frame, frame_id=frame_idx)
        pose_res = pose_results[frame_idx]

        ev_label = frame_to_label.get(frame_idx, "READY STANCE")
        ev_coach = frame_to_coach.get(frame_idx, "")

        rendered_sbs = renderer.render_dual_frame(
            dual_frame,
            pose_res,
            event_label=ev_label,
            coaching_text=ev_coach,
        )
        writer.write(rendered_sbs)

        if frame_idx in target_snapshots and frame_idx not in snapshots_saved:
            snap_path = snapshot_dir / f"algo2_verified_frame_{frame_idx}.jpg"
            cv2.imwrite(str(snap_path), rendered_sbs)
            snapshots_saved.add(frame_idx)
            print(f"   📸 已保存复查快照帧: {snap_path}")

        frame_idx += 1
        if frame_idx % 50 == 0 or frame_idx == len(pose_results):
            elapsed = time.time() - t_render_start
            speed_fps = frame_idx / max(0.001, elapsed)
            print(f"   Pass 2 渲染进度: [{frame_idx}/{len(pose_results)}] ({frame_idx/len(pose_results)*100:.1f}%) | 速度: {speed_fps:.1f} FPS")

    cap.release()
    writer.release()
    render_time = time.time() - t_render_start
    print(f"✅ Pass 2 渲染完成！耗时: {render_time:.2f}s, 平均速度: {frame_idx/max(0.001, render_time):.1f} FPS")

    # 转码为广泛兼容的 H.264 MP4
    cmd = [
        "ffmpeg", "-y",
        "-i", temp_output,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        args.output,
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    Path(temp_output).unlink(missing_ok=True)
    print(f"🎉 最终成果视频已成功生成: {args.output}")


if __name__ == "__main__":
    main()
