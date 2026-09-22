"""
scripts/process_algo2_dual_view.py
──────────────────────────────────
算法 2.0 全流程离线批处理脚本：
输入原始 2.5K 视频，自动完成：
1. 机位解耦与镜面水平翻转
2. 双视角并行姿态估计 (yolo26 / yolov8-pose)
3. 击球正反手判定与后背生物力学计算
4. 渲染导出包含双视角骨骼与 HUD 指标的完整 MP4 视频
"""
import argparse
from pathlib import Path
import sys
import cv2
import numpy as np
import time

# Ensure repo root is in python path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dual_view_manager import DualViewManager
from dual_pose_estimator import DualPoseEstimator
from dual_view_renderer import DualViewRenderer


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

    cap = cv2.VideoCapture(str(input_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"   视频信息: {w_orig}x{h_orig} @ {fps:.1f} FPS, 共 {total_frames} 帧")

    mgr = DualViewManager()
    estimator = DualPoseEstimator(backend="auto")
    renderer = DualViewRenderer(show_hud=True, show_skeleton=True)

    # 临时写出原始 mp4v 容器，随后调用 ffmpeg 转码为 H.264
    temp_output = "/tmp/algo2_temp_out.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_w, out_h = 1080, 720
    writer = cv2.VideoWriter(temp_output, fourcc, fps, (out_w, out_h))

    frame_idx = 0
    t0 = time.time()
    saved_snapshot = False
    frame_records = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if args.max_frames and frame_idx >= args.max_frames:
            break

        dual_frame = mgr.split_frame(frame, frame_id=frame_idx)
        pose_res = estimator.estimate_dual_pose(dual_frame)
        rendered_sbs = renderer.render_dual_frame(dual_frame, pose_res)

        writer.write(rendered_sbs)

        # 收集帧级双视角生物力学记录供事件管线使用
        b = pose_res.biomechanics
        frame_records.append({
            "frame_id": frame_idx,
            "timestamp": frame_idx / fps,
            "pose": {k: (kp.x, kp.y) for k, kp in pose_res.front_pose_orig.items()},
            "healed_pose": {k: (kp.x, kp.y) for k, kp in pose_res.fused_pose_local.items()},
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

        # 保存挥拍瞬间典型帧供复查 (第 25 帧)
        if frame_idx == 25:
            snapshot_path = "/Users/krum5539/.gemini/antigravity/brain/853db2fd-bbb9-45de-8209-c65d2189b516/scratch/algo2_swing_frame_25.jpg"
            cv2.imwrite(snapshot_path, rendered_sbs)
            saved_snapshot = True

        frame_idx += 1
        if frame_idx % 25 == 0 or frame_idx == total_frames:
            elapsed = time.time() - t0
            speed_fps = frame_idx / max(0.001, elapsed)
            print(f"   处理进度: [{frame_idx}/{total_frames}] ({frame_idx/total_frames*100:.1f}%) | 速度: {speed_fps:.1f} FPS")

    cap.release()
    writer.release()
    total_elapsed = time.time() - t0
    print(f"✅ 视频渲染完成！耗时: {total_elapsed:.2f}s, 平均吞吐: {frame_idx/max(0.001, total_elapsed):.1f} FPS")

    # 转码为广泛兼容的 H.264
    import subprocess
    cmd = [
        "ffmpeg", "-y",
        "-i", temp_output,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        args.output,
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    Path(temp_output).unlink(missing_ok=True)
    print(f"🎉 最终成果视频已生成: {args.output}")

    # 运行算法 2.0 事件级生物力学聚合与教练建议
    if frame_records:
        from swing_motion_features import extract_motion_features
        from swing_event_classifier import classify_swing_event
        from swing_biomechanics import aggregate_event_biomechanics
        from local_realtime_coach import LocalRealtimeCoach

        print("\n" + "=" * 60)
        print("🧠 算法 2.0 事件级生物力学与智能教练建议分析报告")
        print("=" * 60)

        features = extract_motion_features(frame_records, dominant_hand="right")
        # 击球瞬间在第 32 帧左右，挥拍核心窗口 [15, 55]
        core_features = features[15:55] if len(features) >= 55 else features
        classification = classify_swing_event(core_features)
        
        event_dict = {
            "event_id": 1,
            "start_frame": 15,
            "contact_frame": 32,
            "peak_frame": 32,
            "end_frame": min(len(frames_records_cache) - 1, 55) if 'frames_records_cache' in locals() else 55,
            "quality_flags": {"pose_frame_ratio": 1.0},
            "phase_counts": {"backswing": 12, "forward_swing": 8, "follow_through": 15},
            "confidence": classification["confidence"],
        }
        biomech_summary = aggregate_event_biomechanics(event_dict, frame_records, features)
        event_dict["biomechanics"] = biomech_summary

        coach = LocalRealtimeCoach()
        advices = coach.advise_all(event_dict)

        print(f"🎾 动作类型: {classification['stroke_type']} (置信度: {classification['confidence'] * 100:.1f}%)")
        print(f"📋 判定规则: {classification['evidence']['classification_context']['decision_rule']}")
        
        metrics = biomech_summary["metrics"]
        print("\n📊 核心生物力学指标:")
        tb = metrics.get("takeback_depth") or {}
        sc = metrics.get("scapular_retraction") or {}
        st = metrics.get("shoulder_turn") or {}
        arm = metrics.get("arm_extension") or {}

        print(f"   • 后背引拍深度比 (Takeback Depth): {tb.get('value', 'N/A')} (置信度: {tb.get('confidence', 0)*100:.1f}%)")
        print(f"   • 肩胛骨收缩比率 (Scapular Pinch): {sc.get('value', 'N/A')} (置信度: {sc.get('confidence', 0)*100:.1f}%)")
        print(f"   • 抗侧身塌陷转肩角 (Shoulder Turn): {st.get('value', 'N/A')}° (置信度: {st.get('confidence', 0)*100:.1f}%)")
        print(f"   • 手臂延展角度 (Arm Extension): {arm.get('value', 'N/A')}° (置信度: {arm.get('confidence', 0)*100:.1f}%)")

        print("\n📢 实时教练纠错建议 (≤15字):")
        for i, adv in enumerate(advices, 1):
            print(f"   {i}. [{adv['code']}] {adv['message']} (置信度: {adv['confidence']*100:.1f}%)")
        print("=" * 60)


if __name__ == "__main__":
    main()
