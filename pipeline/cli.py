"""CLI argument parsing and entrypoint for main_pipe."""

from __future__ import annotations

import argparse
import yaml


def build_argument_parser():
    parser = argparse.ArgumentParser(description='网球分析多进程流水线')
    parser.add_argument('--config', '-c', default='configs/yolo26_tennis_config.yaml',
                        help='配置文件路径')
    parser.add_argument('--input', '-i', default=None, help='输入视频路径，覆盖配置文件')
    parser.add_argument('--output', '-o', default=None, help='输出视频路径，覆盖配置文件')
    parser.add_argument('--inference-schedule', choices=('barrier','interframe'), default=None,
                        help='启动推理调度；interframe 支持 offline 或新版独占连续两阶段重叠流水线')
    parser.add_argument('--racket-ipc', choices=('queue','shared_memory'), default=None,
                        help='球拍图像传输；共享源帧槽仅用于连续模式')
    parser.add_argument('--racket-pose-mode', choices=('legacy','exclusive','shadow','active'), default=None,
                        help='启动方案：legacy旧版／exclusive新版七点独占；shadow/active保留兼容诊断')
    parser.add_argument('--processing-mode', choices=('realtime','buffered','offline'), default=None,
                        help='buffered 连续处理不丢帧，允许启动缓冲；realtime 可丢旧帧；offline 完整分析')
    parser.add_argument(
        '--session-id',
        default=None,
        help='本次分析会话ID；默认自动生成并写入所有JSON/事件记录',
    )
    parser.add_argument(
        '--session-output-root',
        default=None,
        help='会话输出根目录；指定后默认产物写入 <root>/<session-id>/，显式输出路径仍保持不变',
    )
    parser.add_argument(
        '--evidence-manifest',
        default=None,
        help='可重放证据包清单；写入FrameRecord/Event哈希与重放参数',
    )
    parser.add_argument('--max-frames', type=int, default=None,
                        help='最大处理帧数；直播流建议设置以便安全结束。默认使用 video_processing.max_frames')
    parser.add_argument('--live-mode', action='store_true',
                        help='直播优化：关闭双视图、详细指标文本和逐帧 JSON，并优先处理最新帧')
    parser.add_argument('--drop-stale-frames', action='store_true',
                        help='处理跟不上输入时丢弃排队旧帧，降低直播延迟')
    parser.add_argument('--no-dual-view', action='store_true', help='关闭原始/处理结果双窗口显示')
    parser.add_argument(
        '--hdmi-output',
        action='store_true',
        help='将处理后单路画面无边框全屏输出；自动关闭 dual-view',
    )
    parser.add_argument(
        '--display-origin',
        type=int,
        nargs=2,
        metavar=('X', 'Y'),
        default=None,
        help='扩展桌面模式下目标显示器左上角坐标；镜像模式使用默认 0 0',
    )
    parser.add_argument('--no-metrics-text', action='store_true', help='关闭逐帧详细指标文字绘制')
    parser.add_argument('--no-frame-results', action='store_true', help='关闭逐帧 JSON 与诊断 JSON 写入')
    parser.add_argument('--no-save-video', action='store_true', help='禁用处理后视频录制')
    parser.add_argument(
        '--headless', '--no-render',
        dest='headless',
        action='store_true',
        help='纯计算/纯JSON离线极速模式：关闭录制、窗口显示、视觉叠画与默认实时HTML/切片，零拷贝释放源帧槽',
    )
    parser.add_argument(
        '--pose-model',
        default=None,
        help='覆写人体姿态模型路径（如 yolo26m-pose.mlpackage / yolo26n-pose.mlpackage）',
    )
    parser.add_argument(
        '--racket-model-profile',
        choices=('configured', '640_shadow'),
        default=None,
        help='选择预设 7 点球拍模型配置（640_shadow 仅限 shadow 模式）',
    )
    parser.add_argument('--output-fps', type=float, default=None, help='覆盖输出视频帧率')
    parser.add_argument(
        '--inference-workers',
        type=int,
        default=None,
        help='推理线程池 worker 数；当前执行图每帧最多并行 2 个任务',
    )
    parser.add_argument(
        '--det-workers',
        type=int,
        default=None,
        help='独立 Core ML 检测子进程数（offline+legacy+dual_view 饱和 ANE；0=关闭，默认离线headless自动选择）',
    )
    parser.add_argument(
        '--shm-slots',
        type=int,
        default=None,
        help='覆盖共享内存帧缓冲槽数（offline headless 默认自动压缩至最低安全槽数）',
    )
    parser.add_argument('--analyze-swings', action='store_true',
                        help='逐帧 JSON 写完后自动生成 Swing 事件 JSON/CSV；不可与 --live-mode 或 --no-frame-results 同用')
    parser.add_argument('--swing-output-json',
                        help='自动 Swing 分析的事件 JSON 输出路径；默认与逐帧 JSON 同目录')
    parser.add_argument('--swing-events-csv',
                        help='自动 Swing 分析的事件摘要 CSV 输出路径')
    parser.add_argument('--swing-frames-csv',
                        help='自动 Swing 分析的逐帧审计 CSV 输出路径')
    parser.add_argument('--dominant-hand', choices=['right', 'left'], default='right',
                        help='Swing 分析的持拍手，默认 right')
    parser.add_argument('--min-peak-energy', type=float, default=9.0,
                        help='事件峰值最小运动能量，默认 9.0')
    parser.add_argument('--active-energy', type=float, default=5.5,
                        help='进入挥拍事件的最小运动能量，默认 5.5')
    parser.add_argument('--min-event-frames', type=int, default=8,
                        help='有效挥拍事件的最小帧数，默认 8')
    parser.add_argument('--max-internal-gap', type=int, default=3,
                        help='同一挥拍内允许的最大非活跃间隔帧数，默认 3')
    parser.add_argument('--min-event-gap', type=int, default=18,
                        help='间隔启发式参数：源PTS按25Hz参考换算；无源时钟按帧顺序，默认18')
    parser.add_argument('--refractory-frames', type=int, default=20,
                        help='不应期参数：源PTS按25Hz参考换算，默认20=0.8秒；无源时钟按帧启发式')
    parser.add_argument('--min-wrist-sweep', type=float, default=120.0,
                        help='有效挥拍手腕最小空间轨迹跨度（像素），默认 120.0')
    parser.add_argument('--min-arm-extension-range', type=float, default=65.0,
                        help='有效挥拍手腕相对于肩部的最小伸缩距离跨度（像素），默认 65.0')
    parser.add_argument('--realtime-swing-events', action='store_true',
                        help='对直播码流或按时间线播放的视频滚动识别完整挥拍，并更新事件 JSON、HTML 与异步事件片段')
    parser.add_argument('--realtime-swing-json',
                        help='实时 Swing 事件 JSON；默认 <output_stem>_swing_events.json')
    parser.add_argument('--realtime-swing-html',
                        help='实时 Swing HTML 页面；默认 <output_stem>_swing_report.html')
    parser.add_argument('--realtime-swing-clips-dir',
                        help='实时 Swing 独立片段目录；默认 <output_stem>_swing_clips')
    parser.add_argument(
        '--realtime-swing-event-log',
        help='追加式实时事件JSONL；默认与实时事件JSON同名并使用 .jsonl 后缀',
    )
    parser.add_argument('--realtime-analysis-interval', type=int, default=None,
                        help='每隔多少个已处理帧运行一次滚动事件分析，默认 5')
    parser.add_argument('--realtime-settle-frames', type=int, default=None,
                        help='等待参数：源PTS按25Hz参考换算，默认0.6秒；无源时钟按帧启发式')
    parser.add_argument('--realtime-window-frames', type=int, default=None,
                        help='滚动FrameRecord数量上限，默认200；源PTS上下文最多8秒，容量不是时间精度')
    parser.add_argument('--realtime-clip-workers', type=int, default=None,
                        help='异步 Swing 片段编码线程数，默认 1')
    parser.add_argument('--realtime-frame-output', action='store_true',
                        help='异步保存已完成推理的逐帧 JSONL 与最近帧 JSON 快照')
    parser.add_argument('--realtime-frame-jsonl',
                        help='实时逐帧 JSONL；默认 <output_stem>_frames.jsonl')
    parser.add_argument('--realtime-frame-snapshot-json',
                        help='最近帧原子快照 JSON；默认 <output_stem>_frames_latest.json')
    parser.add_argument('--realtime-frame-snapshot-size', type=int, default=None,
                        help='最近帧 JSON 保留的记录数，默认 200')
    parser.add_argument('--realtime-frame-flush-interval', type=int, default=None,
                        help='每多少个已处理帧刷新 JSONL 与最近帧快照，默认 5')
    parser.add_argument('--realtime-coach', action='store_true',
                        help='为每个确认挥拍生成1-3条不超过15字的本地实时动作纠错')
    parser.add_argument('--realtime-coach-max-chars', type=int, default=None,
                        help='本地实时指导最大字数，范围 1-15，默认 15')
    parser.add_argument('--realtime-coach-max-suggestions', type=int, choices=[1, 2, 3], default=None,
                        help='每次挥拍输出建议数上限，范围 1-3，默认 3')
    parser.add_argument('--realtime-coach-min-confidence', type=float, default=None,
                        help='采用生物力学指标的最低置信度，范围 0-1，默认 0.45')
    parser.add_argument('--deepseek-coach', action='store_true',
                        help='异步调用 DeepSeek V4 Flash 生成旁路指导；本地建议不等待')
    parser.add_argument('--deepseek-model',
                        help='DeepSeek 模型名，默认 deepseek-v4-flash')
    parser.add_argument('--deepseek-base-url',
                        help='OpenAI兼容地址，默认 https://api.deepseek.com')
    parser.add_argument('--deepseek-api-key-env',
                        help='保存API密钥的环境变量名，默认 DEEPSEEK_API_KEY')
    parser.add_argument('--deepseek-timeout-seconds', type=float, default=None,
                        help='单次DeepSeek请求超时秒数，默认 3.0')
    parser.add_argument('--deepseek-workers', type=int, default=None,
                        help='DeepSeek旁路并发请求数，默认 2')
    parser.add_argument('--deepseek-coach-max-chars', type=int, default=None,
                        help='DeepSeek建议最大字数，范围 1-15，默认 15')
    parser.add_argument('--realtime-coach-tts', action='store_true',
                        help='用本机 Qwen3-TTS（MLX）异步播报本地 Coach 建议')
    parser.add_argument('--realtime-coach-tts-model',
                        help='MLX Qwen3-TTS 模型，默认 mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16')
    parser.add_argument('--realtime-coach-tts-voice',
                        help='Qwen3-TTS 声音，默认 Vivian（中文）')
    parser.add_argument('--realtime-coach-tts-no-playback', action='store_true',
                        help='只生成报告内可播放 WAV，不通过本机扬声器播报')
    parser.add_argument('--realtime-coach-tts-verbose', action='store_true',
                        help='禁用极简教练口令，使用完整长句播报（如“第X次Forehand：...”）')
    parser.add_argument('--realtime-coach-tts-no-warmup', action='store_true',
                        help='禁用 Qwen3-TTS 启动时后台静默预热')
    parser.add_argument('--realtime-open-report', action='store_true',
                        help='启动实时 Swing 输出时在系统浏览器打开 HTML 页面')
    parser.add_argument(
        '--algo2-dual-view', '--dual-view',
        dest='algo2_dual_view',
        action='store_true',
        help='启用算法2.0虚拟双机位实时解耦、自愈姿态推理、3梯队生物力学与Side-by-Side渲染',
    )
    parser.add_argument(
        '--algo2-config',
        default='configs/dual_view_config.yaml',
        help='算法2.0双机位配置文件路径，默认 configs/dual_view_config.yaml',
    )
    parser.add_argument(
        '--stream-id', '--stream-profile',
        dest='stream_id',
        default=None,
        help='指定码流ID（如 court01-main, camera04-main），用于本地视频文件或未匹配视频映射到指定的ROI与镜面配置',
    )
    return parser


def main_cli(argv=None, pipeline_cls=None):
    args = build_argument_parser().parse_args(argv)

    # 如果没有指定输入且配置文件里也没有，给个默认值
    if not args.input:
        # 尝试从配置加载看有没有
        with open(args.config, 'r') as f:
            tmp_cfg = yaml.safe_load(f)
            if 'video_input_path' not in tmp_cfg:
                args.input = "data/16.10.mp4"

    if pipeline_cls is None:
        from main_pipe import MultiprocessPipeline as pipeline_cls
    pipeline_cls(
        args.config,
        input_path=args.input,
        output_path=args.output,
        max_frames=args.max_frames,
        live_mode=args.live_mode,
        drop_stale_frames=args.drop_stale_frames,
        no_dual_view=args.no_dual_view,
        hdmi_output=args.hdmi_output,
        display_origin=args.display_origin,
        no_metrics_text=args.no_metrics_text,
        no_frame_results=args.no_frame_results,
        no_save_video=args.no_save_video,
        headless=args.headless,
        pose_model=args.pose_model,
        racket_model_profile=args.racket_model_profile,
        output_fps=args.output_fps,
        inference_workers=args.inference_workers,
        det_workers=args.det_workers,
        shm_slots=args.shm_slots,
        analyze_swings=args.analyze_swings,
        swing_output_json=args.swing_output_json,
        swing_events_csv=args.swing_events_csv,
        swing_frames_csv=args.swing_frames_csv,
        dominant_hand=args.dominant_hand,
        min_peak_energy=args.min_peak_energy,
        active_energy=args.active_energy,
        min_event_frames=args.min_event_frames,
        max_internal_gap=args.max_internal_gap,
        min_event_gap=args.min_event_gap,
        refractory_frames=args.refractory_frames,
        min_wrist_sweep=args.min_wrist_sweep,
        min_arm_extension_range=args.min_arm_extension_range,
        realtime_swing_events=args.realtime_swing_events,
        realtime_swing_json=args.realtime_swing_json,
        realtime_swing_html=args.realtime_swing_html,
        realtime_swing_clips_dir=args.realtime_swing_clips_dir,
        realtime_swing_event_log=args.realtime_swing_event_log,
        realtime_analysis_interval=args.realtime_analysis_interval,
        realtime_settle_frames=args.realtime_settle_frames,
        realtime_window_frames=args.realtime_window_frames,
        realtime_clip_workers=args.realtime_clip_workers,
        realtime_frame_output=args.realtime_frame_output,
        realtime_frame_jsonl=args.realtime_frame_jsonl,
        realtime_frame_snapshot_json=args.realtime_frame_snapshot_json,
        realtime_frame_snapshot_size=args.realtime_frame_snapshot_size,
        realtime_frame_flush_interval=args.realtime_frame_flush_interval,
        realtime_coach=args.realtime_coach,
        realtime_coach_max_chars=args.realtime_coach_max_chars,
        realtime_coach_max_suggestions=args.realtime_coach_max_suggestions,
        realtime_coach_min_confidence=args.realtime_coach_min_confidence,
        deepseek_coach_options={
            'enabled': args.deepseek_coach,
            'model': args.deepseek_model,
            'base_url': args.deepseek_base_url,
            'api_key_env': args.deepseek_api_key_env,
            'timeout_seconds': args.deepseek_timeout_seconds,
            'workers': args.deepseek_workers,
            'max_chars': args.deepseek_coach_max_chars,
        },
        coach_tts_options={
            'enabled': args.realtime_coach_tts,
            'model': args.realtime_coach_tts_model,
            'voice': args.realtime_coach_tts_voice,
            'playback': False if args.realtime_coach_tts_no_playback else None,
            'concise_speech': False if args.realtime_coach_tts_verbose else None,
            'warmup': False if args.realtime_coach_tts_no_warmup else None,
        },
        realtime_open_report=args.realtime_open_report,
        session_id=args.session_id,
        session_output_root=args.session_output_root,
        evidence_manifest=args.evidence_manifest,
        algo2_dual_view=args.algo2_dual_view,
        algo2_config=args.algo2_config,
        stream_id=args.stream_id,
        racket_pose_mode=args.racket_pose_mode,
        processing_mode=args.processing_mode,
        inference_schedule=args.inference_schedule,
        racket_ipc=args.racket_ipc,
    ).run()


