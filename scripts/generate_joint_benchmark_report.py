"""Generate comprehensive independent joint benchmark report from human annotations.

Evaluates predictions against blind human truth under both 5px (strict) and 8px (nominal) tolerances,
producing stratified error metrics across motion regimes (setup, acceleration, follow-through)
and camera views (front, mirror).
"""
from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from joint_annotation_evaluation import evaluate_joint_labels


def run_benchmark_analysis(labels_path: Path, predictions_path: Path, output_dir: Path) -> dict:
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    predictions = json.loads(predictions_path.read_text(encoding="utf-8"))

    eval_5px = evaluate_joint_labels(labels, predictions, tolerance_px=5.0)
    eval_8px = evaluate_joint_labels(labels, predictions, tolerance_px=8.0)

    # Motion regime mapping
    regime_map = {
        10: "clear_setup",
        18: "forward_acceleration",
        24: "follow_through_overlap",
        96: "clear_setup",
        108: "forward_acceleration",
        114: "follow_through_overlap",
        186: "clear_setup",
        189: "forward_acceleration",
        194: "follow_through_overlap",
    }

    # Aggregate sample-level errors
    samples_5px = []
    for g in eval_5px.get("groups", []):
        if g.get("joint") == "all":
            continue
        for s in g.get("samples", []):
            samples_5px.append({
                "view": g["view"],
                "joint": g["joint"],
                "frame_id": s["frame_id"],
                "regime": regime_map.get(s["frame_id"], "unknown"),
                "identifiable": s["identifiable"],
                "qualified": s["model_output_qualified"],
                "error_px": s["error_px"],
            })

    # Stratified metrics
    def summarize_errors(items):
        errs = [x["error_px"] for x in items if x["error_px"] is not None]
        if not errs:
            return {
                "count": 0,
                "paired_count": 0,
                "mean_px": None,
                "median_px": None,
                "max_px": None,
                "pass_rate_5px": None,
                "pass_rate_8px": None,
            }
        errs_sorted = sorted(errs)
        med = errs_sorted[len(errs_sorted) // 2]
        return {
            "count": len(items),
            "paired_count": len(errs),
            "mean_px": round(sum(errs) / len(errs), 2),
            "median_px": round(med, 2),
            "max_px": round(max(errs), 2),
            "pass_rate_5px": round(sum(e <= 5.0 for e in errs) / len(errs) * 100, 1),
            "pass_rate_8px": round(sum(e <= 8.0 for e in errs) / len(errs) * 100, 1),
        }

    overall = summarize_errors(samples_5px)
    by_view = {
        v: summarize_errors([x for x in samples_5px if x["view"] == v])
        for v in ("front", "back")
    }
    by_regime = {
        r: summarize_errors([x for x in samples_5px if x["regime"] == r])
        for r in ("clear_setup", "forward_acceleration", "follow_through_overlap")
    }
    by_joint = {
        j: summarize_errors([x for x in samples_5px if x["joint"] == j])
        for j in ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
    }

    report_data = {
        "schema": "tennis.joint-benchmark-consolidated.v1",
        "status": eval_5px.get("status"),
        "annotator_id": labels.get("annotator_id"),
        "tolerance_evaluations": {
            "5px": eval_5px,
            "8px": eval_8px,
        },
        "stratified_metrics": {
            "overall": overall,
            "by_view": by_view,
            "by_regime": by_regime,
            "by_joint": by_joint,
        },
        "samples": samples_5px,
    }

    # Save JSON
    (output_dir / "joint_benchmark_evaluation.json").write_text(
        json.dumps(report_data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8"
    )

    # Generate HTML
    html_content = build_html_report(report_data, labels)
    (output_dir / "joint_benchmark_report.html").write_text(html_content, encoding="utf-8")

    return report_data


def build_html_report(data: dict, labels: dict) -> str:
    m = data["stratified_metrics"]
    ov = m["overall"]

    regime_names = {
        "clear_setup": "静态准备姿态 (Setup)",
        "forward_acceleration": "前挥加速动态 (Acceleration)",
        "follow_through_overlap": "随挥重叠遮挡 (Follow-through)",
    }
    joint_cn = {
        "left_shoulder": "左肩 (Left Shoulder)",
        "right_shoulder": "右肩 (Right Shoulder)",
        "left_hip": "左髋 (Left Hip)",
        "right_hip": "右髋 (Right Hip)",
    }

    regime_rows = "".join(
        f"""<tr>
            <td><strong>{regime_names.get(r, r)}</strong></td>
            <td>{stats['paired_count']}</td>
            <td>{stats['mean_px']} px</td>
            <td>{stats['median_px']} px</td>
            <td>{stats['max_px']} px</td>
            <td><strong style="color: {'#10b981' if (stats['pass_rate_5px'] or 0)>=80 else '#f59e0b'}">{stats['pass_rate_5px']}%</strong></td>
            <td><strong style="color: {'#10b981' if (stats['pass_rate_8px'] or 0)>=90 else '#f59e0b'}">{stats['pass_rate_8px']}%</strong></td>
        </tr>"""
        for r, stats in m["by_regime"].items()
    )

    joint_rows = "".join(
        f"""<tr>
            <td><strong>{joint_cn.get(j, j)}</strong></td>
            <td>{stats['paired_count']}</td>
            <td>{stats['mean_px']} px</td>
            <td>{stats['median_px']} px</td>
            <td>{stats['max_px']} px</td>
            <td><strong>{stats['pass_rate_5px']}%</strong></td>
            <td><strong>{stats['pass_rate_8px']}%</strong></td>
        </tr>"""
        for j, stats in m["by_joint"].items()
    )

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Stage 3 · 独立躯干关节点 9 帧基准评测报告</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background: #0d1117;
            color: #f0f6fc;
            margin: 0;
            padding: 28px;
            line-height: 1.6;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        header {{
            background: linear-gradient(135deg, #1f2937, #111827);
            border: 1px solid #30363d;
            border-radius: 12px;
            padding: 24px 32px;
            margin-bottom: 24px;
        }}
        h1 {{ margin: 0 0 10px 0; font-size: 26px; color: #fff; }}
        .header-sub {{ color: #8b949e; font-size: 15px; margin: 0; }}
        .stat-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 16px;
            margin-bottom: 28px;
        }}
        .stat-card {{
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 10px;
            padding: 16px 20px;
        }}
        .stat-val {{ font-size: 28px; font-weight: bold; color: #58a6ff; margin: 6px 0; }}
        .stat-label {{ color: #8b949e; font-size: 14px; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 16px 0 28px 0;
            background: #161b22;
            border-radius: 8px;
            overflow: hidden;
            border: 1px solid #30363d;
        }}
        th, td {{
            padding: 12px 16px;
            border-bottom: 1px solid #30363d;
            text-align: left;
        }}
        th {{ background: #21262d; color: #c9d1d9; font-size: 14px; }}
        section {{
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 12px;
            padding: 20px 24px;
            margin-bottom: 24px;
        }}
        h2 {{ font-size: 19px; color: #fff; margin-top: 0; }}
        a {{ color: #58a6ff; text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🎾 Stage 3 · 独立躯干关节点 9 帧基准评测报告</h1>
            <p class="header-sub">
                本报告对标 9 帧独立人工盲测真值（覆盖静态准备、前挥加速、随挥遮挡三大阶段）。
                标注者：<strong>{html.escape(str(data['annotator_id'] or '待确认'))}</strong> · 
                评测状态：<strong>{data['status']}</strong>
            </p>
        </header>

        <div class="stat-grid">
            <div class="stat-card">
                <div class="stat-label">有效配对评测点数</div>
                <div class="stat-val">{ov['paired_count']} / 72</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">平均绝对误差 (Mean Error)</div>
                <div class="stat-val">{ov['mean_px'] or 'N/A'} px</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">中位数误差 (Median Error)</div>
                <div class="stat-val">{ov['median_px'] or 'N/A'} px</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">5px 容差符合率 (严格)</div>
                <div class="stat-val">{ov['pass_rate_5px'] or 'N/A'}%</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">8px 容差符合率 (常规)</div>
                <div class="stat-val">{ov['pass_rate_8px'] or 'N/A'}%</div>
            </div>
        </div>

        <section>
            <h2>运动动力学阶段误差分层 (Motion Regimes)</h2>
            <table>
                <thead>
                    <tr>
                        <th>运动阶段</th>
                        <th>配对样本数</th>
                        <th>平均误差</th>
                        <th>中位数误差</th>
                        <th>最大误差</th>
                        <th>5px 达标率</th>
                        <th>8px 达标率</th>
                    </tr>
                </thead>
                <tbody>
                    {regime_rows}
                </tbody>
            </table>
        </section>

        <section>
            <h2>躯干关节点细分误差 (Per-Joint Breakdown)</h2>
            <table>
                <thead>
                    <tr>
                        <th>关节名称</th>
                        <th>配对样本数</th>
                        <th>平均误差</th>
                        <th>中位数误差</th>
                        <th>最大误差</th>
                        <th>5px 达标率</th>
                        <th>8px 达标率</th>
                    </tr>
                </thead>
                <tbody>
                    {joint_rows}
                </tbody>
            </table>
        </section>

        <section>
            <h2>双视角误差对比 (Front vs Mirror View)</h2>
            <table>
                <thead>
                    <tr>
                        <th>观测视角</th>
                        <th>配对样本数</th>
                        <th>平均误差</th>
                        <th>中位数误差</th>
                        <th>最大误差</th>
                        <th>5px 达标率</th>
                        <th>8px 达标率</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>正面人物 (Front View)</strong></td>
                        <td>{m['by_view']['front']['paired_count']}</td>
                        <td>{m['by_view']['front']['mean_px']} px</td>
                        <td>{m['by_view']['front']['median_px']} px</td>
                        <td>{m['by_view']['front']['max_px']} px</td>
                        <td><strong>{m['by_view']['front']['pass_rate_5px']}%</strong></td>
                        <td><strong>{m['by_view']['front']['pass_rate_8px']}%</strong></td>
                    </tr>
                    <tr>
                        <td><strong>镜面背影 (Mirror View)</strong></td>
                        <td>{m['by_view']['back']['paired_count']}</td>
                        <td>{m['by_view']['back']['mean_px']} px</td>
                        <td>{m['by_view']['back']['median_px']} px</td>
                        <td>{m['by_view']['back']['max_px']} px</td>
                        <td><strong>{m['by_view']['back']['pass_rate_5px']}%</strong></td>
                        <td><strong>{m['by_view']['back']['pass_rate_8px']}%</strong></td>
                    </tr>
                </tbody>
            </table>
        </section>

        <p>
            <a href="joint_benchmark_evaluation.json">下载评测数据 JSON</a> · 
            <a href="index.html">返回独立关节点标注控制台</a>
        </p>
    </div>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", required=True, help="Path to independent joint labels JSON")
    parser.add_argument("--predictions", required=True, help="Path to predictions audit JSON")
    parser.add_argument("--output", required=True, help="Path to output directory")
    args = parser.parse_args()

    labels_path = Path(args.labels).resolve()
    predictions_path = Path(args.predictions).resolve()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    res = run_benchmark_analysis(labels_path, predictions_path, output_dir)
    print(f"Benchmark report generated successfully at: {output_dir / 'joint_benchmark_report.html'}")
    print(f"Status: {res['status']}")


if __name__ == "__main__":
    main()
