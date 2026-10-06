    var activeVideoPlayers = typeof activeVideoPlayers !== "undefined" ? activeVideoPlayers : new Set();
    var lastEventsJson = typeof lastEventsJson !== "undefined" ? lastEventsJson : "";
    var filterOnlyValidSwings = typeof filterOnlyValidSwings !== "undefined" ? filterOnlyValidSwings : true;
    var latestRawEvents = typeof latestRawEvents !== "undefined" ? latestRawEvents : [];

    function isShadowEvent(ev) {
      if (!ev) return false;
      if (ev.is_shadow_swing === true) return true;
      if (ev.is_shadow_swing === false) return false;
      const ca = ev.evidence?.contact_analysis || ev.evidence?.classification_context?.contact_analysis;
      if (ca && typeof ca.is_shadow_swing === "boolean") return ca.is_shadow_swing;
      if (ca && ca.has_ball === false) return true;
      return false;
    }

    function extractSubScoresFromEvent(ev) {
      const sqs = ev?.extended_biomechanics?.swing_quality_score || ev?.swing_quality_score || {};
      if (sqs.sub_scores && Object.keys(sqs.sub_scores).length > 0) return sqs.sub_scores;
      const bio = ev?.biomechanics || {};
      const metrics = bio.metrics || {};
      const ext = ev?.extended_biomechanics || bio.extended_biomechanics || {};
      const getVal = (m, key) => {
        if (!m) return null;
        let v = key ? m[key] : (typeof m === 'object' && 'value' in m ? m.value : m);
        if (typeof v === 'object' && v !== null && 'value' in v) v = v.value;
        const n = parseFloat(v);
        return Number.isFinite(n) ? n : null;
      };
      const curveScore = (val, pts) => {
        if (val <= pts[0][0]) return pts[0][1];
        if (val >= pts[pts.length - 1][0]) return pts[pts.length - 1][1];
        for (let i = 0; i < pts.length - 1; i++) {
          const [x0, y0] = pts[i], [x1, y1] = pts[i + 1];
          if (val >= x0 && val <= x1) return y0 + (y1 - y0) * ((val - x0) / Math.max(1e-9, x1 - x0));
        }
        return pts[pts.length - 1][1];
      };
      const sub = {};
      const stc = getVal(metrics.shoulder_turn_change) ?? getVal(ext.shoulder_turn_change) ?? getVal(metrics.shoulder_turn);
      if (stc != null) sub.shoulder_turn = Math.round(curveScore(stc, [[0, 35], [12, 50], [30, 80], [45, 100]]));
      const tb = getVal(metrics.takeback_depth) ?? getVal(ext.takeback_depth) ?? getVal(metrics.scapular_retraction);
      if (tb != null) sub.takeback = Math.round(curveScore(tb, [[0.0, 35], [0.6, 50], [1.2, 75], [1.8, 90], [2.2, 100]]));
      const ae = getVal(metrics.arm_extension) ?? getVal(ext.arm_extension);
      if (ae != null) sub.arm_extension = Math.round(curveScore(ae, [[60, 35], [120, 62], [145, 80], [165, 100]]));
      const rkt = ext.racket_head_speed || ev.racket_speed || {};
      const spd = getVal(rkt, 'max_px_s') ?? getVal(rkt, 'contact_px_s') ?? getVal(metrics.racket_head_speed);
      if (spd != null) sub.racket_speed = Math.round(curveScore(spd, [[0, 30], [800, 50], [1600, 70], [2400, 85], [3200, 100]]));
      const pkf = getVal(metrics.preparation_knee_flexion) ?? getVal(ext.preparation_knee_flexion);
      const legRatio = getVal(ext.leg_drive, 'drive_ratio') ?? getVal(metrics.leg_drive);
      if (pkf != null) sub.leg_drive = Math.round(curveScore(pkf, [[0, 35], [12, 50], [25, 75], [45, 100]]));
      else if (legRatio != null) sub.leg_drive = Math.round(curveScore(legRatio, [[0.0, 35], [0.03, 55], [0.06, 75], [0.10, 100]]));
      return sub;
    }

    function generateRadarSvg(ev) {
      if (isShadowEvent(ev)) {
        return `<div class="radar-box"><span style="color:var(--muted);font-size:11px;text-align:center;">空挥试拍<br><small>无球免除</small></span></div>`;
      }
      const sub = extractSubScoresFromEvent(ev);
      const axes = [
        ["shoulder_turn", "转肩"],
        ["takeback", "引拍"],
        ["arm_extension", "延展"],
        ["racket_speed", "挥速"],
        ["leg_drive", "蹬地"]
      ];
      const hasAny = axes.some(([k]) => sub[k] != null);
      if (!hasAny) {
        return `<div class="radar-box"><span style="color:var(--muted);font-size:11px;text-align:center;">5维力学雷达<br><small>待动作识别</small></span></div>`;
      }
      const width = 140, height = 135;
      const cx = width / 2, cy = height / 2 + 2, rMax = 44;
      const n = axes.length;
      let ringsSvg = [0.33, 0.66, 1.0].map(lvl => {
        let pts = [];
        for (let i = 0; i < n; i++) {
          let ang = -Math.PI / 2 + i * (2 * Math.PI / n);
          pts.push(`${(cx + rMax * lvl * Math.cos(ang)).toFixed(1)},${(cy + rMax * lvl * Math.sin(ang)).toFixed(1)}`);
        }
        let dash = lvl < 1.0 ? ' stroke-dasharray="2,2"' : '';
        let col = lvl < 1.0 ? 'rgba(255,255,255,0.08)' : 'rgba(255,255,255,0.18)';
        return `<polygon points="${pts.join(' ')}" fill="none" stroke="${col}" stroke-width="1"${dash}/>`;
      }).join('');

      let dataPts = [], spokes = [], dots = [], labels = [];
      for (let i = 0; i < n; i++) {
        let [k, label] = axes[i];
        let ang = -Math.PI / 2 + i * (2 * Math.PI / n);
        let cosA = Math.cos(ang), sinA = Math.sin(ang);
        let ox = cx + rMax * cosA, oy = cy + rMax * sinA;
        spokes.push(`<line x1="${cx.toFixed(1)}" y1="${cy.toFixed(1)}" x2="${ox.toFixed(1)}" y2="${oy.toFixed(1)}" stroke="rgba(255,255,255,0.1)" stroke-width="1"/>`);
        let score = sub[k] != null ? Number(sub[k]) : 50;
        let clamped = Math.max(0, Math.min(100, score));
        let rVal = Math.max(5, rMax * (clamped / 100));
        let dx = cx + rVal * cosA, dy = cy + rVal * sinA;
        dataPts.push(`${dx.toFixed(1)},${dy.toFixed(1)}`);
        dots.push(`<circle cx="${dx.toFixed(1)}" cy="${dy.toFixed(1)}" r="2.5" fill="#00f0ff" stroke="#0a0f1a" stroke-width="1"/>`);
        let lx = cx + (rMax + 14) * cosA, ly = cy + (rMax + 14) * sinA;
        let anchor = Math.abs(cosA) < 0.15 ? "middle" : (cosA > 0 ? "start" : "end");
        let displayScore = sub[k] != null ? Math.round(score) : '—';
        labels.push(`<text x="${lx.toFixed(1)}" y="${ly.toFixed(1)}" text-anchor="${anchor}" dominant-baseline="central" font-size="8" font-weight="600" fill="#94a3b8">${escapeHtml(label)}${displayScore}</text>`);
      }
      const poly = `<polygon points="${dataPts.join(' ')}" fill="rgba(0, 240, 255, 0.22)" stroke="#00f0ff" stroke-width="1.8"/>`;
      return `
        <div class="radar-box">
          <div style="font-size:9.5px;font-weight:600;color:var(--muted);margin-bottom:1px;letter-spacing:0.3px;">5维生物力学雷达</div>
          <svg viewBox="0 0 ${width} ${height}" style="width:130px;height:125px;display:block;">
            ${ringsSvg}${spokes.join('')}${poly}${dots.join('')}${labels.join('')}
          </svg>
        </div>`;
    }

    function renderSessionSummary(summary, events) {
      const box = $("session-summary-box");
      if (!box) return;
      if (!summary || !summary.total_swings || summary.total_swings === 0) {
        box.hidden = true;
        box.innerHTML = "";
        return;
      }
      box.hidden = false;

      const total = summary.total_swings || 0;
      const valid = summary.valid_shots_count || 0;
      const dist = summary.distribution || {};
      const qm = summary.quality_metrics || {};
      const defs = summary.common_deficiencies || [];
      const trends = summary.score_trends || [];

      // Generate SVG trendline
      const validTrends = trends.filter(t => !t.is_shadow && t.score != null);
      let trendSvgHtml = "";
      if (validTrends.length >= 1) {
        const svgW = 420;
        const svgH = 110;
        const padX = 35;
        const padY = 20;
        const minS = 40;
        const maxS = 100;
        const rangeS = maxS - minS;

        const getY = (s) => svgH - padY - ((s - minS) / rangeS) * (svgH - 2 * padY);
        const getX = (idx, totalPts) => totalPts <= 1 ? svgW / 2 : padX + (idx / (totalPts - 1)) * (svgW - 2 * padX);

        const avgY = qm.average_score != null ? getY(qm.average_score) : null;
        const avgLine = avgY != null ? `<line x1="${padX}" y1="${avgY}" x2="${svgW - padX}" y2="${avgY}" stroke="rgba(0, 240, 255, 0.45)" stroke-dasharray="4,4" stroke-width="1.5" /><text x="${svgW - padX + 4}" y="${avgY + 3}" fill="#00f0ff" font-size="9">均分 ${qm.average_score}</text>` : "";

        let gridLines = [60, 80, 100].map(val => {
          const y = getY(val);
          return `<line x1="${padX}" y1="${y}" x2="${svgW - padX}" y2="${y}" stroke="rgba(255,255,255,0.06)" stroke-width="1" /><text x="${padX - 22}" y="${y + 3}" fill="rgba(255,255,255,0.3)" font-size="9">${val}</text>`;
        }).join("");

        const points = validTrends.map((t, idx) => ({
          x: getX(idx, validTrends.length),
          y: getY(t.score),
          score: t.score,
          eid: t.event_id,
          speed: t.speed_kmh,
          stroke: t.stroke_type,
        }));

        const polylineD = points.map(p => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(" ");
        const polylineArea = `${points[0].x.toFixed(1)},${svgH - padY} ` + polylineD + ` ${points[points.length - 1].x.toFixed(1)},${svgH - padY}`;

        const dots = points.map(p => `
          <g class="trend-point" data-eid="${p.eid}">
            <circle cx="${p.x.toFixed(1)}" cy="${p.y.toFixed(1)}" r="4" fill="#00f0ff" stroke="#0a101a" stroke-width="2" />
            <title>Event #${p.eid} · ${p.stroke}: ${p.score}分 (${p.speed || '—'} km/h)</title>
          </g>
        `).join("");

        trendSvgHtml = `
          <svg viewBox="0 0 ${svgW} ${svgH}" class="score-trend-svg" preserveAspectRatio="none">
            <defs>
              <linearGradient id="trendGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stop-color="#00f0ff" stop-opacity="0.25" />
                <stop offset="100%" stop-color="#00f0ff" stop-opacity="0.0" />
              </linearGradient>
            </defs>
            ${gridLines}
            ${avgLine}
            <polygon points="${polylineArea}" fill="url(#trendGrad)" />
            <polyline points="${polylineD}" fill="none" stroke="#00f0ff" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" />
            ${dots}
          </svg>
        `;
      } else {
        trendSvgHtml = `<div style="text-align:center; padding: 30px; color: var(--muted); font-size:12px;">暂无足够的有效击球评分数据</div>`;
      }

      const defPills = defs.length > 0 ? defs.map(d => {
        const cls = d.severity === "HIGH" ? "severity-high" : (d.severity === "MEDIUM" ? "severity-medium" : "severity-low");
        return `<span class="deficiency-pill ${cls}">⚠️ ${escapeHtml(d.message)} · <strong>${d.occurrence_rate_percent}%</strong> (${d.count}次)</span>`;
      }).join("") : `<span style="font-size:12px; color: var(--muted);">当前证据未产生技术纠错建议，仍需复核</span>`;

      box.innerHTML = `
        <div class="summary-top-row">
          <div class="summary-title-group">
            <h3>📊 训练会话全景宏观诊断 <span>(Session Coaching Summary)</span></h3>
            <p>事件窗口指标统计与教练建议；二维代理指标仅供回放参考</p>
          </div>
          <div class="summary-stats-pills">
            <span class="stat-pill highlight">可评分击球: ${valid}次</span>
            <span class="stat-pill" style="opacity: 0.85;">空挥试拍: ${dist.shadow_count || 0}次</span>
            <span class="stat-pill highlight">均分: ${qm.average_score != null ? qm.average_score + "分" : "—"}</span>
            <span class="stat-pill success">${escapeHtml(qm.stability_label || "—")}</span>
          </div>
        </div>

        <div class="dist-bar-wrapper">
          <div class="dist-bar-labels">
            <span>正手: ${dist.forehand_count || 0}球 (${dist.forehand_ratio || 0}%)</span>
            <span>反手: ${dist.backhand_count || 0}球 (${dist.backhand_ratio || 0}%)</span>
            <span>空挥试拍: ${dist.shadow_count || 0}次 (${dist.shadow_ratio || 0}%)</span>
          </div>
          <div class="dist-stacked-bar">
            <div class="dist-segment forehand" style="width: ${dist.forehand_ratio || 0}%;" title="正手 ${dist.forehand_ratio || 0}%"></div>
            <div class="dist-segment backhand" style="width: ${dist.backhand_ratio || 0}%;" title="反手 ${dist.backhand_ratio || 0}%"></div>
            <div class="dist-segment shadow" style="width: ${dist.shadow_ratio || 0}%;" title="空挥 ${dist.shadow_ratio || 0}%"></div>
          </div>
        </div>

        <div class="summary-body-grid">
          <div class="trend-chart-box">
            <div class="chart-header">
              <strong>📈 击球质量得分稳定性趋势 (Score Stability)</strong>
              <span>Std: ±${qm.score_std != null ? qm.score_std : '—'}分</span>
            </div>
            ${trendSvgHtml}
          </div>

          <div class="macro-diagnosis-box">
            <div class="chart-header">
              <strong>🎯 学员共性技术短板归纳 (Common Weaknesses)</strong>
            </div>
            <div class="macro-deficiencies-list">
              ${defPills}
            </div>
            <div class="macro-narrative-text">
              ${escapeHtml(summary.macro_diagnosis || "")}
            </div>
          </div>
        </div>
        ${renderMeasurementSummary(summary.analysis_metrics || [])}
      `;
    }

    function handleSwingsData(data, force = false) {
      if (!data) return;
      const events = data.events || [];
      const summary = data.summary || {};
      latestRawEvents = events;
      const currentHash = JSON.stringify([data.session_id, events, summary, filterOnlyValidSwings]);
      if (!force && currentHash === lastEventsJson) return;
      lastEventsJson = currentHash;
      renderSessionSummary(summary, events);
      renderSwingsFeed(events);
    }

    async function pollSwings(force = false) {
      try {
        const response = await api("/api/session/events");
        if (!response.ok) return;
        const data = await response.json();
        handleSwingsData(data, force);
      } catch (err) {}
    }

    function kinematicDisplay(seq, metric = {}) {
      const confidence = metric.confidence ?? seq.confidence ?? 0;
      const qualified = (metric.coach_eligible ?? seq.coach_eligible) === true && confidence > 0;
      const cross = seq.cross_validation || {};
      const unresolved = seq.sequence_quality === "UNRESOLVED_AT_FRAME_RATE" || (typeof seq.latency_hip_to_shoulder_ms === "number" && typeof seq.peak_time_uncertainty_ms === "number" && Math.abs(seq.latency_hip_to_shoulder_ms) <= seq.peak_time_uncertainty_ms);
      const labels = {
        agree: "双视角一致 · 未验证",
        disagree: "双视角冲突 · 需复核",
        single_view: "单视角参考 · 未验证",
        unavailable: "证据不足",
        legacy_single_view: "历史单视角 · 未验证",
      };
      let label = labels[cross.status] || "未验证 · 需复核";
      let badgeClass = "unvalidated";
      if (cross.status === "unavailable" && cross.reason === "cadence_sensitive_peak") {
        label = "短时间间隔敏感 · 暂停判定";
        badgeClass = "cadence_sensitive";
      } else if (unresolved) {
        label = "先后难以分辨" + (cross.status === "single_view" ? " · 单视角" : "");
      } else if (qualified && ["OPTIMAL", "ACCEPTABLE", "DISCONNECTED"].includes(seq.sequence_quality)) {
        label = {OPTIMAL: "时序符合已校准规则", ACCEPTABLE: "时序基本符合规则", DISCONNECTED: "时序需复核"}[seq.sequence_quality];
        badgeClass = seq.sequence_quality === "DISCONNECTED" ? "disconnected" : "optimal";
      }
      const hasCandidateRacket = seq.racket_candidate_peak_frame != null || seq.candidate_latency_shoulder_to_racket_ms != null;
      if (seq.racket_peak_frame === null && (seq.latency_hip_to_shoulder_ms != null || seq.candidate_latency_hip_to_shoulder_ms != null)) {
        label += hasCandidateRacket ? " · 含候选拍峰" : " · 缺拍峰";
      }
      const intervals = [];
      const hipVal = seq.latency_hip_to_shoulder_ms;
      const candHipVal = seq.candidate_latency_hip_to_shoulder_ms;
      if (typeof hipVal === "number" && Number.isFinite(hipVal)) {
        intervals.push(`髋—肩 ${hipVal > 0 ? "+" : ""}${hipVal.toFixed(1)} ms`);
      } else if (typeof candHipVal === "number" && Number.isFinite(candHipVal)) {
        intervals.push(`髋—肩(候选F${seq.candidate_hip_peak_frame ?? "—"}) ${candHipVal > 0 ? "+" : ""}${candHipVal.toFixed(1)} ms`);
      }
      const rktVal = seq.latency_shoulder_to_racket_ms;
      const candRktVal = seq.candidate_latency_shoulder_to_racket_ms;
      if (typeof rktVal === "number" && Number.isFinite(rktVal)) {
        intervals.push(`肩—拍 ${rktVal > 0 ? "+" : ""}${rktVal.toFixed(1)} ms`);
      } else if (typeof candRktVal === "number" && Number.isFinite(candRktVal)) {
        const candSpeed = typeof seq.racket_candidate_peak_speed === "number" ? ` @ ${seq.racket_candidate_peak_speed.toFixed(0)}px/s` : "";
        intervals.push(`肩—拍(候选F${seq.racket_candidate_peak_frame ?? "—"}) ${candRktVal > 0 ? "+" : ""}${candRktVal.toFixed(1)} ms${candSpeed}`);
      }
      let explanation = "二维峰值参考；未验证结果不用于技术纠错。";
      if (unresolved) explanation = "峰值落在同帧或定位范围重叠，当前采样无法分辨真实先后；0 ms 不代表动作错误或真实同步。";
      if (typeof seq.sampling_interval_ms === "number") explanation += ` 采样间隔约 ${seq.sampling_interval_ms.toFixed(1)} ms。`;
      if (typeof seq.peak_time_uncertainty_ms === "number") explanation += ` 峰值定位范围约 ${seq.peak_time_uncertainty_ms.toFixed(1)} ms（非统计置信区间）。`;
      const peakReasons = {usable:"峰值可用", boundary_peak:"峰值在窗口边界", ambiguous_peak:"峰值过宽或多峰", discontinuous_evidence:"有效片段不连续", low_coverage:"有效覆盖不足", insufficient_samples:"有效样本不足", insufficient_motion:"未形成明确运动峰值", cadence_sensitive_peak:"峰值对短时间间隔敏感"};
      for (const [view, title] of [["front","正面"],["back","背面"]]) {
        const segments = seq.views?.[view]?.segments;
        if (segments) explanation += ` ${title}：髋 ${peakReasons[segments.hip?.status] || "证据不足"}；肩 ${peakReasons[segments.shoulder?.status] || "证据不足"}。`;
      }
      if (seq.racket_evidence) explanation += ` 球拍：${peakReasons[seq.racket_evidence.status] || "证据不足"}。`;
      if (seq.racket_candidate_peak_frame != null && seq.racket_peak_frame == null) {
        const candSpeedStr = typeof seq.racket_candidate_peak_speed === 'number' ? `${seq.racket_candidate_peak_speed.toFixed(1)} px/s` : '—';
        const candLatStr = typeof seq.candidate_latency_shoulder_to_racket_ms === 'number' ? `${seq.candidate_latency_shoulder_to_racket_ms > 0 ? '+' : ''}${seq.candidate_latency_shoulder_to_racket_ms.toFixed(1)} ms` : '—';
        explanation += ` 球拍候选峰值：第 ${seq.racket_candidate_peak_frame} 帧（速度 ${candSpeedStr}，候选肩—拍时差 ${candLatStr}，诊断性参考）。`;
      }
      for (const [key, title] of [["hip_to_shoulder","髋—肩"],["shoulder_to_racket","肩—拍"],["candidate_shoulder_to_racket","肩—拍(候选)"]]) {
        const range = seq.pair_timing?.[key]?.latency_range_ms;
        if (Array.isArray(range) && range.length === 2 && range.every(Number.isFinite)) explanation += ` ${title}时差范围 ${range[0]}～${range[1]} ms（非统计置信区间）。`;
      }
      return {label, badgeClass, intervals: intervals.join(" · ") || "暂无可用峰值间隔", explanation};
    }

    function observationReason(metric) {
      const reasons = metric.measurement_evidence?.reasons || [];
      if (reasons.includes("below_motion_resolution_guard")) return "无显著下蹲蓄力 (上移极微)";
      if (reasons.includes("contact_anchor_unconfirmed")) return "触球候选待确认";
      if (reasons.includes("path_endpoints_coincide")) {
        const drop = metric?.drop_depth_ratio ?? metric?.measurement_evidence?.fields?.drop_depth_ratio;
        return (drop === 0 || drop === "0") ? "平击推进 (无下沉提拉)" : "轨迹方向不明确";
      }
      if (reasons.includes("unstable_ankle_line")) return "动态步法跨步中";
      return "观测证据不足";
    }

    function qualifiedObservation(metric, field) {
      const evidence = metric.measurement_evidence || {};
      const qualification = evidence.fields?.[field] || evidence;
      return qualification.display_eligible === true ? metric[field] : null;
    }

    function measurementValue(value, unit = "") {
      const units = {image_plane_deg:"°（像面）", deg_360:"°", deg_2d:"°（二维）", deg:"°", ratio:"x", body_width:"投影躯干宽度"};
      const labels = {UNRESOLVED_AT_FRAME_RATE:"当前帧率下先后难辨",DISCONNECTED:"投影峰值顺序反向（待复核）",OPTIMAL:"投影峰值同序（未验证）"};
      const text = typeof value === "number" && Number.isFinite(value) ? value.toFixed(2).replace(/\.?0+$/, "") : (labels[value] ?? String(value ?? "—"));
      return escapeHtml(`${text} ${units[unit] ?? unit}`.trim());
    }

    function renderGroundReference(event) {
      const reference = event.ground_reference;
      if (!reference || typeof reference !== 'object') return '';
      const text = ['脚踝地面投影参考 · 着地未确认 · 不参与评分'];
      if (reference.application?.scope === 'camera_profile') text.push('机位共享标定：' + (reference.application.camera_binding?.stream_id || ''));
      if (reference.dimensions_measured !== true) text.push('尺寸待实测');
      const reasonLabels = {camera_geometry_not_confirmed:'当前画面四角尚未核对',source_binding_mismatch:'输入来源不匹配',source_image_size_mismatch:'原图尺寸不匹配',observation_coordinate_space_unverified:'观测坐标系未确认',mixed_calibration_versions:'窗口含不同标定版本',duplicate_source_frames:'窗口含重复源帧'};
      for (const key of Object.keys(reference.rejected_reasons || {})) if (reasonLabels[key]) text.push(reasonLabels[key]);
      for (const [key, label] of [['left_ankle','左脚踝'],['right_ankle','右脚踝']]) {
        const row = reference.feet?.[key] || {}, value = row.median_projection_difference_m;
        text.push(typeof value === 'number' && Number.isFinite(value) ? `${label}双视角投影差 ${value.toFixed(3)} m（${row.paired_frames}对）` : `${label}缺少合格双视角配对`);
        if (row.reasons?.corner_correspondence_not_confirmed) text.push('镜中四角实体对应尚未核对');
      }
      return `<p class="ground-reference" style="color:var(--muted);font-size:13px">${escapeHtml(text.join('；'))}</p>`;
    }

    function renderEventMeasurements(event) {
      const rows = event.analysis_metrics || [];
      const ground = renderGroundReference(event);
      if (!rows.length) return ground;
      return ground + `<details class="event-measurements" style="margin:12px 0"><summary>全部分析指标 · 事件窗口统计 (${rows.length}项)</summary>
        <p style="font-size:12px;color:var(--muted)">来自当前事件 JSON；视频底部为逐帧数值，统计窗口不同。代理指标不直接代表真实力学或技术评分。</p>
        <div class="swing-telemetry-grid">${rows.map(m => `<div class="telemetry-cell">
          <span class="t-label">${escapeHtml(m.label || m.key)}</span><span class="t-val">${measurementValue(m.value, m.unit)}</span>
          <small style="color:var(--muted)" title="${escapeHtml((m.source_frames || []).join(', '))}">${escapeHtml(m.method_label || "观测参考")} · ${m.display_eligible === true ? "观测资格通过 · 准确性未验证" : m.display_eligible === false ? "资格不通过" : "观测未验证"} · ${m.coach_eligible ? "需复核" : "不用于纠错"}${m.sample_count != null ? ` · ${m.sample_count}个样本` : ""}</small>
        </div>`).join("")}</div></details>`;
    }

    function renderMeasurementSummary(rows) {
      if (!rows.length) return "";
      const metricKinds = new Set(rows.map(m => m.key)).size;
      return `<details class="session-measurements" open style="margin-top:16px"><summary>观测统计 · ${rows.length}行 / ${metricKinds}种指标</summary>
        <p style="font-size:12px;color:var(--muted)">排除空挥，每行按动作类型、单位和观测方法分组，同一指标可能有多行。显示中位数与范围；缺失值不补零。指标种类数不代表独立验证或评分维度数量。</p>
        <div style="overflow:auto"><table style="width:100%;font-size:12px;text-align:left;border-spacing:10px">
          <thead><tr><th>动作</th><th>指标</th><th>中位数 / 分类计数</th><th>范围</th><th>有数据事件</th></tr></thead><tbody>
          ${rows.map(m => `<tr><td>${escapeHtml(m.stroke_type)}</td><td title="${escapeHtml(m.observability || '')}">${escapeHtml(m.label)}<small style="display:block;color:var(--muted)">${escapeHtml(m.method_label || "观测参考")}</small></td>
          <td>${m.median != null ? measurementValue(m.median,m.unit) : escapeHtml(Object.entries(m.categories || {}).map(([k,v])=>`${k}: ${v}次`).join(" · "))}</td>
          <td>${m.min != null ? `${measurementValue(m.min,m.unit)}–${measurementValue(m.max,m.unit)}` : "—"}</td><td>${m.count}/${m.total_events}</td></tr>`).join("")}
        </tbody></table></div></details>`;
    }

    function renderSwingsFeed(events) {
      const emptyEl = $("swings-empty");
      const listEl = $("swings-feed-list");
      const badgeEl = $("swings-count-badge");
      if (!listEl) return;

      const allEvents = events || [];
      const validEvents = allEvents.filter(ev => !isShadowEvent(ev));
      const shadowCount = allEvents.length - validEvents.length;
      const displayEvents = filterOnlyValidSwings ? validEvents : allEvents;

      if (badgeEl) {
        if (filterOnlyValidSwings) {
          badgeEl.textContent = `${validEvents.length} 次挥拍候选`;
          if (shadowCount > 0) {
            badgeEl.title = `共检测到 ${allEvents.length} 次挥拍，已自动过滤 ${shadowCount} 次空挥试拍`;
          } else {
            badgeEl.removeAttribute("title");
          }
        } else {
          badgeEl.textContent = `${allEvents.length} 次挥拍 (含空挥)`;
          badgeEl.removeAttribute("title");
        }
      }

      if (displayEvents.length === 0) {
        if (emptyEl) {
          emptyEl.hidden = false;
          if (allEvents.length > 0 && filterOnlyValidSwings) {
            emptyEl.innerHTML = `
              <strong>暂无有效实战击球</strong>
              <span>当前会话已自动过滤 ${shadowCount} 次空挥试拍。点击右上角“全部记录”可查看试拍动作，或在击球后查看针对性教练诊断。</span>
            `;
          } else {
            emptyEl.innerHTML = `
              <strong>暂无挥拍击球事件</strong>
              <span>启动流水线分析后，检测到的击球动作、高级遥测卡片与慢动作视频切片将在此处实时显示。</span>
            `;
          }
        }
        listEl.innerHTML = "";
        return;
      }

      if (emptyEl) emptyEl.hidden = true;

      const reversed = [...displayEvents].reverse();
      listEl.innerHTML = reversed.map(ev => {
        const eid = ev.event_id;
        const isShadow = isShadowEvent(ev);
        const cardClass = isShadow ? "swing-item-card is-shadow-card" : "swing-item-card";
        const strokeType = ev.stroke_type || "Swing";
        const strokeText = strokeType === "Forehand" ? "正手 (Forehand)" : (strokeType === "Backhand" ? "反手 (Backhand)" : strokeType);
        const score = ev.swing_score ?? (ev.biomechanics?.swing_score ?? "—");
        const grade = ev.swing_grade || (ev.biomechanics?.swing_grade || "REVIEW_REQUIRED");
        const badgeClass = isShadow ? "badge-shadow" : (score === "—" ? "badge-provisional" : `badge-${grade.toLowerCase()}`);
        const badgeText = isShadow ? "空挥试拍 · 无来球" : (score === "—" ? "诊断参考 · 待教练评分" : `${grade} · ${score}分`);

        const ext = ev.extended_biomechanics || (ev.biomechanics?.extended_biomechanics || {});
        const reasonNames = {
          automatic_rubric_not_independently_validated: "自动评分标准待教练独立标定",
          contact_not_confirmed: "触球为视觉候选推断（待人工确认）",
          missing_observations: "缺少相关专项观测",
          shadow_swing: "空挥试拍"
        };
        const blockersHtml = (ev.scoring_blockers || []).length ? `
          <details style="margin:10px 0; background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.07); border-radius:6px; padding:6px 10px;">
            <summary style="cursor:pointer; color:var(--muted); font-size:12px;">教练五维评审 · 待评定说明 (人工专项)</summary>
            <div style="font-size:12px; margin-top:8px; line-height:1.6;">
              ${ev.scoring_blockers.map(b => `<p style="margin:4px 0;"><strong>${escapeHtml(b.label)}</strong>：${(b.reasons || []).map(r => escapeHtml(reasonNames[r] || r)).join('；')}${b.missing_observations?.length ? `（待测：${escapeHtml(b.missing_observations.join(', '))}）` : ''}</p>`).join('')}
              <div style="color:var(--muted); font-size:11px; margin-top:6px; border-top:1px dashed rgba(255,255,255,0.08); padding-top:6px;">
                💡 五维综合评分（准备、到位、击球、协调、随挥）属于教练人工专项评审；单目视觉已在右侧【5维生物力学技术雷达】中完整透出转肩、引拍、延展、挥速、蹬地等客观表现。
              </div>
            </div>
          </details>` : "";
        const speed = ext.racket_head_speed || {};
        const speedContact = speed.contact_px_s != null ? `${speed.contact_px_s.toFixed(0)} <small>px/s · 候选触球帧<br>原始峰值 ${speed.max_px_s?.toFixed(0) ?? "—"} px/s（未验证）</small>` : "未观测 · km/h 未标定";
        
        const brush = ext.brush_angle || {};
        const brushAngle = qualifiedObservation(brush, 'low_to_high_angle_deg');
        const riseRatio = qualifiedObservation(brush, 'drop_depth_ratio');
        let brushDirection = "";
        if (brushAngle != null) {
          brushDirection = `${brushAngle >= 0 ? "+" : ""}${brushAngle}°`;
        } else if (riseRatio === 0) {
          brushDirection = "平击推进 (无下沉提拉)";
        } else {
          brushDirection = escapeHtml(observationReason(brush));
        }
        const brushText = `${brushDirection}<small> · 上升比 ${riseRatio != null ? escapeHtml(String(riseRatio)) + "x" : "未观测"}</small>`;

        const seqMetric = ev.biomechanics?.metrics?.kinematic_sequence || {};
        const seq = seqMetric.details || ext.kinematic_sequence || {};
        const seqDisplay = kinematicDisplay(seq, seqMetric);

        const leg = ext.leg_drive || {};
        const hipRise = qualifiedObservation(leg, 'drive_px');
        let legText = "";
        if (hipRise != null) {
          legText = `${hipRise} <small>px · 像面髋中心</small>`;
        } else if (leg?.measurement_evidence?.reasons?.includes("below_motion_resolution_guard")) {
          legText = `平立击球 <small>· 无显著下蹲蓄力 (上移极微)</small>`;
        } else {
          legText = escapeHtml(observationReason(leg));
        }

        const advices = ev.coach_advices || (ev.coach_advice ? [ev.coach_advice] : []);
        const coachHtml = advices.length > 0 ? `
          <div class="swing-coach-box">
            <span class="coach-label">AI 教练建议</span>
            ${advices.map(a => `<span class="coach-pill">🎯 ${escapeHtml(a.message || a.code)}</span>`).join("")}
          </div>
        ` : (isShadow ? `
          <div class="swing-coach-box" style="background: rgba(255,255,255,0.02); border-color: rgba(255,255,255,0.06);">
            <span class="coach-label" style="color: var(--muted);">提示</span>
            <span class="coach-pill" style="color: var(--muted); font-size: 11px;">空挥试拍（未检测到触球）· 已免除纠错建议以保持诊断纯净</span>
          </div>
        ` : "");

        const freezeUrl = ev.impact_freeze_url || "";
        const clipUrl = ev.clip_url || "";
        const isPlayerOpen = activeVideoPlayers.has(eid);

        return `
          <div class="${cardClass}" id="swing-card-${eid}">
            <div class="swing-header">
              <div class="swing-title">
                <strong>Event #${eid} · ${strokeText}</strong>
                <span class="badge-grade ${badgeClass}">${badgeText}</span>
              </div>
              <div class="swing-frames">
                第 ${ev.start_frame || 0}–${ev.end_frame || 0} 帧 (触球候选 ${ev.contact_frame ?? "—"}F)
              </div>
            </div>

            ${blockersHtml}
            <div class="swing-telemetry-container">
              <div class="swing-telemetry-grid">
                <div class="telemetry-cell">
                  <span class="t-label">球拍框中心像素速度</span>
                  <span class="t-val">${speedContact}</span>
                </div>
                <div class="telemetry-cell">
                  <span class="t-label">球拍像面轨迹 / 上升比</span>
                  <span class="t-val">${brushText}</span>
                </div>
                <div class="telemetry-cell">
                  <span class="t-label">动力链峰值时序 · 二维参考</span>
                  <span class="t-val" title="${escapeHtml(seqDisplay.explanation)}">
                    <span class="badge-seq ${seqDisplay.badgeClass}">${escapeHtml(seqDisplay.label)}</span>
                    <small>${escapeHtml(seqDisplay.intervals)}</small>
                  </span>
                  <details><summary>查看各段证据与时差范围</summary><small>${escapeHtml(seqDisplay.explanation)}</small></details>
                </div>
                <div class="telemetry-cell">
                  <span class="t-label">髋部像面上移</span>
                  <span class="t-val">${legText}</span>
                </div>
              </div>
              ${generateRadarSvg(ev)}
            </div>

            ${renderEventMeasurements(ev)}
            ${coachHtml}

            <div class="swing-media-row">
              ${freezeUrl ? `
                <div class="media-thumb-box" title="点击放大触球定格" onclick="showFreezeModal('${freezeUrl}')">
                  <img src="${freezeUrl}" class="freeze-thumb" alt="触球定格">
                  <span class="thumb-tag">触球定格 🔍</span>
                </div>
              ` : ""}
              ${clipUrl ? `
                <button class="button play-clip-btn" type="button" onclick="toggleClipPlayer(${eid}, '${clipUrl}')">
                  ▶ 行内慢放
                </button>
                <button class="button" style="font-size: 11px; padding: 6px 12px; background: rgba(0, 240, 255, 0.12); border: 1px solid var(--blue); color: var(--text); cursor: pointer;" type="button" onclick="openClipModal(${eid}, '${clipUrl}', '${escapeHtml(strokeText)}', '${score}')">
                  ⛶ 全屏弹窗复盘
                </button>
              ` : ""}
            </div>

            <div class="inline-clip-player" id="clip-player-${eid}" ${isPlayerOpen ? "" : "hidden"}>
              <video controls loop playsinline class="clip-video-el" src="${isPlayerOpen ? clipUrl : ""}"></video>
              <div class="player-controls-bar">
                <span>慢放倍速：</span>
                <button class="btn-speed active" type="button" onclick="setSpeed(this, ${eid}, 1.0)">1.0x 正常</button>
                <button class="btn-speed" type="button" onclick="setSpeed(this, ${eid}, 0.5)">0.5x 慢放</button>
                <button class="btn-speed" type="button" onclick="setSpeed(this, ${eid}, 0.25)">0.25x 极慢</button>
                <span style="margin: 0 4px;">|</span>
                <button class="btn-frame" type="button" onclick="stepFrame(${eid}, -0.04)">⏮ -1帧</button>
                <button class="btn-frame" type="button" onclick="stepFrame(${eid}, +0.04)">+1帧 ⏭</button>
              </div>
            </div>
          </div>
        `;
      }).join("");

      activeVideoPlayers.forEach(eid => {
        const c = document.getElementById(`clip-player-${eid}`);
        if (c) {
          const v = c.querySelector(".clip-video-el");
          if (v && v.paused) v.play().catch(() => {});
        }
      });
    }

    function handleLogsData() {
      // Compatibility anchor for test slicing if needed
    }
